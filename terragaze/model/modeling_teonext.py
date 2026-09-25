# Adopted from https://github.com/opengvlab/internvl. Below is the orignial copyright:
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoModelForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import load_balancing_loss_func

from teonext.model.configuration_teonext import TeoNextConfig
from teonext.model.configuration_teonext import normalize_attention_pooling_query_type
from teonext.model.conversation import get_conv_template
from teonext.utils.utils import get_decoder_layers

logger = logging.get_logger(__name__)

MODALITY_BRIDGE_MLP = "mlp"
MODALITY_BRIDGE_ATTENTION_POOLING = "attention_pooling"
ATTENTION_POOLING_QUERY_MEAN = "mean"
ATTENTION_POOLING_QUERY_MEAN_LEARNED = "mean_learned"
ATTENTION_POOLING_QUERY_MEAN_GATED_LEARNED = "mean_gated_learned"
ATTENTION_POOLING_QUERY_LEARNED = "learned"
SUPPORTED_MODALITY_BRIDGES = {
    MODALITY_BRIDGE_MLP,
    MODALITY_BRIDGE_ATTENTION_POOLING,
}


def normalize_modality_bridge_type(value: Optional[str]) -> str:
    bridge_type = value or MODALITY_BRIDGE_MLP
    if bridge_type not in SUPPORTED_MODALITY_BRIDGES:
        supported = ", ".join(sorted(SUPPORTED_MODALITY_BRIDGES))
        raise ValueError(f"Unsupported modality_bridge_type={bridge_type!r}. Supported values: {supported}.")
    return bridge_type


def infer_square_grid(num_tokens: int) -> Tuple[int, int]:
    grid = int(math.sqrt(num_tokens))
    if grid * grid != num_tokens:
        raise ValueError(
            f"Vision patch tokens must form a square grid for attention pooling, got {num_tokens} tokens."
        )
    return grid, grid


def compute_num_image_tokens(
    *,
    image_size: int,
    patch_size: int,
    modality_bridge_type: str,
    downsample_ratio: float,
    use_pixel_shuffle: bool,
    pooling_h: int,
    pooling_w: int,
) -> int:
    grid = image_size // patch_size
    if modality_bridge_type == MODALITY_BRIDGE_MLP:
        token_ratio = downsample_ratio ** 2 if use_pixel_shuffle else 1
        return int((grid ** 2) * token_ratio)

    if modality_bridge_type == MODALITY_BRIDGE_ATTENTION_POOLING:
        return math.ceil(grid / pooling_h) * math.ceil(grid / pooling_w)

    raise ValueError(f"Unsupported modality_bridge_type={modality_bridge_type!r}.")


class SwiGLUProjector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.up = nn.Linear(input_size, 2 * hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * x)


class AttentionPoolingConnector(nn.Module):
    """Jina-VLM style 2D local attention-pooling connector."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        pooling_h: int = 2,
        pooling_w: int = 2,
        num_heads: int = 16,
        head_dim: Optional[int] = None,
        projector_hidden_size: Optional[int] = None,
        query_type: str = ATTENTION_POOLING_QUERY_MEAN,
        query_grid_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        if pooling_h <= 0 or pooling_w <= 0:
            raise ValueError("pooling_h and pooling_w must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size
        self.pooling_h = pooling_h
        self.pooling_w = pooling_w
        self.num_heads = num_heads
        self.head_dim = head_dim or intermediate_size // num_heads
        self.query_type = normalize_attention_pooling_query_type(query_type)
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if self.query_type != ATTENTION_POOLING_QUERY_MEAN:
            if query_grid_size is None:
                raise ValueError("query_grid_size is required when attention_pooling_query_type uses learned queries.")
            query_h, query_w = query_grid_size
            if query_h <= 0 or query_w <= 0:
                raise ValueError("query_grid_size dimensions must be positive.")
            self.query_grid_size = (query_h, query_w)
            self.query_embed = nn.Parameter(torch.empty(query_h * query_w, input_size))
            nn.init.trunc_normal_(self.query_embed, mean=0.0, std=0.02)
            if self.query_type == ATTENTION_POOLING_QUERY_MEAN_GATED_LEARNED:
                self.query_gate = nn.Parameter(torch.zeros(1))
            else:
                self.query_gate = None
        else:
            self.query_grid_size = None
            self.query_embed = None
            self.query_gate = None

        attention_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(input_size, attention_size, bias=False)
        self.kv_proj = nn.Linear(input_size, 2 * attention_size, bias=False)
        self.out_proj = nn.Linear(attention_size, intermediate_size, bias=True)
        self.q_bias = nn.Parameter(torch.zeros(attention_size))
        self.k_bias = nn.Parameter(torch.zeros(attention_size))
        self.v_bias = nn.Parameter(torch.zeros(attention_size))
        self.projector = SwiGLUProjector(
            intermediate_size,
            projector_hidden_size or 3 * output_size,
            output_size,
        )

    def _pool_blocks(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        batch_size, num_tokens, hidden_size = image_features.shape
        grid_h, grid_w = infer_square_grid(num_tokens)
        pad_h = (-grid_h) % self.pooling_h
        pad_w = (-grid_w) % self.pooling_w

        image_features = image_features.reshape(batch_size, grid_h, grid_w, hidden_size)
        if pad_h or pad_w:
            image_features = F.pad(image_features, (0, 0, 0, pad_w, 0, pad_h))

        pooled_h = (grid_h + pad_h) // self.pooling_h
        pooled_w = (grid_w + pad_w) // self.pooling_w
        image_features = image_features.reshape(
            batch_size,
            pooled_h,
            self.pooling_h,
            pooled_w,
            self.pooling_w,
            hidden_size,
        )
        image_features = image_features.permute(0, 1, 3, 2, 4, 5).contiguous()
        blocks = image_features.reshape(
            batch_size * pooled_h * pooled_w,
            self.pooling_h * self.pooling_w,
            hidden_size,
        )
        return blocks, pooled_h, pooled_w

    def _learned_query(self, batch_size: int, pooled_h: int, pooled_w: int, dtype, device) -> torch.Tensor:
        query = self.query_embed.to(device=device)
        query_h, query_w = self.query_grid_size
        if (pooled_h, pooled_w) != (query_h, query_w):
            query = query.reshape(1, query_h, query_w, self.input_size).permute(0, 3, 1, 2)
            query = F.interpolate(query.float(), size=(pooled_h, pooled_w), mode="bilinear", align_corners=False)
            query = query.permute(0, 2, 3, 1).reshape(pooled_h * pooled_w, self.input_size)

        query = query.to(dtype=dtype, device=device)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        return query.reshape(batch_size * pooled_h * pooled_w, 1, self.input_size)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        batch_size = image_features.shape[0]
        blocks, pooled_h, pooled_w = self._pool_blocks(image_features)
        mean_query = blocks.mean(dim=1, keepdim=True)
        if self.query_type == ATTENTION_POOLING_QUERY_MEAN:
            query = mean_query
        else:
            learned_query = self._learned_query(
                batch_size,
                pooled_h,
                pooled_w,
                dtype=mean_query.dtype,
                device=mean_query.device,
            )
            if self.query_type == ATTENTION_POOLING_QUERY_MEAN_LEARNED:
                query = mean_query + learned_query
            elif self.query_type == ATTENTION_POOLING_QUERY_MEAN_GATED_LEARNED:
                query = mean_query + self.query_gate.tanh().to(dtype=mean_query.dtype) * learned_query
            elif self.query_type == ATTENTION_POOLING_QUERY_LEARNED:
                query = learned_query
            else:
                raise ValueError(f"Unsupported attention_pooling_query_type={self.query_type!r}.")

        q = F.linear(query, self.q_proj.weight, self.q_bias)
        kv = F.linear(blocks, self.kv_proj.weight, torch.cat((self.k_bias, self.v_bias)))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        pooled = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        pooled = pooled.transpose(1, 2).reshape(blocks.shape[0], 1, self.num_heads * self.head_dim)
        pooled = self.out_proj(pooled)

        pooled = pooled.reshape(batch_size, pooled_h * pooled_w, self.intermediate_size)
        return self.projector(pooled)

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

def _detect_vision_backend(config) -> Optional[str]:
    architecture = (getattr(config, "architectures", None) or [None])[0]
    model_type = getattr(config, "model_type", None)
    if architecture == "SiglipVisionModel" or model_type in {"siglip", "siglip_vision_model"}:
        return "siglip"
    elif architecture in {"CLIPVisionModel", "CLIPModel"} or model_type in {"clip", "clip_vision_model"}:
        return "clip"
    return None

class TeoNextModel(PreTrainedModel):
    config_class = TeoNextConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False
    _no_split_modules = [
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
        "SiglipMultiheadAttentionPoolingHead",
        "CLIPVisionEmbeddings",
        "CLIPEncoderLayer",
        "Qwen3DecoderLayer",
        "LlamaDecoderLayer",
        "AttentionPoolingConnector",
    ]
    _tp_plan = ''

    def __init__(self, config: TeoNextConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.vision_select_layers = list(getattr(config, "vision_select_layers", []) or [])
        self.template = config.template
        self.use_pixel_shuffle = config.use_pixel_shuffle
        self.modality_bridge_type = normalize_modality_bridge_type(getattr(config, "modality_bridge_type", None))
        self.downsample_ratio = config.downsample_ratio
        if self.modality_bridge_type == MODALITY_BRIDGE_ATTENTION_POOLING and self.use_pixel_shuffle:
            raise ValueError("attention_pooling modality bridge does not support use_pixel_shuffle=True.")
        if self.modality_bridge_type == MODALITY_BRIDGE_ATTENTION_POOLING and not self.vision_select_layers:
            self.vision_select_layers = [-4, -10]
            self.config.vision_select_layers = self.vision_select_layers
        self.num_image_token = self.compute_num_image_tokens(image_size)
        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'use_pixel_shuffle: {self.use_pixel_shuffle}')
        logger.info(f'modality_bridge_type: {self.modality_bridge_type}')

        #| if compose vision_model and language_model from scratch, this branch will be activated
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if _detect_vision_backend(config.vision_config) == "siglip":
                from transformers.models.siglip.modeling_siglip import SiglipVisionModel
                self.vision_model = SiglipVisionModel(config.vision_config)
            elif _detect_vision_backend(config.vision_config) == "clip":
                from transformers.models.clip.modeling_clip import CLIPVisionModel
                self.vision_model = CLIPVisionModel(config.vision_config)
            else:
                raise NotImplementedError(f"Unsupported vision model architecture in config: {config.vision_config}")

        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = AutoModelForCausalLM.from_config(config.llm_config)
            logger.info(f"language_model type: {type(self.language_model)}")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = self.build_modality_bridge(vit_hidden_size, llm_hidden_size)

        self.num_samples = 0
        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    @staticmethod
    def _all_reduce_loss_denominator(denominator):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.AVG)
        return denominator

    def _cut_cross_entropy_loss(self, logits, labels, loss_weight=None):
        vocab_size = self.language_model.config.vocab_size
        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        if loss_weight is not None:
            loss_weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=logits.device)
            shift_weights = loss_weight[..., 1:]
        else:
            effective_tokens = (labels != -100).sum(dim=-1).to(dtype=torch.float32, device=logits.device)
            sample_weights = torch.zeros_like(effective_tokens)
            has_effective_tokens = effective_tokens > 0
            sample_weights[has_effective_tokens] = effective_tokens[has_effective_tokens].rsqrt()
            weights = torch.where(labels != -100, sample_weights.unsqueeze(1), 0.0)
            shift_weights = weights[..., 1:]

        shift_weights = shift_weights.to(dtype=torch.float32, device=logits.device)
        shift_weights_sum = shift_weights.reshape(-1).sum()
        shift_weights_sum = self._all_reduce_loss_denominator(shift_weights_sum)

        active = (shift_labels != -100) & (shift_weights != 0)
        active_logits = shift_logits[active]
        active_labels = shift_labels[active]
        active_weights = shift_weights[active]

        if active_labels.numel() == 0:
            return logits.sum() * 0.0

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(active_logits, active_labels)
        return (loss * active_weights).sum() / shift_weights_sum

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        """Resize position embeddings for SigLIP vision model via bicubic interpolation.

        SigLIP uses nn.Embedding (shape [num_patches, embed_dim]) without a CLS token,
        unlike InternViT which uses nn.Parameter (shape [1, num_patches+1, embed_dim]) with CLS.
        This method handles the SigLIP-specific structure.

        Args:
            old_size: Original image resolution the model was pretrained on.
            new_size: Target image resolution for fine-tuning.
            patch_size: Patch size of the vision model.
        """
        if _detect_vision_backend(self.config.vision_config) == "siglip":
            #| Navigate through SigLIP's nested structure:
            #| model.vision_model -> SiglipVisionModel
            #|   .vision_model   -> SiglipVisionTransformer
            #|     .embeddings   -> SiglipVisionEmbeddings
            #|       .position_embedding -> nn.Embedding(num_patches, embed_dim)
            embeddings = self.vision_model.vision_model.embeddings
            old_pos_emb = embeddings.position_embedding.weight.data  # [num_patches, embed_dim]
            embed_dim = old_pos_emb.shape[-1]

            old_grid = old_size // patch_size
            new_grid = new_size // patch_size

            # Reshape to 2D spatial grid, interpolate, then flatten back
            # [num_patches, embed_dim] -> [1, old_grid, old_grid, embed_dim] -> [1, embed_dim, old_grid, old_grid]
            pos_emb_2d = old_pos_emb.float().reshape(1, old_grid, old_grid, embed_dim).permute(0, 3, 1, 2)
            pos_emb_2d = F.interpolate(pos_emb_2d, size=(new_grid, new_grid), mode='bicubic', align_corners=False)
            # [1, embed_dim, new_grid, new_grid] -> [new_grid*new_grid, embed_dim]
            new_pos_emb = pos_emb_2d.permute(0, 2, 3, 1).reshape(-1, embed_dim).to(old_pos_emb.dtype)

            new_num_patches = new_grid * new_grid
            new_position_embedding = nn.Embedding(new_num_patches, embed_dim)
            new_position_embedding.weight = nn.Parameter(new_pos_emb)
            embeddings.position_embedding = new_position_embedding

            # Update related attributes
            embeddings.num_patches = new_num_patches
            embeddings.num_positions = new_num_patches
            embeddings.image_size = new_size
            #| Rebuild position_ids buffer for the new grid size
            embeddings.register_buffer(
                'position_ids', torch.arange(new_num_patches).expand((1, -1)), persistent=False
            )
            logger.info(f'Resized SigLIP position embeddings: {old_size}({old_grid}x{old_grid}) -> {new_size}({new_grid}x{new_grid})')
            return 
        elif _detect_vision_backend(self.config.vision_config) == "clip":
            embeddings = self.vision_model.vision_model.embeddings
            old_pos_emb = embeddings.position_embedding.weight.data
            embed_dim = old_pos_emb.shape[-1]
            old_grid = old_size // patch_size
            new_grid = new_size // patch_size

            cls_pos_emb = old_pos_emb[:1]
            patch_pos_emb = old_pos_emb[1:]
            patch_pos_emb = patch_pos_emb.float().reshape(1, old_grid, old_grid, embed_dim).permute(0, 3, 1, 2)
            patch_pos_emb = F.interpolate(
                patch_pos_emb, size=(new_grid, new_grid), mode="bicubic", align_corners=False
            )
            patch_pos_emb = patch_pos_emb.permute(0, 2, 3, 1).reshape(-1, embed_dim).to(old_pos_emb.dtype)
            new_pos_emb = torch.cat([cls_pos_emb, patch_pos_emb], dim=0)

            new_num_patches = new_grid * new_grid
            new_num_positions = new_num_patches + 1
            new_position_embedding = nn.Embedding(new_num_positions, embed_dim)
            new_position_embedding.weight = nn.Parameter(new_pos_emb)
            embeddings.position_embedding = new_position_embedding
            embeddings.num_patches = new_num_patches
            embeddings.num_positions = new_num_positions
            embeddings.image_size = new_size
            embeddings.register_buffer(
                "position_ids", torch.arange(new_num_positions).expand((1, -1)), persistent=False
            )
            logger.info(
                f"Resized CLIP position embeddings: {old_size}({old_grid}x{old_grid}) -> {new_size}({new_grid}x{new_grid})"
            )
            return
        else:
            raise NotImplementedError("Unsupported vision model architecture for resizing position embeddings.")

    def compute_num_image_tokens(self, image_size):
        return compute_num_image_tokens(
            image_size=image_size,
            patch_size=self.patch_size,
            modality_bridge_type=self.modality_bridge_type,
            downsample_ratio=self.downsample_ratio,
            use_pixel_shuffle=self.use_pixel_shuffle,
            pooling_h=getattr(self.config, "attention_pooling_h", 2),
            pooling_w=getattr(self.config, "attention_pooling_w", 2),
        )

    def build_modality_bridge(self, vit_hidden_size, llm_hidden_size):
        if self.modality_bridge_type == MODALITY_BRIDGE_ATTENTION_POOLING:
            image_size = self.config.force_image_size or self.config.vision_config.image_size
            grid = image_size // self.patch_size
            pooling_h = getattr(self.config, "attention_pooling_h", 2)
            pooling_w = getattr(self.config, "attention_pooling_w", 2)
            return AttentionPoolingConnector(
                input_size=vit_hidden_size * len(self.vision_select_layers),
                intermediate_size=vit_hidden_size,
                output_size=llm_hidden_size,
                pooling_h=pooling_h,
                pooling_w=pooling_w,
                num_heads=getattr(self.config, "attention_pooling_num_heads", 16),
                head_dim=getattr(self.config, "attention_pooling_head_dim", None),
                projector_hidden_size=getattr(self.config, "attention_pooling_projector_hidden_size", None),
                query_type=getattr(self.config, "attention_pooling_query_type", ATTENTION_POOLING_QUERY_MEAN),
                query_grid_size=(math.ceil(grid / pooling_h), math.ceil(grid / pooling_w)),
            ).to(torch.bfloat16)

        if self.modality_bridge_type != MODALITY_BRIDGE_MLP:
            raise ValueError(f"Unsupported modality_bridge_type={self.modality_bridge_type!r}.")

        vit_mlp_input_dim = vit_hidden_size * int(1 / self.downsample_ratio) ** 2 if self.use_pixel_shuffle else vit_hidden_size
        return nn.Sequential(
            nn.LayerNorm(vit_mlp_input_dim), #| dimension after optional pixel shuffle
            nn.Linear(vit_mlp_input_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        ).to(torch.bfloat16)

    def forward(
            self,
            pixel_values: torch.FloatTensor, #| from LazySupervisedDataset.__getitem__
            input_ids: torch.LongTensor = None, #| from LazySupervisedDataset.__getitem__
            attention_mask: Optional[torch.Tensor] = None, #| from LazySupervisedDataset.__getitem__
            position_ids: Optional[torch.LongTensor] = None, #| from LazySupervisedDataset.__getitem__
            image_flags: Optional[torch.LongTensor] = None, #| from LazySupervisedDataset.__getitem__
            past_key_values: Optional[List[torch.FloatTensor]] = None, #| same as llava implementation from trainer
            labels: Optional[torch.LongTensor] = None, #| from LazySupervisedDataset.__getitem__
            use_cache: Optional[bool] = None, #| same as llava implementation from trainer
            output_attentions: Optional[bool] = None, #| same as llava implementation from trainer
            output_hidden_states: Optional[bool] = None, #| same as llava implementation from trainer
            return_dict: Optional[bool] = None, #| same as llava implementation from trainer
            statistics: Optional[torch.LongTensor] = None, #| from packed_collate_fn in dataloader
            cu_seqlens: Optional[torch.LongTensor] = None, #| from packed_collate_fn in dataloader
            loss_weight: Optional[List] = None, #| from packed_collate_fn in dataloader
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #| How to pack results, whether a tuple or a dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1) #| only one dimension
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone() #| [Batch, Seq, Hidden]

        #| default to use pixel_shuffle, may need to turn off
        vit_embeds = self.extract_feature(pixel_values) #| visual_features after pixel_shuffle, mlp
        vit_embeds = vit_embeds[image_flags == 1] #| only take real images [num_images, seq_len, Hidden]
        vit_batch_size = pixel_values.shape[0] #| all images dynamic patches

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            # print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_effective_tokens, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}, {num_effective_tokens=}')

        ignore = False
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        if vit_embeds.shape[-1] != C:
            raise ValueError(
                f"Visual feature hidden size ({vit_embeds.shape[-1]}) does not match LLM hidden size ({C})."
            )

        vit_embeds = vit_embeds.reshape(-1, C)
        selected_count = int(selected.sum().item())
        if selected_count != vit_embeds.shape[0]:
            raise ValueError(
                "Image token mismatch: "
                f"found {selected_count} IMG_CONTEXT tokens but extracted {vit_embeds.shape[0]} visual tokens. "
                f"num_image_token={self.num_image_token}, modality_bridge_type={self.modality_bridge_type}."
            )

        #| merge vit_embeds into input_embeds
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)

        #| corporate with qwen3_flash_monkey_patch
        #| set cu_seqlens for flash attention
        #| TODO: need to simplify
        #| no matter llm type
        for layer in get_decoder_layers(self.language_model):
            if hasattr(layer, "self_attn"):
                layer.self_attn.cu_seqlens = cu_seqlens

        outputs = self.language_model(
            inputs_embeds=input_embeds, #| not input_ids, embeded manually
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        logits = outputs.logits

        loss = None
        aux_loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            if getattr(self.config, "cut_cross_entropy", False):
                loss = self._cut_cross_entropy_loss(logits, labels, loss_weight=loss_weight)
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous() #| build logits without last token(not used to predict)
                shift_labels = labels[..., 1:].contiguous() #| build labels without first token(not predicted)
                shift_weights = loss_weight[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none') #| reduction='none' means return loss for each token, not mean loss
                shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_weights = shift_weights.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_weights = shift_weights.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                shift_weights_sum = shift_weights.sum()
                torch.distributed.all_reduce(shift_weights_sum, op=torch.distributed.ReduceOp.AVG)

                loss = loss * shift_weights
                loss = loss.sum() / shift_weights_sum

        elif labels is not None: #| compute loss weight along the batch dimension
            if getattr(self.config, "cut_cross_entropy", False):
                loss = self._cut_cross_entropy_loss(logits, labels)
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if (shift_labels == -100).all():
                    ignore = True
                    shift_labels = shift_labels * 0

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                loss_weight = (labels != -100).sum(dim=-1).float()
                loss_weight = 1 / loss_weight.sqrt()
                loss_weight = torch.where(labels != -100, loss_weight.unsqueeze(1), 0.0)

                shift_weights = loss_weight[..., 1:].contiguous()
                shift_weights = shift_weights.view(-1)
                shift_weights = shift_weights.to(shift_logits.device)
                shift_weights_sum = shift_weights.sum()
                torch.distributed.all_reduce(shift_weights_sum, op=torch.distributed.ReduceOp.AVG)

                loss = loss * shift_weights
                loss = loss.sum() / shift_weights_sum

            # debug_input = self.tokenizer.batch_decode(input_ids.reshape(B, N), skip_special_tokens=False)[0]
            # debug_label = self.tokenizer.batch_decode(torch.where(labels >= 0, labels, self.tokenizer.pad_token_id), skip_special_tokens=False)[0]

            # debug_input = debug_input.replace("<IMG_CONTEXT>", "")
            # debug_input = debug_input.replace(self.tokenizer.pad_token, "")
            # debug_label = debug_label.replace(self.tokenizer.pad_token, "")

            # if torch.distributed.get_rank() == 0:
            #     print(
            #         f'[Debug]\n'
            #         f'input ({input_ids.reshape(B, N).shape}): {debug_input}\n'
            #         f'label ({labels.shape}): {debug_label}\n'
            #         f'pad_token: {self.tokenizer.pad_token}\n'
            #         f'[/Debug]\n\n'
            #     )

        #| If MoE Model, compute auxiliary loss
        if getattr(outputs, 'router_logits', None) is not None:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.language_model.num_experts,
                self.language_model.num_experts_per_tok,
                attention_mask,
            )

            if loss is not None:
                if torch.distributed.get_rank() == 0:
                    print(f"[Debug] lm_loss: {loss.item()}, aux_loss: {aux_loss.item()},  weighted_aux_loss: {self.language_model.router_aux_loss_coef * aux_loss.item()}")

                loss = loss + self.language_model.router_aux_loss_coef * aux_loss.to(loss.device)

        if ignore and loss is not None:
            print(f"[Debug] ignore curr loss")
            loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if aux_loss is not None:
            return MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _remove_non_patch_tokens(self, vit_embeds):
        if _detect_vision_backend(self.config.vision_config) == "siglip":
            return vit_embeds #| already in [num_images, seq_len, hidden] format
        elif _detect_vision_backend(self.config.vision_config) == "clip":
            return vit_embeds[:, 1:, :] #| remove [cls] token, only patch tokens are used for fusion
        else:
            raise NotImplementedError(f"Unsupported vision model architecture in config: {self.config.vision_config}")

    def _extract_mlp_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]

        vit_embeds = self._remove_non_patch_tokens(vit_embeds)
        
        #| TODO: current siglip2-so400M-14-384 has 729 tokens can not be divided by 4
        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1) #| [num_images, h, w, c]
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) #| [num_images, h * scale, w * scale, c // (scale ** 2)]
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) #| [num_images, seqence_len, c // (scale ** 2)]
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _extract_attention_pooling_feature(self, pixel_values):
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        features = []
        for layer_idx in self.vision_select_layers:
            features.append(self._remove_non_patch_tokens(outputs.hidden_states[layer_idx]))
        vit_embeds = torch.cat(features, dim=-1)
        return self.mlp1(vit_embeds)

    def extract_feature(self, pixel_values):
        if self.modality_bridge_type == MODALITY_BRIDGE_ATTENTION_POOLING:
            return self._extract_attention_pooling_feature(pixel_values)
        return self._extract_mlp_feature(pixel_values)

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)

        sep = template.sep.strip() if template.sep2 is None else template.sep2.strip()
        eos_token_id = tokenizer.convert_tokens_to_ids(sep)

        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=False)
        responses = [response.split(sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        #| if first conversation turn
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message

        #| TODO: does this logic common enough for different cases?
        sep = template.sep.strip() if template.sep2 is None else template.sep2.strip()
        eos_token_id = tokenizer.convert_tokens_to_ids(sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
        response = response.split(sep)[0].strip()

        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print + response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            selected_count = int(selected.sum().item())
            if selected_count == 0:
                raise ValueError("No IMG_CONTEXT tokens found in input_ids for image generation.")
            if vit_embeds.shape[-1] != C:
                raise ValueError(
                    f"Visual feature hidden size ({vit_embeds.shape[-1]}) does not match LLM hidden size ({C})."
                )
            vit_embeds = vit_embeds.reshape(-1, C)
            if selected_count != vit_embeds.shape[0]:
                raise ValueError(
                    "Image token mismatch: "
                    f"found {selected_count} IMG_CONTEXT tokens but extracted {vit_embeds.shape[0]} visual tokens. "
                    f"num_image_token={self.num_image_token}, modality_bridge_type={self.modality_bridge_type}."
                )
            input_embeds[selected] = vit_embeds.to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
