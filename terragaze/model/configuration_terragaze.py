# Adopted from https://github.com/opengvlab/internvl. Below is the orignial copyright:
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from typing import Dict, Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

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
SUPPORTED_ATTENTION_POOLING_QUERY_TYPES = {
    ATTENTION_POOLING_QUERY_MEAN,
    ATTENTION_POOLING_QUERY_MEAN_LEARNED,
    ATTENTION_POOLING_QUERY_MEAN_GATED_LEARNED,
    ATTENTION_POOLING_QUERY_LEARNED,
}


def normalize_modality_bridge_type(value: Optional[str]) -> str:
    bridge_type = value or MODALITY_BRIDGE_MLP
    if bridge_type not in SUPPORTED_MODALITY_BRIDGES:
        supported = ", ".join(sorted(SUPPORTED_MODALITY_BRIDGES))
        raise ValueError(f"Unsupported modality_bridge_type={bridge_type!r}. Supported values: {supported}.")
    return bridge_type


def normalize_attention_pooling_query_type(value: Optional[str]) -> str:
    query_type = value or ATTENTION_POOLING_QUERY_MEAN
    if query_type not in SUPPORTED_ATTENTION_POOLING_QUERY_TYPES:
        supported = ", ".join(sorted(SUPPORTED_ATTENTION_POOLING_QUERY_TYPES))
        raise ValueError(f"Unsupported attention_pooling_query_type={query_type!r}. Supported values: {supported}.")
    return query_type


logger = logging.get_logger(__name__)


class TeoNextConfig(PretrainedConfig):
    model_type = 'teonext'
    #| Important: is_composition is set to True to indicate that this model is a composition of multiple models
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        use_backbone_lora=0,
        use_llm_lora=0,
        llm_lora_alpha=None,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        use_pixel_shuffle=True,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        add_grounding_special_tokens=True,
        use_bbox_coord_tokens=False,
        bbox_coord_token_max=1000,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        pad2square=False,
        modality_bridge_type=MODALITY_BRIDGE_MLP,
        vision_select_layers=None,
        attention_pooling_h=2,
        attention_pooling_w=2,
        attention_pooling_num_heads=16,
        attention_pooling_head_dim=None,
        attention_pooling_projector_hidden_size=None,
        attention_pooling_query_type=ATTENTION_POOLING_QUERY_MEAN,
        cut_cross_entropy=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['SiglipVisionModel']}
            logger.info('vision_config is None. Initializing the SiglipVisionModel with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['Qwen3ForCausalLM']}
            logger.info('llm_config is None. Initializing the Qwen3ForCausalLM config with default values (`Qwen3ForCausalLM`).')
        
        assert 'architectures' in llm_config, "Should specify architecture in llm_config"

        #| build vision config and llm config
        if not isinstance(vision_config, dict):
            self.vision_config = vision_config
        else:
            architecture = (vision_config.get("architectures") or [None])[0]
            model_type = vision_config.get("model_type")

            #| no achitectures field found in siglip2
            if architecture == "SiglipVisionModel" or model_type in {"siglip", "siglip_vision_model"}:
                from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
                self.vision_config = SiglipVisionConfig(**vision_config)
            elif architecture in {"CLIPVisionModel", "CLIPModel"} or model_type in {"clip", "clip_vision_model"}:
                from transformers.models.clip.configuration_clip import CLIPVisionConfig
                self.vision_config = CLIPVisionConfig(**vision_config)
            else:
                raise ValueError(f"Unsupported vision architecture: arch={architecture}, model_type={model_type}")
        
        if not isinstance(llm_config, dict):
            self.llm_config = llm_config
        else:
            architecture = (llm_config.get("architectures") or [None])[0]
            model_type = llm_config.get("model_type")

            if architecture == "Qwen3MoeForCausalLM" or model_type == "qwen3_moe":
                from transformers import Qwen3MoeConfig
                self.llm_config = Qwen3MoeConfig(**llm_config)
            elif architecture == "Qwen3ForCausalLM" or model_type == "qwen3":
                from transformers import Qwen3Config
                self.llm_config = Qwen3Config(**llm_config)
            elif architecture == "LlamaForCausalLM" or model_type == "llama":
                from transformers import LlamaConfig
                self.llm_config = LlamaConfig(**llm_config)
            else:
                raise ValueError(f"Unsupported llm architecture: arch={architecture}, model_type={model_type}")
        self.text_config = self.llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.llm_lora_alpha = llm_lora_alpha
        self.select_layer = select_layer
        self.modality_bridge_type = normalize_modality_bridge_type(modality_bridge_type)
        if vision_select_layers is None:
            vision_select_layers = [-4, -10] if self.modality_bridge_type == "attention_pooling" else []
        if isinstance(vision_select_layers, int):
            vision_select_layers = [vision_select_layers]
        self.vision_select_layers = list(vision_select_layers)
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.use_pixel_shuffle = use_pixel_shuffle
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.add_grounding_special_tokens = add_grounding_special_tokens
        self.use_bbox_coord_tokens = use_bbox_coord_tokens
        self.bbox_coord_token_max = bbox_coord_token_max
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.pad2square = pad2square
        self.attention_pooling_h = attention_pooling_h
        self.attention_pooling_w = attention_pooling_w
        self.attention_pooling_num_heads = attention_pooling_num_heads
        self.attention_pooling_head_dim = attention_pooling_head_dim
        self.attention_pooling_projector_hidden_size = attention_pooling_projector_hidden_size
        self.attention_pooling_query_type = normalize_attention_pooling_query_type(attention_pooling_query_type)
        self.cut_cross_entropy = cut_cross_entropy
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'modality_bridge_type: {self.modality_bridge_type}')
        logger.info(f'vision_select_layers: {self.vision_select_layers}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        vision_config = getattr(self, "vision_config", None)
        llm_config = getattr(self, "llm_config", None) or getattr(self, "text_config", None)
        output['vision_config'] = vision_config.to_dict() if hasattr(vision_config, "to_dict") else vision_config
        output['llm_config'] = llm_config.to_dict() if hasattr(llm_config, "to_dict") else llm_config
        output['text_config'] = output['llm_config']
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['llm_lora_alpha'] = self.llm_lora_alpha
        output['select_layer'] = self.select_layer
        output['modality_bridge_type'] = self.modality_bridge_type
        output['vision_select_layers'] = self.vision_select_layers
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['use_pixel_shuffle'] = self.use_pixel_shuffle
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['add_grounding_special_tokens'] = self.add_grounding_special_tokens
        output['use_bbox_coord_tokens'] = self.use_bbox_coord_tokens
        output['bbox_coord_token_max'] = self.bbox_coord_token_max
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch
        output['pad2square'] = self.pad2square
        output['attention_pooling_h'] = self.attention_pooling_h
        output['attention_pooling_w'] = self.attention_pooling_w
        output['attention_pooling_num_heads'] = self.attention_pooling_num_heads
        output['attention_pooling_head_dim'] = self.attention_pooling_head_dim
        output['attention_pooling_projector_hidden_size'] = self.attention_pooling_projector_hidden_size
        output['attention_pooling_query_type'] = self.attention_pooling_query_type
        output['cut_cross_entropy'] = self.cut_cross_entropy
        output['tie_word_embeddings'] = self.tie_word_embeddings

        return output
