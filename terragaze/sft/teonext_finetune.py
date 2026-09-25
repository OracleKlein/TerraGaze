# Adopted from https://github.com/opengvlab/internvl. Below is the orignial copyright:
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import json
import random
import logging
import warnings
import traceback

from copy import deepcopy
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import numpy as np
import transformers

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer,
    HfArgumentParser, TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers.utils.logging import enable_default_handler, enable_explicit_format, set_verbosity
from transformers.models.siglip import SiglipVisionConfig, SiglipVisionModel
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from teonext.utils.dist_utils import destroy_dist_process_group, init_dist
from teonext.utils.utils import apply_llm_lora
from teonext.utils.utils import expand2square
from teonext.utils.utils import has_flash_attn

from teonext.model import (
    TeoNextConfig, 
    TeoNextModel,
)
from teonext.patch import (
    concat_pad_data_collator,
    replace_qwen3_attention_class,
)
from teonext.utils.constants import (
    GROUNDING_SPECIAL_TOKENS,
    IMAGE_SPECIAL_TOKENS,
    IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
)
from teonext.utils.bbox import (
    bbox_coord_tokens,
    convert_conversation_bboxes_to_coord_tokens,
)
from teonext.utils.utils import sort_media_by_timestamps
from teonext.sft.dataset import (
    ConcatDataset,
    dynamic_preprocess, jpeg_degrade_functions, qualities,
    preprocess_pretrain, preprocess_internvl2_5, preprocess_fastchat_chatml
)
from teonext.sft.dataset_packed import PackedDataset, packed_collate_fn
from teonext.sft.trainer import TeoNextTrainer, TeoNextTrainingArguments

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def _detect_vision_backend(vision_path):
    config_path = os.path.join(vision_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    model_type = raw_config.get("model_type")
    if model_type == "siglip":
        return "siglip"
    if model_type == "clip":
        return "clip"
    raise ValueError(f"Unsupported vision backbone config model_type={model_type} from {config_path}")

def _resolve_attention_implementation():
    return 'flash_attention_2' if has_flash_attn else 'eager'


def _apply_dtype_policy(model, compute_dtype):
    """Apply a stable mixed-precision policy similar to train.py.

    - Keep normalization modules in fp32 for numeric stability.
    - Keep lm_head/embed_tokens in fp32 for fp16 stability, and bf16 for bf16 runs.
    - Keep LoRA layers in bf16 for bf16 runs (fp16 runs keep existing precision).
    """
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:
        LoraLayer = tuple()

    for name, module in model.named_modules():
        name_lower = name.lower()

        if isinstance(module, LoraLayer) and compute_dtype == torch.bfloat16:
            module.to(torch.bfloat16)

        if 'norm' in name_lower:
            module.to(torch.float32)

        if ('lm_head' in name_lower or 'embed_tokens' in name_lower) and hasattr(module, 'weight'):
            if compute_dtype == torch.float16 and module.weight.dtype != torch.float32:
                module.to(torch.float32)
            elif compute_dtype == torch.bfloat16 and module.weight.dtype == torch.float32:
                module.to(torch.bfloat16)

def _maybe_set_wandb_run_name(training_args):
    report_to = getattr(training_args, 'report_to', []) or []
    if isinstance(report_to, str):
        report_to = [report_to]
    if 'wandb' not in report_to or getattr(training_args, 'run_name', None):
        return

    logging_dir = os.path.normpath(training_args.logging_dir)
    output_name = os.path.basename(os.path.dirname(os.path.dirname(logging_dir)))
    run_name = f'{output_name}-{os.path.basename(logging_dir)}'
    object.__setattr__(training_args, 'run_name', run_name)
    logger.info(f'Set wandb run_name to: {run_name}')


def _unwrap_model(model):
    while hasattr(model, 'module'):
        model = model.module
    return model


def _set_llm_use_cache(model, use_cache):
    model = _unwrap_model(model)
    if not hasattr(model.language_model.config, 'use_cache'):
        raise AttributeError('language_model.config must define use_cache.')
    if not hasattr(model.config.llm_config, 'use_cache'):
        raise AttributeError('model.config.llm_config must define use_cache.')
    model.language_model.config.use_cache = use_cache
    model.config.llm_config.use_cache = use_cache


class SaveImageProcessorCallback(TrainerCallback):
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def on_save(self, args, state, control, **kwargs):
        if self.image_processor is None or not getattr(args, 'should_save', True):
            return control

        checkpoint_dir = os.path.join(
            args.output_dir,
            f'{PREFIX_CHECKPOINT_DIR}-{state.global_step}',
        )
        self.image_processor.save_pretrained(checkpoint_dir)
        return control


def parse_vision_select_layers(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None

    layers = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            raise ValueError(f"Invalid vision_select_layers={value!r}: empty layer index.")
        try:
            layers.append(int(item))
        except ValueError as exc:
            raise ValueError(
                f"Invalid vision_select_layers={value!r}: each comma-separated item must be an integer."
            ) from exc
    return layers


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained modality bridge weights.'}
    )
    modality_bridge_type: str = field(
        default='attention_pooling',
        metadata={'help': 'Modality bridge type: mlp or attention_pooling. New training defaults to attention_pooling.'}
    )
    attention_pooling_query_type: str = field(
        default='mean',
        metadata={
            'help': (
                'Attention-pooling query type: mean, mean_learned, mean_gated_learned, or learned. '
                'mean preserves the legacy connector behavior.'
            )
        },
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the modality bridge (mlp1). Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    vision_select_layers: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                'Comma-separated ViT hidden-state layers for attention_pooling, for example "-4,-10". '
                'When unset, TeoNextConfig keeps its bridge-specific default.'
            )
        },
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    llm_lora_alpha: Optional[int] = field(
        default=None,
        metadata={'help': 'Set the LLM LoRA alpha. When unset, defaults to 2 * use_llm_lora for backward compatibility.'}
    )
    merge_llm_lora_before_save: bool = field(
        default=False,
        metadata={'help': 'Merge LLM LoRA weights into the base language model before saving the final checkpoint.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_fast_image_processor: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the image processor.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )
    cut_cross_entropy: Optional[bool] = field(
        default=None,
        metadata={
            'help': (
                'When set, override model.config.cut_cross_entropy. '
                'When unset, keep the checkpoint/config value.'
            )
        },
    )

@dataclass
class DataArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=384,
        metadata={'help': 'Set the desired size for the image. Default is 384.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    use_pixel_shuffle: bool = field(
        default=False,
        metadata={'help': 'Use pixel shuffle to reduce the number of visual tokens. Default is True.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internvl2_5', metadata={'help': 'Prompt style for a conversation.'}
    )
    add_grounding_special_tokens: bool = field(
        default=True,
        metadata={'help': 'Whether to add dedicated grounding tokens like <quad>, <ref>, and <box>.'},
    )
    use_bbox_coord_tokens: bool = field(
        default=False,
        metadata={'help': 'Convert bracket bbox text like [0, 100, 3, 900] to coordinate tokens like <0><100><3><900>.'},
    )
    bbox_coord_token_max: int = field(
        default=1000,
        metadata={'help': 'Maximum inclusive coordinate token value for bbox coord-token mode.'},
    )
    #| InternVL uses a meta file to store all dataset information
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution for single-image samples only.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail when single-image dynamic preprocessing produces multiple patches.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches for single-image dynamic preprocessing. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches for single-image dynamic preprocessing. Default is 12.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=32,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=False,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    split_annotations: bool = field(
        default=False,
        metadata={'help': 'Whether to split annotations to save memory usage. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        ds_name,
        num_image_token,
        image_size=384,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        group_by_modality_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        repeat_time=1,
        image_processor=None,
        split_annotations=False,
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        use_bbox_coord_tokens=False,
        bbox_coord_token_max=1000,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        # logger.info(f'[Dataset] num_image_token: {num_image_token}')
        # logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        # logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        # logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode #| same as use_packed_ds, means whether use multiple workers to load data
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle #| same as use_packed_ds
        # TODO: quick resume
        self._state_dict = {}

        # logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.split_annotations = split_annotations #| determine data_rank and data_world_size

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                repeat_time = int(repeat_time)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        if self.split_annotations:
            total_lines = len(self.raw_data)
            # logger.info(f'world_size: {self.world_size}, rank: {self.rank}, total_lines: {total_lines}')
            lines_per_rank = total_lines // self.world_size  # Number of lines each rank should process
            lines_per_rank = max(1, lines_per_rank)

            # Calculate the start and end line numbers for the current rank
            start_line = lines_per_rank * self.rank  # Starting line for the current rank
            end_line = start_line + lines_per_rank  # Ending line for the current rank

            # Assign the appropriate lines to the current rank
            self.raw_data = self.raw_data[start_line:end_line]

        #| Create a private random number generator for this dataset, notice that this random number between
        #| different workers is same, but this only used as shuffle seed
        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle: #| use_packed_ds=True -> force_shuffle=True
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.group_by_length = group_by_length
        self.group_by_modality_length = group_by_modality_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.image_processor = image_processor
        self.num_fake_dump = 0
        self.use_bbox_coord_tokens = use_bbox_coord_tokens
        self.bbox_coord_token_max = bbox_coord_token_max

        if self.image_processor is None:
            raise ValueError(f'[{self.ds_name}] image_processor is required for Siglip training.')
        logger.info(f'[{self.ds_name}] image transform backend: hf_image_processor ({type(self.image_processor).__name__})')

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length or self.group_by_modality_length:
            self.length = []
            self.modality_length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                token_length = self._estimate_grouping_length(data_item)
                self.length.append(token_length)
                self.modality_length.append(token_length if self._has_image(data_item) else -token_length)

    def __len__(self):
        return len(self.raw_data)

    def _has_image(self, data_item):
        return 'image' in data_item and len(data_item['image']) != 0

    def _estimate_grouping_length(self, data_item):
        if 'length' in data_item and not self.use_bbox_coord_tokens:
            return data_item['length']

        conversations = self._prepare_conversations(data_item['conversations'])
        conversations = '\n'.join([temp['value'] for temp in conversations])
        token_length = self.tokenizer(
            conversations, return_tensors='pt', padding=False, truncation=False,
        ).input_ids.size(1)
        if self._has_image(data_item):
            image_value = data_item['image']
            image_count = len(image_value) if isinstance(image_value, list) else 1
            if self.dynamic_image_size and image_count == 1:
                max_image_patches = self.max_dynamic_patch + int(self.use_thumbnail)
            else:
                max_image_patches = 1
            token_length += self.num_image_token * max_image_patches * image_count
        return token_length

    def _prepare_conversations(self, conversations):
        if not self.use_bbox_coord_tokens:
            return deepcopy(conversations)
        return convert_conversation_bboxes_to_coord_tokens(
            conversations,
            max_coord=self.bbox_coord_token_max,
            sample_hint=f"dataset={self.ds_name}",
        )

    #| use_pretrain is determined by whether this data item's first round of conversation has 'from' == 'pretrain'
    #| In this case, conv_style passed in config sh is not activated, fallback to pretrain template instead, generally speaking,
    #| pretrain stage uses Image Captioning task, and usually not using sft template to wrap the input image and text
    def get_preprocess_function(self, use_pretrain=False):
        # Select the appropriate preprocessing function based on the template name
        if use_pretrain:
            return preprocess_pretrain

        if self.template_name == 'internvl2_5':
            return preprocess_internvl2_5

        if self.template_name == "TinyLlama":
            return preprocess_fastchat_chatml

        raise NotImplementedError(f'Unsupported template: {self.template_name}')

    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        target_path = os.path.join(self.root, image_path)
        if os.path.exists(target_path):
            return target_path
            
        # Fallback to other common extensions if the specific file does not exist
        base, _ = os.path.splitext(target_path)
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF']:
            test_path = base + ext
            if os.path.exists(test_path):
                return test_path
                
        return target_path

    def _get_pad_color(self):
        image_mean = getattr(self.image_processor, 'image_mean', None)
        if image_mean is None:
            return (127, 127, 127)
        if isinstance(image_mean, (int, float)):
            image_mean = [image_mean] * 3
        try:
            return tuple(int(x * 255) for x in image_mean[:3])
        except Exception:
            return (127, 127, 127)

    def get_transform(self):
        pad_color = self._get_pad_color()

        def transform(image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.is_train:
                quality = random.choice(qualities)
                image = jpeg_degrade_functions[quality](image)
            if self.pad2square:
                image = expand2square(image, pad_color)
            pixel_values = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            return pixel_values[0]

        return transform

    def _should_use_dynamic_preprocess(self, data_item):
        if not self.dynamic_image_size:
            return False
        image_value = data_item.get('image')
        image_count = len(image_value) if isinstance(image_value, list) else 1
        return image_count == 1

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        first_turn_idx = 1 if data_item['conversations'][0].get('from') == 'system' else 0
        if '<image>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = '<image>\n' + data_item['conversations'][first_turn_idx]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        image = self.load_image(image_path)

        use_dynamic_preprocess = self._should_use_dynamic_preprocess(data_item)
        if use_dynamic_preprocess:
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        #| images: [num_patches, 3, image_size, image_size]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not use_dynamic_preprocess:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        conversations = self._prepare_conversations(data_item['conversations'])
        ret = preprocess_function(self.template_name, [conversations],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        actual_image_count = int((ret['input_ids'][0] == image_end_token_id).sum().item())
        assert actual_image_count == 1, (
            f'image tokens are truncated: expected_images=1 actual_images={actual_image_count} '
            f'max_seq_length={self.tokenizer.model_max_length} num_image_token={self.num_image_token} '
            f'num_patches={num_patches} dataset={self.ds_name}'
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        first_turn_idx = 1 if data_item['conversations'][0].get('from') == 'system' else 0
        if '<image>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = '<image>\n' * len(data_item['image']) + data_item['conversations'][first_turn_idx]['value']

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        use_dynamic_preprocess = self._should_use_dynamic_preprocess(data_item)
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            image = self.load_image(image_path)
            if use_dynamic_preprocess:
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        conversations = self._prepare_conversations(data_item['conversations'])
        ret = preprocess_function(self.template_name, [conversations],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        actual_image_count = int((ret['input_ids'][0] == image_end_token_id).sum().item())
        assert actual_image_count == num_image, (
            f'image tokens are truncated: expected_images={num_image} actual_images={actual_image_count} '
            f'max_seq_length={self.tokenizer.model_max_length} num_image_token={self.num_image_token} '
            f'num_patches={num_patches} num_tiles={num_tiles} dataset={self.ds_name}'
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][0]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        conversations = self._prepare_conversations(data_item['conversations'])
        ret = preprocess_function(self.template_name, [conversations],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def fake_data_get_item(self):
        # Build transformation function
        self.num_fake_dump += 1
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        #| min_num=max_num=1 means no dynamic patch, just one fake dummy image
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function(use_pretrain=True)

        conversations = [
            {"from": "pretrain", "value": '我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'},
        ]

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(conversations)],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1) #| fill the padding tokens with 1, escape from index error

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=torch.ones_like(ret['input_ids'][0]) * -100, #| fake data, no need to train
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long) #| used to mark whether data is real or fake
        )
        logger.warning(f'Dumping a fake data, the dataset is: {self.ds_name} ({self.num_fake_dump})')
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed #| ensure self.worker_distributed is only enabled once
            and self.worker_id is not None #| cpu worker used to load data, this variable is assigned by PackedDataset
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                #| Distributed Sampler will pad the last batch with the first few samples
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                if self.use_packed_ds:
                    raise StopIteration
                return self.fake_data_get_item()
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                data_item = json.loads(self.raw_data[i])
                image_value = data_item.get('image')
                image_count = len(image_value) if isinstance(image_value, list) else int(bool(image_value))
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.get_image_path(item) for item in data_item['image']]
                        print(
                            f'Failed to process sample id={data_item.get("id")} task={data_item.get("task")} '
                            f'image_count={image_count} images={images}, the dataset is: {self.ds_name}, '
                            f'error: {type(e).__name__}: {e}'
                        )
                    else:
                        data_path = self.get_image_path(data_item['image'])
                        print(
                            f'Failed to process sample id={data_item.get("id")} task={data_item.get("task")} '
                            f'image_count={image_count} image={data_path}, the dataset is: {self.ds_name}, '
                            f'error: {type(e).__name__}: {e}'
                        )
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError, OSError)):
                    traceback.print_exc()
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        #| When dataloader launches multiple workers to load data, each worker will call __iter__() to get the iterator,
        #| at this time, we can enable the worker distributed to let each worker load a portion of data, this happens
        #| whenever split_annotations is True or False, because it when split_annotations is True, data_rank will be set to 0.
        self._enable_worker_distributed()
        start_idx = 0

        #| self.worker_state_key is initialized in PackedDataset according to the worker id.
        assert self.worker_state_key is not None
        #| try to resume from the last state
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=} (total={len(self)})'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


MAX_IMAGE_LENGTH = 8

def order_pick_k(lst, k):
    if len(lst) <= k:
        return lst, None
    rng = np.random.random(len(lst))
    index = np.argsort(rng)[:k]
    index_sort = sorted(index)
    new_lst = [lst[i] for i in index_sort]
    logger.warning(
        f"WARNING: total file: {len(lst)}, random pick: {k}."
        f" (ignored)"
    )
    return new_lst, index_sort

#| NOTE: In answers, there's bbx, no need to include bbx in answers according to RSCoVLM
class TEOChatlasDataset(LazySupervisedDataset):
    """Dataset for TEOChatlas with special data processing (e.g., sorting by timestamp)."""

    def __init__(self, *args, **kwargs):
        meta = kwargs.get('meta')
        self.rewrite_times_prompt = bool(meta.get('rewrite_times_prompt', False)) if meta else False
        if meta and str(meta.get('annotation', '')).endswith('.json'):
            jsonl_path = meta['annotation'][:-5] + '.jsonl'
            if not os.path.exists(jsonl_path):
                logger.info(f"TEOChatlasDataset: Generating JSONL from {meta['annotation']}")
                import json
                try:
                    with open(meta['annotation'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    with open(jsonl_path, 'w', encoding='utf-8') as f:
                        for item in data:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    logger.info(f"TEOChatlasDataset: Successfully generated {jsonl_path}")
                except Exception as e:
                    logger.error(f"TEOChatlasDataset: Failed to generate JSONL: {e}")
                    raise e
            # Replace the annotation path with the generated jsonl file
            kwargs['meta']['annotation'] = jsonl_path
            
        super().__init__(*args, **kwargs)
        self._normalize_video_key_to_image()

    def _normalize_video_key_to_image(self):
        """Keep base __getitem__ unchanged by normalizing TEOChatlas samples once after load."""
        normalized = 0
        normalized_raw_data = []

        for line in self.raw_data:
            try:
                data_item = json.loads(line)
            except json.JSONDecodeError:
                normalized_raw_data.append(line)
                continue

            if 'video' in data_item and 'image' not in data_item:
                data_item['image'] = data_item['video']
                normalized += 1
                normalized_raw_data.append(json.dumps(data_item, ensure_ascii=False) + '\n')
            else:
                normalized_raw_data.append(line)

        if normalized > 0:
            self.raw_data = normalized_raw_data
            logger.info(f'[{self.ds_name}] Added image alias from video for {normalized} samples.')

    def multi_modal_multi_image_get_item(self, data_item):
        if 'video' in data_item and 'image' not in data_item:
            data_item['image'] = data_item.pop('video')
            
        image_files = data_item['image']
        timestamps = data_item.get('timestamp', [])
        
        has_matched_timestamps = len(timestamps) > 0 and len(image_files) == len(timestamps)

        # Apply order_pick_k
        image_files, indices = order_pick_k(image_files, MAX_IMAGE_LENGTH)
        
        if has_matched_timestamps:
            if indices is not None:
                timestamps = [timestamps[i] for i in indices]

            image_files, timestamps = sort_media_by_timestamps(image_files, timestamps)
            data_item['timestamp'] = timestamps

        data_item['image'] = list(image_files)

        # Prepare interleave tokens
        num_video_images = len(image_files)
        replace_token = "Image: <image>\n"
        vid_replace_token = ''.join(f"Image {i+1}: <image>\n" for i in range(num_video_images))

        # Check if conversation has any image/video tokens. If not, inject <video> to the beginning.
        first_turn_idx = 1 if data_item.get('conversations', [{'from': ''}])[0].get('from', '') == 'system' else 0
        if '<image>' not in data_item['conversations'][first_turn_idx]['value'] and '<video>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = '<video>\n' + data_item['conversations'][first_turn_idx]['value']

        # Apply chronological replace and interleave text labels
        for sentence in data_item.get('conversations', []):
            if self.rewrite_times_prompt and 'times:' in sentence.get('value', ''):
                sentence['value'] = sentence['value'].replace("times:", "times in chronological order:")
                 
            if '<image>' in sentence.get('value', ''):
                sentence['value'] = sentence['value'].replace('<image>', replace_token)
            if '<video>' in sentence.get('value', ''):
                sentence['value'] = sentence['value'].replace('<video>', vid_replace_token)

        return super().multi_modal_multi_image_get_item(data_item)


def build_datasets(
    data_args,
    tokenizer,
    model,
    image_processor=None,
    group_by_length=False,
    group_by_modality_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    split_annotations=False,
):
    datasets = []
    lengths = []

    data_rank = 0 if split_annotations else dist.get_rank()
    data_world_size = 1 if split_annotations else dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch

        dataset_format = str(ds_collections[ds_name].get('dataset_format', '')).lower()
        dataset_cls = (
            TEOChatlasDataset
            if 'TEOChatlas' in ds_name or dataset_format == 'teochatlas'
            else LazySupervisedDataset
        )

        dataset = dataset_cls(
            template_name=data_args.conv_style,
            meta=ds_collections[ds_name],
            tokenizer=tokenizer,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name].get('data_augment', False),
            pad2square=data_args.pad2square,
            group_by_length=(group_by_length or group_by_modality_length) and not data_args.use_packed_ds,
            group_by_modality_length=group_by_modality_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            image_processor=image_processor,
            split_annotations=split_annotations,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
            use_bbox_coord_tokens=data_args.use_bbox_coord_tokens,
            bbox_coord_token_max=data_args.bbox_coord_token_max,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)} ({data_rank=}, {data_world_size=})')
        datasets.append(dataset)
        lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            # allow_empty_data=data_args.allow_empty_data, #| no need
            allow_deduplicated_ds_name=False,
        )
        if dist.get_rank() == 0:
            logger.info(f"Wrapping datasets into PackedDataset with max_packed_tokens={data_args.max_packed_tokens}")
        return train_dataset
    
    #| Fallback to Map-style Dataset(__getitem__ instead of __iter__)
    train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if group_by_length or group_by_modality_length:
        train_dataset.length = [item for dataset in datasets for item in dataset.length]
        train_dataset.modality_length = [item for dataset in datasets for item in dataset.modality_length]
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square': #| default
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def train():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataArguments, TeoNextTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #| use_packed_ds is passed to trainer to keep PackedDataset dataloader behavior.
    training_args.use_packed_ds = data_args.use_packed_ds

    # if data_args.use_packed_ds and training_args.max_steps <= 0:
    #     raise ValueError(
    #         'PackedDataset is IterableDataset, so `--max_steps` must be > 0 when `--use_packed_ds True`.'
    #     )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    #| TrainingArguments.should_log controls Whether or not the current process should produce log.
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info() #| only in main process, print transformers log info

    #| different process has different log level, only main process has log level info
    #| During overall training, use args to pass all parameters including log level, rank, world size, etc. same as
    #| prismatic-vl's design
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()
    _maybe_set_wandb_run_name(training_args)

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    #| compatible with llava's precision load logic
    compute_dtype = torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32
    )
    model_torch_dtype = torch.float32 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32
    )
    attn_implementation = _resolve_attention_implementation()
    vision_select_layers = parse_vision_select_layers(model_args.vision_select_layers)
    logger.info(f'Using compute dtype: {compute_dtype}')
    logger.info(f'Using attention implementation: {attn_implementation}')
    logger.info(f'Using vision_select_layers: {vision_select_layers}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length

    # if model_args.model_name_or_path is not None:
    #     checkpoint_grounding_special_tokens = TeoNextConfig.from_pretrained(
    #         model_args.model_name_or_path
    #     ).add_grounding_special_tokens
    #     if checkpoint_grounding_special_tokens != data_args.add_grounding_special_tokens:
    #         raise ValueError(
    #             'add_grounding_special_tokens mismatch: '
    #             f'checkpoint={checkpoint_grounding_special_tokens}, '
    #             f'args={data_args.add_grounding_special_tokens}'
    #         )

    token_list = list(IMAGE_SPECIAL_TOKENS)
    if data_args.add_grounding_special_tokens:
        token_list.extend(GROUNDING_SPECIAL_TOKENS)
    if data_args.use_bbox_coord_tokens:
        token_list.extend(bbox_coord_tokens(data_args.bbox_coord_token_max))
    #| Initialization is conducted after llm or vlm is loaded
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    processor_source = model_args.model_name_or_path or model_args.vision_path
    if processor_source is None:
        raise ValueError(
            "model_name_or_path or vision_path is required to load the image processor."
        )
    image_processor = AutoImageProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        use_fast=model_args.use_fast_image_processor,
    )
    logger.info(f'Loaded image processor from: {processor_source}')

    if model_args.use_liger:
        raise NotImplementedError

    # `model_name_or_path` resumes from an existing TeoNext/VLM checkpoint.
    # Otherwise, build a fresh TeoNext model from `vision_path` + `llm_path`.
    if model_args.model_name_or_path is not None:
        logger.info('Loading TeoNextModel...')
        config = TeoNextConfig.from_pretrained(model_args.model_name_or_path)
        config.llm_config._attn_implementation = attn_implementation
        #| use_flash_attn is set to true by default in teonextmodel
        config.vision_config._attn_implementation = attn_implementation
        if hasattr(config.vision_config, 'use_flash_attn'):
            config.vision_config.use_flash_attn = attn_implementation == 'flash_attention_2'

        config.template = data_args.conv_style #| from conversation.py, different stages of training may use different templates, for mlp pretrain, maybe plain, sft use internvl2_5, etc
        config.select_layer = model_args.vision_select_layer #| default to -1, not same as llava using -2
        if vision_select_layers is not None:
            config.vision_select_layers = vision_select_layers
        config.dynamic_image_size = data_args.dynamic_image_size #| critical! if you want to use dynamic high-resolution, take this
        config.use_thumbnail = data_args.use_thumbnail
        config.add_grounding_special_tokens = data_args.add_grounding_special_tokens
        config.use_bbox_coord_tokens = data_args.use_bbox_coord_tokens
        config.bbox_coord_token_max = data_args.bbox_coord_token_max
        if config.use_pixel_shuffle != data_args.use_pixel_shuffle:
            raise ValueError(
                f'use_pixel_shuffle mismatch: checkpoint={config.use_pixel_shuffle}, '
                f'args={data_args.use_pixel_shuffle}'
            )
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = TeoNextModel.from_pretrained(model_args.model_name_or_path, torch_dtype=model_torch_dtype, config=config)
    else:
        backend = _detect_vision_backend(model_args.vision_path)
        if backend == "siglip":
            logger.info("Loading SiglipVisionModel...")
            vision_config = SiglipVisionConfig.from_pretrained(model_args.vision_path)
            vision_config.torch_dtype = model_torch_dtype
            vision_config._attn_implementation = attn_implementation
            if hasattr(vision_config, 'use_flash_attn'):
                vision_config.use_flash_attn = attn_implementation == 'flash_attention_2'
            vision_model = SiglipVisionModel.from_pretrained(
                    model_args.vision_path, torch_dtype=model_torch_dtype, config=vision_config
                )
        elif backend == "clip":
            logger.info("Loading CLIPVisionModel...")
            vision_config = CLIPVisionConfig.from_pretrained(model_args.vision_path)
            vision_config.torch_dtype = model_torch_dtype
            vision_config._attn_implementation = attn_implementation
            if hasattr(vision_config, 'use_flash_attn'):
                vision_config.use_flash_attn = attn_implementation == 'flash_attention_2'
            vision_model = CLIPVisionModel.from_pretrained(
                    model_args.vision_path, torch_dtype=model_torch_dtype, config=vision_config
                )
        else:
            raise NotImplementedError(f"Unsupported vision model backend: {backend}")

        logger.info('Loading LLM...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        llm_config._attn_implementation = attn_implementation

        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_path, 
            torch_dtype=model_torch_dtype, 
            config=llm_config, 
            trust_remote_code=True
        )

        logger.info('Building TeoNextConfig...')
        teonext_config = TeoNextConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            use_backbone_lora=model_args.use_backbone_lora,
            use_llm_lora=model_args.use_llm_lora,
            llm_lora_alpha=model_args.llm_lora_alpha,
            select_layer=model_args.vision_select_layer,
            force_image_size=data_args.force_image_size,
            downsample_ratio=data_args.down_sample_ratio, # 0-1, used in pixel shuffle to eliminate the number of image tokens
            use_pixel_shuffle=data_args.use_pixel_shuffle,
            template=data_args.conv_style,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            add_grounding_special_tokens=data_args.add_grounding_special_tokens,
            use_bbox_coord_tokens=data_args.use_bbox_coord_tokens,
            bbox_coord_token_max=data_args.bbox_coord_token_max,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
            pad2square=data_args.pad2square, #| teochat use pad2square, but internvl use dynamic_image_size
            modality_bridge_type=model_args.modality_bridge_type,
            vision_select_layers=vision_select_layers,
            attention_pooling_query_type=model_args.attention_pooling_query_type,
        )
        #| force_image_size is the desired image size for each image
        teonext_config.force_image_size = data_args.force_image_size

        logger.info('Building TeoNextModel...')
        model = TeoNextModel(teonext_config, vision_model, llm)

    model.img_context_token_id = img_context_token_id #| placeholder token id for image context
    model.tokenizer = tokenizer #| this logic replaces initialize_vision_tokenizer in llava_arch.py
    model.config.torch_dtype = model_torch_dtype
    model.config.add_grounding_special_tokens = data_args.add_grounding_special_tokens
    model.config.use_bbox_coord_tokens = data_args.use_bbox_coord_tokens
    model.config.bbox_coord_token_max = data_args.bbox_coord_token_max
    if model_args.cut_cross_entropy is not None:
        model.config.cut_cross_entropy = model_args.cut_cross_entropy
    logger.info(f'model.config.cut_cross_entropy: {getattr(model.config, "cut_cross_entropy", False)}')

    if data_args.use_packed_ds:
        if not getattr(model.language_model.config, "model_type", None) == "qwen3":
            raise NotImplementedError(f"PackedDataset is currently only compatible with Qwen3-based language models. Now is {getattr(model.language_model.config, 'model_type', 'Unknown')}")
        replace_qwen3_attention_class(model.language_model)

    #| Here, down_sample_ratio should stay the same across different stages of training
    assert model.config.downsample_ratio == data_args.down_sample_ratio
    assert model.config.use_pixel_shuffle == data_args.use_pixel_shuffle

    #| TODO: investigate whether a full VLM (for example Qwen3-VL) can provide a
    #| dimension-compatible projector to initialize this bridge. This is research-only
    #| for now, because finding a matching hidden-size/layout is uncommon.
    if model_args.mlp_path is not None:
        logger.info('Loading pretrained modality bridge...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict, strict=False)
        allowed_missing_keys = set()
        if getattr(model.mlp1, "query_embed", None) is not None:
            allowed_missing_keys.add("query_embed")
        if getattr(model.mlp1, "query_gate", None) is not None:
            allowed_missing_keys.add("query_gate")
        unexpected_keys = set(message.unexpected_keys)
        missing_keys = set(message.missing_keys) - allowed_missing_keys
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Failed to load pretrained modality bridge. "
                f"missing_keys={sorted(missing_keys)}, unexpected_keys={sorted(unexpected_keys)}"
            )
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    #| If the dedicated image size is not compatible with the original vision model, resize the position embedding
    #| via bicubic interpolation (adapted for SigLIP's nn.Embedding without CLS token)
    #| Keep image_size same as original vision model
    if model.config.vision_config.image_size != data_args.force_image_size:
        raise ValueError(
            f'image_size mismatch: checkpoint={model.config.vision_config.image_size}, '
            f'args={data_args.force_image_size}'
        )
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                    new_size=data_args.force_image_size,
                                    patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
        
    model.config.force_image_size = data_args.force_image_size
    
    model.num_image_token = model.compute_num_image_tokens(data_args.force_image_size)

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        resized_vocab_size = model.language_model.get_input_embeddings().weight.size(0)
        if resized_vocab_size != len(tokenizer):
            raise ValueError(
                f'Resized embedding rows ({resized_vocab_size}) do not match tokenizer size ({len(tokenizer)}).'
            )
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
        logger.info(
            f'Resized token embeddings to {resized_vocab_size} '
            f'(tokenizer size: {len(tokenizer)}).'
        )

    original_use_cache = model.language_model.config.use_cache
    _set_llm_use_cache(model, False)
    if model_args.grad_checkpoint:
        if hasattr(model.language_model, 'gradient_checkpointing_enable'):
            model.language_model.gradient_checkpointing_enable()
        elif hasattr(model.language_model, '_set_gradient_checkpointing'):
            model.language_model._set_gradient_checkpointing()
        else:
            logger.warning('language_model does not expose gradient checkpointing APIs.')

        #| Keep gradients flowing through embeddings when checkpointing is active.
        #| In original llava implementation, embeddings are frozen
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        elif hasattr(model.language_model, 'enable_input_require_grads'):
            model.language_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        logger.info('gradient_checkpointing is enabled')

    #| num_images_expected is set to 96 with max_dynamic_patch=12, means about 8 conversations in one buffer
    #| TODO: if i want to use this packeddataset, need to ensure images not bypass the expected number, or this
    #| packed sample may be deprecated directly, need to statically set the number of images and max_packed_tokens
    train_dataset = build_datasets(
        data_args,
        tokenizer,
        model,
        image_processor=image_processor,
        group_by_length=training_args.group_by_length,
        group_by_modality_length=training_args.group_by_modality_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        split_annotations=data_args.split_annotations,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        #| eval() is not used here, because maybe last several layers are unfrozen need DropPath to be active,
        #| and even if all layers are frozen, DropPath can also make training of mlp more stable
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    #| Not wrap model with lora manually in train.py, instead, put corresponding logic in model's class function
    if model_args.use_backbone_lora:
        if hasattr(model, 'wrap_backbone_lora'):
            model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        else:
            raise NotImplementedError('wrap_backbone_lora is not implemented in TeoNextModel.')
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        apply_llm_lora(
            model,
            model_args.use_llm_lora,
            llm_lora_alpha=model_args.llm_lora_alpha,
            logger_=logger,
        )

    _apply_dtype_policy(model, compute_dtype)

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction), #| set to square
        )
    else: #| corporate with ConcatDataset(only mixing dataset, no sequence packing)
        collator = concat_pad_data_collator

    #| After PackedDataset and packed_collate_fn, batch is defined as input_ids, labels,
    #| attention_mask, position_ids, loss_weight, cu_seqlens, pixel_values, image_flags, statistics
    #| If no use_packed_ds is enabled, build_datasets will fall back to ConcatDataset and concat_pad_data_collator, get
    #| input_ids, labels, attention_mask, position_ids, pixel_values, image_flags
    #| ======================= BATCH SHAPE CHEATSHEET =======================
    #| [PATH 1] PackedDataset + packed_collate_fn (use_packed_ds = True)
    #|   - input_ids, labels, attention_mask, loss_weight : [batch_size=1, max_packed_tokens] (usually 1 due to micro_num constraint, dynamic padding)
    #|   - cu_seqlens   : [batch_size=1, num_packed_samples + 1] (for Flash Attention 2 var-len masking)
    #|   - pixel_values : [total_num_images_in_buffer, C, H, W] (flattened, NOT [batch_size, ...])
    #|   - image_flags  : [total_num_images_in_buffer]
    #|   - statistics   : [4] (num_samples, effective_tokens, pad_tokens, image_count)
    #|
    #| [PATH 2] ConcatDataset + concat_pad_data_collator (use_packed_ds = False)
    #|   - input_ids, labels, attention_mask : [batch_size, max_seq_len_in_this_batch] (dynamic padding)
    #|   - pixel_values : [total_num_images_in_batch, C, H, W] (flattened and concatenated)
    #|   - image_flags  : [total_num_images_in_batch]
    #|   * Note: No cu_seqlens, no statistics, and no loss_weight (uniform average loss fallback).
    #| ======================================================================
    training_args.split_annotations = data_args.split_annotations
    trainer = TeoNextTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.add_callback(SaveImageProcessorCallback(image_processor))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(f'[Memory Usage before training] {torch.cuda.memory_allocated()/1024/1024/1024:.2f}GB')
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if model_args.merge_llm_lora_before_save:
            if not model_args.use_llm_lora:
                raise ValueError('merge_llm_lora_before_save requires use_llm_lora > 0.')

            save_target_model = _unwrap_model(trainer.model)

            if not hasattr(save_target_model.language_model, 'merge_and_unload'):
                raise RuntimeError('The current language_model does not support merge_and_unload().')

            logger.info('Merging LLM LoRA weights before saving...')
            save_target_model.language_model = save_target_model.language_model.merge_and_unload()
            save_target_model.config.use_llm_lora = 0
            save_target_model.config.llm_lora_alpha = None

        _set_llm_use_cache(trainer.model, original_use_cache)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        image_processor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


def main():
    try:
        train()
    finally:
        destroy_dist_process_group()

if __name__ == '__main__':
    main()
