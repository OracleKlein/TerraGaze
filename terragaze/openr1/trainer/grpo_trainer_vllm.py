from __future__ import annotations

import logging
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Iterable
from unittest.mock import patch

from accelerate.utils import broadcast_object_list
from accelerate.utils import gather_object
from accelerate.utils import set_seed
from datasets import features
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers import Trainer
from transformers import TrainerCallback
from trl import GRPOConfig
from trl.models import prepare_deepspeed

from teonext.model import TeoNextModel
from teonext.model import TeoNextConfig
from teonext.model.conversation import get_conv_template
from teonext.openr1.arguments import TeoNextGRPOScriptArguments
from teonext.openr1.dataset import SREGRPODataset
from teonext.openr1.vllm_model import allow_vllm_aimv2_duplicate_registration
from teonext.openr1.vllm_model import register_teonext_vllm_model
from teonext.utils.constants import GROUNDING_SPECIAL_TOKENS
from teonext.utils.constants import IMAGE_SPECIAL_TOKENS
from teonext.utils.constants import IMG_CONTEXT_TOKEN
from teonext.utils.constants import IMG_END_TOKEN
from teonext.utils.constants import IMG_START_TOKEN
from teonext.utils.bbox import bbox_coord_tokens
from teonext.utils.bbox import bbox_text_format_from_config
from teonext.utils.openr1_trace import trace_openr1
from teonext.utils.openr1_trace import trace_train_runtime
from teonext.utils.utils import apply_llm_lora
from teonext.utils.utils import get_tensor_dtype
from teonext.utils.utils import has_flash_attn
from teonext.utils.utils import preprocess_images

try:
    import deepspeed
except ImportError:
    deepspeed = None

try:
    with allow_vllm_aimv2_duplicate_registration():
        from vllm import LLM
        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
except Exception as exc:
    EngineArgs = None
    LLM = None
    SamplingParams = None
    VLLM_IMPORT_ERROR = exc
else:
    VLLM_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


RewardFunc = Callable[..., Any]


class RepeatRandomSampler:
    """Repeat each sampled index consecutively for GRPO prompt groups."""

    def __init__(self, data_source: Iterable[Any], repeat_count: int, seed: int | None = None):
        if repeat_count <= 0:
            raise ValueError("repeat_count must be positive.")
        self.num_samples = len(data_source)
        self.repeat_count = repeat_count
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        indexes = list(range(self.num_samples))
        rng.shuffle(indexes)
        for index in indexes:
            for _ in range(self.repeat_count):
                yield index

    def __len__(self):
        return self.num_samples * self.repeat_count


def _dist():
    return dist


def dist_is_initialized() -> bool:
    dist = _dist()
    return bool(dist is not None and dist.is_available() and dist.is_initialized())


def get_rank() -> int:
    if not dist_is_initialized():
        return 0
    return _dist().get_rank()


def get_world_size() -> int:
    if not dist_is_initialized():
        return 1
    return _dist().get_world_size()


def gather_rank_objects(local_values: list[Any]) -> tuple[list[Any], int]:
    if not dist_is_initialized():
        return list(local_values), 0
    gathered: list[list[Any] | None] = [None for _ in range(get_world_size())]
    _dist().all_gather_object(gathered, list(local_values))
    rank = get_rank()
    local_start = sum(len(gathered[idx] or []) for idx in range(rank))
    values = [value for rank_values in gathered for value in (rank_values or [])]
    return values, local_start


def register_rank_cuda_device() -> None:
    # OpenR1 differs from the PR1 baseline by keeping a colocated vLLM engine on
    # a dedicated rollout GPU while the training ranks are wrapped later by
    # DeepSpeed/Accelerate.  After vLLM touches CUDA, NCCL may not be able to
    # infer the intended rank-to-device mapping at the next barrier.  Explicitly
    # setting each process to cuda:LOCAL_RANK and running a tiny CUDA collective
    # registers that mapping early, avoiding the "device used by this process is
    # currently unknown" hang path.  Plain SFT does not need this because it has
    # no separate rollout engine living on another GPU in the same job.
    if not torch.cuda.is_available():
        return
    local_rank_text = os.environ.get("LOCAL_RANK")
    if local_rank_text is None:
        if get_world_size() > 1:
            raise ValueError("LOCAL_RANK is required for distributed CUDA training.")
        local_rank = torch.cuda.current_device()
    else:
        try:
            local_rank = int(local_rank_text)
        except ValueError as exc:
            raise ValueError(f"Invalid LOCAL_RANK value: {local_rank_text!r}.") from exc

    cuda_device_count = torch.cuda.device_count()
    if local_rank >= cuda_device_count:
        raise ValueError(
            f"LOCAL_RANK={local_rank} but torch only sees {cuda_device_count} CUDA device(s)."
        )

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    trace_openr1(f"set current CUDA device to {device}")

    if not dist_is_initialized():
        return
    trace_openr1(f"before NCCL device mapping all_reduce on {device}")
    marker = torch.zeros(1, device=device)
    _dist().all_reduce(marker)
    torch.cuda.synchronize(device)
    trace_openr1(f"after NCCL device mapping all_reduce on {device}")


def compute_group_advantages(
    rewards: list[float],
    num_generations: int,
    *,
    normalize: bool = True,
) -> tuple[list[float], float]:
    if num_generations <= 1:
        raise ValueError("GRPO requires num_generations > 1.")
    if len(rewards) % num_generations != 0:
        raise ValueError("Reward count must be divisible by num_generations.")

    advantages: list[float] = []
    zero_std_groups = 0
    for start in range(0, len(rewards), num_generations):
        group = rewards[start:start + num_generations]
        mean = sum(group) / len(group)
        variance = sum((value - mean) ** 2 for value in group) / len(group)
        std = math.sqrt(variance)
        if std == 0.0:
            zero_std_groups += 1
            advantages.extend([0.0 for _ in group])
        elif not normalize:
            advantages.extend([value - mean for value in group])
        else:
            advantages.extend([(value - mean) / (std + 1.0e-6) for value in group])
    group_count = len(rewards) // num_generations
    zero_ratio = zero_std_groups / group_count if group_count else 0.0
    return advantages, zero_ratio


def validate_prompt_groups(sample_ids: list[str], num_generations: int) -> None:
    if num_generations <= 1:
        raise ValueError("GRPO requires num_generations > 1.")
    if len(sample_ids) % num_generations != 0:
        raise ValueError("Sample id count must be divisible by num_generations.")
    for start in range(0, len(sample_ids), num_generations):
        group = sample_ids[start:start + num_generations]
        if len(set(group)) != 1:
            raise ValueError(f"GRPO prompt group contains multiple sample ids: {group!r}")


def compute_group_zero_ratios(
    rewards: list[float],
    labels: list[str],
    num_generations: int,
) -> dict[str, float]:
    if len(rewards) != len(labels):
        raise ValueError("Reward and label counts must match for grouped zero-advantage metrics.")
    if len(rewards) % num_generations != 0:
        raise ValueError("Reward count must be divisible by num_generations.")

    zero_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    for start in range(0, len(rewards), num_generations):
        group_rewards = rewards[start:start + num_generations]
        group_labels = labels[start:start + num_generations]
        if len(set(group_labels)) != 1:
            raise ValueError(f"GRPO prompt group spans multiple metric labels: {group_labels!r}")
        label = group_labels[0]
        group_counts[label] += 1
        if all(value == group_rewards[0] for value in group_rewards[1:]):
            zero_counts[label] += 1
    return {
        label: zero_counts[label] / count
        for label, count in group_counts.items()
    }


def pad_token_sequences(sequences, *, pad_token_id: int, device):
    if not sequences:
        raise ValueError("Expected at least one token sequence.")
    max_len = max(sequence.numel() for sequence in sequences)
    if max_len <= 0:
        raise ValueError("Completion token sequence must not be empty.")
    padded = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        length = sequence.numel()
        padded[row, :length] = sequence
        mask[row, :length] = 1
    return padded, mask


def mask_after_eos(token_ids, base_mask, *, eos_token_id: int | None):
    if eos_token_id is None:
        return base_mask

    is_eos = token_ids == eos_token_id
    if not is_eos.any():
        return base_mask
    sequence_indices = torch.arange(token_ids.shape[1], device=token_ids.device).expand_as(token_ids)
    eos_idx = torch.full((token_ids.shape[0],), token_ids.shape[1] - 1, dtype=torch.long, device=token_ids.device)
    eos_rows = is_eos.any(dim=1)
    eos_idx[eos_rows] = is_eos.int().argmax(dim=1)[eos_rows]
    eos_mask = (sequence_indices <= eos_idx.unsqueeze(1)).long()
    return base_mask * eos_mask


def compute_kl_terms(per_token_logps, ref_per_token_logps) -> dict[str, Any]:
    if per_token_logps.shape != ref_per_token_logps.shape:
        raise ValueError(
            f"Policy/ref logprob shape mismatch: {tuple(per_token_logps.shape)} vs "
            f"{tuple(ref_per_token_logps.shape)}"
        )
    log_ratio = ref_per_token_logps - per_token_logps
    return {
        "k1": per_token_logps - ref_per_token_logps,
        "k3": torch.exp(log_ratio) - log_ratio - 1,
        "kimikl": 0.5 * (per_token_logps - ref_per_token_logps) ** 2,
    }


def select_kl_approximator(terms: dict[str, Any], name: str):
    if name == "fullkimi":
        return terms["kimikl"]
    if name not in {"k1", "k3", "kimikl"}:
        raise ValueError("kl_approximator must be one of: k3, k1, kimikl, fullkimi.")
    return terms[name]


def masked_mean(values, mask):
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _unwrap_model(model):
    if hasattr(model, "module"):
        return _unwrap_model(model.module)
    return model


def _set_requires_grad(module, requires_grad: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _set_teonext_use_cache(model, use_cache: bool) -> None:
    if not hasattr(model, "config") or not hasattr(model.config, "llm_config"):
        raise AttributeError("TeoNext model config must define llm_config to set use_cache.")
    language_model = getattr(model, "language_model", None)
    if language_model is None or not hasattr(language_model, "config"):
        raise AttributeError("TeoNext model must define language_model.config to set use_cache.")
    model.config.use_cache = use_cache
    model.config.llm_config.use_cache = use_cache
    language_model.config.use_cache = use_cache


def _enable_input_require_grads_for_checkpointing(model) -> None:
    language_model = getattr(model, "language_model", None)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    if language_model is not None and hasattr(language_model, "enable_input_require_grads"):
        language_model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("Gradient checkpointing requires input embeddings to expose gradients.")

    def make_inputs_require_grad(_module, _inputs, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)


def _iter_vllm_sync_weights(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield full-rank training weights with names accepted by the vLLM model."""
    adapter_weight_markers = (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
        ".lora_magnitude_vector.",
    )
    peft_language_prefix = "language_model.base_model.model."

    for name, tensor in weights:
        if any(marker in name for marker in adapter_weight_markers):
            continue

        clean_name = name
        if clean_name.startswith(peft_language_prefix):
            clean_name = "language_model." + clean_name[len(peft_language_prefix):]
        clean_name = clean_name.replace(".base_layer.", ".")
        yield clean_name, tensor


def _load_teonext_processors(
    model_name_or_path: str,
    *,
    model_init_kwargs: dict[str, Any],
    processing_class: PreTrainedTokenizerBase | None = None,
) -> tuple[PreTrainedTokenizerBase, Any]:
    cache_dir = model_init_kwargs.get("cache_dir")
    trust_remote_code = model_init_kwargs.get("trust_remote_code", True)
    if processing_class is None:
        processing_class = AutoTokenizer.from_pretrained(
            model_name_or_path,
            add_eos_token=False,
            trust_remote_code=trust_remote_code,
            use_fast=bool(model_init_kwargs.get("use_fast_tokenizer", False)),
            cache_dir=cache_dir,
        )
    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=bool(model_init_kwargs.get("use_fast_image_processor", False)),
        cache_dir=cache_dir,
    )
    return processing_class, image_processor


def _prepare_teonext_tokenizer(model, tokenizer, config) -> None:
    token_list = list(IMAGE_SPECIAL_TOKENS)
    if getattr(config, "add_grounding_special_tokens", True):
        token_list.extend(GROUNDING_SPECIAL_TOKENS)
    if getattr(config, "use_bbox_coord_tokens", False):
        token_list.extend(bbox_coord_tokens(getattr(config, "bbox_coord_token_max", 1000)))
    tokenizer.add_tokens(token_list, special_tokens=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.tokenizer = tokenizer

    previous_vocab_size = model.language_model.get_input_embeddings().weight.size(0)
    target_vocab_size = len(tokenizer)
    if previous_vocab_size == target_vocab_size:
        model.config.llm_config.vocab_size = target_vocab_size
        model.language_model.config.vocab_size = target_vocab_size
        return
    model.language_model.resize_token_embeddings(len(tokenizer))
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    added_rows = target_vocab_size - previous_vocab_size
    if added_rows > 0:
        input_embeddings_avg = input_embeddings[:-added_rows].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-added_rows].mean(dim=0, keepdim=True)
        input_embeddings[-added_rows:] = input_embeddings_avg
        output_embeddings[-added_rows:] = output_embeddings_avg

    resized_vocab_size = model.language_model.get_input_embeddings().weight.size(0)
    if resized_vocab_size != len(tokenizer):
        raise ValueError(
            f"Resized embedding rows ({resized_vocab_size}) do not match tokenizer size ({len(tokenizer)})."
        )
    model.config.llm_config.vocab_size = len(tokenizer)
    model.language_model.config.vocab_size = len(tokenizer)


def _load_teonext_grpo_model(
    model_name_or_path: str,
    *,
    script_args,
    model_init_kwargs: dict[str, Any],
    processing_class: PreTrainedTokenizerBase | None = None,
):
    model_init_kwargs = dict(model_init_kwargs)
    load_dtype = model_init_kwargs.get("torch_dtype")
    use_cache = model_init_kwargs.pop("use_cache", None)
    if load_dtype is not None and not isinstance(load_dtype, torch.dtype):
        dtype_name = str(load_dtype).lower()
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name.split(".", 1)[1]
        dtype_mapping = {
            "auto": None,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_mapping:
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        load_dtype = dtype_mapping[dtype_name]
    tokenizer, image_processor = _load_teonext_processors(
        model_name_or_path,
        model_init_kwargs=model_init_kwargs,
        processing_class=processing_class,
    )

    cache_dir = model_init_kwargs.get("cache_dir", getattr(script_args, "cache_dir", None))
    trust_remote_code = model_init_kwargs.get("trust_remote_code", True)
    config = TeoNextConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    attn_implementation = model_init_kwargs.get("attn_implementation") or ("flash_attention_2" if has_flash_attn else "eager")
    config.llm_config._attn_implementation = attn_implementation
    config.vision_config._attn_implementation = attn_implementation
    if hasattr(config.vision_config, "use_flash_attn"):
        config.vision_config.use_flash_attn = attn_implementation == "flash_attention_2"

    if script_args.conv_style is not None:
        config.template = script_args.conv_style
    if script_args.select_layer is not None:
        config.select_layer = script_args.select_layer
    if script_args.dynamic_image_size is not None:
        config.dynamic_image_size = script_args.dynamic_image_size
    if script_args.use_thumbnail is not None:
        config.use_thumbnail = script_args.use_thumbnail
    if script_args.min_dynamic_patch is not None:
        config.min_dynamic_patch = script_args.min_dynamic_patch
    if script_args.max_dynamic_patch is not None:
        config.max_dynamic_patch = script_args.max_dynamic_patch
    if use_cache is not None:
        config.use_cache = use_cache
        config.llm_config.use_cache = use_cache

    model_kwargs = dict(model_init_kwargs)
    for key in ("attn_implementation", "use_fast_tokenizer", "use_fast_image_processor"):
        model_kwargs.pop(key, None)
    model_kwargs.update({"config": config, "cache_dir": cache_dir})
    if load_dtype is not None:
        model_kwargs["torch_dtype"] = load_dtype
    else:
        model_kwargs.pop("torch_dtype", None)
    model = TeoNextModel.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.torch_dtype = load_dtype or getattr(model.config, "torch_dtype", torch.float32)
    if use_cache is not None:
        _set_teonext_use_cache(model, bool(use_cache))

    _prepare_teonext_tokenizer(model, tokenizer, config)

    device = script_args.device
    if device == "cuda" and torch.cuda.is_available():
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    if load_dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=load_dtype)
    return tokenizer, model, image_processor


def render_teonext_prompt(
    model,
    tokenizer,
    question: str,
    num_patches_list: list[int],
    *,
    expand_image_tokens: bool = True,
) -> tuple[str, str, int | None]:
    if question and "<image>" not in question:
        question = "<image>\n" + question

    base_model = _unwrap_model(model)
    base_model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    template = get_conv_template(base_model.template)
    template.system_message = base_model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    sep = template.sep.strip() if template.sep2 is None else template.sep2.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(sep)
    if eos_token_id == tokenizer.unk_token_id:
        eos_token_id = tokenizer.eos_token_id
    if expand_image_tokens:
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * base_model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
    return query, sep, eos_token_id


class TeoNextVLLMGRPOTrainer(Trainer):
    def __init__(
        self,
        model: str | TeoNextModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: SREGRPODataset | None = None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        peft_config: Any | None = None,
        script_args: TeoNextGRPOScriptArguments | None = None,
    ) -> None:
        if script_args is None:
            raise ValueError("script_args is required for TeoNextVLLMGRPOTrainer.")
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(f"{model_name.split('/')[-1]}-GRPO")
        args.remove_unused_columns = False
        if args.gradient_checkpointing:
            # OpenR1 uses non-reentrant gradient checkpointing for the DeepSpeed
            # ZeRO + LoRA path.  Reentrant checkpointing recomputes full forward
            # segments during backward, which can make the same LoRA parameters
            # appear ready for reduction more than once in this multi-forward
            # GRPO step (rollout, policy/ref logprobs, and vLLM weight sync).
            # The simpler SFT path does not hit this interaction because it does
            # not combine rollout generation, adapter state changes, and repeated
            # policy/reference forwards inside a single trainer step.
            gradient_checkpointing_kwargs = dict(getattr(args, "gradient_checkpointing_kwargs", None) or {})
            if gradient_checkpointing_kwargs.get("use_reentrant") is True:
                raise ValueError(
                    "TeoNext GRPO with DeepSpeed and LoRA requires non-reentrant gradient checkpointing. "
                    "Set gradient_checkpointing_kwargs.use_reentrant=false."
                )
            gradient_checkpointing_kwargs["use_reentrant"] = False
            args.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

        # Args
        self.script_args = script_args
        model_init_kwargs = dict(args.model_init_kwargs or {})
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Models
        if isinstance(model, str):
            self.model_name_or_path = model
            processing_class, model, image_processor = _load_teonext_grpo_model(
                self.model_name_or_path,
                script_args=script_args,
                model_init_kwargs=model_init_kwargs,
                processing_class=processing_class,
            )
        else:
            self.model_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
            if not self.model_name_or_path:
                raise ValueError("An instantiated TeoNext model must define config._name_or_path.")
            if model_init_kwargs:
                raise ValueError("args.model_init_kwargs can only be used when `model` is a string path.")
            processing_class, image_processor = _load_teonext_processors(
                self.model_name_or_path,
                model_init_kwargs={"cache_dir": script_args.cache_dir, "trust_remote_code": True},
                processing_class=processing_class,
            )
            _prepare_teonext_tokenizer(model, processing_class, model.config)
            if args.gradient_checkpointing:
                _set_teonext_use_cache(model, False)

        # PEFT / trainable policy
        bridge_trainable = not bool(getattr(script_args, "freeze_mlp", False))
        use_llm_lora = 0
        llm_lora_alpha = None
        if peft_config is not None:
            use_llm_lora = int(getattr(peft_config, "r", 0) or 0)
            llm_lora_alpha = getattr(peft_config, "lora_alpha", None)
        if peft_config is not None and use_llm_lora <= 0:
            raise ValueError("peft_config must define a positive LoRA rank for TeoNext GRPO.")
        if not bridge_trainable and peft_config is None:
            raise ValueError("TeoNext GRPO requires bridge training, LM LoRA, or both.")

        # Apply training configuration to model
        _set_requires_grad(model, False)
        if bridge_trainable:
            seen_bridge_modules = set()
            for name in ("mlp1", "connector", "projector"):
                module = getattr(model, name, None)
                if module is None or id(module) in seen_bridge_modules:
                    continue
                seen_bridge_modules.add(id(module))
                _set_requires_grad(module, True)
        if use_llm_lora:
            apply_llm_lora(model, use_llm_lora, llm_lora_alpha=llm_lora_alpha)

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters remain after applying freeze and LoRA arguments.")
        if args.gradient_checkpointing:
            _enable_input_require_grads_for_checkpointing(model)
            trace_openr1("enabled input requires_grad for gradient checkpointing")

        # Reference model
        self.ref_model = None
        language_model = getattr(model, "language_model", None)
        self.use_adapter_reference = bool(
            peft_config is not None
            and not bridge_trainable
            and (hasattr(model, "disable_adapter") or hasattr(language_model, "disable_adapter"))
        )
        if not self.use_adapter_reference:
            _tokenizer, self.ref_model, _image_processor = _load_teonext_grpo_model(
                self.model_name_or_path,
                script_args=script_args,
                model_init_kwargs=model_init_kwargs,
                processing_class=processing_class,
            )
            self.ref_model.eval()
            _set_requires_grad(self.ref_model, False)
        if self.ref_model is None and not self.use_adapter_reference:
            raise ValueError(
                "TeoNext GRPO requires a reference model path or an LM LoRA adapter that can be disabled."
            )

        # Processing class, rewards
        self.image_processor = image_processor

        if reward_funcs is None:
            raise ValueError("TeoNextVLLMGRPOTrainer requires at least one reward function.")
        self.reward_funcs = list(reward_funcs) if isinstance(reward_funcs, list) else [reward_funcs]
        for index, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, str):
                self.reward_funcs[index] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func,
                    num_labels=1,
                    **model_init_kwargs,
                )

        if reward_processing_classes is None:
            self.reward_processing_classes = [None] * len(self.reward_funcs)
        elif isinstance(reward_processing_classes, list):
            self.reward_processing_classes = list(reward_processing_classes)
        else:
            self.reward_processing_classes = [reward_processing_classes]
        if len(self.reward_processing_classes) != len(self.reward_funcs):
            raise ValueError("The number of reward processing classes must match the number of reward functions.")
        
        cache_dir = model_init_kwargs.get("cache_dir")
        trust_remote_code = model_init_kwargs.get("trust_remote_code", True)
        for index, (reward_processing_class, reward_func) in enumerate(
            zip(self.reward_processing_classes, self.reward_funcs)
        ):
            if not isinstance(reward_func, PreTrainedModel):
                continue
            if reward_processing_class is None:
                reward_processing_class = AutoTokenizer.from_pretrained(
                    reward_func.config._name_or_path,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir,
                )
            if reward_processing_class.pad_token_id is None:
                reward_processing_class.pad_token = reward_processing_class.eos_token
            reward_func.config.pad_token_id = reward_processing_class.pad_token_id
            self.reward_processing_classes[index] = reward_processing_class

        def data_collator(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return features
        
        # GRPO arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = int(args.num_generations)
        if self.num_generations <= 1:
            raise ValueError("GRPO requires num_generations > 1.")
        
        self.beta = float(args.beta)
        
        #| inject GRPO KL approximator argument
        self.kl_approximator = script_args.kl_approximator
        if self.kl_approximator not in {"k3", "k1", "kimikl", "fullkimi"}:
            raise ValueError("kl_approximator must be one of: k3, k1, kimikl, fullkimi.")
        self.use_kl = bool(script_args.use_kl or args.beta > 0.0 or self.kl_approximator == "fullkimi")
              
        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)

        self._last_loaded_step = 0
        self._resume_from_checkpoint = bool(getattr(args, "resume_from_checkpoint", None))
        self._force_next_vllm_sync = self._resume_from_checkpoint
        self.llm = None
        self.sampling_params = None
        
        # Trainer
        trace_openr1("before Trainer.__init__")
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        trace_openr1("after Trainer.__init__")
        self.model_accepts_loss_kwargs = False
        register_rank_cuda_device()

        if self.ref_model is not None:
            trace_openr1("before preparing ref_model")
            self.ref_model.eval()
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            trace_openr1("after preparing ref_model")
        
        for index, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[index] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # In a micro-batch, num_generations must divide the global batch size.
        global_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes
        possible_values = [
            value for value in range(2, global_batch_size + 1)
            if global_batch_size % value == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({self.accelerator.num_processes} x "
                f"{self.args.per_device_train_batch_size}) must be evenly divisible by "
                f"num_generations ({self.num_generations}). Valid values: {possible_values}."
            )

        if self.args.eval_strategy != "no":
            global_eval_batch_size = self.args.per_device_eval_batch_size * self.accelerator.num_processes
            possible_eval_values = [
                value for value in range(2, global_eval_batch_size + 1)
                if global_eval_batch_size % value == 0
            ]
            if self.num_generations not in possible_eval_values:
                raise ValueError(
                    f"The global eval batch size ({self.accelerator.num_processes} x "
                    f"{self.args.per_device_eval_batch_size}) must be evenly divisible by "
                    f"num_generations ({self.num_generations}). Valid values: {possible_eval_values}."
                )

        set_seed(self.args.seed, device_specific=True)

        temperature_func = getattr(self.script_args, "temperature_func", None)
        if temperature_func is None:
            raise ValueError("temperature_func is required for PR1-aligned TeoNext GRPO.")
        if temperature_func not in {"constant", "linear"}:
            raise ValueError("temperature_func must be one of: constant, linear.")
        init_temp = float(self.script_args.temperature_begin)
        final_temp = float(self.script_args.temperature_end)
        if temperature_func == "linear" and init_temp > final_temp:
            raise ValueError("temperature_begin must be less than or equal to temperature_end for linear schedule.")

        if temperature_func == "constant":
            self.temperature_func = lambda _step: float(self.args.temperature)
        else:
            total_steps = int(getattr(self.args, "max_steps", -1) or -1)
            if total_steps <= 0:
                if self.train_dataset is None:
                    raise ValueError("train_dataset is required to estimate linear temperature schedule steps.")
                sample_count = len(self.train_dataset) * int(self.args.num_generations)
                denominator = (
                    int(self.args.per_device_train_batch_size)
                    * int(self.accelerator.num_processes)
                    * int(self.args.gradient_accumulation_steps)
                )
                if denominator <= 0:
                    raise ValueError("per-device batch size, process count, and gradient accumulation must be positive.")
                total_steps = max(1, math.ceil(sample_count / denominator))

            def linear_temperature(step: int) -> float:
                bounded_step = min(max(int(step), 0), total_steps)
                return init_temp + (final_temp - init_temp) * (bounded_step / total_steps)

            self.temperature_func = linear_temperature

        self._vllm_initialized = False
        trace_openr1("vLLM init deferred until after DeepSpeed prepare")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "images",
                "ground_truth",
                "is_negative",
                "num_images",
                "sample_id",
                "source_name",
                "dataset",
                "task",
                "_openr1_index",
            ]

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        return RepeatRandomSampler(dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _prepare_inputs(self, inputs):
        return inputs

    def _max_images_per_prompt(self) -> int:
        max_images_per_prompt = 1
        for dataset in (self.train_dataset, self.eval_dataset):
            if dataset is None:
                continue
            samples = getattr(dataset, "samples", None)
            if samples is None:
                samples = [dataset[index] for index in range(len(dataset))]
            for sample in samples:
                max_images_per_prompt = max(max_images_per_prompt, len(sample["images"]))
        return max_images_per_prompt

    def _resolve_rollout_device(self) -> str:
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count <= 0:
            raise ValueError("vLLM rollout requires at least one visible CUDA device.")
        if cuda_device_count <= self.accelerator.num_processes:
            raise ValueError(
                "OpenR1 vLLM rollout reserves the last visible CUDA device for generation. "
                f"Torch sees {cuda_device_count} CUDA device(s), but training uses "
                f"{self.accelerator.num_processes} process(es). Reduce NUM_GPUS/torchrun "
                "processes so at least one visible GPU remains dedicated to vLLM."
            )
        return f"cuda:{cuda_device_count - 1}"

    def _resolve_rollout_max_model_len(self) -> int:
        if self.max_prompt_length is None or self.max_completion_length is None:
            raise ValueError("max_prompt_length and max_completion_length are required for vLLM rollout.")
        rollout_max_model_len = int(self.max_prompt_length) + int(self.max_completion_length)
        if rollout_max_model_len <= 0:
            raise ValueError("max_prompt_length + max_completion_length must be positive for vLLM rollout.")
        return rollout_max_model_len

    def _initialize_vllm_rollout(self) -> None:
        if not self.accelerator.is_main_process:
            return
        if self.llm is not None and self.sampling_params is not None:
            return

        trace_openr1("main process before deferred vLLM init")
        if not self.model_name_or_path:
            raise ValueError("vLLM rollout requires --model_name_or_path to point at a saved TeoNext checkpoint.")
        if LLM is None or SamplingParams is None or EngineArgs is None:
            raise RuntimeError("vLLM rollout requires the optional vllm package to be installed.") from VLLM_IMPORT_ERROR

        register_teonext_vllm_model()
        rollout_device = self._resolve_rollout_device()
        rollout_max_model_len = self._resolve_rollout_max_model_len()
        max_images_per_prompt = self._max_images_per_prompt()
        llm_kwargs = {
            "model": self.model_name_or_path,
            "trust_remote_code": True,
            "device": rollout_device,
            "max_model_len": rollout_max_model_len,
            "gpu_memory_utilization": self.args.vllm_gpu_memory_utilization,
            "enable_prefix_caching": True,
            "limit_mm_per_prompt": {"image": max_images_per_prompt},
            # Limit concurrent sequences so that max_num_batched_tokens/max_num_seqs
            # exceeds the per-image token count (392), avoiding the vLLM multimodal
            # profiling warning about embeddings not fitting in the profiling window.
            # "max_num_seqs": 128,
        }
        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None,
        )
        original_create_engine_config = EngineArgs.create_engine_config

        def create_engine_config_on_rollout_device(engine_args, *args_, **kwargs_):
            vllm_config = original_create_engine_config(engine_args, *args_, **kwargs_)
            if rollout_device.startswith("cuda:"):
                vllm_config.device_config.device = torch.device(rollout_device)
                vllm_config.device_config.device_type = "cuda"
            return vllm_config

        engine_config_patch = patch.object(
            EngineArgs,
            "create_engine_config",
            create_engine_config_on_rollout_device,
        )
        previous_cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        try:
            with world_size_patch, profiling_patch, engine_config_patch:
                self.llm = LLM(**llm_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Installed vLLM does not accept the expected LLM constructor arguments for TeoNext rollout."
            ) from exc
        finally:
            if previous_cuda_device is not None:
                torch.cuda.set_device(previous_cuda_device)

        self.sampling_params = SamplingParams(
            n=1,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=50,
            max_tokens=self.max_completion_length,
        )

        trace_openr1("main process after deferred vLLM init")

    def _ensure_vllm_rollout(self) -> None:
        if self._vllm_initialized:
            return
        self._initialize_vllm_rollout()
        trace_openr1("before accelerator.wait_for_everyone after deferred vLLM init")
        self.accelerator.wait_for_everyone()
        trace_openr1("after accelerator.wait_for_everyone after deferred vLLM init")
        self._vllm_initialized = True

    def train(self, *args, **kwargs):
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint")
        if resume_from_checkpoint is None and args:
            resume_from_checkpoint = args[0]
        if resume_from_checkpoint is None:
            resume_from_checkpoint = getattr(self.args, "resume_from_checkpoint", None)
        if resume_from_checkpoint:
            self._resume_from_checkpoint = True
            self._force_next_vllm_sync = True

        trace_openr1("before Trainer.train")
        with trace_train_runtime(self.accelerator, deepspeed):
            result = super().train(*args, **kwargs)
            trace_openr1("after Trainer.train")
            return result

    def get_train_dataloader(self):
        trace_openr1(
            f"before get_train_dataloader dataset_len={len(self.train_dataset) if self.train_dataset is not None else 'none'}"
        )
        dataloader = super().get_train_dataloader()
        trace_openr1(
            "after get_train_dataloader "
            f"dataloader={type(dataloader).__name__} "
            f"sampler={type(getattr(dataloader, 'sampler', None)).__name__} "
            f"batch_sampler={type(getattr(dataloader, 'batch_sampler', None)).__name__}"
        )
        return dataloader

    def create_optimizer(self):
        trace_openr1(f"before create_optimizer deepspeed={self.is_deepspeed_enabled}")
        optimizer = super().create_optimizer()
        trace_openr1(f"after create_optimizer optimizer={type(optimizer).__name__}")
        return optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        trace_openr1(
            f"before create_scheduler num_training_steps={num_training_steps} optimizer={type(optimizer).__name__}"
        )
        scheduler = super().create_scheduler(num_training_steps, optimizer)
        trace_openr1(f"after create_scheduler scheduler={type(scheduler).__name__}")
        return scheduler

    def _wrap_model(self, *args, **kwargs):
        trace_openr1("before _wrap_model")
        wrapped_model = super()._wrap_model(*args, **kwargs)
        trace_openr1(f"after _wrap_model model={type(wrapped_model).__name__}")
        return wrapped_model

    def training_step(self, model, inputs, num_items_in_batch=None):
        trace_openr1(f"before training_step batch_size={len(inputs) if isinstance(inputs, list) else 'non-list'}")
        loss = super().training_step(model, inputs, num_items_in_batch)
        trace_openr1("after training_step")
        return loss

    def _decode_vllm_image_inputs(self, inputs_vllm: list[dict[str, Any]]) -> list[dict[str, Any]]:
        decoded_inputs = []
        for request in inputs_vllm:
            mm_data = request.get("multi_modal_data")
            if not isinstance(mm_data, dict) or "image" not in mm_data:
                raise ValueError("Each vLLM rollout request must include multi_modal_data['image'].")
            decoded_images = []
            for image in mm_data["image"]:
                if isinstance(image, Image.Image):
                    decoded_images.append(image.convert("RGB") if image.mode != "RGB" else image)
                    continue
                if not isinstance(image, (str, os.PathLike)):
                    raise TypeError(f"vLLM rollout expected image path or PIL image, got {type(image).__name__}.")
                with Image.open(os.fspath(image)) as pil_image:
                    decoded_images.append(pil_image.convert("RGB").copy())
            decoded_request = dict(request)
            decoded_mm_data = dict(mm_data)
            decoded_mm_data["image"] = decoded_images
            decoded_request["multi_modal_data"] = decoded_mm_data
            decoded_inputs.append(decoded_request)
        return decoded_inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        trace_openr1(f"compute_loss start batch_size={len(inputs) if isinstance(inputs, list) else 'non-list'}")
        if return_outputs:
            raise ValueError("TeoNextVLLMGRPOTrainer does not support return_outputs.")
        if not isinstance(inputs, list):
            raise TypeError("TeoNext GRPO data collator must return a list of raw sample dictionaries.")

        self._ensure_vllm_rollout()

        if self.accelerator.is_main_process:
            if self.llm is None or self.sampling_params is None:
                raise RuntimeError("Main process has no vLLM LLM instance.")
            self.sampling_params.temperature = self.temperature_func(self.state.global_step)

        prompts = [sample["prompt"] for sample in inputs]
        images = [list(sample["images"]) for sample in inputs]
        for sample, prompt_text, sample_images in zip(inputs, prompts, images):
            image_token_count = prompt_text.count("<image>")
            if image_token_count != len(sample_images):
                sample_id = sample.get("sample_id", "<unknown>")
                raise ValueError(
                    f"Prepared GRPO prompt for sample {sample_id!r} contains {image_token_count} <image> token(s) "
                    f"but has {len(sample_images)} image(s). The dataset must prepare image placeholders before training."
                )

        batch_size = 1
        inputs_vllm = []
        for prompt_text, sample_images in zip(prompts, images):
            prompt, _sep, _eos_token_id = render_teonext_prompt(
                self.model,
                self.processing_class,
                prompt_text,
                [],
                expand_image_tokens=False,
            )
            for _ in range(batch_size):
                inputs_vllm.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": sample_images},
                    }
                )

        # First, have main process load weights if needed.  Fresh runs skip
        # step 0 because vLLM starts from the same base checkpoint.  Resumed
        # runs force one sync because Trainer restores checkpoint weights later.
        step = int(self.state.global_step)
        sync_steps = max(1, int(self.script_args.vllm_sync_steps))
        force_vllm_sync = self._force_next_vllm_sync
        if force_vllm_sync or (step != self._last_loaded_step and step % sync_steps == 0):
            unwrapped = self.accelerator.unwrap_model(model)
            # GatheredParameters is only needed for ZeRO stage 3 where parameters
            # are sharded across ranks.  For stage 0/1/2 parameters are fully
            # replicated; using GatheredParameters there is a no-op at best and
            # can introduce an unintended barrier at worst.
            if deepspeed is not None and self.is_deepspeed_enabled:
                ds_stage = int(
                    (self.accelerator.deepspeed_config or {})
                    .get("zero_optimization", {})
                    .get("stage", 0)
                )
                gather_context = (
                    deepspeed.zero.GatheredParameters(unwrapped.parameters())
                    if ds_stage >= 3
                    else nullcontext()
                )
            else:
                gather_context = nullcontext()

            with gather_context:
                if self.accelerator.is_main_process:
                    if self.llm is None:
                        raise RuntimeError("Main process has no vLLM LLM instance.")
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    # Merge LoRA deltas into base weights before building the
                    # state-dict so that the vLLM model (which uses vLLM parallel
                    # linears that PEFT cannot wrap) receives plain full-rank tensors.
                    lm = getattr(unwrapped, "language_model", None)
                    has_lora = lm is not None and hasattr(lm, "merge_adapter")
                    if has_lora:
                        lm.merge_adapter()
                    try:
                        trace_openr1(f"main process before vLLM weight sync step={step}")
                        llm_model.load_weights(_iter_vllm_sync_weights(unwrapped.state_dict().items()))
                        trace_openr1(f"main process after vLLM weight sync step={step}")
                    finally:
                        if has_lora:
                            lm.unmerge_adapter()

            self._last_loaded_step = step
            self._force_next_vllm_sync = False


        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        trace_openr1("before gather_object input counts")
        all_input_counts = gather_object([len(inputs_vllm)])
        trace_openr1(f"after gather_object input counts counts={all_input_counts}")
        local_start = sum(int(value) for value in all_input_counts[:self.accelerator.process_index])
        local_end = local_start + len(inputs_vllm)
        trace_openr1("before gather_object vLLM inputs")
        all_inputs_vllm = gather_object(inputs_vllm)
        trace_openr1(f"after gather_object vLLM inputs total={len(all_inputs_vllm)}")
        if self.accelerator.is_main_process:
            trace_openr1("main process before decoding vLLM image inputs")
            all_inputs_vllm = self._decode_vllm_image_inputs(all_inputs_vllm)
            trace_openr1("main process after decoding vLLM image inputs")
            trace_openr1("main process before vLLM generate")
            outputs = self.llm.generate(all_inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)
            trace_openr1("main process after vLLM generate")
            completion_id_lists = [
                list(output.token_ids)
                for completions in outputs
                for output in completions.outputs
            ]
            if len(completion_id_lists) != len(all_inputs_vllm):
                raise RuntimeError(
                    f"vLLM returned {len(completion_id_lists)} completions for "
                    f"{len(all_inputs_vllm)} prompts; expected one completion per prompt."
                )
        else:
            completion_id_lists = [None] * len(all_inputs_vllm)
        
        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # corresponding slice.
        trace_openr1("before broadcast_object_list completions")
        completion_id_lists = broadcast_object_list(completion_id_lists, from_process=0)
        trace_openr1("after broadcast_object_list completions")
        completion_id_lists = completion_id_lists[local_start:local_end]
        if len(completion_id_lists) != len(inputs):
            raise RuntimeError(
                f"Received {len(completion_id_lists)} vLLM completions for {len(inputs)} local GRPO samples."
            )

        prompt_completion_records = [
            self._prepare_teonext_prompt_completion_inputs(sample, completion_ids)
            for sample, completion_ids in zip(inputs, completion_id_lists)
        ]
        if not prompt_completion_records:
            raise ValueError("GRPO compute_loss received an empty batch.")

        logprob_records = []
        k1_terms = []
        k3_terms = []
        kimikl_terms = []
        ppl_terms = []
        ref_ppl_terms = []
        completion_lengths = []
        for sample, prompt_completion_inputs, prompt_length, completion_mask in prompt_completion_records:
            trace_openr1("before policy logps forward")
            policy_logps, _policy_entropy = self._get_per_token_logps(
                model,
                prompt_completion_inputs,
                prompt_length=prompt_length,
            )
            trace_openr1("after policy logps forward")
            completion_lengths.append(float(completion_mask.sum().detach().cpu()))
            ppl_terms.append(masked_mean(-policy_logps.unsqueeze(0), completion_mask.unsqueeze(0)).mean())

            with torch.inference_mode():
                if self.ref_model is not None:
                    trace_openr1("before ref_model logps forward")
                    ref_logps, _ref_entropy = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_inputs,
                        prompt_length=prompt_length,
                    )
                    trace_openr1("after ref_model logps forward")
                elif self.use_adapter_reference:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    language_model = getattr(unwrapped_model, "language_model", None)
                    if hasattr(unwrapped_model, "disable_adapter"):
                        adapter_context = unwrapped_model.disable_adapter()
                    elif hasattr(language_model, "disable_adapter"):
                        adapter_context = language_model.disable_adapter()
                    else:
                        raise RuntimeError("No TeoNext or language_model adapter is available to disable.")
                    with adapter_context:
                        ref_logps, _ref_entropy = self._get_per_token_logps(
                            model,
                            prompt_completion_inputs,
                            prompt_length=prompt_length,
                        )
                else:
                    raise RuntimeError("No TeoNext GRPO reference source is available.")
            kl_terms = compute_kl_terms(policy_logps, ref_logps)
            per_token_kl = select_kl_approximator(kl_terms, self.kl_approximator)
            k1_terms.append(masked_mean(kl_terms["k1"].unsqueeze(0), completion_mask.unsqueeze(0)).mean())
            k3_terms.append(masked_mean(kl_terms["k3"].unsqueeze(0), completion_mask.unsqueeze(0)).mean())
            kimikl_terms.append(masked_mean(kl_terms["kimikl"].unsqueeze(0), completion_mask.unsqueeze(0)).mean())
            ref_ppl_terms.append(masked_mean(-ref_logps.unsqueeze(0), completion_mask.unsqueeze(0)).mean())
            logprob_records.append((policy_logps, completion_mask, per_token_kl))

        completion_texts = self.processing_class.batch_decode(completion_id_lists, skip_special_tokens=False)
        clean_completions = [
            completion.split(self.processing_class.eos_token or "")[0].strip()
            if self.processing_class.eos_token else completion.strip()
            for completion in completion_texts
        ]
        rewards = [0.0 for _ in clean_completions]
        reward_breakdowns = [dict() for _ in clean_completions]
        reward_kwargs = {
            key: [sample.get(key) for sample in inputs]
            for key in (
                "prompt",
                "ground_truth",
                "is_negative",
                "num_images",
                "sample_id",
                "source_name",
                "dataset",
                "task",
            )
        }
        reward_kwargs["current_step"] = self.state.global_step
        reward_kwargs.update({
            "grounding_format_reward_weight": self.script_args.grounding_format_reward_weight,
            "grounding_iou_reward_weight": self.script_args.grounding_iou_reward_weight,
            "detection_format_reward_weight": self.script_args.detection_format_reward_weight,
            "detection_f1_reward_weight": self.script_args.detection_f1_reward_weight,
            "detection_iou_reward_weight": self.script_args.detection_iou_reward_weight,
            "detection_missing_penalty_weight": self.script_args.detection_missing_penalty_weight,
            "temporal_format_reward_weight": self.script_args.temporal_format_reward_weight,
            "temporal_exact_reward_weight": self.script_args.temporal_exact_reward_weight,
            "negative_no_box_reward_weight": self.script_args.negative_no_box_reward_weight,
            "negative_text_reward_weight": self.script_args.negative_text_reward_weight,
            "negative_sample_reward_scale": self.script_args.negative_sample_reward_scale,
            "iou_threshold": self.script_args.iou_threshold,
            "bbox_format": bbox_text_format_from_config(self.model.config),
        })
        for index, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_name = getattr(
                    getattr(reward_func, "config", None),
                    "_name_or_path",
                    reward_func.__class__.__name__,
                ).split("/")[-1]
            else:
                reward_name = getattr(reward_func, "__name__", reward_func.__class__.__name__)
            if isinstance(reward_func, PreTrainedModel):
                reward_processing_class = self.reward_processing_classes[index]
                if reward_processing_class is None:
                    raise ValueError("A reward processing class is required for pretrained reward models.")
                reward_inputs = reward_processing_class(
                    [
                        f"{prompt}{completion}"
                        for prompt, completion in zip(prompts, clean_completions)
                    ],
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    logits = reward_func(**reward_inputs).logits
                raw_rewards = [float(value) for value in logits[:, 0].detach().cpu().tolist()]
            else:
                raw_rewards = reward_func(prompts=prompts, completions=clean_completions, **reward_kwargs)
            if len(raw_rewards) != len(clean_completions):
                raise ValueError(
                    f"Reward function {reward_name} returned {len(raw_rewards)} rewards for "
                    f"{len(clean_completions)} completions."
                )
            for sample_index, raw_reward in enumerate(raw_rewards):
                if isinstance(raw_reward, dict):
                    reward_value = float(raw_reward.get("total_reward", raw_reward.get("reward", 0.0)))
                    for key, value in raw_reward.items():
                        reward_breakdowns[sample_index][key] = (
                            reward_breakdowns[sample_index].get(key, 0.0) + float(value)
                        )
                else:
                    reward_value = float(raw_reward)
                    reward_breakdowns[sample_index][reward_name] = reward_value
                rewards[sample_index] += reward_value

        all_rewards, local_reward_start = gather_rank_objects(rewards)
        all_sample_ids, _sample_id_start = gather_rank_objects([
            str(sample.get("sample_id") or "")
            for sample in inputs
        ])
        all_task_labels, _task_start = gather_rank_objects([
            str(sample.get("task") or "unknown")
            for sample in inputs
        ])
        validate_prompt_groups([str(value) for value in all_sample_ids], self.num_generations)
        advantages, zero_ratio = compute_group_advantages(
            [float(value) for value in all_rewards],
            self.num_generations,
            normalize=self.kl_approximator != "fullkimi",
        )
        local_advantages = advantages[local_reward_start:local_reward_start + len(rewards)]

        loss_terms = []
        for (policy_logps, completion_mask, per_token_kl), advantage in zip(logprob_records, local_advantages):
            if self.kl_approximator == "fullkimi":
                per_token_loss = -torch.exp(policy_logps) * float(advantage)
            else:
                per_token_loss = -torch.exp(policy_logps - policy_logps.detach()) * float(advantage)
            if self.use_kl:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            loss_terms.append(masked_mean(per_token_loss.unsqueeze(0), completion_mask.unsqueeze(0)).mean())

        loss = torch.stack(loss_terms).mean()
        metrics = {
            "completion_length": sum(completion_lengths) / max(1, len(completion_lengths)),
            "ppl": float(torch.stack(ppl_terms).mean().detach().cpu()),
        }
        if k1_terms:
            metrics["k1_kl"] = float(torch.stack(k1_terms).mean().detach().cpu())
            metrics["k3_kl"] = float(torch.stack(k3_terms).mean().detach().cpu())
            metrics["kimikl_kl"] = float(torch.stack(kimikl_terms).mean().detach().cpu())
            ref_ppl = torch.stack(ref_ppl_terms).mean()
            metrics["delta_ref_ppl"] = float((torch.stack(ppl_terms).mean() - ref_ppl).detach().cpu())

        all_breakdowns, _start = gather_rank_objects(reward_breakdowns)
        reward_count = max(1, len(all_rewards))
        reward_mean = sum(float(value) for value in all_rewards) / reward_count
        reward_std = math.sqrt(
            sum((float(value) - reward_mean) ** 2 for value in all_rewards) / reward_count
        )
        self._metrics["reward"].append(reward_mean)
        self._metrics["reward_std"].append(reward_std)
        self._metrics["advantages"].append(sum(local_advantages) / max(1, len(local_advantages)))
        self._metrics["zero_advantage_group_ratio"].append(zero_ratio)
        task_zero_ratios = compute_group_zero_ratios(
            [float(value) for value in all_rewards],
            [str(value) for value in all_task_labels],
            self.num_generations,
        )
        for label, ratio in task_zero_ratios.items():
            self._metrics[f"zero_advantage_group_ratio/task/{label}"].append(ratio)
        for key, value in metrics.items():
            self._metrics[key].append(float(value))
        if self.accelerator.is_main_process:
            self._metrics["temperature"].append(self.sampling_params.temperature)

        breakdown_sums: dict[str, float] = {}
        for breakdown in all_breakdowns:
            for key, value in breakdown.items():
                breakdown_sums[key] = breakdown_sums.get(key, 0.0) + float(value)
        for key, value in breakdown_sums.items():
            self._metrics[f"rewards/{key}"].append(value / max(1, len(all_breakdowns)))
        return loss

    def _prepare_teonext_prompt_completion_inputs(self, sample: dict[str, Any], completion_id_list: list[int]):
        if not completion_id_list:
            raise ValueError("vLLM returned an empty completion token list.")

        policy_base_model = _unwrap_model(self.model)
        question = sample["prompt"]
        pixel_values, num_patches_list = preprocess_images(
            sample["images"],
            self.image_processor,
            policy_base_model.config,
        )
        if pixel_values is None:
            raise ValueError(f"GRPO sample has no preprocessed image tensors: {sample.get('sample_id')}")

        device = self.accelerator.device
        pixel_values = pixel_values.to(device=device, dtype=get_tensor_dtype(policy_base_model))

        prompt, _sep, eos_token_id = render_teonext_prompt(
            policy_base_model,
            self.processing_class,
            question,
            num_patches_list,
        )
        prompt_inputs = self.processing_class(prompt, return_tensors="pt")
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_attention = prompt_inputs["attention_mask"].to(device)
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_attention = prompt_attention[:, -self.max_prompt_length:]

        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else self.processing_class.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for GRPO padding.")

        # Pad the completions, and concatenate them with the prompts.
        completion_ids = [torch.tensor(completion_id_list, dtype=torch.long, device=device)]
        completion_ids, completion_mask = pad_token_sequences(
            completion_ids,
            pad_token_id=pad_token_id,
            device=device,
        )
        completion_mask = mask_after_eos(completion_ids, completion_mask, eos_token_id=eos_token_id)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_attention, completion_mask], dim=1)
        image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device=device)
        prompt_completion_inputs = {
            "pixel_values": pixel_values,
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
            "labels": None,
        }
        prompt_length = prompt_ids.shape[1]
        return sample, prompt_completion_inputs, prompt_length, completion_mask.squeeze(0)

    def _get_per_token_logps(
        self,
        forward_model,
        prompt_completion_inputs: dict[str, Any],
        *,
        prompt_length: int,
    ):
        model_inputs = dict(prompt_completion_inputs)
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(
            dtype=get_tensor_dtype(_unwrap_model(forward_model))
        )
        outputs = forward_model(**model_inputs)
        logits = outputs.logits[:, :-1, :]
        shifted_ids = model_inputs["input_ids"][:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        probs = logits.softmax(dim=-1)
        entropies = -torch.sum(log_probs * probs, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=shifted_ids.unsqueeze(-1)).squeeze(-1)
        completion_logps = per_token_logps[:, prompt_length - 1:]
        completion_entropies = entropies[:, prompt_length - 1:]
        return completion_logps.squeeze(0), completion_entropies.squeeze(0)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        metrics = {key: sum(values) / len(values) for key, values in self._metrics.items() if values}
        logs = {**logs, **metrics}
        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)
        self._metrics.clear()


VLLMTeoNextGRPOTrainer = TeoNextVLLMGRPOTrainer
