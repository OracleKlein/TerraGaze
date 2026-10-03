import importlib.util
import json
import re
from collections import deque
from datetime import date, datetime
from pathlib import Path


# Dependency helpers

def _detect_flash_attn() -> bool:
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


has_flash_attn = _detect_flash_attn()


# Collection / prompt helpers

DEFAULT_VIDEO_TOKEN = "<video>"


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def replace_video_token(prompt, num_images, prompt_strategy):
    if prompt_strategy == "interleave":
        replacement = "".join(f"Image {idx + 1}: <image>\n" for idx in range(num_images))
    else:
        replacement = "<image>\n" * num_images
    return prompt.replace(DEFAULT_VIDEO_TOKEN, replacement.rstrip())


def ensure_media_placeholders(prompt, num_images, prompt_strategy):
    if num_images <= 0:
        return prompt
    if DEFAULT_VIDEO_TOKEN in prompt:
        prompt = replace_video_token(prompt, num_images, prompt_strategy)
    if "<image>" in prompt:
        if prompt.count("<image>") != num_images:
            raise ValueError(
                f"Prompt contains {prompt.count('<image>')} image token(s) but sample has {num_images} image(s)."
            )
        return prompt
    if prompt_strategy == "interleave" and num_images > 1:
        prefix = "".join(f"Image {idx + 1}: <image>\n" for idx in range(num_images))
    else:
        prefix = "<image>\n" * num_images
    return prefix + prompt


# Timestamp / media ordering helpers

_TIMESTAMP_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y")


def timestamp_sort_key(timestamp, index):
    if isinstance(timestamp, datetime):
        return (0, timestamp, index)
    if isinstance(timestamp, date):
        return (0, datetime.combine(timestamp, datetime.min.time()), index)

    text = str(timestamp).strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return (0, datetime.strptime(text, fmt), index)
        except ValueError:
            pass

    match = re.fullmatch(r"[Tt](\d+)", text)
    if match:
        return (1, int(match.group(1)), index)

    expected = ", ".join(_TIMESTAMP_FORMATS) + ", or t<N>"
    raise ValueError(f"Unsupported timestamp format {timestamp!r}; expected {expected}.")


def sort_media_by_timestamps(image_paths, timestamps):
    if not timestamps:
        return list(image_paths), []
    if len(image_paths) != len(timestamps):
        raise ValueError(
            f"Cannot sort media by timestamps with different lengths: "
            f"{len(image_paths)} image paths and {len(timestamps)} timestamps."
        )
    pairs = sorted(
        enumerate(zip(image_paths, timestamps)),
        key=lambda item: timestamp_sort_key(item[1][1], item[0]),
    )
    sorted_image_paths, sorted_timestamps = zip(*(item[1] for item in pairs))
    return list(sorted_image_paths), list(sorted_timestamps)


# Image processing helpers

def get_pad_color(image_processor):
    image_mean = getattr(image_processor, "image_mean", None)
    if image_mean is None:
        return (127, 127, 127)
    if isinstance(image_mean, (int, float)):
        image_mean = [image_mean] * 3
    try:
        return tuple(int(x * 255) for x in image_mean[:3])
    except Exception:
        return (127, 127, 127)


def get_image_size(config, image_processor):
    force_image_size = getattr(config, "force_image_size", None)
    if force_image_size is not None:
        return force_image_size
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        return size.get("height") or size.get("shortest_edge") or next(iter(size.values()))
    if isinstance(size, (tuple, list)):
        return size[0]
    if isinstance(size, int):
        return size
    return config.vision_config.image_size


def expand2square(pil_img, background_color):
    from PIL import Image

    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def preprocess_images(image_paths, image_processor, config):
    import torch
    from PIL import Image

    image_size = get_image_size(config, image_processor)
    pad_color = get_pad_color(image_processor)
    dynamic_image_size = getattr(config, "dynamic_image_size", False)
    use_thumbnail = getattr(config, "use_thumbnail", False)
    min_dynamic_patch = getattr(config, "min_dynamic_patch", 1)
    max_dynamic_patch = getattr(config, "max_dynamic_patch", 6)
    pad2square = getattr(config, "pad2square", False)

    pixel_values_list = []
    num_patches_list = []
    use_dynamic_preprocess = dynamic_image_size and len(image_paths) == 1
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        if use_dynamic_preprocess:
            images = dynamic_preprocess(
                image,
                min_num=min_dynamic_patch,
                max_num=max_dynamic_patch,
                image_size=image_size,
                use_thumbnail=use_thumbnail,
            )
        else:
            images = [expand2square(image, pad_color)] if pad2square else [image]

        tensor_patches = [image_processor.preprocess(patch, return_tensors="pt")["pixel_values"][0] for patch in images]
        pixel_values = torch.stack(tensor_patches)
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    if not pixel_values_list:
        return None, []
    return torch.cat(pixel_values_list, dim=0), num_patches_list


# Model inspection helpers

def get_tensor_dtype(model):
    import torch

    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def get_decoder_layers(module):
    """Resolve decoder layers across PEFT/base-model wrapper stacks."""
    queue = deque([module])
    seen = set()

    while queue:
        current = queue.popleft()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        layers = getattr(current, "layers", None)
        if layers is not None:
            return layers

        for attr in ("model", "base_model", "language_model"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                queue.append(child)

    raise AttributeError(f"Unable to locate decoder layers from model type {type(module).__name__}.")


def find_linear_module_names(module, exclude_keywords=None):
    import torch

    exclude_keywords = exclude_keywords or []
    names = set()
    for name, submodule in module.named_modules():
        if any(keyword in name for keyword in exclude_keywords):
            continue
        if isinstance(submodule, torch.nn.Linear):
            names.add(name.split(".")[-1])
    return sorted(names)


# LoRA / checkpoint helpers

def apply_llm_lora(model, rank, *, llm_lora_alpha=None, logger_=None):
    if rank <= 0:
        return None

    alpha = llm_lora_alpha if llm_lora_alpha is not None else 2 * rank
    if alpha <= 0:
        raise ValueError("llm_lora_alpha must be a positive integer when use_llm_lora > 0.")

    if hasattr(model, "wrap_llm_lora"):
        model.wrap_llm_lora(r=rank, lora_alpha=alpha)
    else:
        from peft import LoraConfig, get_peft_model

        target_modules = find_linear_module_names(
            model.language_model,
            exclude_keywords=["lm_head"],
        )
        if not target_modules:
            raise RuntimeError("No linear modules found for llm LoRA target_modules.")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        if logger_ is not None:
            logger_.info("Applied PEFT llm LoRA fallback with target_modules=%s", target_modules)

    model.config.use_llm_lora = rank
    model.config.llm_lora_alpha = alpha
    if logger_ is not None:
        logger_.info("Applied LLM LoRA with rank=%s alpha=%s", rank, alpha)
    return alpha


def resolve_checkpoint_files(src_dir):
    src_dir = Path(src_dir)
    index_path = src_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        checkpoint_keys = set(index["weight_map"].keys())
        return [src_dir / shard_name for shard_name in shard_names], checkpoint_keys

    single_shard = src_dir / "model.safetensors"
    if single_shard.exists():
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError("safetensors is required to load this checkpoint.") from exc
        with safe_open(str(single_shard), framework="pt", device="cpu") as f:
            checkpoint_keys = set(f.keys())
        return [single_shard], checkpoint_keys

    raise FileNotFoundError(f"No safetensors checkpoint found under: {src_dir}")


def load_sharded_state_dict(model, checkpoint_files, checkpoint_keys):
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError("safetensors is required to load this checkpoint.") from exc

    model_keys = set(model.state_dict().keys())
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    missing_keys = sorted(model_keys - checkpoint_keys)
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:10])
        raise RuntimeError(f"Unexpected keys in checkpoint: {preview}")
    if missing_keys:
        preview = ", ".join(missing_keys[:10])
        raise RuntimeError(f"Missing keys in checkpoint: {preview}")

    for checkpoint_file in checkpoint_files:
        shard_state = load_file(str(checkpoint_file), device="cpu")
        model.load_state_dict(shard_state, strict=False)


def is_merged_lora_conversion(model_path):
    meta_path = Path(model_path) / "conversion_meta.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return False
    return bool(metadata.get("merged_llm_lora"))


def checkpoint_uses_unmerged_lora(model_path, config):
    if not getattr(config, "use_llm_lora", 0):
        return False
    if is_merged_lora_conversion(model_path):
        return False

    try:
        _, checkpoint_keys = resolve_checkpoint_files(model_path)
    except FileNotFoundError:
        return False

    return any(".lora_A." in key or ".lora_B." in key or ".base_layer." in key for key in checkpoint_keys)


# Parsing helpers

def extract_bboxes(bbox_str, bbox_format="bracket"):
    from teonext.utils.bbox import extract_bboxes as parse_bboxes

    return [list(box) for box in parse_bboxes(bbox_str, bbox_format=bbox_format)]
