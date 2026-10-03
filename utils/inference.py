import argparse
import json
from pathlib import Path

from teonext.utils.system_prompt import use_system_prompt

_MISSING_INFERENCE_DEPENDENCIES = []

try:
    import torch
except ImportError:
    torch = None
    _MISSING_INFERENCE_DEPENDENCIES.append("torch")

try:
    from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
except ImportError:
    Image = None
    ImageFile = None
    PngImagePlugin = None
    UnidentifiedImageError = None
    _MISSING_INFERENCE_DEPENDENCIES.append("pillow")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    AutoConfig = None
    AutoImageProcessor = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    _MISSING_INFERENCE_DEPENDENCIES.append("transformers")

try:
    from teonext.model import TeoNextConfig, TeoNextModel
except ImportError:
    TeoNextConfig = None
    TeoNextModel = None
    _MISSING_INFERENCE_DEPENDENCIES.append("teonext.model")

try:
    from teonext.utils.constants import (
        GROUNDING_SPECIAL_TOKENS,
        IMAGE_SPECIAL_TOKENS,
    )
except ImportError:
    GROUNDING_SPECIAL_TOKENS = ()
    IMAGE_SPECIAL_TOKENS = ()
    _MISSING_INFERENCE_DEPENDENCIES.append("teonext.utils.constants")

from teonext.utils.bbox import (
    bbox_coord_tokens,
    bbox_text_format_from_config,
    convert_bracket_bboxes_to_coord_tokens,
    extract_bboxes,
)
from teonext.utils.answer_matching import annotate_answer_metadata
from teonext.utils.utils import apply_llm_lora
from teonext.utils.utils import checkpoint_uses_unmerged_lora
from teonext.utils.utils import ensure_media_placeholders
from teonext.utils.utils import get_image_size
from teonext.utils.utils import get_pad_color
from teonext.utils.utils import get_tensor_dtype
from teonext.utils.utils import has_flash_attn
from teonext.utils.utils import load_sharded_state_dict
from teonext.utils.utils import normalize_to_list
from teonext.utils.utils import preprocess_images
from teonext.utils.utils import resolve_checkpoint_files
from teonext.utils.utils import sort_media_by_timestamps

if Image is not None:
    Image.MAX_IMAGE_PIXELS = None
if ImageFile is not None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
if PngImagePlugin is not None:
    PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def _require_inference_dependencies():
    if not _MISSING_INFERENCE_DEPENDENCIES:
        return
    missing = ", ".join(dict.fromkeys(_MISSING_INFERENCE_DEPENDENCIES))
    raise RuntimeError(
        "TEONext inference requires missing dependency/module(s): "
        f"{missing}. Install the package dependencies before loading a model."
    )


def _flash_attn_available():
    return has_flash_attn

def infer_model_dtype(config):
    dtype = getattr(config, "torch_dtype", None) or getattr(config.llm_config, "torch_dtype", None)
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    return parse_torch_dtype(dtype)

def parse_torch_dtype(value):
    if isinstance(value, torch.dtype):
        return value
    value = value.lower()
    mapping = {
        "auto": None,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[value]


def get_model_name(model_path, vision_path=None, llm_path=None):
    if model_path:
        return Path(model_path).name
    vision_name = Path(vision_path).name if vision_path else "vision"
    llm_name = Path(llm_path).name if llm_path else "llm"
    return f"teonext_{vision_name}_{llm_name}"


def build_model_with_lora(config, model_dtype):
    model = TeoNextModel(config)
    model.to(dtype=model_dtype)

    if config.use_llm_lora:
        try:
            apply_llm_lora(
                model,
                config.use_llm_lora,
                llm_lora_alpha=getattr(config, "llm_lora_alpha", None),
            )
        except ImportError as exc:
            raise RuntimeError("peft is required to rebuild and load an unmerged LoRA checkpoint.") from exc
        model.language_model.to(dtype=model_dtype)

    return model


def add_special_tokens(
    tokenizer,
    model,
    add_grounding_special_tokens=True,
    use_bbox_coord_tokens=False,
    bbox_coord_token_max=1000,
):
    special_tokens = list(IMAGE_SPECIAL_TOKENS)
    if add_grounding_special_tokens:
        special_tokens.extend(GROUNDING_SPECIAL_TOKENS)
    if use_bbox_coord_tokens:
        special_tokens.extend(bbox_coord_tokens(bbox_coord_token_max))

    num_new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
    if num_new_tokens <= 0:
        return

    model.language_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    new_token_start = len(tokenizer) - num_new_tokens
    new_token_end = len(tokenizer)
    input_embeddings = model.language_model.get_input_embeddings().weight.data
    output_embeddings = model.language_model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:new_token_start].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:new_token_start].mean(dim=0, keepdim=True)
    input_embeddings[new_token_start:new_token_end] = input_embeddings_avg
    output_embeddings[new_token_start:new_token_end] = output_embeddings_avg

    resized_vocab_size = model.language_model.get_input_embeddings().weight.size(0)
    model.config.llm_config.vocab_size = resized_vocab_size
    model.language_model.config.vocab_size = resized_vocab_size


def _apply_inference_device(model, device, torch_dtype, device_map, quantization_config):
    if device_map is not None or quantization_config is not None:
        return model
    if torch_dtype is None:
        return model.to(device)
    return model.to(device=device, dtype=torch_dtype)


def _has_image_processor_config(path):
    path = Path(path)
    if not path.is_dir():
        return False
    return any(
        (path / filename).exists()
        for filename in ("preprocessor_config.json", "image_processor_config.json")
    )


def _image_processor_candidates(model_path=None, vision_path=None):
    candidates = []

    def add_candidate(source):
        if source is None:
            return
        source = str(source)
        if source not in candidates:
            candidates.append(source)

    add_candidate(model_path)

    if model_path is not None:
        checkpoint_dir = Path(model_path)
        if checkpoint_dir.name.startswith("checkpoint-"):
            parent_dir = checkpoint_dir.parent
            if _has_image_processor_config(parent_dir):
                add_candidate(parent_dir)

    add_candidate(vision_path)
    return candidates


def load_image_processor(
    model_path=None,
    vision_path=None,
    use_fast_image_processor=False,
    cache_dir=None,
):
    candidates = _image_processor_candidates(model_path=model_path, vision_path=vision_path)
    if not candidates:
        raise ValueError("model_path or vision_path is required to load the image processor.")

    errors = []
    for processor_source in candidates:
        try:
            return AutoImageProcessor.from_pretrained(
                processor_source,
                trust_remote_code=True,
                use_fast=use_fast_image_processor,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            errors.append(f"{processor_source}: {exc}")

    details = "\n".join(errors)
    raise RuntimeError(
        "Could not load a TeoNext image processor from model_path or vision_path. "
        "If you are evaluating an intermediate checkpoint, pass the original "
        "--vision_path or copy preprocessor_config.json into the checkpoint directory. "
        f"Tried:\n{details}"
    )


def load_model(
    model_path=None,
    vision_path=None,
    llm_path=None,
    device="cuda",
    device_map=None,
    torch_dtype="bfloat16",
    load_4bit=False,
    load_8bit=False,
    use_fast_tokenizer=False,
    use_fast_image_processor=False,
    cache_dir=None,
    conv_style=None,
    select_layer=None,
    modality_bridge_type="attention_pooling",
    attention_pooling_query_type="mean",
    dynamic_image_size=None,
    use_thumbnail=None,
    min_dynamic_patch=None,
    max_dynamic_patch=None,
    use_flash_attn=None,
):
    _require_inference_dependencies()
    if model_path is None and (vision_path is None or llm_path is None):
        raise ValueError("Please provide model_path, or both vision_path and llm_path.")
    if device.startswith("cuda") and not torch.cuda.is_available() and device_map is None:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    resolved_dtype = parse_torch_dtype(torch_dtype)

    quantization_config = None
    if load_4bit and load_8bit:
        raise ValueError("Only one of load_4bit/load_8bit can be enabled.")
    if load_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    if load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    flash_attn_available = _flash_attn_available()
    if use_flash_attn is None:
        use_flash_attn = flash_attn_available
    attn_implementation = "flash_attention_2" if use_flash_attn and flash_attn_available else "eager"

    tokenizer_path = model_path or llm_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=use_fast_tokenizer,
        cache_dir=cache_dir,
    )

    image_processor = load_image_processor(
        model_path=model_path,
        vision_path=vision_path,
        use_fast_image_processor=use_fast_image_processor,
        cache_dir=cache_dir,
    )

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "cache_dir": cache_dir,
    }
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if model_path is not None:
        config = TeoNextConfig.from_pretrained(model_path, cache_dir=cache_dir)
        config.llm_config._attn_implementation = attn_implementation

        config.vision_config._attn_implementation = attn_implementation
        if hasattr(config.vision_config, "use_flash_attn"):
            config.vision_config.use_flash_attn = attn_implementation == "flash_attention_2"

        if conv_style is not None:
            config.template = conv_style
        if select_layer is not None:
            config.select_layer = select_layer
        if dynamic_image_size is not None:
            config.dynamic_image_size = dynamic_image_size
        if use_thumbnail is not None:
            config.use_thumbnail = use_thumbnail
        if min_dynamic_patch is not None:
            config.min_dynamic_patch = min_dynamic_patch
        if max_dynamic_patch is not None:
            config.max_dynamic_patch = max_dynamic_patch

        if checkpoint_uses_unmerged_lora(model_path, config):
            if device_map is not None or quantization_config is not None:
                raise NotImplementedError(
                    "Loading an unmerged LoRA checkpoint currently does not support device_map/load_4bit/load_8bit."
                )
            print(
                "[TeoNext inference] Detected an unmerged LLM LoRA checkpoint. "
                "Loading can be slow because the model is rebuilt before weights are loaded. "
                "For evaluation, merge it first with scripts/support/merge_lora_checkpoint.sh.",
                flush=True,
            )
            load_dtype = resolved_dtype or infer_model_dtype(config)
            model = build_model_with_lora(config, load_dtype)
            checkpoint_files, checkpoint_keys = resolve_checkpoint_files(model_path)
            load_sharded_state_dict(model, checkpoint_files, checkpoint_keys)
        else:
            model = TeoNextModel.from_pretrained(model_path, config=config, **model_kwargs)
    else:
        config_path = Path(vision_path) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        model_type = raw_config.get("model_type")
        if model_type == "siglip":
            from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
            from transformers.models.siglip.modeling_siglip import SiglipVisionModel
            vision_config = SiglipVisionConfig.from_pretrained(vision_path, cache_dir=cache_dir)
            vision_config._attn_implementation = attn_implementation
            if hasattr(vision_config, "use_flash_attn"):
                vision_config.use_flash_attn = attn_implementation == "flash_attention_2"
            vision_model = SiglipVisionModel.from_pretrained(vision_path, config=vision_config, **model_kwargs)
        elif model_type == "clip":
            from transformers.models.clip.configuration_clip import CLIPVisionConfig
            from transformers.models.clip.modeling_clip import CLIPVisionModel
            vision_config = CLIPVisionConfig.from_pretrained(vision_path, cache_dir=cache_dir)
            vision_config._attn_implementation = attn_implementation
            if hasattr(vision_config, "use_flash_attn"):
                vision_config.use_flash_attn = attn_implementation == "flash_attention_2"
            vision_model = CLIPVisionModel.from_pretrained(vision_path, config=vision_config, **model_kwargs)
        else:
            raise ValueError(f"Unsupported vision backbone config model_type={model_type} from {config_path}")

        llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, cache_dir=cache_dir)
        llm_config._attn_implementation = attn_implementation
        language_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            config=llm_config,
            trust_remote_code=True,
            **model_kwargs,
        )

        config = TeoNextConfig(
            vision_config=vision_config.to_dict(),
            llm_config=llm_config.to_dict(),
            template=conv_style,
            select_layer=select_layer if select_layer is not None else -1,
            modality_bridge_type=modality_bridge_type,
            attention_pooling_query_type=attention_pooling_query_type,
            use_pixel_shuffle=False,
            dynamic_image_size=bool(dynamic_image_size),
            use_thumbnail=bool(use_thumbnail),
            min_dynamic_patch=min_dynamic_patch or 1,
            max_dynamic_patch=max_dynamic_patch or 6,
        )
        model = TeoNextModel(config, vision_model=vision_model, language_model=language_model)

    add_special_tokens(
        tokenizer,
        model,
        add_grounding_special_tokens=config.add_grounding_special_tokens,
        use_bbox_coord_tokens=getattr(config, "use_bbox_coord_tokens", False),
        bbox_coord_token_max=getattr(config, "bbox_coord_token_max", 1000),
    )
    model = _apply_inference_device(model, device, resolved_dtype, device_map, quantization_config)
    model.eval()
    return tokenizer, model, image_processor


def run_inference_single(
    model,
    tokenizer,
    image_processor,
    question,
    image_paths=None,
    timestamps=None,
    prompt_strategy="interleave",
    chronological_prefix=True,
    temperature=0.2,
    max_new_tokens=256,
    system_prompt=None,
):
    if getattr(model.config, "use_bbox_coord_tokens", False):
        question = convert_bracket_bboxes_to_coord_tokens(
            question,
            max_coord=getattr(model.config, "bbox_coord_token_max", 1000),
        )
    image_paths = normalize_to_list(image_paths)
    timestamps = normalize_to_list(timestamps)
    if image_paths and timestamps:
        image_paths, timestamps = sort_media_by_timestamps(image_paths, timestamps)

    if chronological_prefix:
        question = question.replace("times:", "times in chronological order:")

    question = ensure_media_placeholders(question, len(image_paths), prompt_strategy)
    pixel_values = None
    num_patches_list = None
    if image_paths:
        pixel_values, num_patches_list = preprocess_images(image_paths, image_processor, model.config)
        pixel_values = pixel_values.to(device=model.device, dtype=get_tensor_dtype(model))

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_config["temperature"] = temperature
    if tokenizer.pad_token_id is not None:
        generation_config["pad_token_id"] = tokenizer.pad_token_id
    with use_system_prompt(model, system_prompt):
        return model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
        )


def run_inference(
    dataset,
    model,
    tokenizer,
    image_processor,
    prompt_strategy,
    chronological_prefix,
    temperature,
    max_new_tokens,
    sample_ids=None,
    disable_tqdm=False,
    output_callback=None,
):
    if sample_ids is not None and len(sample_ids) != len(dataset):
        raise ValueError("sample_ids length must match dataset length.")

    config = getattr(model, "config", None)
    bbox_format = bbox_text_format_from_config(config)
    use_bbox_coord_tokens = getattr(config, "use_bbox_coord_tokens", False)
    bbox_coord_token_max = getattr(config, "bbox_coord_token_max", 1000)
    outputs = []
    for idx, example in enumerate(tqdm(dataset, disable=disable_tqdm)):
        question = example["conversations"][0]["value"]
        ground_truth = example["conversations"][1]["value"]
        if use_bbox_coord_tokens:
            question = convert_bracket_bboxes_to_coord_tokens(
                question,
                max_coord=bbox_coord_token_max,
                sample_hint=f"sample_index={idx}, role=input",
            )
            ground_truth = convert_bracket_bboxes_to_coord_tokens(
                ground_truth,
                max_coord=bbox_coord_token_max,
                sample_hint=f"sample_index={idx}, role=ground_truth",
            )
        response = run_inference_single(
            model,
            tokenizer,
            image_processor,
            question,
            example.get("video", example.get("image", [])),
            timestamps=example.get("timestamp", []),
            prompt_strategy=prompt_strategy,
            chronological_prefix=chronological_prefix,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **({"system_prompt": example["system_prompt"]} if "system_prompt" in example else {}),
        )
        output = {
            "question": question,
            "response": response,
            "ground_truth": ground_truth,
            "task": example["task"],
            "_sample_id": sample_ids[idx] if sample_ids is not None else idx,
            "bbox_format": bbox_format,
        }
        is_core_teonext_bench = (str(example.get("dataset") or ""), str(example.get("task") or "")) in {
            ("DynamicEarthNet", "question_answering"),
            ("DynamicEarthNet", "region_based_question_answering"),
            ("DynamicEarthNet", "temporal_referring_expression"),
            ("SECOND", "question_answering"),
            ("SECOND", "region_based_question_answering"),
            ("SECOND", "spatial_referring_expression"),
            ("Landsat-SCD", "question_answering"),
            ("Landsat-SCD", "region_based_question_answering"),
        }
        for metadata_key in (
            "cdchat_id",
            "cdchat_num_regions",
            "cdchat_protocol",
            "changechat_id",
            "changechat_image_id",
            "changechat_protocol",
            "changeflag",
            "dataset",
            "sub_task",
            "answer_format",
            "candidate_classes",
            "qag_id",
            "qag_img_id",
            "qag_type",
            "mask_path",
            "dvl_id",
            "landsat30_au_qa_id",
            "system_prompt",
            "challenge_case",
            "vrsbench_question_id",
            "vrsbench_image_id",
            "referring_expression",
            "obj_cls",
            "unique",
            "vrsbench_question_type",
            "vrsbench_question",
        ):
            if is_core_teonext_bench and metadata_key in {"sub_task", "answer_format"}:
                continue
            if metadata_key in example:
                output[metadata_key] = example[metadata_key]

        polygon = example.get("polygon")
        if polygon is not None:
            output["polygon"] = polygon

        input_bboxes = extract_bboxes(question, bbox_format=bbox_format)
        output_bboxes = extract_bboxes(ground_truth, bbox_format=bbox_format)
        if input_bboxes:
            output["input_bboxes"] = input_bboxes
        if output_bboxes:
            output["output_bboxes"] = output_bboxes
        if not is_core_teonext_bench:
            annotate_answer_metadata(output)
        outputs.append(output)
        if output_callback is not None:
            output_callback(output)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--vision_path", default=None)
    parser.add_argument("--llm_path", default=None)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--image_path", type=str, nargs="*", default=None)
    parser.add_argument("--timestamp", type=str, nargs="*", default=None)
    parser.add_argument("--prompt_strategy", type=str, default="interleave")
    parser.add_argument("--chronological_prefix", dest="chronological_prefix", action="store_true")
    parser.add_argument("--no_chronological_prefix", dest="chronological_prefix", action="store_false")
    parser.set_defaults(chronological_prefix=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--use_fast_image_processor", action="store_true")
    parser.add_argument("--conv_style", default=None)
    parser.add_argument("--select_layer", type=int, default=None)
    parser.add_argument("--modality_bridge_type", type=str, default="attention_pooling")
    parser.add_argument("--attention_pooling_query_type", type=str, default="mean")
    parser.add_argument("--dynamic_image_size", dest="dynamic_image_size", action="store_true", default=None)
    parser.add_argument("--no_dynamic_image_size", dest="dynamic_image_size", action="store_false", default=None)
    parser.add_argument("--use_thumbnail", dest="use_thumbnail", action="store_true", default=None)
    parser.add_argument("--no_use_thumbnail", dest="use_thumbnail", action="store_false", default=None)
    parser.add_argument("--min_dynamic_patch", type=int, default=None)
    parser.add_argument("--max_dynamic_patch", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()

    tokenizer, model, image_processor = load_model(
        model_path=args.model_path,
        vision_path=args.vision_path,
        llm_path=args.llm_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        cache_dir=args.cache_dir,
        use_fast_tokenizer=args.use_fast_tokenizer,
        use_fast_image_processor=args.use_fast_image_processor,
        conv_style=args.conv_style,
        select_layer=args.select_layer,
        modality_bridge_type=args.modality_bridge_type,
        attention_pooling_query_type=args.attention_pooling_query_type,
        dynamic_image_size=args.dynamic_image_size,
        use_thumbnail=args.use_thumbnail,
        min_dynamic_patch=args.min_dynamic_patch,
        max_dynamic_patch=args.max_dynamic_patch,
    )

    response = run_inference_single(
        model,
        tokenizer,
        image_processor,
        args.question,
        image_paths=args.image_path,
        timestamps=args.timestamp,
        prompt_strategy=args.prompt_strategy,
        chronological_prefix=args.chronological_prefix,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    output = {
        "model_name": get_model_name(args.model_path, args.vision_path, args.llm_path),
        "question": args.question,
        "image_path": args.image_path,
        "response": response,
    }
    print(response)

    if args.out_path is not None:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
