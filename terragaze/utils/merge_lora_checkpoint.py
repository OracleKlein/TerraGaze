from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from teonext.utils.utils import checkpoint_uses_unmerged_lora


@dataclass
class OutputPlan:
    final_dir: Path
    save_dir: Path
    copy_to_existing_root: bool


def _load_teonext_merge_dependencies():
    try:
        from teonext.model import TeoNextConfig
        from teonext.utils.inference import load_model
    except ImportError as exc:
        raise RuntimeError(
            "merge_lora_checkpoint.py requires the TeoNext model/inference dependencies. "
            "Install the package with the training or OpenR1 extras before merging LoRA checkpoints."
        ) from exc
    return TeoNextConfig, load_model


def _new_hidden_sibling(path: Path, label: str) -> Path:
    return path.parent / f".{path.name}.{label}.{os.getpid()}.{uuid4().hex[:8]}"


def _prepare_output_plan(input_dir: Path, output_dir: Path, overwrite: bool) -> OutputPlan:
    resolved_input = input_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_input == resolved_output:
        raise ValueError("input_dir and output_dir must be different.")

    output_has_content = output_dir.exists() and any(output_dir.iterdir())
    if output_has_content and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Set OVERWRITE=1 or pass --overwrite to copy merged model files into it."
        )

    if output_has_content:
        save_dir = _new_hidden_sibling(output_dir, "merge-staging")
        save_dir.mkdir(parents=True)
        return OutputPlan(
            final_dir=output_dir,
            save_dir=save_dir,
            copy_to_existing_root=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return OutputPlan(final_dir=output_dir, save_dir=output_dir, copy_to_existing_root=False)


def _commit_output_plan(plan: OutputPlan) -> None:
    if not plan.copy_to_existing_root:
        return

    print(f"[merge_lora_checkpoint] Copying merged model files into output root: {plan.final_dir}", flush=True)
    for pattern in (
        "model*.safetensors",
        "model*.safetensors.index.json",
        "pytorch_model*.bin",
        "pytorch_model.bin.index.json",
        "adapter_model*.safetensors",
        "adapter_model.bin",
    ):
        for stale_file in plan.final_dir.glob(pattern):
            if stale_file.is_file():
                stale_file.unlink()

    for staged_item in plan.save_dir.iterdir():
        target = plan.final_dir / staged_item.name
        if target.exists():
            if target.is_dir():
                if target.name.startswith("checkpoint-"):
                    raise RuntimeError(f"Refusing to overwrite checkpoint directory: {target}")
                shutil.rmtree(target)
            else:
                target.unlink()
        if staged_item.is_dir():
            shutil.copytree(staged_item, target)
        else:
            shutil.copy2(staged_item, target)

    shutil.rmtree(plan.save_dir)


def merge_lora_checkpoint(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    cache_dir: str | None = None,
    max_shard_size: str = "5GB",
    overwrite: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input checkpoint directory does not exist: {input_dir}")

    TeoNextConfig, load_model = _load_teonext_merge_dependencies()
    config = TeoNextConfig.from_pretrained(input_dir, cache_dir=cache_dir)
    if not getattr(config, "use_llm_lora", 0):
        raise ValueError(f"Checkpoint does not declare an active LLM LoRA adapter: {input_dir}")
    if not checkpoint_uses_unmerged_lora(input_dir, config):
        raise ValueError(f"Checkpoint does not contain unmerged LLM LoRA weights: {input_dir}")

    plan = _prepare_output_plan(input_dir, output_dir, overwrite=overwrite)

    print(f"[merge_lora_checkpoint] Loading unmerged checkpoint: {input_dir}", flush=True)
    tokenizer, model, image_processor = load_model(
        model_path=str(input_dir),
        device=device,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )

    if not hasattr(model.language_model, "merge_and_unload"):
        raise RuntimeError("Loaded language_model does not support merge_and_unload().")

    print("[merge_lora_checkpoint] Merging LLM LoRA adapter into base language model.", flush=True)
    model.language_model = model.language_model.merge_and_unload()
    model.config.use_llm_lora = 0
    model.config.llm_lora_alpha = None

    print(f"[merge_lora_checkpoint] Saving merged checkpoint: {plan.save_dir}", flush=True)
    model.save_pretrained(
        plan.save_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(plan.save_dir)
    image_processor.save_pretrained(plan.save_dir)

    metadata = {
        "merged_llm_lora": True,
        "source_checkpoint": str(input_dir),
        "final_output_dir": str(output_dir),
        "copied_to_existing_root": plan.copy_to_existing_root,
    }
    with open(plan.save_dir / "conversion_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    _commit_output_plan(plan)
    print("[merge_lora_checkpoint] Done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-off helper to merge a TeoNext unmerged LLM LoRA checkpoint.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merge_lora_checkpoint(
        args.input_dir,
        args.output_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        cache_dir=args.cache_dir,
        max_shard_size=args.max_shard_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
