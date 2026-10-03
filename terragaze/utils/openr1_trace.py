from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


def is_openr1_trace_enabled() -> bool:
    return os.environ.get("TEONEXT_OPENR1_TRACE", "0").lower() in {"1", "true", "yes"}


def _get_trace_rank() -> int | str:
    try:
        import torch.distributed as dist
    except Exception:
        return os.environ.get("RANK", "0")
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return os.environ.get("RANK", "0")


def trace_openr1(message: str) -> None:
    if not is_openr1_trace_enabled():
        return
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[teonext-openr1-trace rank={_get_trace_rank()} local_rank={local_rank}] {message}", flush=True)


def _describe_objects(values: tuple[Any, ...]) -> str:
    return ", ".join(type(value).__name__ for value in values) or "none"


@contextmanager
def trace_train_runtime(accelerator, deepspeed_module) -> Iterator[None]:
    if not is_openr1_trace_enabled():
        yield
        return

    original_prepare = accelerator.prepare
    original_deepspeed_initialize = (
        getattr(deepspeed_module, "initialize", None)
        if deepspeed_module is not None
        else None
    )

    def traced_prepare(*prepare_args, **prepare_kwargs):
        trace_openr1(
            f"before accelerator.prepare args={_describe_objects(prepare_args)} "
            f"kwargs={sorted(prepare_kwargs)}"
        )
        result = original_prepare(*prepare_args, **prepare_kwargs)
        result_values = result if isinstance(result, tuple) else (result,)
        trace_openr1(f"after accelerator.prepare result={_describe_objects(result_values)}")
        return result

    def traced_deepspeed_initialize(*ds_args, **ds_kwargs):
        trace_openr1(
            f"before deepspeed.initialize args={_describe_objects(ds_args)} "
            f"kwargs={sorted(ds_kwargs)}"
        )
        result = original_deepspeed_initialize(*ds_args, **ds_kwargs)
        result_values = result if isinstance(result, tuple) else (result,)
        trace_openr1(f"after deepspeed.initialize result={_describe_objects(result_values)}")
        return result

    accelerator.prepare = traced_prepare
    if original_deepspeed_initialize is not None:
        deepspeed_module.initialize = traced_deepspeed_initialize
    try:
        yield
    finally:
        accelerator.prepare = original_prepare
        if original_deepspeed_initialize is not None:
            deepspeed_module.initialize = original_deepspeed_initialize
