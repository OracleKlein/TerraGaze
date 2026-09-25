import json
import os
import torch

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import Trainer, TrainingArguments
from transformers.trainer import has_length, is_datasets_available, is_sagemaker_mp_enabled, logger, seed_worker
from transformers.trainer_pt_utils import LengthGroupedSampler, get_length_grouped_indices


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


SPECIAL_LR_GROUPS = (
    ("mlp1", "mlp1.", "mlp_lr"),
    ("vision_model", "vision_model.", "vision_lr"),
    ("language_model", "language_model.", "llm_lr"),
)


def _get_special_lr_specs(args):
    specs = []
    for group_name, prefix, arg_name in SPECIAL_LR_GROUPS:
        lr = getattr(args, arg_name, None)
        if lr is not None:
            specs.append((group_name, prefix, lr, arg_name))
    return specs


def _has_special_lrs(args):
    return any(getattr(args, arg_name, None) is not None for _, _, arg_name in SPECIAL_LR_GROUPS)


def _load_deepspeed_config(args):
    hf_deepspeed_config = getattr(args, "hf_deepspeed_config", None)
    config = getattr(hf_deepspeed_config, "config", None)
    if isinstance(config, dict):
        return config

    deepspeed_config = getattr(args, "deepspeed", None)
    if isinstance(deepspeed_config, dict):
        return deepspeed_config
    if isinstance(deepspeed_config, str) and deepspeed_config and os.path.isfile(deepspeed_config):
        with open(deepspeed_config, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def validate_special_lr_deepspeed_config(args):
    if not _has_special_lrs(args):
        return

    deepspeed_config = _load_deepspeed_config(args)
    if isinstance(deepspeed_config, dict) and "optimizer" in deepspeed_config:
        raise ValueError(
            "mlp_lr, vision_lr, and llm_lr require Trainer-created optimizer parameter groups. "
            "The current DeepSpeed config defines an optimizer block, so DeepSpeed will replace custom "
            "groups with a single optimizer lr. Use teonext/config/deepspeed/zero_stage1_lr_config.json for multi-LR "
            "training, or remove the optimizer block from the DeepSpeed config."
        )


def build_teonext_optimizer_groups(opt_model, args, decay_parameters):
    trainable_named_params = [(name, param) for name, param in opt_model.named_parameters() if param.requires_grad]
    special_lr_specs = _get_special_lr_specs(args)
    matched_special_counts = {arg_name: 0 for _, _, _, arg_name in special_lr_specs}

    grouped = {}

    def add_param(group_name, decay_key, lr, param):
        group_key = (group_name, decay_key, lr)
        if group_key not in grouped:
            grouped[group_key] = {
                "params": [],
                "weight_decay": args.weight_decay if decay_key == "decay" else 0.0,
                "lr": lr,
                "name": f"{group_name}_{decay_key}",
            }
        grouped[group_key]["params"].append(param)

    for name, param in trainable_named_params:
        group_name = "default"
        group_lr = args.learning_rate
        matched_special_arg = None

        for special_group_name, prefix, lr, arg_name in special_lr_specs:
            if name.startswith(prefix):
                group_name = special_group_name
                group_lr = lr
                matched_special_arg = arg_name
                break

        if matched_special_arg is not None:
            matched_special_counts[matched_special_arg] += 1

        decay_key = "decay" if name in decay_parameters else "no_decay"
        add_param(group_name, decay_key, group_lr, param)

    missing = [arg_name for arg_name, count in matched_special_counts.items() if count == 0]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            f"Special learning rate(s) were set but matched no trainable parameters: {missing_names}. "
            "Check freeze flags and parameter-name prefixes before launching training."
        )

    return list(grouped.values())


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[idx::num_chunks] for idx in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]

    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_world_size_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[idx:idx + megabatch_size].tolist() for idx in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda idx: lengths[idx], reverse=True) for megabatch in megabatches]
    if megabatches:
        megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
        max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
        megabatches[0], megabatches[max_idx] = megabatches[max_idx], megabatches[0]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [idx for megabatch in megabatches for chunk in megabatch for idx in chunk]


def get_modality_length_grouped_indices(lengths, batch_size, world_size=1, mega_batch_mult=None, generator=None):
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    if any(length == 0 for length in lengths):
        raise ValueError("modality_length must not contain zero.")

    abs_lengths = [abs(length) for length in lengths]
    if all(length > 0 for length in lengths) or all(length < 0 for length in lengths):
        if world_size == 1:
            return get_length_grouped_indices(
                abs_lengths,
                batch_size,
                mega_batch_mult=mega_batch_mult,
                generator=generator,
            )
        return get_world_size_length_grouped_indices(abs_lengths, batch_size, world_size, generator=generator)

    mm_pairs = [(idx, length) for idx, length in enumerate(lengths) if length > 0]
    text_pairs = [(idx, -length) for idx, length in enumerate(lengths) if length < 0]
    mm_indices, mm_lengths = zip(*mm_pairs)
    text_indices, text_lengths = zip(*text_pairs)

    mm_shuffle = [
        mm_indices[idx]
        for idx in get_world_size_length_grouped_indices(list(mm_lengths), batch_size, world_size, generator=None)
    ]
    text_shuffle = [
        text_indices[idx]
        for idx in get_world_size_length_grouped_indices(list(text_lengths), batch_size, world_size, generator=None)
    ]

    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (world_size * batch_size * 4), 50)
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    megabatch_size = mega_batch_mult * world_size * batch_size
    mm_megabatches = [mm_shuffle[idx:idx + megabatch_size] for idx in range(0, len(mm_shuffle), megabatch_size)]
    text_megabatches = [
        text_shuffle[idx:idx + megabatch_size] for idx in range(0, len(text_shuffle), megabatch_size)
    ]
    additional_batch = mm_megabatches[-1] + text_megabatches[-1]
    megabatches = mm_megabatches[:-1] + text_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[idx] for idx in megabatch_indices]

    if additional_batch:
        megabatches.append(sorted(additional_batch, key=lambda idx: abs(lengths[idx]), reverse=True))

    if megabatches:
        megabatch_maximums = [abs(lengths[megabatch[0]]) for megabatch in megabatches]
        max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
        megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [idx for megabatch in megabatches for idx in megabatch]


class TeoNextLengthGroupedSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size, world_size, lengths, generator=None, group_by_modality=False):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                world_size=self.world_size,
                generator=self.generator,
            )
        else:
            indices = get_world_size_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        return iter(indices)


@dataclass
class TeoNextTrainingArguments(TrainingArguments):
    group_by_modality_length: bool = field(
        default=False,
        metadata={'help': 'Group training samples by modality first, then by approximate sequence length.'},
    )
    mlp_lr: Optional[float] = field(
        default=None,
        metadata={'help': 'Optional learning rate for the TeoNext modality bridge (mlp1).'},
    )
    vision_lr: Optional[float] = field(
        default=None,
        metadata={'help': 'Optional learning rate for the vision encoder.'},
    )
    llm_lr: Optional[float] = field(
        default=None,
        metadata={'help': 'Optional learning rate for the language model.'},
    )


class TeoNextTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_special_lr_deepspeed_config(self.args)

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = train_dataset.modality_length
            return TeoNextLengthGroupedSampler(
                self.args.train_batch_size,
                self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )

        if self.args.group_by_length:
            if is_datasets_available():
                import datasets
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = train_dataset.length
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        return RandomSampler(train_dataset)

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        data_collator = self.data_collator
        if is_datasets_available():
            import datasets
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = DataLoader(dataset, **dataloader_params)
        if not (self.args.split_annotations or getattr(self.args, 'use_packed_ds', False)):
            dataloader = self.accelerator.prepare(dataloader)

        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    def create_optimizer(self):
        validate_special_lr_deepspeed_config(self.args)
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = set(self.get_decay_parameter_names(opt_model))
            optimizer_grouped_parameters = build_teonext_optimizer_groups(
                opt_model,
                self.args,
                decay_parameters,
            )

            #| compatiable with both custom optimizer and default optimizer
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            override_keys = ("params", "model", "optimizer_dict")
            present_override_keys = [key for key in override_keys if key in optimizer_kwargs]
            if present_override_keys and _has_special_lrs(self.args):
                keys = ", ".join(present_override_keys)
                raise ValueError(
                    f"Optimizer kwargs contain {keys}, which would override TeoNext multi-LR parameter groups."
                )
            for key in present_override_keys:
                optimizer_grouped_parameters = optimizer_kwargs.pop(key)

            if getattr(self.args, "process_index", 0) == 0:
                for group in optimizer_grouped_parameters:
                    param_count = sum(param.numel() for param in group["params"])
                    logger.info(
                        "optimizer group %s: lr=%s, weight_decay=%s, params=%s",
                        group.get("name", "unnamed"),
                        group.get("lr", self.args.learning_rate),
                        group.get("weight_decay", 0.0),
                        param_count,
                    )

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
