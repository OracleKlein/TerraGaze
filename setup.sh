#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${SCRIPT_DIR}
cd "${REPO_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n terragaze python=3.12 -y
conda activate terragaze

pip install --upgrade pip setuptools wheel

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install \
  transformers==4.55.0 \
  tokenizers==0.21.4 \
  huggingface-hub==0.34.3 \
  safetensors==0.5.3 \
  sentencepiece==0.2.0 \
  einops==0.8.2 \
  timm==1.0.19 \
  numpy==2.2.6 \
  pillow==11.3.0 \
  pyyaml==6.0.2 \
  tqdm==4.67.1 \
  requests==2.32.4 \
  packaging==25.0 \
  psutil==7.2.2 \
  accelerate==1.9.0 \
  datasets==4.0.0 \
  triton==3.3.0 \
  ninja==1.13.0 \
  mmengine==0.10.7 \
  opencv-python==4.12.0.88 \
  imageio==2.37.0 \
  decord==0.6.0 \
  scipy==1.15.3 \
  pandas==2.3.1 \
  pyarrow==21.0.0 \
  fsspec==2025.3.0 \
  regex==2025.7.34 \
  protobuf==6.31.1 \
  shapely==2.1.1 \
  wandb==0.25.1 \
  tensorboard==2.20.0 \
  peft==0.17.1 \
  trl==0.20.0

pip install deepspeed==0.17.4 bitsandbytes==0.46.1

pip install vllm==0.9.2 \
  --extra-index-url https://download.pytorch.org/whl/cu128

pip install --upgrade --force-reinstall --no-deps nvidia-nccl-cu12==2.30.7

MAX_JOBS=16 NVCC_THREADS=1 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  pip install -v flash-attn==2.8.3 --no-build-isolation

pip install --no-deps -e .
