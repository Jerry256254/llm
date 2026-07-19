#!/usr/bin/env bash
# Rebuild training Docker image after torch/torchvision fix.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-llm-finetune/unsloth:cuda12.1.0-r5}"
echo "Removing old broken tags…"
docker rmi llm-finetune/unsloth:cuda12.1.0 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r2 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r3 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r4 2>/dev/null || true
docker rmi "$TAG" 2>/dev/null || true
echo "Building $TAG (Qwen3.5-ready PEFT stack)…"
docker build -f docker/Dockerfile.unsloth -t "$TAG" .
echo "Runtime smoke (with GPU)…"
docker run --rm --gpus all "$TAG" python -c "
import torch
from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')
print('READY')
"
echo "Hotovo: $TAG"
