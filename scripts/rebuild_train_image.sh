#!/usr/bin/env bash
# Rebuild training Docker image after torch/torchvision fix.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-llm-finetune/unsloth:cuda12.1.0-r7}"
echo "Removing old broken tags…"
docker rmi llm-finetune/unsloth:cuda12.1.0 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r2 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r3 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r4 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r5 2>/dev/null || true
docker rmi llm-finetune/unsloth:cuda12.1.0-r6 2>/dev/null || true
docker rmi "$TAG" 2>/dev/null || true
echo "Building $TAG (Gemma4 + Qwen3.5 / transformers 5.14)…"
docker build -f docker/Dockerfile.unsloth -t "$TAG" .
echo "Runtime smoke (with GPU)…"
docker run --rm --gpus all "$TAG" python -c "
import torch
from transformers import TrainingArguments
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from peft import LoraConfig
from trl import SFTTrainer
assert 'qwen3_5' in CONFIG_MAPPING, 'missing qwen3_5'
assert 'gemma4' in CONFIG_MAPPING, 'missing gemma4'
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')
print('qwen3_5 + gemma4 OK')
print('READY')
"
echo "Hotovo: $TAG"
