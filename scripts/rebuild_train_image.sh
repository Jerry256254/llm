#!/usr/bin/env bash
# Rebuild training Docker image after torch/torchvision fix.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-llm-finetune/unsloth:cuda12.1.0-r2}"
echo "Removing old images (optional)…"
docker rmi llm-finetune/unsloth:cuda12.1.0 2>/dev/null || true
docker rmi "$TAG" 2>/dev/null || true
echo "Building $TAG …"
docker build -f docker/Dockerfile.unsloth \
  --build-arg CUDA_VERSION=12.1.0 \
  -t "$TAG" .
echo "Smoke test imports inside image…"
docker run --rm --gpus all "$TAG" python3 -c "
import torch, torchvision
from transformers import TrainingArguments
print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())
try:
    import unsloth
    print('unsloth OK')
except Exception as e:
    print('unsloth', e)
print('OK')
"
echo "Hotovo: $TAG"
