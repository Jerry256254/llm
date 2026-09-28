#!/usr/bin/env bash
set -e
cd /workspace_models/llama.cpp
export LD_LIBRARY_PATH=/workspace_models/llama.cpp/build/bin:$LD_LIBRARY_PATH
OUT=/workspace_models/gguf_out
echo "=== quantize Q4_K_M ==="
./build/bin/llama-quantize "$OUT/glimmer24l-f16.gguf" "$OUT/glimmer24l-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -n 12
echo "=== QUANT-DONE ==="
ls -lh "$OUT/glimmer24l-Q4_K_M.gguf"
