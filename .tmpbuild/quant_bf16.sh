set -e
cd /workspace_models/llama.cpp
export LD_LIBRARY_PATH=/workspace_models/llama.cpp/build/bin:$LD_LIBRARY_PATH
M=/workspace_models/muse-glimmer-24L
OUT=/workspace_models/gguf_out
echo "=== convert bf16 ==="
python convert_hf_to_gguf.py "$M" --outtype bf16 --outfile "$OUT/g24-bf16.gguf" 2>&1 | tail -n 8
echo "=== quantize Q4_K_M ==="
./build/bin/llama-quantize "$OUT/g24-bf16.gguf" "$OUT/glimmer24l-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -n 12
echo "=== DONE ==="; ls -lh "$OUT/glimmer24l-Q4_K_M.gguf"
