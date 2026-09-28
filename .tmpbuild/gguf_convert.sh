set -e
cd /workspace_models/llama.cpp
M=/workspace_models/muse-glimmer-24L
OUT=/workspace_models/gguf_out
mkdir -p $OUT
echo "=== convert f16 ==="
python convert_hf_to_gguf.py "$M" --outtype f16 --outfile "$OUT/glimmer24l-f16.gguf" 2>&1 | tail -n 40
echo "=== F16-DONE ==="
ls -lh $OUT/
echo "=== quantize Q4_K_M ==="
./build/bin/llama-quantize "$OUT/glimmer24l-f16.gguf" "$OUT/glimmer24l-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -n 15
echo "=== QUANT-DONE ==="
ls -lh $OUT/
