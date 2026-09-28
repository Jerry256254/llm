set -e
cd /workspace_build/llama.cpp
export LD_LIBRARY_PATH=/workspace_build/llama.cpp/build/bin:$LD_LIBRARY_PATH
M=/workspace_build/muse-glimmer-30b
OUT=/workspace_build/teacher_gguf
mkdir -p $OUT
echo "=== convert teacher f16 (52L, full) ==="
python convert_hf_to_gguf.py "$M" --outtype bf16 --outfile "$OUT/glimmer-full-bf16.gguf" 2>&1 | grep -iE "success|error|Traceback|Writing the following" | tail -n 5
echo "=== quantize Q4_K_M ==="
./build/bin/llama-quantize "$OUT/glimmer-full-bf16.gguf" "$OUT/glimmer-full-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -n 4
echo "TEACHER-DONE"; ls -lh "$OUT"/*.gguf
