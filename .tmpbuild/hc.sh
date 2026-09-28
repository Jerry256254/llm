cd /workspace_build/llama.cpp
export LD_LIBRARY_PATH=/workspace_build/llama.cpp/build/bin:$LD_LIBRARY_PATH
G=/workspace_build/teacher_gguf/glimmer-full-Q4_K_M.gguf
OUT=/workspace_build/teacher_gguf/hc.txt
: > "$OUT"
./build/bin/llama-completion -m "$G" --jinja -ngl 99 -c 4096 -t 8 --temp 0.2 -n 130 \
  -p "Reasoning strength: high. Uzivateli: Kolik je 17 krat 23? Odpoved:" >> "$OUT" 2>&1
echo "=====SPLIT=====" >> "$OUT"
./build/bin/llama-completion -m "$G" --jinja -ngl 99 -c 4096 -t 8 --temp 0.2 -n 130 \
  -p "Reasoning strength: high. User: What is the boiling point of water at 1 atm? Answer:" >> "$OUT" 2>&1
echo "=====SPLIT=====" >> "$OUT"
tail -n 5 "$OUT"
