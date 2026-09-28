cd /workspace_models/llama.cpp
export LD_LIBRARY_PATH=/workspace_models/llama.cpp/build/bin:$LD_LIBRARY_PATH
G=/workspace_models/gguf_out/glimmer24l-Q4_K_M.gguf
R=/workspace_models/gguf_out/chat.txt
printf 'Kolik je 17 krat 23? Odpovedz jen cislem.\n/exit\n' | timeout 240 ./build/bin/llama-cli -m "$G" -n 120 -t 12 --seed 0 -p "" 2>/dev/null | tail -n 200 > "$R"
echo "==== PHYS ====" >> "$R"
printf 'Vysvetli moment hybnosti jednou vetou.\n/exit\n' | timeout 240 ./build/bin/llama-cli -m "$G" -n 120 -t 12 --seed 0 -p "" 2>/dev/null | tail -n 200 >> "$R"
echo ALLDONE >> "$R"
