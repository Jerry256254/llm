cd /workspace_build/llama.cpp
export LD_LIBRARY_PATH=/workspace_build/llama.cpp/build/bin:$LD_LIBRARY_PATH
G=/workspace_build/gguf_out/glimmer24l-Q4_K_M.gguf
R=/workspace_build/gguf_out/comp.txt
: > "$R"
run(){ echo "### $1" >> "$R"; timeout 150 ./build/bin/llama-completion -m "$G" --jinja -p "$1" -n 130 -t 12 --temp 0.1 2>/dev/null >> "$R"; echo -e "\n-----" >> "$R"; }
run "Kolik je 17 krat 23? Vysledek:"
run "Vysvetli moment hybnosti jednou vetou:"
run "What is entropy? Answer:"
echo ALLDONE >> "$R"
