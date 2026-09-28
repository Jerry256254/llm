cd /workspace_models/llama.cpp
export LD_LIBRARY_PATH=/workspace_models/llama.cpp/build/bin:$LD_LIBRARY_PATH
G=/workspace_models/gguf_out/glimmer24l-Q4_K_M.gguf
R=/workspace_models/gguf_out/smoke.txt
: > "$R"; : > /workspace_models/gguf_out/smoke_err.txt
ask() { echo "===== Q: $2" >> "$R"; timeout 260 ./build/bin/llama-cli -m "$G" --jinja -sys "$1" -p "$2" --no-display-prompt -n 180 -t 12 -ngl 99 </dev/null >> "$R" 2>>/workspace_models/gguf_out/smoke_err.txt; echo >> "$R"; }
ask "" "Uzivateli: Kolik je 17 krat 23? Odpovedz nejdriiv samotnym cislem, potom ukaz postup. (reasoning strength: high)"
ask "" "Uzivateli: Vysvetli strucne a odborně cesky, co je moment hybnosti. (reasoning strength: high)"
ask "" "User: Explain in English what entropy means in thermodynamics. (reasoning strength: high)"
echo "SMOKE-COMPLETE" >> "$R"
