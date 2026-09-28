cd /workspace_build/llama.cpp
export LD_LIBRARY_PATH=/workspace_build/llama.cpp/build/bin:$LD_LIBRARY_PATH
G=/workspace_build/gguf_out/glimmer24l-Q4_K_M.gguf
./build/bin/llama-completion -m "$G" -p "test:" -n 8 --temp 0 2>&1 | grep -aiE "assert|error|abort|what|exception|window|layer|rope|vision|peg|invalid|check|terminate|ggml|GGUF" | tail -n 15
