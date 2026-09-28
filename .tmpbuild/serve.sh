cd /workspace_build/llama.cpp
export LD_LIBRARY_PATH=/workspace_build/llama.cpp/build/bin:$LD_LIBRARY_PATH
G=/workspace_build/teacher_gguf/glimmer-full-Q4_K_M.gguf
./build/bin/llama-server -m "$G" --jinja -ngl 99 -c 8192 -t 10 -np 4 --host 127.0.0.1 --port 11435 > /workspace_build/teacher_gguf/server.log 2>&1
