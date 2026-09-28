#!/usr/bin/env bash
# Serve the full Muse-Glimmer-30B Q4 teacher on the L4 GPU, OpenAI-ish API on :11435
set -e
cd /workspace_build/llama.cpp
export LD_LIBRARY_PATH=/workspace_build/llama.cpp/build/bin:$LD_LIBRARY_PATH
G=/workspace_build/teacher_gguf/glimmer-full-Q4_K_M.gguf
./build/bin/llama-server -m "$G" --jinja -ngl 99 -c 8192 -t 8 --host 0.0.0.0 --port 11435 \
  >> /workspace_build/teacher_gguf/server.log 2>&1 &
echo $! > /workspace_build/teacher_gguf/server.pid
sleep 3
echo "server launching, pid $(cat /workspace_build/teacher_gguf/server.pid)"
