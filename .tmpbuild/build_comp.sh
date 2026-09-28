set -e
cd /workspace_build/llama.cpp
cmake --build build -j10 --target llama-completion 2>&1 | tail -n 3
ls -lh build/bin/llama-completion && echo COMP-BUILT
