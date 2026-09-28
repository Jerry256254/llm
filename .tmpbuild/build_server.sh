set -e
cd /workspace_build/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -aiE "cuda|config done|error" | tail -n 6
cmake --build build -j12 --target llama-server llama-cli llama-completion llama-quantize 2>&1 | tail -n 6
ls -lh build/bin/llama-server && echo SERVER-BUILT
