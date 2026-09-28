set -e
cd /workspace_build
rm -rf llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git 2>&1 | tail -n 3
cd llama.cpp
echo "=== muse file ==="; ls src/models/ | grep -i muse || echo NO_MUSE_FILE
echo "=== convert has muse ==="; grep -c -iE "muse[-_]glimmer" convert_hf_to_gguf.py || echo NONE
echo "=== pip req ==="; pip install -q -r requirements.txt 2>&1 | tail -n 2 || true
echo "=== cmake ==="; cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -n 3
cmake --build build -j10 --target llama-quantize llama-cli 2>&1 | tail -n 6
echo "=== BUILD-DONE ==="; ls -lh build/bin/llama-quantize build/bin/llama-cli
