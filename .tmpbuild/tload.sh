cd /workspace_build/llama.cpp
export LD_LIBRARY_PATH=/workspace_build/llama.cpp/build/bin:$LD_LIBRARY_PATH
./build/bin/llama-cli -m /workspace_build/gguf_test.gguf --jinja -p "Ahoj" -n 5 --no-display-prompt 2>&1 | grep -aiE "error|fail|abort|load|tokens|system_info" | head -n 12
echo "EXIT:$?"
