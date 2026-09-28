
set -e
MODEL_DIR="/workspace/merged"
OUT_DIR="/workspace/gguf"
mkdir -p "$OUT_DIR"

# Prefer llama.cpp convert script shipped in image
CONVERT=""
for c in   /opt/llama.cpp/convert_hf_to_gguf.py   /opt/llama.cpp/convert-hf-to-gguf.py   /usr/local/bin/convert_hf_to_gguf.py; do
  if [ -f "$c" ]; then CONVERT="$c"; break; fi
done

if [ -z "$CONVERT" ]; then
  echo "llama.cpp convert script not found in image" >&2
  exit 1
fi

python "$CONVERT" "$MODEL_DIR" --outfile "$OUT_DIR/model-f16.gguf" --outtype f16

QUANT=$(command -v quantize || true)
if [ -z "$QUANT" ]; then
  for q in /opt/llama.cpp/llama-quantize /opt/llama.cpp/quantize /usr/local/bin/llama-quantize; do
    if [ -x "$q" ]; then QUANT="$q"; break; fi
  done
fi

if [ "q4_k_m" = "f16" ]; then
  cp "$OUT_DIR/model-f16.gguf" "$OUT_DIR/model-q4_k_m.gguf" || mv "$OUT_DIR/model-f16.gguf" "$OUT_DIR/model-q4_k_m.gguf"
else
  if [ -z "$QUANT" ]; then
    echo "llama-quantize not found" >&2
    exit 1
  fi
  "$QUANT" "$OUT_DIR/model-f16.gguf" "$OUT_DIR/model-q4_k_m.gguf" q4_k_m
  # keep f16 optional — delete to save disk
  rm -f "$OUT_DIR/model-f16.gguf"
fi
ls -lh "$OUT_DIR"
