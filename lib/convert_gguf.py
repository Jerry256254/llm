"""Export Hugging Face / merged weights to GGUF via llama.cpp in Docker."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

from .env_setup import docker_run_base_args, build_or_pull_image, recommend_cuda_tag, GpuInfo

console = Console()

QUANT_MAP = {
    "f16": "f16",
    "q8_0": "q8_0",
    "q5_k_m": "q5_k_m",
    "q4_k_m": "q4_k_m",
    "q3_k_m": "q3_k_m",
}


def find_model_dir(run_dir: Path) -> Path:
    """Prefer fully merged HF model; fall back to adapter (needs merge)."""
    merged = run_dir / "merged"
    if merged.exists() and any(merged.iterdir()):
        # require config.json
        if (merged / "config.json").exists() or any(merged.glob("*.safetensors")):
            return merged
    adapter = run_dir / "adapter"
    if adapter.exists() and any(adapter.iterdir()):
        return adapter
    raise FileNotFoundError(f"No model weights found under {run_dir}")


def convert_to_gguf(
    run_dir: Path,
    *,
    quant: str = "q4_k_m",
    gpus: Optional[list[GpuInfo]] = None,
    image: Optional[str] = None,
    out_name: str = "model",
) -> Path:
    """
    Convert HF model directory to GGUF using llama.cpp tools inside the training image.
    Returns path to the quantized .gguf file.
    """
    quant = QUANT_MAP.get(quant.lower(), quant.lower())
    model_dir = find_model_dir(run_dir)
    gguf_dir = run_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Relative paths inside container (/workspace is run_dir)
    rel_model = model_dir.relative_to(run_dir)
    f16_name = f"{out_name}-f16.gguf"
    q_name = f"{out_name}-{quant}.gguf"

    if image is None:
        cuda_tag = recommend_cuda_tag(gpus or [])
        # conversion works on CPU too but image already has llama.cpp
        image = build_or_pull_image(framework="unsloth", cuda_tag=cuda_tag)

    convert_script = """
set -e
MODEL_DIR="/workspace/{rel_model}"
OUT_DIR="/workspace/gguf"
mkdir -p "$OUT_DIR"

# Prefer llama.cpp convert script shipped in image
CONVERT=""
for c in \
  /opt/llama.cpp/convert_hf_to_gguf.py \
  /opt/llama.cpp/convert-hf-to-gguf.py \
  /usr/local/bin/convert_hf_to_gguf.py; do
  if [ -f "$c" ]; then CONVERT="$c"; break; fi
done

if [ -z "$CONVERT" ]; then
  echo "llama.cpp convert script not found in image" >&2
  exit 1
fi

python "$CONVERT" "$MODEL_DIR" --outfile "$OUT_DIR/{f16_name}" --outtype f16

QUANT=$(command -v quantize || true)
if [ -z "$QUANT" ]; then
  for q in /opt/llama.cpp/llama-quantize /opt/llama.cpp/quantize /usr/local/bin/llama-quantize; do
    if [ -x "$q" ]; then QUANT="$q"; break; fi
  done
fi

if [ "{quant}" = "f16" ]; then
  cp "$OUT_DIR/{f16_name}" "$OUT_DIR/{q_name}" || mv "$OUT_DIR/{f16_name}" "$OUT_DIR/{q_name}"
else
  if [ -z "$QUANT" ]; then
    echo "llama-quantize not found" >&2
    exit 1
  fi
  "$QUANT" "$OUT_DIR/{f16_name}" "$OUT_DIR/{q_name}" {quant}
  # keep f16 optional — delete to save disk
  rm -f "$OUT_DIR/{f16_name}"
fi
ls -lh "$OUT_DIR"
""".format(rel_model=rel_model.as_posix(), f16_name=f16_name, q_name=q_name, quant=quant)

    script_path = run_dir / "convert_gguf.sh"
    script_path.write_text(convert_script, encoding="utf-8")
    script_path.chmod(0o755)

    extra_env = {}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(key):
            extra_env[key] = os.environ[key]
            break
    args = docker_run_base_args(image, run_dir, gpus="all", extra_env=extra_env or None)
    # CPU-only conversion is fine; still allow GPU image
    cmd = args + ["bash", "/workspace/convert_gguf.sh"]

    console.print(f"[bold cyan]Konverze do GGUF ({quant})…[/]")
    log = run_dir / "convert_gguf.log"
    with log.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        console.print(log.read_text(encoding="utf-8", errors="replace")[-4000:])
        raise RuntimeError(f"GGUF conversion failed (see {log})")

    out = gguf_dir / q_name
    if not out.exists():
        # pick any gguf
        found = list(gguf_dir.glob("*.gguf"))
        if not found:
            raise FileNotFoundError(f"No GGUF produced in {gguf_dir}")
        out = found[0]

    console.print(f"[green]GGUF hotovo:[/] {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out
