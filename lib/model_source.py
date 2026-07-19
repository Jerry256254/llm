"""Resolve training base model from HF, local path, or local Ollama install."""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console

console = Console()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fix_hf_cache_permissions() -> None:
    """Avoid PermissionError on ~/.cache/huggingface/stored_tokens (often root-owned)."""
    cache = Path.home() / ".cache" / "huggingface"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        # If we can't write, try chown via sudo -n
        test = cache / ".write_test"
        try:
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return
        except PermissionError:
            pass
        subprocess.run(
            ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(cache)],
            check=False,
            capture_output=True,
        )
        cache.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        console.print(f"[yellow]HF cache permissions: {e}[/]")


def list_ollama_models() -> list[dict]:
    """Return local Ollama models [{name, size, ...}]."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models") or []
        out = []
        for m in models:
            out.append(
                {
                    "name": m.get("name") or m.get("model"),
                    "size": m.get("size"),
                    "digest": m.get("digest"),
                    "details": m.get("details") or {},
                }
            )
        return out
    except Exception as e:
        console.print(f"[dim]Ollama list failed: {e}[/]")
        return []


def _ollama_show(name: str) -> dict:
    r = requests.post(f"{OLLAMA_HOST}/api/show", json={"name": name}, timeout=60)
    r.raise_for_status()
    return r.json()


def _find_ollama_blobs_dir() -> Optional[Path]:
    candidates = [
        Path.home() / ".ollama" / "models" / "blobs",
        Path("/usr/share/ollama/.ollama/models/blobs"),
        Path("/root/.ollama/models/blobs"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # search
    home_o = Path.home() / ".ollama"
    if home_o.exists():
        for p in home_o.rglob("blobs"):
            if p.is_dir():
                return p
    return None


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def resolve_ollama_gguf(name: str, dest_dir: Path) -> Path:
    """
    Locate GGUF weights for a local Ollama model and copy/symlink into dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = name.removeprefix("ollama:").removeprefix("ollama/")

    # Ensure model exists locally
    try:
        info = _ollama_show(name)
    except Exception as e:
        console.print(f"[yellow]ollama show failed ({e}), trying pull…[/]")
        subprocess.run(["ollama", "pull", name], check=False)
        info = _ollama_show(name)

    blobs = _find_ollama_blobs_dir()
    if not blobs:
        raise FileNotFoundError(
            "Nenalezena složka Ollama blobs (~/.ollama/models/blobs). "
            "Je Ollama nainstalovaná a má stažené modely?"
        )

    # Prefer digests from model_info / details
    digests: list[str] = []
    for key in ("model_info", "details"):
        block = info.get(key) or {}
        if isinstance(block, dict):
            for k, v in block.items():
                if "digest" in k.lower() and isinstance(v, str):
                    digests.append(v)
    # Manifest-style: look for sha256- files referenced in modelfile FROM
    modelfile = info.get("modelfile") or ""
    digests += re.findall(r"sha256:([a-f0-9]{64})", modelfile)
    digests += re.findall(r"sha256-([a-f0-9]{64})", modelfile)

    # Also scan largest blob files that look like GGUF
    candidates: list[Path] = []
    for d in digests:
        d = d.replace("sha256:", "").replace("sha256-", "")
        for prefix in (f"sha256-{d}", f"sha256:{d}", d):
            p = blobs / prefix
            if p.exists():
                candidates.append(p)

    if not candidates:
        # fallback: largest GGUF-looking blobs (heuristic)
        files = sorted(blobs.iterdir(), key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True)
        for p in files[:30]:
            if p.is_file() and p.stat().st_size > 50_000_000 and _is_gguf(p):
                candidates.append(p)
                break

    gguf_src = None
    for p in candidates:
        if _is_gguf(p):
            gguf_src = p
            break
    if gguf_src is None and candidates:
        # some ollama blobs are raw without checking
        gguf_src = max(candidates, key=lambda p: p.stat().st_size)

    if gguf_src is None:
        raise FileNotFoundError(
            f"Nepodařilo se najít GGUF pro Ollama model '{name}'. "
            f"Blobs: {blobs}"
        )

    dest = dest_dir / f"{name.replace(':', '_').replace('/', '_')}.gguf"
    if not dest.exists() or dest.stat().st_size != gguf_src.stat().st_size:
        console.print(f"[cyan]Kopíruji Ollama váhy → {dest} ({gguf_src.stat().st_size/1e9:.2f} GB)…[/]")
        shutil.copy2(gguf_src, dest)
    return dest


def gguf_to_hf(gguf_path: Path, out_dir: Path, image: str, gpus: str = "all") -> Path:
    """
    Convert GGUF → HuggingFace folder using training image (transformers).
    Quality: dequantized from GGUF — OK for continued full-tune experiments, not ideal vs original FP16.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "config.json"
    if marker.exists():
        console.print(f"[green]HF model už existuje:[/] {out_dir}")
        return out_dir

    gguf_path = gguf_path.resolve()
    out_dir = out_dir.resolve()
    # parent mount
    work = out_dir.parent
    rel_gguf = gguf_path.name if gguf_path.parent == work else str(gguf_path)

    script = r"""
import sys
from pathlib import Path
gguf = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
print("Converting", gguf, "->", out, flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
# transformers GGUF load (architecture inferred from GGUF metadata)
try:
    tok = AutoTokenizer.from_pretrained(str(gguf), gguf_file=str(gguf.name) if False else str(gguf))
except Exception as e:
    print("tokenizer from gguf failed:", e, flush=True)
    tok = None
model = AutoModelForCausalLM.from_pretrained(str(gguf.parent if False else gguf))
# Better API in recent transformers:
try:
    model = AutoModelForCausalLM.from_pretrained(str(gguf))
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(".", gguf_file=str(gguf))
except Exception:
    model = AutoModelForCausalLM.from_pretrained(str(gguf), gguf_file=gguf.name)

if tok is None:
    try:
        tok = AutoTokenizer.from_pretrained(str(gguf))
    except Exception:
        from transformers import PreTrainedTokenizerFast
        raise RuntimeError("Cannot build tokenizer from GGUF — pick HF base instead")

model.save_pretrained(str(out), safe_serialization=True)
tok.save_pretrained(str(out))
print("SAVED", out, flush=True)
"""
    conv = work / "gguf_to_hf_tmp.py"
    # put gguf next to out parent for simple mounts
    mount_dir = out_dir.parent
    local_gguf = mount_dir / gguf_path.name
    if local_gguf.resolve() != gguf_path.resolve():
        if not local_gguf.exists():
            shutil.copy2(gguf_path, local_gguf)

    conv_script = mount_dir / "_gguf2hf.py"
    conv_script.write_text(
        f"""
from pathlib import Path
import sys
gguf = Path("/workspace/{local_gguf.name}")
out = Path("/workspace/{out_dir.name}")
out.mkdir(parents=True, exist_ok=True)
print("GGUF", gguf, "size", gguf.stat().st_size, flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading GGUF (may take a while)…", flush=True)
# Primary path used by recent transformers for local GGUF files
model = AutoModelForCausalLM.from_pretrained(
    str(gguf),
    torch_dtype="auto",
    device_map="cpu",
)
try:
    tok = AutoTokenizer.from_pretrained(str(gguf))
except Exception as e:
    print("tok from gguf failed", e, flush=True)
    # minimal fallback: require user HF tokenizer later
    tok = None
model.save_pretrained(str(out), safe_serialization=True)
if tok is not None:
    tok.save_pretrained(str(out))
else:
    # write a note
    (out / "TOKENIZER_MISSING.txt").write_text("Load tokenizer from original HF base model")
print("DONE", out, flush=True)
""",
        encoding="utf-8",
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        gpus,
        "-v",
        f"{mount_dir}:/workspace",
        "-w",
        "/workspace",
        image,
        "python",
        f"/workspace/{conv_script.name}",
    ]
    console.print("[cyan]Konverze Ollama GGUF → HF (CPU, může trvat)…[/]")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0 or not (out_dir / "config.json").exists():
        raise RuntimeError(
            "Konverze GGUF→HF selhala. "
            "Pro trénink raději použijte otevřený HF model (např. unsloth/llama-3.2-1b) "
            "nebo nastavte HF_TOKEN a gated model s přijatou licencí."
        )
    return out_dir


def resolve_model_for_training(
    model_id: str,
    *,
    work_dir: Path,
    docker_image: Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    Returns (model_path_or_id_for_container, extra_docker_volume_args).

    Supports:
      - HF id: google/gemma-3-1b-pt
      - local HF dir: /path/to/model
      - ollama:gemma2:2b  /  ollama/gemma2:2b
    """
    mid = (model_id or "").strip()
    extra: list[str] = []

    # Local directory with config.json
    p = Path(mid).expanduser()
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        host = p.resolve()
        extra = ["-v", f"{host}:/models/base:ro"]
        return "/models/base", extra

    # Local GGUF file
    if p.exists() and p.is_file() and (mid.endswith(".gguf") or _is_gguf(p)):
        out = work_dir / "base_from_gguf"
        if docker_image:
            gguf_to_hf(p, out, docker_image)
            extra = ["-v", f"{out.resolve()}:/models/base:ro"]
            return "/models/base", extra
        raise RuntimeError("GGUF conversion needs docker image")

    # Ollama source
    if mid.startswith("ollama:") or mid.startswith("ollama/"):
        ollama_name = mid.split(":", 1)[-1] if mid.startswith("ollama:") else mid.split("/", 1)[-1]
        if ollama_name.startswith("ollama/"):
            ollama_name = ollama_name[7:]
        base_root = work_dir / "ollama_bases" / ollama_name.replace(":", "_").replace("/", "_")
        gguf = resolve_ollama_gguf(ollama_name, base_root)
        if not docker_image:
            raise RuntimeError("Ollama→HF conversion requires training docker image")
        hf_dir = base_root / "hf"
        gguf_to_hf(gguf, hf_dir, docker_image)
        extra = ["-v", f"{hf_dir.resolve()}:/models/base:ro"]
        return "/models/base", extra

    # Plain HF hub id
    return mid, extra
