"""Resolve & download training bases: HF hub (preferred) or local path.

Ollama names like qwen3.5:0.8b are mapped to official HF Base IDs and downloaded
via huggingface_hub — NOT passed into transformers as repo ids.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TOKEN_FILE = PROJECT_ROOT / "outputs" / ".hf_token"

# Ollama name → HF training Base (Apache-2.0 / easy full-train on L4)
OLLAMA_TO_HF: dict[str, str] = {
    "qwen3.5:0.8b": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3.5:0.8b-mlx": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3.5:2b": "Qwen/Qwen3.5-2B-Base",
    "qwen3.5:2b-mlx": "Qwen/Qwen3.5-2B-Base",
    "qwen3.5:4b": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5:4b-mlx": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5:9b": "Qwen/Qwen3.5-9B-Base",
    "qwen3.5:9b-mlx": "Qwen/Qwen3.5-9B-Base",
    "qwen3.5:latest": "Qwen/Qwen3.5-0.8B-Base",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B",
    "llama3.2:1b": "unsloth/llama-3.2-1b",
    "llama3.2:3b": "unsloth/llama-3.2-3b",
}

# Easy full-train bases for L4 24GB (recommended UI list)
EASY_BASE_MODELS: list[dict] = [
    {"id": "Qwen/Qwen3.5-0.8B-Base", "label": "Qwen3.5 · 0.8B Base ★ L4", "params_b": 0.8, "ollama_equiv": "qwen3.5:0.8b"},
    {"id": "Qwen/Qwen3.5-2B-Base", "label": "Qwen3.5 · 2B Base ★ L4", "params_b": 2.0, "ollama_equiv": "qwen3.5:2b"},
    {"id": "Qwen/Qwen3.5-4B-Base", "label": "Qwen3.5 · 4B Base · L4", "params_b": 4.0, "ollama_equiv": "qwen3.5:4b"},
    {"id": "unsloth/llama-3.2-1b", "label": "Llama 3.2 · 1B", "params_b": 1.0, "ollama_equiv": "llama3.2:1b"},
    {"id": "Qwen/Qwen2.5-1.5B", "label": "Qwen2.5 · 1.5B", "params_b": 1.5, "ollama_equiv": None},
    {"id": "Qwen/Qwen2.5-3B", "label": "Qwen2.5 · 3B", "params_b": 3.0, "ollama_equiv": None},
]


def fix_hf_cache_permissions() -> None:
    cache = Path.home() / ".cache" / "huggingface"
    try:
        cache.mkdir(parents=True, exist_ok=True)
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
    except Exception as e:
        console.print(f"[yellow]HF cache: {e}[/]")


def get_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit.strip():
        return explicit.strip()
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    if TOKEN_FILE.exists():
        try:
            t = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if t:
                return t
        except OSError:
            pass
    # huggingface saved token
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        try:
            return tok.read_text(encoding="utf-8").strip() or None
        except OSError:
            pass
    return None


def save_hf_token(token: str) -> None:
    """Save token to project + user cache (no sudo, no git)."""
    token = token.strip()
    if not token:
        raise ValueError("Prázdný token")
    fix_hf_cache_permissions()
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    TOKEN_FILE.chmod(0o600)
    tok_dir = Path.home() / ".cache" / "huggingface"
    tok_dir.mkdir(parents=True, exist_ok=True)
    try:
        p = tok_dir / "token"
        p.write_text(token + "\n", encoding="utf-8")
        p.chmod(0o600)
    except OSError:
        pass
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def ensure_hf_cli() -> dict:
    """Ensure huggingface_hub + hf CLI available in current env."""
    import importlib
    import sys

    out = {"huggingface_hub": False, "hf_cli": False, "installed": False, "error": None}
    try:
        importlib.import_module("huggingface_hub")
        out["huggingface_hub"] = True
    except ImportError:
        pass
    out["hf_cli"] = shutil.which("hf") is not None or shutil.which("huggingface-cli") is not None
    if out["huggingface_hub"]:
        return out
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_xet"],
        )
        out["installed"] = True
        out["huggingface_hub"] = True
        out["hf_cli"] = shutil.which("hf") is not None
    except Exception as e:
        out["error"] = str(e)
    return out


def ensure_ollama() -> dict:
    """Install Ollama if missing (Linux curl installer)."""
    out = {"present": False, "installed": False, "error": None, "models": []}
    if shutil.which("ollama"):
        out["present"] = True
    else:
        try:
            console.print("[cyan]Instaluji Ollama…[/]")
            r = subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                check=False,
                timeout=300,
            )
            out["installed"] = r.returncode == 0 and shutil.which("ollama") is not None
            out["present"] = bool(shutil.which("ollama"))
            if not out["present"]:
                out["error"] = "install script finished but ollama not in PATH"
        except Exception as e:
            out["error"] = str(e)
    if out["present"]:
        # start serve best-effort
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        out["models"] = list_ollama_models()
    return out


def list_ollama_models() -> list[dict]:
    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        r.raise_for_status()
        return [
            {"name": m.get("name"), "size": m.get("size"), "source": "ollama"}
            for m in (r.json().get("models") or [])
            if m.get("name")
        ]
    except Exception:
        return []


def map_ollama_to_hf(name: str) -> Optional[str]:
    n = name.strip().removeprefix("ollama:").removeprefix("ollama/").lower()
    if n in OLLAMA_TO_HF:
        return OLLAMA_TO_HF[n]
    # strip tags like :latest
    base = n.split(":")[0] if ":" in n else n
    for k, v in OLLAMA_TO_HF.items():
        if k.startswith(base):
            return v
    return None


def local_model_dir(hf_id: str) -> Path:
    safe = hf_id.replace("/", "__")
    return MODELS_DIR / safe


def is_model_downloaded(hf_id: str) -> bool:
    d = local_model_dir(hf_id)
    return d.is_dir() and (d / "config.json").exists()


def download_hf_model(hf_id: str, token: Optional[str] = None, progress_cb=None) -> Path:
    """Download HF model to models/<id> and return path."""
    ensure_hf_cli()
    fix_hf_cache_permissions()
    tok = get_hf_token(token)
    dest = local_model_dir(hf_id)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").exists():
        console.print(f"[green]Model už stažen:[/] {dest}")
        return dest

    console.print(f"[cyan]Stahuji {hf_id} → {dest} …[/]")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=hf_id,
        local_dir=str(dest),
        token=tok,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if not (dest / "config.json").exists():
        raise RuntimeError(f"Download incomplete: missing config.json in {dest}")
    console.print(f"[green]Staženo:[/] {dest}")
    return dest


def normalize_model_id(model_id: str) -> str:
    """Convert ollama:… to HF id when known."""
    mid = (model_id or "").strip()
    if mid.startswith("ollama:") or mid.startswith("ollama/"):
        name = mid.split(":", 1)[-1] if mid.startswith("ollama:") else mid.split("/", 1)[-1]
        mapped = map_ollama_to_hf(name)
        if mapped:
            console.print(f"[cyan]Ollama {name} → HF {mapped}[/]")
            return mapped
        raise ValueError(
            f"Neznámé Ollama jméno '{name}'. "
            f"Použijte HF ID (např. Qwen/Qwen3.5-0.8B-Base) nebo podporované: "
            f"{', '.join(sorted(OLLAMA_TO_HF.keys())[:8])}…"
        )
    return mid


def resolve_model_for_training(
    model_id: str,
    *,
    work_dir: Path,
    docker_image: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    Returns (path_or_id_inside_container, extra docker -v args).
    Always prefers local download under models/ for reliability.
    """
    mid = normalize_model_id(model_id)

    # Local HF directory
    p = Path(mid).expanduser()
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        host = p.resolve()
        return "/models/base", ["-v", f"{host}:/models/base:ro"]

    # Already downloaded under models/
    if is_model_downloaded(mid):
        host = local_model_dir(mid).resolve()
        return "/models/base", ["-v", f"{host}:/models/base:ro"]

    # Download from HF
    host = download_hf_model(mid, token=hf_token).resolve()
    return "/models/base", ["-v", f"{host}:/models/base:ro"]
