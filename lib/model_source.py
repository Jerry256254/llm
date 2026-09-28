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
# Project-owned HF home — avoids root-owned ~/.cache/huggingface (Permission denied)
PROJECT_HF_HOME = PROJECT_ROOT / "outputs" / ".hf_home"

# Ollama name → HF training Base (prefer *Base* / non-it for full SFT when available)
OLLAMA_TO_HF: dict[str, str] = {
    # Gemma 4 family
    "gemma4:e2b": "google/gemma-4-E2B",
    "gemma4:e2b-it": "google/gemma-4-E2B-it",
    "gemma4:e4b": "google/gemma-4-E4B",
    "gemma4:e4b-it": "google/gemma-4-E4B-it",
    "gemma4:12b": "google/gemma-4-12B",
    "gemma4:12b-it": "google/gemma-4-12B-it",
    "gemma4:26b": "google/gemma-4-26B-A4B",
    "gemma4:26b-a4b": "google/gemma-4-26B-A4B",
    "gemma4:26b-it": "google/gemma-4-26B-A4B-it",
    "gemma4:31b": "google/gemma-4-31B",
    "gemma4:31b-it": "google/gemma-4-31B-it",
    "gemma4:latest": "google/gemma-4-E2B",
    # Qwen 3.5 family
    "qwen3.5:0.8b": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3.5:0.8b-mlx": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3.5:2b": "Qwen/Qwen3.5-2B-Base",
    "qwen3.5:2b-mlx": "Qwen/Qwen3.5-2B-Base",
    "qwen3.5:4b": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5:4b-mlx": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5:9b": "Qwen/Qwen3.5-9B-Base",
    "qwen3.5:9b-mlx": "Qwen/Qwen3.5-9B-Base",
    "qwen3.5:27b": "Qwen/Qwen3.5-27B",
    "qwen3.5:35b": "Qwen/Qwen3.5-35B-A3B-Base",
    "qwen3.5:35b-a3b": "Qwen/Qwen3.5-35B-A3B-Base",
    "qwen3.5:122b": "Qwen/Qwen3.5-122B-A10B",
    "qwen3.5:122b-a10b": "Qwen/Qwen3.5-122B-A10B",
    "qwen3.5:latest": "Qwen/Qwen3.5-0.8B-Base",
    # legacy aliases
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B",
    "llama3.2:1b": "unsloth/llama-3.2-1b",
    "llama3.2:3b": "unsloth/llama-3.2-3b",
    "gemma2:2b": "google/gemma-2-2b",
    "gemma2:9b": "google/gemma-2-9b",
    "gemma2:9b-it": "google/gemma-2-9b-it",
    "gemma2:9b-instruct": "google/gemma-2-9b-it",
}

# Main UI model list (order = dropdown order). Custom/local paths → web "Vlastní model" field.
# L4 24GB: full FT ok for ≤~5B; 9B full = těsné / staged; QLoRA continue via custom path.
EASY_BASE_MODELS: list[dict] = [
    {
        "id": "Qwen/Qwen3.5-9B",
        "label": "qwen3.5:9b · Qwen3.5 9B ★ KucLab Hertz 0.1 (STEM)",
        "params_b": 9.0,
        "ollama_equiv": "qwen3.5:9b",
    },
    {
        "id": "google/gemma-2-9b-it",
        "label": "gemma2:9b-it · Gemma 2 9B Instruct · KucLab V1.0 (starší řada)",
        "params_b": 9.0,
        "ollama_equiv": "gemma2:9b-it",
    },
    {
        "id": "google/gemma-2-9b",
        "label": "gemma2:9b · Gemma 2 9B Base ★ L4",
        "params_b": 9.0,
        "ollama_equiv": "gemma2:9b",
    },
    {
        "id": "Qwen/Qwen3.5-0.8B-Base",
        "label": "qwen3.5:0.8b · Qwen3.5 0.8B Base ★ L4 full",
        "params_b": 0.8,
        "ollama_equiv": "qwen3.5:0.8b",
    },
    {
        "id": "google/gemma-4-E2B",
        "label": "gemma4:e2b · Gemma 4 E2B (~5B weights) ★ L4 full seq≤512",
        "params_b": 5.1,
        "ollama_equiv": "gemma4:e2b",
    },
    {
        "id": "google/gemma-4-E4B",
        "label": "gemma4:e4b · Gemma 4 E4B · L4 full těsně / QLoRA",
        "params_b": 8.0,
        "ollama_equiv": "gemma4:e4b",
    },
    {
        "id": "Qwen/Qwen3.5-9B-Base",
        "label": "qwen3.5:9b · Qwen3.5 9B Base · full těsně / QLoRA",
        "params_b": 9.0,
        "ollama_equiv": "qwen3.5:9b",
    },
    {
        "id": "google/gemma-4-12B",
        "label": "gemma4:12b · Gemma 4 12B · potřebuje víc VRAM / QLoRA",
        "params_b": 12.0,
        "ollama_equiv": "gemma4:12b",
    },
    {
        "id": "google/gemma-4-26B-A4B",
        "label": "gemma4:26b · Gemma 4 26B-A4B MoE · velký GPU / QLoRA",
        "params_b": 26.0,
        "ollama_equiv": "gemma4:26b",
    },
    {
        "id": "Qwen/Qwen3.5-27B",
        "label": "qwen3.5:27b · Qwen3.5 27B · velký GPU / QLoRA",
        "params_b": 27.0,
        "ollama_equiv": "qwen3.5:27b",
    },
    {
        "id": "google/gemma-4-31B",
        "label": "gemma4:31b · Gemma 4 31B dense · multi-GPU",
        "params_b": 31.0,
        "ollama_equiv": "gemma4:31b",
    },
    {
        "id": "Qwen/Qwen3.5-35B-A3B-Base",
        "label": "qwen3.5:35b · Qwen3.5 35B-A3B MoE Base · multi-GPU",
        "params_b": 35.0,
        "ollama_equiv": "qwen3.5:35b",
    },
    {
        "id": "Qwen/Qwen3.5-122B-A10B",
        "label": "qwen3.5:122b · Qwen3.5 122B-A10B MoE · cluster",
        "params_b": 122.0,
        "ollama_equiv": "qwen3.5:122b",
    },
]


def ensure_project_hf_home() -> Path:
    """Writable HF cache under the project (never depends on root-owned ~/.cache)."""
    PROJECT_HF_HOME.mkdir(parents=True, exist_ok=True)
    # Point HF libs at project cache for this process
    os.environ.setdefault("HF_HOME", str(PROJECT_HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(PROJECT_HF_HOME / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_HF_HOME / "hub"))
    # Xet can fail with Permission denied on root-owned global cache
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return PROJECT_HF_HOME


def fix_hf_cache_permissions() -> None:
    """Prefer project HF home; try to fix ~/.cache if writable via sudo -n."""
    ensure_project_hf_home()
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
        console.print(f"[yellow]HF cache (~/.cache): {e} — using {PROJECT_HF_HOME}[/]")


def _token_paths() -> list[Path]:
    return [
        TOKEN_FILE,
        PROJECT_HF_HOME / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ]


def get_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit.strip():
        return explicit.strip()
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    for p in _token_paths():
        if p.exists():
            try:
                t = p.read_text(encoding="utf-8").strip()
                if t:
                    return t
            except OSError:
                pass
    return None


def save_hf_token(token: str) -> None:
    """Save token to project paths (writable without sudo)."""
    token = token.strip()
    if not token:
        raise ValueError("Prázdný token")
    ensure_project_hf_home()
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    TOKEN_FILE.chmod(0o600)
    for tok_dir in (PROJECT_HF_HOME, Path.home() / ".cache" / "huggingface"):
        try:
            tok_dir.mkdir(parents=True, exist_ok=True)
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


def _has_weight_files(d: Path) -> bool:
    """True if directory has actual model weights (not only config/tokenizer)."""
    if not d.is_dir():
        return False
    for pat in ("*.safetensors", "*.bin", "*.gguf", "pytorch_model*.bin"):
        if any(d.glob(pat)):
            return True
    # sharded single-file sometimes named without extension pattern above
    for p in d.iterdir():
        if p.is_file() and p.suffix in (".safetensors", ".bin") and p.stat().st_size > 1_000_000:
            return True
    return False


def is_model_downloaded(hf_id: str) -> bool:
    # Local absolute/relative checkpoint dir (continue FT from previous run)
    p = Path(hf_id).expanduser()
    try:
        if p.is_dir() and (p / "config.json").exists() and _has_weight_files(p):
            return True
    except OSError:
        pass
    d = local_model_dir(hf_id)
    return d.is_dir() and (d / "config.json").exists() and _has_weight_files(d)


def download_hf_model(hf_id: str, token: Optional[str] = None, progress_cb=None) -> Path:
    """Download HF model to models/<id> and return path."""
    ensure_hf_cli()
    fix_hf_cache_permissions()
    ensure_project_hf_home()
    tok = get_hf_token(token)
    dest = local_model_dir(hf_id)
    dest.mkdir(parents=True, exist_ok=True)
    if is_model_downloaded(hf_id):
        console.print(f"[green]Model už stažen:[/] {dest}")
        return dest

    if (dest / "config.json").exists() and not _has_weight_files(dest):
        console.print(f"[yellow]Neúplný download (chybí váhy) — stahuji znovu: {dest}[/]")

    console.print(f"[cyan]Stahuji {hf_id} → {dest} …[/]")
    from huggingface_hub import snapshot_download

    # Prefer classic HTTP download over xet when project/global xet cache is broken
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    snapshot_download(
        repo_id=hf_id,
        local_dir=str(dest),
        token=tok,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if not (dest / "config.json").exists():
        raise RuntimeError(f"Download incomplete: missing config.json in {dest}")
    if not _has_weight_files(dest):
        raise RuntimeError(
            f"Download incomplete: no weight files (*.safetensors/*.bin) in {dest}. "
            f"Zkuste smazat složku a stáhnout znovu s platným HF tokenem."
        )
    console.print(f"[green]Staženo:[/] {dest}")
    return dest


def normalize_model_id(model_id: str) -> str:
    """Convert ollama:… / gemma4:e2b-style tags to HF id when known."""
    mid = (model_id or "").strip()
    if not mid:
        return mid
    # Explicit ollama: prefix
    if mid.startswith("ollama:") or mid.startswith("ollama/"):
        name = mid.split(":", 1)[-1] if mid.startswith("ollama:") else mid.split("/", 1)[-1]
        mapped = map_ollama_to_hf(name)
        if mapped:
            console.print(f"[cyan]Ollama {name} → HF {mapped}[/]")
            return mapped
        raise ValueError(
            f"Neznámé Ollama jméno '{name}'. "
            f"Použijte HF ID (např. google/gemma-4-E2B) nebo podporované: "
            f"{', '.join(sorted(OLLAMA_TO_HF.keys())[:10])}…"
        )
    # Bare Ollama tags: "gemma4:e2b", "qwen3.5:0.8b" (have colon, not HF org/name)
    if ":" in mid and "/" not in mid:
        mapped = map_ollama_to_hf(mid)
        if mapped:
            console.print(f"[cyan]Ollama {mid} → HF {mapped}[/]")
            return mapped
    # Also accept keys without colon match via map
    mapped = map_ollama_to_hf(mid)
    if mapped and mapped != mid:
        console.print(f"[cyan]Alias {mid} → HF {mapped}[/]")
        return mapped
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
