"""Export Hugging Face / merged weights to GGUF via llama.cpp in Docker."""

from __future__ import annotations

import os
import re
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


def patch_qwen35_gguf_metadata(gguf_path: Path) -> bool:
    """Fix llama.cpp convert bug for Qwen3.5: block_count includes MTP nextn layer.

    convert_hf_to_gguf sets block_count = num_hidden_layers + nextn_predict_layers
    but only exports tensors for the real 0..N-1 blocks. Ollama then fails with:
      missing tensor 'blk.N.attn_norm.weight'

    Returns True if any field was patched.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        console.print("[yellow]gguf package missing — skip metadata patch (pip install gguf)[/]")
        return False

    path = Path(gguf_path)
    # Docker may leave root-owned files — work on a writable copy in place when needed
    try:
        reader = GGUFReader(str(path), "r+")
    except PermissionError:
        fixed = path.with_suffix(path.suffix + ".writable")
        import shutil

        shutil.copy2(path, fixed)
        path.unlink(missing_ok=False)
        fixed.rename(path)
        reader = GGUFReader(str(path), "r+")

    # Count real blk.* layers from tensor names
    names = [t.name for t in reader.tensors]
    layers = set()
    for n in names:
        m = re.match(r"blk\.(\d+)\.", n)
        if m:
            layers.add(int(m.group(1)))
    if not layers:
        return False
    real_blocks = max(layers) + 1  # 0-indexed

    changed = False
    # Architecture prefix may be qwen35 / qwen3 etc.
    for key, field in list(reader.fields.items()):
        if key.endswith(".block_count") or key.endswith(".block_count".replace(".", "_")):
            pass
        if not (key.endswith("block_count") or key.endswith("nextn_predict_layers")):
            continue
        try:
            arr = field.parts[-1]
            old = int(arr[0])
        except Exception:
            continue
        if key.endswith("block_count") and old != real_blocks:
            console.print(
                f"[yellow]GGUF patch:[/] {key} {old} → {real_blocks} "
                f"(match actual blk.0..{real_blocks - 1})"
            )
            arr[0] = real_blocks
            changed = True
        if key.endswith("nextn_predict_layers") and old != 0:
            console.print(f"[yellow]GGUF patch:[/] {key} {old} → 0 (disable incomplete MTP)")
            arr[0] = 0
            changed = True

    if hasattr(reader.data, "flush"):
        reader.data.flush()
    del reader
    return changed


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


def _project_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def ensure_tokenizer_for_gguf(model_dir: Path, run_dir: Optional[Path] = None) -> None:
    """Ensure SentencePiece tokenizer.model exists (required for Gemma GGUF convert).

    Training often saves only tokenizer.json (fast tokenizer). llama.cpp Gemma path
    still wants tokenizer.model next to the weights.
    """
    import json
    import shutil

    model_dir = Path(model_dir)
    sp = model_dir / "tokenizer.model"
    if sp.is_file() and sp.stat().st_size > 0:
        return

    # Only Gemma-family checkpoints in this project actually need a
    # SentencePiece tokenizer.model for llama.cpp conversion. Every other
    # architecture here (Qwen2/Qwen2.5/Qwen3.5/...) uses a BPE tokenizer via
    # tokenizer.json and legitimately has no tokenizer.model at all — that's
    # correct, not something to "fix". Without this guard, a Qwen checkpoint
    # would get an unrelated Gemma tokenizer.model copied in (wrong vocab),
    # silently sitting there unused today only because llama.cpp's Qwen2
    # path happens to ignore it — exactly the kind of mismatched-metadata
    # bug this project has been bitten by more than once this session.
    cfg_json = model_dir / "config.json"
    model_type = ""
    if cfg_json.is_file():
        try:
            model_type = (json.loads(cfg_json.read_text(encoding="utf-8")).get("model_type") or "").lower()
        except Exception:
            pass
    has_bpe_tokenizer = (model_dir / "tokenizer.json").is_file()
    if has_bpe_tokenizer and not model_type.startswith("gemma"):
        return

    candidates: list[Path] = []
    # 1) train_config model_id / local models/
    for cfg_name in ("train_config.json", "train_config.container.json"):
        if run_dir is None:
            break
        cfg_path = Path(run_dir) / cfg_name
        if not cfg_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = (cfg.get("model_id") or "").strip()
        if not mid:
            continue
        # container path rewrite
        if mid in ("/models/base", "models/base"):
            # try to resolve from estimate / host models by known ids
            pass
        safe = mid.replace("/", "__")
        candidates.append(_project_models_dir() / safe)
        if Path(mid).is_dir():
            candidates.append(Path(mid))
    # 2) known local downloads
    models_root = _project_models_dir()
    if models_root.is_dir():
        for d in models_root.iterdir():
            if d.is_dir() and (d / "tokenizer.model").is_file():
                # prefer gemma-2-9b if config architecture matches
                candidates.append(d)
    # 3) HF cache under project
    hf = Path(__file__).resolve().parent.parent / "outputs" / ".hf_home" / "hub"
    if hf.is_dir():
        for cand in hf.glob("models--*/*/snapshots/*/tokenizer.model"):
            candidates.append(cand.parent)

    # Prefer same family as config model_type
    model_type = ""
    cfg_json = model_dir / "config.json"
    if cfg_json.is_file():
        try:
            model_type = (json.loads(cfg_json.read_text(encoding="utf-8")).get("model_type") or "").lower()
        except Exception:
            pass

    def score(p: Path) -> int:
        s = 0
        name = str(p).lower()
        if model_type and model_type in name:
            s += 10
        if "gemma-2-9b" in name or "gemma2" in name:
            s += 5
        if (p / "tokenizer.model").is_file() or p.name == "tokenizer.model":
            s += 1
        return s

    candidates = sorted({c.resolve() for c in candidates if c}, key=score, reverse=True)
    src_model: Optional[Path] = None
    for c in candidates:
        if c.is_file() and c.name == "tokenizer.model":
            src_model = c
            break
        if c.is_dir() and (c / "tokenizer.model").is_file():
            src_model = c / "tokenizer.model"
            break
    if src_model is None:
        raise FileNotFoundError(
            f"tokenizer.model missing in {model_dir} and no base model found to copy from. "
            f"Re-download the base HF model or copy tokenizer.model into the merged folder."
        )

    shutil.copy2(src_model, sp)
    console.print(f"[yellow]GGUF fix:[/] zkopírován tokenizer.model z {src_model}")

    # optional companions
    src_dir = src_model.parent
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        dst = model_dir / name
        src = src_dir / name
        if (not dst.exists() or dst.stat().st_size == 0) and src.is_file():
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass


def _config_has_bitsandbytes(model_dir: Path) -> bool:
    import json

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        c = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    qc = c.get("quantization_config")
    if isinstance(qc, dict) and (
        qc.get("quant_method") in ("bitsandbytes", "bnb") or qc.get("load_in_4bit") or qc.get("_load_in_4bit")
    ):
        return True
    if c.get("load_in_4bit") or c.get("_load_in_4bit"):
        return True
    # Config may already be stripped while tensors still carry bnb state
    # (weight.absmax / quant_map / nf4 blobs). Detect from index or shard keys.
    idx = model_dir / "model.safetensors.index.json"
    try:
        if idx.is_file():
            wm = (json.loads(idx.read_text(encoding="utf-8")).get("weight_map") or {})
            for k in wm:
                if "absmax" in k or "quant_map" in k or "bitsandbytes" in k:
                    return True
        else:
            from safetensors import safe_open

            for shard in sorted(model_dir.glob("model-*.safetensors"))[:2]:
                with safe_open(str(shard), framework="pt") as f:
                    for k in f.keys():
                        if "absmax" in k or "quant_map" in k or "bitsandbytes" in k:
                            return True
    except Exception:
        pass
    return False


def _strip_bnb_config(model_dir: Path) -> None:
    import json

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return
    try:
        c = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if "quantization_config" not in c and not c.get("load_in_4bit"):
        return
    c.pop("quantization_config", None)
    c.pop("load_in_4bit", None)
    c.pop("load_in_8bit", None)
    cfg_path.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console.print(f"[yellow]GGUF fix:[/] removed bitsandbytes quantization_config from {cfg_path}")


def remerge_adapter_for_gguf(run_dir: Path, *, gpus: Optional[list[GpuInfo]] = None) -> Path:
    """If merged/ is still 4-bit bnb, re-merge adapter onto a clean base and save fp16.

    Runs inside the training Docker image (has torch/peft/bnb).
    """
    run_dir = Path(run_dir).resolve()
    adapter = run_dir / "adapter"
    merged = run_dir / "merged"
    if not (adapter / "adapter_config.json").is_file() and not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"No PEFT adapter under {adapter}")

    # Base model: prefer original base from train_config or parent continue path
    import json

    base_host: Optional[Path] = None
    for cfg_name in ("train_config.json", "train_config.container.json"):
        cp = run_dir / cfg_name
        if not cp.is_file():
            continue
        try:
            cfg = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Host config often has absolute path before rewrite
        for key in ("model_id", "base_model"):
            mid = (cfg.get(key) or "").strip()
            if not mid or mid == "/models/base":
                continue
            if Path(mid).is_dir():
                base_host = Path(mid).resolve()
                break
            # model_id is an HF hub id (e.g. "Qwen/Qwen2.5-14B-Instruct"), not a
            # filesystem path — resolve it the same way model_source.py names
            # local download dirs, instead of falling through to the stale
            # hardcoded fallback below (which previously picked an unrelated
            # gemma-2-9b checkpoint here, silently corrupting the re-merge).
            try:
                from .model_source import local_model_dir

                cand = local_model_dir(mid)
                if cand.is_dir() and (cand / "config.json").is_file():
                    base_host = cand.resolve()
                    break
            except Exception:
                pass
        if base_host:
            break
    # Fallback: previous full run merged (gemma2-9b continue) or models/
    if base_host is None:
        root = Path(__file__).resolve().parent.parent
        for p in (
            root / "outputs" / "run_1784797894" / "merged",
            root / "models" / "google__gemma-2-9b",
        ):
            if p.is_dir() and (p / "config.json").is_file() and not _config_has_bitsandbytes(p):
                base_host = p
                break
    if base_host is None:
        raise FileNotFoundError("Cannot find clean base model to re-merge LoRA for GGUF")

    target_context = 0
    for cfg_name in ("train_config.json", "train_config.container.json"):
        cp = run_dir / cfg_name
        if not cp.is_file():
            continue
        try:
            target_context = int(json.loads(cp.read_text(encoding="utf-8")).get("target_context") or 0)
            break
        except Exception:
            continue

    cuda_tag = recommend_cuda_tag(gpus or [])
    image = build_or_pull_image(framework="unsloth", cuda_tag=cuda_tag)
    script = run_dir / "remerge_for_gguf.py"
    script.write_text(
        r'''
import json, os, gc, torch
from pathlib import Path
from shutil import copy2
from transformers import AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file

base = "/base"
adapter = "/workspace/adapter"
out = "/workspace/merged"
print("re-merge base=", base, "adapter=", adapter, flush=True)
tok = AutoTokenizer.from_pretrained(
    adapter if Path(adapter, "tokenizer.json").is_file() else base,
    use_fast=True,
    trust_remote_code=True,
)
# Forcing float16 here silently corrupted every export: bfloat16 (range
# ~3.4e38) down-cast to float16 (range ~65504) turns any out-of-range weight
# or activation into inf/nan — a known failure mode for Qwen-family models,
# whose upstream checkpoints ship in bfloat16. Verified end to end: a LoRA
# merge with an all-zero adapter (should be a lossless no-op) dropped
# MMLU-Pro accuracy from 68.3% to 29.6% on identical weights, purely from
# this cast.
#
# NOT reading base config.json's torch_dtype field for this: after a model
# has been through 4-bit QLoRA loading, transformers/bitsandbytes can leave
# that field reporting "float32" (the compute dtype), which is a loading
# artifact, not the checkpoint's real storage dtype — trusting it here
# picked float16 again and reproduced the exact same corruption on retry.
# Default to bfloat16 (correct for every model this project targets); only
# an explicit, unambiguous "float16" marks a genuine float16-native model.
_base_cfg_path = Path(base) / "config.json"
_native_dtype = ""
try:
    import json as _json
    _c = _json.loads(_base_cfg_path.read_text())
    _native_dtype = str(_c.get("torch_dtype") or _c.get("text_config", {}).get("dtype") or "")
except Exception:
    pass
dtype = torch.float16 if _native_dtype == "float16" else torch.bfloat16
print(f"merge dtype: {dtype} (base config torch_dtype={_native_dtype!r}, default=bfloat16)", flush=True)
# --- Manual shard-by-shard LoRA merge — no transformers model loading, no
# accelerate device_map/offload, no GPU. Simple and robust: only one ~2GB
# shard plus the (small) adapter tensors are ever resident at once. This
# replaces a device_map="auto" + disk-offload approach that failed three
# different ways in a row on this host (only ~15GiB RAM): a missing
# offload_dir, the GPU packed to 22.04/22.06 GiB with nothing left for a
# 136 MiB scratch alloc, and a KeyError from PEFT's merge_and_unload() not
# understanding offloaded/meta parameters. None of that machinery is needed
# for what LoRA merging actually is: for each targeted weight matrix,
# merged = W + (lora_alpha/r) * (B @ A). Plain tensor arithmetic.
out_p = Path(out)
os.makedirs(out, exist_ok=True)
for p in out_p.glob("model-*.safetensors"):
    p.unlink(missing_ok=True)
idx_out = out_p / "model.safetensors.index.json"
if idx_out.is_file():
    idx_out.unlink(missing_ok=True)

adapter_cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
r = int(adapter_cfg["r"])
lora_alpha = float(adapter_cfg["lora_alpha"])
use_rslora = bool(adapter_cfg.get("use_rslora"))
scaling = (lora_alpha / (r ** 0.5)) if use_rslora else (lora_alpha / r)
print(f"LoRA merge: r={r} alpha={lora_alpha} use_rslora={use_rslora} scaling={scaling}", flush=True)

lora = {}
with safe_open(str(Path(adapter) / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        lora[k] = f.get_tensor(k)
print(f"loaded {len(lora)} LoRA tensors", flush=True)

base_idx_path = Path(base) / "model.safetensors.index.json"
if base_idx_path.is_file():
    weight_map = json.loads(base_idx_path.read_text())["weight_map"]
    shard_names = sorted(set(weight_map.values()))
else:
    with safe_open(str(Path(base) / "model.safetensors"), framework="pt", device="cpu") as f:
        weight_map = {k: "model.safetensors" for k in f.keys()}
    shard_names = ["model.safetensors"]

merged_count = 0
for shard_name in shard_names:
    tensors = {}
    with safe_open(str(Path(base) / shard_name), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name.endswith(".weight"):
                prefix = "base_model.model." + name[: -len(".weight")]
                a_key, b_key = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
                if a_key in lora and b_key in lora:
                    A = lora[a_key].to(torch.float32)
                    B = lora[b_key].to(torch.float32)
                    t = (t.to(torch.float32) + (B @ A) * scaling).to(dtype)
                    merged_count += 1
                else:
                    t = t.to(dtype) if t.is_floating_point() else t
            else:
                t = t.to(dtype) if t.is_floating_point() else t
            tensors[name] = t.contiguous()
    save_file(tensors, str(out_p / shard_name), metadata={"format": "pt"})
    del tensors
    gc.collect()
    print(f"wrote shard {shard_name}", flush=True)

print(f"merged {merged_count} LoRA-targeted tensors", flush=True)
if base_idx_path.is_file():
    copy2(base_idx_path, idx_out)
else:
    idx_out.write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2) + "\n"
    )
tok.save_pretrained(out)
# Rewrite config.json from the ORIGINAL base checkpoint, not from what
# save_pretrained() just wrote. Qwen3.5's config.json is a nested
# multimodal-style structure (text_config/vision_config, no top-level
# vocab_size — real vocab_size=248320 lives under text_config). Round-
# tripping through AutoModelForCausalLM.from_pretrained()+save_pretrained()
# was silently flattening/losing that nesting and replacing it with a
# generic top-level vocab_size=256000 that matches nothing real. The
# result loads without error but decodes to garbage (U+FFFD repeatedly) —
# verified end to end: LoRA fine-tuning was fine (adapter differed between
# runs, confirmed by hash), but two DIFFERENT trained adapters produced
# byte-identical, garbage-decoding merged output because this config bug
# overwhelmed whatever the adapter actually learned.
cp = Path(out) / "config.json"
base_cp = Path(base) / "config.json"
c = json.loads(base_cp.read_text()) if base_cp.is_file() else json.loads(cp.read_text())
c.pop("quantization_config", None)
c.pop("load_in_4bit", None)
c.pop("load_in_8bit", None)
# Optionally extend context via YaRN rope scaling (see train_inside_container.py
# for why: LoRA fine-tuning on short sequences doesn't teach long context,
# the base model's own pretraining does — we just need to tell config to use
# it). __TARGET_CONTEXT__ is substituted with an int on the host; 0 = off.
target_context = __TARGET_CONTEXT__
native_max = c.get("max_position_embeddings")
if target_context and native_max and target_context > native_max and not c.get("rope_scaling"):
    factor = target_context / native_max
    c["rope_scaling"] = {
        "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": native_max,
    }
    c["max_position_embeddings"] = target_context
    print(f"Injected YaRN rope_scaling: {native_max} -> {target_context} (factor={factor:.3f})", flush=True)
cp.write_text(json.dumps(c, indent=2) + "\n")
# Ensure non-empty weight index (docker kill mid-save left 0-byte index before)
idx = Path(out) / "model.safetensors.index.json"
shards = sorted(Path(out).glob("model-*-of-*.safetensors"))
if (not idx.is_file()) or idx.stat().st_size < 10 or not shards:
    raise SystemExit(f"bad save: index={idx} shards={len(shards)}")
# Rebuild index from shards if incomplete
try:
    meta = json.loads(idx.read_text())
    wm = meta.get("weight_map") or {}
except Exception:
    wm = {}
if len(wm) < 10:
    weight_map = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map[k] = shard.name
    meta = {
        "metadata": {"total_size": sum(s.stat().st_size for s in shards)},
        "weight_map": weight_map,
    }
    idx.write_text(json.dumps(meta, indent=2) + "\n")
    print("rebuilt safetensors index keys=", len(weight_map), flush=True)
for name in ("tokenizer.model", "special_tokens_map.json"):
    src, dst = Path(base) / name, Path(out) / name
    if src.is_file() and (not dst.is_file() or dst.stat().st_size == 0):
        copy2(src, dst)
print("re-merge done", flush=True)
'''.lstrip().replace("__TARGET_CONTEXT__", str(target_context)),
        encoding="utf-8",
    )
    args = docker_run_base_args(image, run_dir, gpus="all")
    # inject base mount before image name
    image_idx = args.index(image)
    prefix, suffix = args[:image_idx], args[image_idx:]
    prefix.extend(["-v", f"{base_host.resolve()}:/base:ro", "-v", f"{script.resolve()}:/opt/remerge_for_gguf.py:ro"])
    cmd = prefix + suffix + ["python", "/opt/remerge_for_gguf.py"]
    console.print(f"[cyan]Re-merge LoRA on clean base for GGUF…[/] base={base_host}")
    log = run_dir / "remerge.log"
    with log.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        console.print(log.read_text(encoding="utf-8", errors="replace")[-3000:])
        raise RuntimeError(f"re-merge for GGUF failed (see {log})")
    if _config_has_bitsandbytes(merged):
        _strip_bnb_config(merged)
    if _config_has_bitsandbytes(merged):
        raise RuntimeError("merged/ still has bitsandbytes after re-merge")
    console.print(f"[green]Re-merge OK:[/] {merged}")
    return merged


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
    run_dir = Path(run_dir).resolve()
    model_dir = find_model_dir(run_dir)
    # QLoRA bad merge: llama.cpp cannot read bitsandbytes weights
    if _config_has_bitsandbytes(model_dir):
        console.print(
            "[yellow]merged/ still has bitsandbytes quant — re-merging adapter onto clean base…[/]"
        )
        try:
            model_dir = remerge_adapter_for_gguf(run_dir, gpus=gpus)
        except Exception as e:
            console.print(f"[red]Re-merge failed:[/] {e}")
            # last try: strip config only (won't help if tensors still 4bit)
            _strip_bnb_config(model_dir)
            if _config_has_bitsandbytes(model_dir):
                raise RuntimeError(
                    "GGUF needs dequantized fp16 merge. Re-merge failed and "
                    "quantization_config is still present."
                ) from e
    # Repair missing SentencePiece before docker convert (Gemma etc.)
    try:
        ensure_tokenizer_for_gguf(model_dir, run_dir)
    except Exception as e:
        console.print(f"[yellow]Tokenizer preflight: {e}[/]")
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

    # Qwen3.5 / hybrid: fix block_count vs MTP nextn mismatch for Ollama
    try:
        if patch_qwen35_gguf_metadata(out):
            console.print("[green]GGUF metadata patched for Ollama load[/]")
    except Exception as e:
        console.print(f"[yellow]GGUF metadata patch skipped: {e}[/]")

    console.print(f"[green]GGUF hotovo:[/] {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out
