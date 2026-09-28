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
# it). 307200 is substituted with an int on the host; 0 = off.
target_context = 307200
# Multimodal checkpoints (Gemma-4 "gemma4_unified", Qwen3.5) keep the text
# tower's settings in a nested text_config and have NO top-level
# max_position_embeddings. Reading only the top level silently returned None
# there, so the whole YaRN block no-opped and target_context was ignored
# without any error — the export just stayed at native context. Resolve the
# section that actually owns max_position_embeddings and patch that one.
rope_target = c
if "max_position_embeddings" not in rope_target and isinstance(c.get("text_config"), dict):
    rope_target = c["text_config"]
native_max = rope_target.get("max_position_embeddings")
if target_context and native_max and target_context > native_max and not rope_target.get("rope_scaling"):
    factor = target_context / native_max
    rope_target["rope_scaling"] = {
        "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": native_max,
    }
    rope_target["max_position_embeddings"] = target_context
    where = "text_config" if rope_target is not c else "top-level"
    print(f"Injected YaRN rope_scaling in {where}: {native_max} -> {target_context} (factor={factor:.3f})", flush=True)
elif target_context and not native_max:
    print(f"WARNING: target_context={target_context} requested but no max_position_embeddings found — context NOT extended", flush=True)
cp.write_text(json.dumps(c, indent=2) + "\n")
# Ensure the weights actually saved. Sharded bases (e.g. Qwen2.5-14B, 18
# shards) produce model-XXXXX-of-XXXXX.safetensors + an index; small bases
# (e.g. Gemma-4-12B, one file) legitimately produce a single
# model.safetensors with no sharding pattern to match — check for either.
idx = Path(out) / "model.safetensors.index.json"
shards = sorted(Path(out).glob("model-*-of-*.safetensors"))
single_file = Path(out) / "model.safetensors"
has_weights = bool(shards) or (single_file.is_file() and single_file.stat().st_size > 0)
if (not idx.is_file()) or idx.stat().st_size < 10 or not has_weights:
    raise SystemExit(f"bad save: index={idx} shards={len(shards)} single_file={single_file.is_file()}")
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
