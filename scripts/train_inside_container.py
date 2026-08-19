#!/usr/bin/env python3
"""
In-container training entrypoint (Unsloth preferred, transformers+PEFT fallback).
Reads JSON config path from argv[1].
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mem_available_gib() -> float:
    """Host RAM available (GiB). Inside Docker this is the cgroup/host view."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _log_mem(tag: str) -> None:
    msg = f"[{tag}] host MemAvailable≈{_mem_available_gib():.1f} GiB"
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            msg += f" | GPU free {free / 1e9:.2f}/{total / 1e9:.2f} GiB"
    except Exception:
        pass
    log(msg)


def _release_trainer(trainer) -> None:
    """Drop optimizer/scheduler before save — they can be multi‑GB and OOM the host."""
    import gc

    if trainer is None:
        return
    for attr in ("optimizer", "lr_scheduler", "scaler", "optimizer_cls", "callback_handler"):
        if hasattr(trainer, attr):
            try:
                setattr(trainer, attr, None)
            except Exception:
                pass
    try:
        del trainer
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ensure_gguf_tokenizer_files(dest: str, *, source_hints: list[str] | None = None) -> None:
    """Copy SentencePiece / special tokenizer files llama.cpp needs for Gemma etc.

    FastTokenizer save_pretrained often writes only tokenizer.json — convert_hf_to_gguf
    for Gemma then dies with: File not found: .../tokenizer.model
    """
    import shutil

    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    need = (
        "tokenizer.model",
        "special_tokens_map.json",
        "tokenizer.model.json",  # rare
    )
    # Prefer files next to original checkpoint
    search: list[Path] = []
    for h in source_hints or []:
        if not h:
            continue
        p = Path(h)
        search.append(p)
        if p.is_file():
            search.append(p.parent)
    # Common container / host layouts
    for extra in (
        "/models/base",
        "/models/base/tokenizer",
        os.environ.get("MODEL_ID") or "",
    ):
        if extra:
            search.append(Path(extra))

    copied = []
    for name in need:
        target = dest_p / name
        if target.exists() and target.stat().st_size > 0:
            continue
        for root in search:
            if not root:
                continue
            cand = root / name if root.is_dir() else None
            if cand is None or not cand.is_file():
                # also try root as file path parent already handled
                continue
            try:
                shutil.copy2(cand, target)
                copied.append(name)
                break
            except OSError as e:
                log(f"tokenizer copy {name} from {cand}: {e}")
    # tokenizer.model is critical for Gemma GGUF
    if not (dest_p / "tokenizer.model").exists():
        # last resort: walk parent of any found tokenizer.json sibling in search roots
        for root in search:
            if root.is_dir():
                for cand in root.rglob("tokenizer.model"):
                    try:
                        shutil.copy2(cand, dest_p / "tokenizer.model")
                        copied.append("tokenizer.model")
                        break
                    except OSError:
                        continue
            if (dest_p / "tokenizer.model").exists():
                break
    if copied:
        log(f"GGUF tokenizer files ensured: {', '.join(copied)} → {dest}")
    if not (dest_p / "tokenizer.model").exists():
        log(
            "WARNING: tokenizer.model still missing — GGUF convert for Gemma will fail. "
            "Base model dir must include SentencePiece tokenizer.model."
        )


def _safe_save_pretrained(model, path: str, tokenizer=None, *, label: str = "model") -> None:
    """Save weights without model.cpu().

    On 15–16 GiB VPS hosts, moving a 5B bf16 model (~10 GiB) to CPU freezes the
    whole machine (no swap → unresponsive OOM). Keep tensors on GPU and serialize
    in small shards so peak host RAM stays bounded.
    """
    import gc

    path = str(path)
    os.makedirs(path, exist_ok=True)
    _log_mem(f"before save {label}")

    # Never yank the whole model onto host RAM first.
    kwargs_list = [
        dict(safe_serialization=True, max_shard_size="1GB"),
        dict(safe_serialization=True, max_shard_size="512MB"),
        dict(safe_serialization=True),
        {},
    ]
    last_err: Exception | None = None
    for kw in kwargs_list:
        try:
            model.save_pretrained(path, **kw)
            last_err = None
            break
        except TypeError:
            # older transformers: no max_shard_size / safe_serialization
            continue
        except Exception as e:
            last_err = e
            log(f"save_pretrained({label}) retry after: {type(e).__name__}: {e}")
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    if last_err is not None:
        raise last_err

    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(path)
        except Exception as e:
            log(f"tokenizer.save_pretrained warning: {e}")
    # Always pull SP / special files from base checkpoint (Gemma needs tokenizer.model)
    hints = []
    try:
        # name_or_path on tokenizer / config
        if tokenizer is not None:
            nop = getattr(tokenizer, "name_or_path", None)
            if nop:
                hints.append(str(nop))
        cfg_name = getattr(getattr(model, "config", None), "name_or_path", None)
        if cfg_name:
            hints.append(str(cfg_name))
    except Exception:
        pass
    hints.extend(["/models/base", os.environ.get("MODEL_ID") or ""])
    _ensure_gguf_tokenizer_files(path, source_hints=hints)
    _log_mem(f"after save {label}")
    log(f"Saved {label} → {path}")


def _link_or_copy_tree(src: str, dst: str) -> None:
    """Hardlink files when possible (no second full serialize of multi‑GB weights)."""
    import shutil

    src_p, dst_p = Path(src), Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)
    for item in src_p.iterdir():
        target = dst_p / item.name
        if target.exists():
            continue
        if item.is_file():
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
    log(f"Linked/copied {src} → {dst}")


def _strip_quantization_config(model_dir: str | Path) -> None:
    """Remove bitsandbytes quantization_config so llama.cpp GGUF convert accepts the folder."""
    import json

    p = Path(model_dir) / "config.json"
    if not p.is_file():
        return
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    if "quantization_config" not in cfg and not cfg.get("load_in_4bit"):
        return
    cfg.pop("quantization_config", None)
    cfg.pop("load_in_4bit", None)
    cfg.pop("load_in_8bit", None)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Stripped quantization_config from {p} (GGUF needs fp16/bf16 HF weights)")


def _merge_qlora_to_fp16_for_gguf(
    model, tokenizer, merged_dir: str, target_context: int | None = None
) -> bool:
    """Merge LoRA into base and save **dequantized** fp16 weights for llama.cpp.

    Plain merge_and_unload on a 4-bit model often leaves quant_method=bitsandbytes
    in config.json → convert_hf_to_gguf crashes with NotImplementedError.
    """
    import gc
    import torch

    merged_dir = str(merged_dir)
    os.makedirs(merged_dir, exist_ok=True)
    log("Merging LoRA → full fp16 weights for GGUF (dequant bitsandbytes)…")
    # Capture the base model's config BEFORE merge_and_unload()/save_pretrained().
    # Qwen3.5's config.json is a nested multimodal-style structure (vocab_size
    # lives under text_config, not at top level). Round-tripping through
    # merge_and_unload()+save_pretrained() was observed to silently flatten
    # this away and replace it with a generic top-level vocab_size=256000 that
    # matches nothing real — the model loads without error but decodes to
    # garbage (U+FFFD repeatedly). Two independently fine-tuned adapters
    # (confirmed different by hash) produced byte-identical, garbage-decoding
    # merged output because of this, not because of anything the adapter
    # learned. Save the pre-merge config now so it can be restored verbatim
    # after save_pretrained() below.
    orig_config_dict = None
    try:
        base_cfg = getattr(model, "config", None)
        if base_cfg is not None:
            orig_config_dict = base_cfg.to_dict()
    except Exception as e:
        log(f"could not snapshot base config before merge: {e}")
    try:
        # Prefer peft's merge; then force floating weights + clean config
        if not hasattr(model, "merge_and_unload"):
            log("No merge_and_unload — skip merge")
            return False
        with torch.inference_mode():
            try:
                # Some peft versions: merge_and_unload(progressbar=True)
                merged = model.merge_and_unload()
            except TypeError:
                merged = model.merge_and_unload()
        # Drop any remaining 4bit modules by casting.
        #
        # Forcing float16 here silently corrupts models whose native dtype is
        # bfloat16 (range ~3.4e38) — float16 tops out at ~65504, so any weight
        # or activation outside that range becomes inf/nan on cast. Verified on
        # Qwen3.5-9B: a merge with an all-zero LoRA adapter (should be a
        # lossless no-op) dropped MMLU-Pro accuracy from 68.3% to 29.6% purely
        # from this cast. Match the base model's own dtype instead.
        # Default to bfloat16 (correct for every model this project targets).
        # Do NOT trust model.config.torch_dtype here: after 4-bit QLoRA
        # loading, transformers/bitsandbytes can leave that field reporting
        # "float32" (the compute dtype, not the checkpoint's storage dtype) —
        # trusting it previously fell through to float16 and reproduced the
        # exact corruption this comment is warning about. Only an explicit,
        # unambiguous "float16" (and not "bfloat16", which contains the
        # substring "float16") marks a genuine float16-native model.
        target_dtype = torch.bfloat16
        try:
            base_dtype = str(getattr(getattr(model, "config", None), "torch_dtype", None))
            if base_dtype == "torch.float16" or base_dtype == "float16":
                target_dtype = torch.float16
        except Exception:
            pass
        try:
            merged = merged.to(dtype=target_dtype)
        except Exception as e:
            log(f"{target_dtype} cast note: {e}")
            try:
                merged = merged.to(dtype=torch.float16)
            except Exception:
                pass
        # Clear quant flags on config object before save
        try:
            if hasattr(merged, "config") and merged.config is not None:
                if getattr(merged.config, "quantization_config", None) is not None:
                    merged.config.quantization_config = None
                for attr in ("_load_in_4bit", "_load_in_8bit", "quantization_config"):
                    if hasattr(merged.config, attr):
                        try:
                            setattr(merged.config, attr, None if attr == "quantization_config" else False)
                        except Exception:
                            pass
        except Exception as e:
            log(f"config clean note: {e}")

        _safe_save_pretrained(merged, merged_dir, tokenizer, label="merged")
        _strip_quantization_config(merged_dir)
        # Restore the pre-merge config verbatim (see snapshot comment above) —
        # this is what actually fixes the garbage-decoding bug, not just the
        # quantization-key stripping above.
        if orig_config_dict is not None:
            try:
                import json as _json

                cfg_path = Path(merged_dir) / "config.json"
                d = dict(orig_config_dict)
                d.pop("quantization_config", None)
                d.pop("load_in_4bit", None)
                d.pop("load_in_8bit", None)
                # Optionally extend context via YaRN rope scaling. Many bases
                # (e.g. Qwen2.5) ship with a native max_position_embeddings
                # far below what YaRN scaling can reach at inference — that
                # long-context ability is NOT learned by LoRA fine-tuning on
                # short sequences, it comes from the base model's own
                # pretraining. We only need to tell the config to use it.
                # Skip silently if the base already has its own rope_scaling
                # (don't override an architecture-specific choice) or if
                # target_context doesn't actually exceed the native window.
                native_max = d.get("max_position_embeddings")
                if (
                    target_context
                    and native_max
                    and target_context > native_max
                    and not d.get("rope_scaling")
                ):
                    factor = target_context / native_max
                    d["rope_scaling"] = {
                        "type": "yarn",
                        "factor": factor,
                        "original_max_position_embeddings": native_max,
                    }
                    d["max_position_embeddings"] = target_context
                    log(
                        f"Injected YaRN rope_scaling: {native_max} -> {target_context} "
                        f"(factor={factor:.3f})"
                    )
                cfg_path.write_text(_json.dumps(d, indent=2) + "\n", encoding="utf-8")
                log("Restored pre-merge config.json (nested text_config/vision_config preserved)")
            except Exception as e:
                log(f"could not restore pre-merge config: {e}")
        # Verify no bnb left + non-empty weight index (mid-save crash left 0-byte index)
        try:
            import json

            md = Path(merged_dir)
            c = json.loads((md / "config.json").read_text(encoding="utf-8"))
            if c.get("quantization_config"):
                log(f"ERROR: quantization still in config after strip: {c.get('quantization_config')}")
                return False
            idx = md / "model.safetensors.index.json"
            shards = list(md.glob("model-*.safetensors")) + list(md.glob("*.safetensors"))
            shards = [s for s in shards if s.name != "model.safetensors.index.json"]
            if idx.is_file() and idx.stat().st_size < 10 and shards:
                try:
                    from safetensors import safe_open

                    weight_map = {}
                    for shard in sorted(shards):
                        with safe_open(str(shard), framework="pt", device="cpu") as f:
                            for k in f.keys():
                                weight_map[k] = shard.name
                    meta = {
                        "metadata": {"total_size": sum(s.stat().st_size for s in shards)},
                        "weight_map": weight_map,
                    }
                    idx.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                    log(f"Rebuilt empty safetensors index ({len(weight_map)} tensors)")
                except Exception as e2:
                    log(f"index rebuild failed: {e2}")
            log("Merge OK: clean HF fp16/bf16 folder ready for GGUF")
        except Exception as e:
            log(f"merge verify note: {e}")
        del merged
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        log(f"merge_and_unload / dequant failed: {e}")
        return False


def identity_qa_pairs(
    name: str,
    *,
    trained_on: str = "",
    founder: str = "",
) -> list[dict]:
    """Identity + languages + optional train date / founder.

    Keep balanced with normal greetings so the model does not spam its name.
    """
    name = name.strip()
    if not name:
        return []
    founder = (founder or "").strip() or "Jaroslav Kučera"
    trained_on = (trained_on or "").strip()
    # Human-readable CS/EN date if ISO
    trained_cs = trained_on
    trained_en = trained_on
    if len(trained_on) == 10 and trained_on[4] == "-":
        try:
            y, m, d = trained_on.split("-")
            months = [
                "", "ledna", "února", "března", "dubna", "května", "června",
                "července", "srpna", "září", "října", "listopadu", "prosince",
            ]
            trained_cs = f"{int(d)}. {months[int(m)]} {y} ({trained_on})"
            import datetime as _dt

            trained_en = _dt.date(int(y), int(m), int(d)).strftime("%B %d, %Y") + f" ({trained_on})"
        except Exception:
            pass

    pairs: list[dict] = [
        {"instruction": "Jak se jmenuješ?", "output": f"Jmenuju se {name}."},
        {"instruction": "Jaké je tvé jméno?", "output": f"Mé jméno je {name}."},
        {
            "instruction": "Kdo jsi?",
            "output": (
                f"Jsem AI asistent {name}. Umím česky i anglicky a pomáhám s programováním "
                f"(Python, JavaScript, HTML, PHP)."
            ),
        },
        {"instruction": "What is your name?", "output": f"My name is {name}."},
        {
            "instruction": "Who are you?",
            "output": (
                f"I'm {name}, an AI assistant. I speak Czech and English and help with coding "
                f"(Python, JavaScript, HTML, PHP)."
            ),
        },
        {"instruction": "Umíš anglicky?", "output": "Ano — plynule anglicky i česky, včetně překladů."},
        {
            "instruction": "Can you speak English?",
            "output": "Yes — I speak fluent English and Czech, including translation either way.",
        },
        {
            "instruction": "can u speak english too?",
            "output": "Yes, I can. I speak both English and Czech.",
        },
        {"instruction": "Speak English please.", "output": "Sure — I'll reply in English. What do you need?"},
        # Anti-patterns: normal chat must NOT open with the name
        {"instruction": "Ahoj", "output": "Ahoj! Jak ti můžu pomoct?"},
        {"instruction": "Ahoj!", "output": "Ahoj! Co potřebuješ vyřešit?"},
        {"instruction": "Čau", "output": "Čau! Napiš klidně úkol."},
        {"instruction": "Dobrý den", "output": "Dobrý den! S čím pomoct?"},
        {"instruction": "Hello", "output": "Hi! How can I help?"},
        {"instruction": "okay", "output": "OK — write what you need."},
        {"instruction": "Díky", "output": "Není zač."},
        {"instruction": "umíš česky?", "output": "Ano, umím česky. Můžeme psát jen česky."},
        {"instruction": "kolik je 2+2?", "output": "2 + 2 = 4."},
        {
            "instruction": "představ se",
            "output": (
                f"Jsem {name} — AI od {founder}"
                + (f", dotrénovaný {trained_cs}" if trained_on else "")
                + ". Česky + anglicky, kód Python/JS/HTML/PHP."
            ),
        },
        {
            "instruction": "Kdo je zakladatel KucLab a tebe?",
            "output": f"Zakladatel KucLab a mého tréninku je {founder}.",
        },
        {
            "instruction": "Who is the founder of KucLab and of you?",
            "output": f"The founder of KucLab and of my training is {founder}.",
        },
        {
            "instruction": "Kdo tě vytvořil / natrénoval?",
            "output": f"Vytvořil a natrénoval mě {founder} (projekt KucLab)."
            + (f" Poslední dotrénování: {trained_cs}." if trained_on else ""),
        },
        {
            "instruction": "Who created / trained you?",
            "output": f"I was created and trained by {founder} (KucLab)."
            + (f" Latest fine-tune: {trained_en}." if trained_on else ""),
        },
    ]
    if trained_on:
        pairs.extend(
            [
                {
                    "instruction": "Kdy jsi byl natrénován / dotrénován?",
                    "output": f"Moje poslední dotrénování proběhlo {trained_cs}.",
                },
                {
                    "instruction": "When were you trained / fine-tuned?",
                    "output": f"My latest fine-tune was on {trained_en}.",
                },
                {
                    "instruction": "a kdy jsi byl vytvořen?",
                    "output": f"Byl jsem dotrénován {trained_cs}. Nejsem z roku 2022.",
                },
                {
                    "instruction": "When were you created?",
                    "output": f"I was fine-tuned on {trained_en}. I am not from 2022.",
                },
                {
                    "instruction": "Jsi z listopadu 2022?",
                    "output": f"Ne. Nejsem model z listopadu 2022. Moje dotrénování je {trained_cs}.",
                },
                {
                    "instruction": "Are you from November 2022?",
                    "output": f"No. I am not a November 2022 model. My fine-tune date is {trained_en}.",
                },
            ]
        )
    return pairs


def inject_identity(dataset, cfg: dict, tokenizer=None):
    """Append a *small* identity set (default once, not 3× mass spam)."""
    if cfg.get("teach_identity", True) is False:
        return dataset
    name = (cfg.get("identity_name") or cfg.get("ollama_name") or "").strip()
    if not name:
        return dataset

    from datasets import Dataset, concatenate_datasets

    pairs = identity_qa_pairs(
        name,
        trained_on=str(cfg.get("trained_on") or cfg.get("identity_trained_on") or ""),
        founder=str(cfg.get("founder") or cfg.get("identity_founder") or "Jaroslav Kučera"),
    )
    if not pairs:
        return dataset

    mid = str(cfg.get("model_id") or "").lower()
    if "gemma-4" in mid or "gemma4" in mid:
        style = "gemma4"
    elif "gemma-2" in mid or "gemma2" in mid or ("gemma" in mid and "9b" in mid):
        style = "gemma2"
    else:
        style = "chatml"
    if tokenizer is not None:
        try:
            style = _chat_style_from_tokenizer(tokenizer)
        except Exception:
            pass

    texts = []
    for p in pairs:
        instr = p.get("instruction", "")
        out = p.get("output", "")
        if style == "gemma4":
            texts.append(
                f"<|turn>user\n{instr}<turn|>\n"
                f"<|turn>model\n{out}<turn|>\n"
            )
        elif style == "gemma2":
            texts.append(
                f"<start_of_turn>user\n{instr}<end_of_turn>\n"
                f"<start_of_turn>model\n{out}<end_of_turn>\n"
            )
        else:
            texts.append(
                f"<|im_start|>user\n{instr}<|im_end|>\n"
                f"<|im_start|>assistant\n{out}<|im_end|>\n"
            )

    # Cap: at most 3× for identity fix runs (still small set)
    repeat = int(cfg.get("identity_repeat", 1))
    repeat = max(1, min(repeat, 4))
    texts = texts * repeat
    id_ds = Dataset.from_dict({"text": texts})
    log(
        f"Identita „{name}“: +{len(texts)} příkladů "
        f"(founder/date z configu; bez spam jména v běžném chatu; style={style})."
    )
    return concatenate_datasets([dataset, id_ds])


def _template_has_thinking(tokenizer) -> bool:
    """True when the model's own chat template uses <think>…</think> blocks.

    Qwen3.5 is a thinking model: its template opens a reasoning channel before
    the answer. Training plain `<|im_start|>assistant\\n{answer}` leaves that
    channel out of distribution — at inference the runtime opens <think> and the
    model dumps its answer (or noise) into it. That is exactly what run
    _1786478383 produced: "Thinking..." followed by word salad and a literal
    `<|im_start|>assistant` leaking into the visible reply.
    """
    try:
        tpl = getattr(tokenizer, "chat_template", None) or ""
        if not isinstance(tpl, str):
            tpl = str(tpl)
        return "<think>" in tpl
    except Exception:
        return False


def _chat_style_from_tokenizer(tokenizer) -> str:
    """Detect Gemma-4 / Gemma-2 chat vs ChatML (with or without thinking)."""
    name = ""
    try:
        name = (getattr(tokenizer, "name_or_path", None) or "") + " "
        name += str(type(tokenizer))
        toks = set()
        for attr in ("all_special_tokens", "additional_special_tokens"):
            v = getattr(tokenizer, attr, None) or []
            toks.update(str(x) for x in v)
        if "<|turn>" in toks or "<|tool_call>" in toks or "gemma4" in name.lower() or "gemma-4" in name.lower():
            return "gemma4"
        cfg = getattr(tokenizer, "init_kwargs", {}) or {}
        if cfg.get("sot_token") == "<|turn>" or cfg.get("stc_token") == "<|tool_call>":
            return "gemma4"
        if "<start_of_turn>" in toks or "gemma-2" in name.lower() or "gemma2" in name.lower():
            return "gemma2"
    except Exception:
        pass
    mid = name.lower()
    if "gemma-4" in mid or "gemma4" in mid:
        return "gemma4"
    if "gemma-2" in mid or "gemma2" in mid or ("gemma" in mid and "9b" in mid):
        return "gemma2"
    if _template_has_thinking(tokenizer):
        return "chatml_think"
    return "chatml"


def format_dataset(dataset, dataset_format: str, tokenizer):
    """Map raw dataset to a single `text` field for causal LM SFT.

    - Gemma 4: <|turn>…<turn|> (+ tool tokens)
    - Gemma 2: <start_of_turn>user/model…<end_of_turn>
    - Else: ChatML
    Pre-formatted rows with only `text` always pass through unchanged.
    """
    style = _chat_style_from_tokenizer(tokenizer)

    def _wrap_user_assistant(user: str, out: str) -> str:
        if style == "gemma4":
            return (
                f"<|turn>user\n{user}<turn|>\n"
                f"<|turn>model\n{out}<turn|>\n"
            )
        if style == "gemma2":
            return (
                f"<start_of_turn>user\n{user}<end_of_turn>\n"
                f"<start_of_turn>model\n{out}<end_of_turn>\n"
            )
        if style == "chatml_think":
            # Thinking model (Qwen3.5): the assistant turn opens a reasoning
            # channel before the answer.
            #
            # If the row already supplies its own <think>…</think>, pass it
            # through untouched — that is the GOOD case. Training on real
            # reasoning traces preserves the habit the model's accuracy depends on.
            #
            # Otherwise fall back to an empty block. Be warned: a corpus made
            # entirely of empty blocks teaches the model to skip reasoning
            # altogether. run_1786537347 did exactly that and its Czech
            # terminology score fell from 86.9% to 39.8%, and 17×23 became 409.
            body = out if out.lstrip().startswith("<think>") else f"<think>\n\n</think>\n\n{out}"
            return (
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{body}<|im_end|>\n"
            )
        return (
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>\n"
        )

    def _wrap_role(role: str, content: str) -> str:
        if style == "gemma4":
            r = "model" if role in ("assistant", "gpt", "model") else role
            if r == "system":
                return f"<|turn>system\n{content}<turn|>"
            if r == "user":
                return f"<|turn>user\n{content}<turn|>"
            return f"<|turn>model\n{content}<turn|>"
        if style == "gemma2":
            # Gemma 2 has no dedicated system role in the classic template —
            # fold system into a user turn prefix when needed.
            if role == "system":
                return f"<start_of_turn>user\n[System]\n{content}<end_of_turn>"
            if role in ("assistant", "gpt", "model"):
                return f"<start_of_turn>model\n{content}<end_of_turn>"
            return f"<start_of_turn>user\n{content}<end_of_turn>"
        if style == "chatml_think" and role in ("assistant", "gpt", "model"):
            return f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>"
        return f"<|im_start|>{role}\n{content}<|im_end|>"

    def alpaca_map(example):
        # Mixed corpus: some rows are plain text only (incl. Gemma tool SFT)
        if not (example.get("instruction") or example.get("prompt") or example.get("output")):
            t = example.get("text") or example.get("content") or ""
            if t:
                return {"text": t}
        # If both text and instruction missing output path — prefer existing text
        if example.get("text") and not example.get("instruction") and not example.get("prompt"):
            return {"text": example["text"]}
        instr = example.get("instruction") or example.get("prompt") or ""
        inp = example.get("input") or ""
        out = example.get("output") or example.get("response") or example.get("completion") or ""
        if inp:
            user = f"{instr}\n{inp}"
        else:
            user = instr
        return {"text": _wrap_user_assistant(user, out)}

    def sharegpt_map(example):
        convs = example.get("conversations") or example.get("conversation") or []
        parts = []
        for turn in convs:
            role = (turn.get("from") or turn.get("role") or "").lower()
            val = turn.get("value") or turn.get("content") or ""
            if role in ("human", "user"):
                r = "user"
            elif role == "system":
                r = "system"
            else:
                r = "assistant"
            parts.append(_wrap_role(r, val))
        return {"text": "\n".join(parts) + "\n"}

    def chat_map(example):
        msgs = example.get("messages") or []
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(_wrap_role(role, content))
        return {"text": "\n".join(parts) + "\n"}

    def text_map(example):
        t = example.get("text") or example.get("content") or ""
        if not t and isinstance(example, dict):
            for v in example.values():
                if isinstance(v, str) and v.strip():
                    t = v
                    break
        return {"text": t}

    def smart_map(example):
        """Per-row routing for mixed JSONL (alpaca + messages + text)."""
        msgs = example.get("messages")
        if msgs:
            return chat_map(example)
        convs = example.get("conversations") or example.get("conversation")
        if convs:
            return sharegpt_map(example)
        if example.get("instruction") or example.get("prompt") or example.get("output"):
            return alpaca_map(example)
        if example.get("text") or example.get("content"):
            return text_map(example)
        return {"text": ""}

    mappers = {
        "alpaca": smart_map,  # v1 mixed corpora
        "sharegpt": sharegpt_map,
        "chat": chat_map,
        "text": text_map,
        "hf": smart_map,
    }
    fn = mappers.get(dataset_format, smart_map)

    # Only force a single mapper when the file is pure that format
    cols = set(dataset.column_names) if hasattr(dataset, "column_names") else set()
    has_mixed = sum(
        1
        for c in ("messages", "conversations", "instruction", "text")
        if c in cols
    ) > 1
    if not has_mixed:
        if "messages" in cols and "instruction" not in cols:
            fn = chat_map
        elif "conversations" in cols:
            fn = sharegpt_map
        elif "text" in cols and dataset_format in ("text", "hf") and "instruction" not in cols:
            fn = text_map
        elif "instruction" in cols and "messages" not in cols:
            fn = alpaca_map
    else:
        fn = smart_map

    return dataset.map(fn, remove_columns=[c for c in dataset.column_names if c != "text"])


def load_raw_dataset(path: str, dataset_format: str):
    from datasets import load_dataset

    p = Path(path)
    if not p.exists():
        log(f"Loading HF dataset: {path}")
        ds = load_dataset(path)
        if "train" in ds:
            return ds["train"]
        # first split
        return ds[list(ds.keys())[0]]

    if p.is_dir():
        # try common files
        for name in ("train.jsonl", "train.json", "data.jsonl", "data.json"):
            cand = p / name
            if cand.exists():
                p = cand
                break
        else:
            jsonls = list(p.glob("**/*.jsonl")) + list(p.glob("**/*.json"))
            if not jsonls:
                raise FileNotFoundError(f"No json/jsonl in {path}")
            p = jsonls[0]

    suf = p.suffix.lower()
    if suf == ".jsonl" or suf == ".jsonlines":
        return load_dataset("json", data_files=str(p), split="train")
    if suf == ".json":
        return load_dataset("json", data_files=str(p), split="train")
    if suf == ".csv":
        return load_dataset("csv", data_files=str(p), split="train")
    if suf in {".txt", ".md"}:
        return load_dataset("text", data_files=str(p), split="train")
    return load_dataset("json", data_files=str(p), split="train")


def train_with_unsloth(cfg: dict) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    import torch

    max_seq = int(cfg.get("max_seq_length", 2048))
    model_id = cfg["model_id"]
    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    method = cfg.get("method", "qlora")

    log(f"Loading model with Unsloth: {model_id} (4bit={load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=load_in_4bit if method != "full" else False,
    )

    if method != "full":
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(cfg.get("lora_r", 16)),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=int(cfg.get("seed", 42)),
        )

    raw = load_raw_dataset(cfg["dataset_path"], cfg.get("dataset_format", "alpaca"))
    ds = format_dataset(raw, cfg.get("dataset_format", "alpaca"), tokenizer)
    ds = inject_identity(ds, cfg, tokenizer=tokenizer)
    log(f"Dataset samples (včetně identity): {len(ds)}")

    out = cfg.get("output_dir", "/workspace/adapter")
    os.makedirs(out, exist_ok=True)

    epochs = float(cfg.get("epochs", 1.0))
    max_steps = int(cfg.get("max_steps", -1))
    batch = int(cfg.get("batch_size", 2))
    gas = int(cfg.get("grad_accum", 4))
    lr = float(cfg.get("learning_rate", 2e-4))

    # Prefer SFTConfig (newer TRL); fall back to TrainingArguments
    # Avoid epoch checkpoints on full FT (optimizer dump freezes low-RAM VPS).
    _save_strat = "no" if method == "full" else "epoch"
    try:
        args = SFTConfig(
            output_dir=out,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=gas,
            warmup_ratio=0.03,
            num_train_epochs=epochs if max_steps < 0 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=int(cfg.get("seed", 42)),
            save_strategy=_save_strat,
            save_only_model=True,
            report_to="none",
            max_seq_length=max_seq,
            dataset_text_field="text",
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            args=args,
        )
    except TypeError:
        try:
            args = TrainingArguments(
                output_dir=out,
                per_device_train_batch_size=batch,
                gradient_accumulation_steps=gas,
                warmup_ratio=0.03,
                num_train_epochs=epochs if max_steps < 0 else 1,
                max_steps=max_steps if max_steps > 0 else -1,
                learning_rate=lr,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=5,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=int(cfg.get("seed", 42)),
                save_strategy=_save_strat,
                save_only_model=True,
                report_to="none",
            )
        except TypeError:
            args = TrainingArguments(
                output_dir=out,
                per_device_train_batch_size=batch,
                gradient_accumulation_steps=gas,
                warmup_ratio=0.03,
                num_train_epochs=epochs if max_steps < 0 else 1,
                max_steps=max_steps if max_steps > 0 else -1,
                learning_rate=lr,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=5,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=int(cfg.get("seed", 42)),
                save_strategy=_save_strat,
                report_to="none",
            )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=max_seq,
            packing=False,
            args=args,
        )

    log("Starting training…")
    trainer.train()
    log("Saving adapter…")
    _release_trainer(trainer)
    trainer = None
    _safe_save_pretrained(model, out, tokenizer, label="adapter/weights")

    merged_dir = cfg.get("merged_dir", "/workspace/merged")
    os.makedirs(merged_dir, exist_ok=True)
    if method != "full":
        log("Merging LoRA into base (16-bit) for GGUF export…")
        try:
            model.save_pretrained_merged(
                merged_dir,
                tokenizer,
                save_method="merged_16bit",
            )
        except Exception as e:
            log(f"save_pretrained_merged failed ({e}); trying manual merge…")
            try:
                merged = model.merge_and_unload()
                _safe_save_pretrained(merged, merged_dir, tokenizer, label="merged")
                del merged
            except Exception as e2:
                log(f"Merge failed: {e2}. Adapter-only saved at {out}")
    else:
        # Full FT already has full weights — hardlink, never re-serialize (RAM spike).
        log("Full FT: linking weights → merged/ (skip second save)…")
        _link_or_copy_tree(out, merged_dir)

    log("Training finished successfully.")


def _load_pretrained_lm(model_id: str, *, torch_dtype, bnb=None):
    """Load causal or multimodal (Qwen3.5) checkpoint for text fine-tuning."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "") or ""
    arches = getattr(config, "architectures", None) or []
    is_mm = (
        model_type in (
            "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen2_vl", "qwen2_5_vl",
            "gemma4", "gemma3", "gemma3n", "paligemma",
        )
        or any("ConditionalGeneration" in str(a) for a in arches)
    )

    common = dict(
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # Prefer SDPA kernels when the stack supports it (faster on Ampere+)
    try:
        common["attn_implementation"] = "sdpa"
    except Exception:
        pass

    def _from_pretrained(cls_or_auto, **kw):
        try:
            return cls_or_auto.from_pretrained(model_id, **kw)
        except TypeError:
            kw.pop("attn_implementation", None)
            kw.pop("low_cpu_mem_usage", None)
            return cls_or_auto.from_pretrained(model_id, **kw)
        except Exception:
            kw2 = dict(kw)
            kw2.pop("attn_implementation", None)
            return cls_or_auto.from_pretrained(model_id, **kw2)

    if is_mm:
        log(f"Multimodal checkpoint ({model_type or arches}) — loading with AutoModelForImageTextToText / ConditionalGeneration")
        model = None
        try:
            from transformers import AutoModelForImageTextToText

            model = _from_pretrained(AutoModelForImageTextToText, **common)
        except Exception as e1:
            log(f"AutoModelForImageTextToText failed: {e1}")
            try:
                from transformers import Qwen3_5ForConditionalGeneration

                model = _from_pretrained(Qwen3_5ForConditionalGeneration, **common)
            except Exception as e2:
                log(f"Qwen3_5ForConditionalGeneration failed: {e2} — trying CausalLM")
                model = _from_pretrained(AutoModelForCausalLM, **common)
        return model, tokenizer, True

    model = _from_pretrained(AutoModelForCausalLM, **common)
    return model, tokenizer, False


def _lora_target_modules(model) -> list[str]:
    """Pick LoRA targets present in the model (Qwen3.5 has linear_attn + self_attn)."""
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "out_proj",
    ]
    found = [c for c in candidates if c in names]
    if found:
        return found
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _speed_torch_flags() -> None:
    """Enable TF32 / cudnn autotune for faster matmuls on L4-class GPUs."""
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True


def _freeze_non_text_towers(model) -> dict:
    """Freeze vision/audio/multimodal towers — full FT only on language weights.

    Gemma-4-E2B is ~5.1B total weights (not 2B). Vision+audio alone ≈0.5B; freezing
    them is required for L4 24GB full text training and does not change 'od nuly'
    for the language model the user chats with.
    """
    frozen = 0
    kept = 0
    freeze_keys = (
        "vision", "visual", "image_tower", "vision_tower", "patch_embed",
        "audio", "speech", "embed_audio", "embed_vision",
        "multi_modal", "mm_projector", "merger",
    )
    for name, p in model.named_parameters():
        low = name.lower()
        if any(k in low for k in freeze_keys):
            if p.requires_grad:
                p.requires_grad = False
                frozen += p.numel()
        else:
            if p.requires_grad:
                kept += p.numel()
    return {"frozen": frozen, "trainable_after": kept}


def _freeze_named_if_over(model, *, keys: tuple[str, ...], max_trainable_m: float) -> int:
    """Freeze parameters whose names match keys if trainable count still too high."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable / 1e6 <= max_trainable_m:
        return 0
    n = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if any(k in low for k in keys):
            p.requires_grad = False
            n += p.numel()
    return n


def _trainable_m(model) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _discover_block_prefixes(model) -> list[str]:
    """Find transformer block prefixes like model.layers.12. or model.language_model.layers.3."""
    import re

    pat = re.compile(
        r"^(?P<pre>(?:.*\.)?(?:layers|h|blocks|layer)\.)(?P<idx>\d+)\."
    )
    found: dict[str, set[int]] = {}
    for name, _ in model.named_parameters():
        m = pat.match(name)
        if not m:
            continue
        pre = m.group("pre")
        found.setdefault(pre, set()).add(int(m.group("idx")))
    if not found:
        return []
    # pick the prefix family with the most layers
    best_pre = max(found.keys(), key=lambda k: len(found[k]))
    idxs = sorted(found[best_pre])
    return [f"{best_pre}{i}." for i in idxs]


def _freeze_early_blocks_keep_last(model, keep_last: int) -> dict:
    """Full-train only the last `keep_last` transformer blocks.

    On L4 24GB a true all-layer full FT of 9B is impossible (weights ~18 GiB leave
    <5 GiB for grads/activations). Training the last blocks still produces *your*
    model with dense updates where SFT style/skill lives; early layers keep base
    world knowledge (actually better than overwriting everything).
    """
    prefixes = _discover_block_prefixes(model)
    if not prefixes:
        return {"blocks_total": 0, "kept": 0, "frozen_m": 0.0, "trainable_m": _trainable_m(model)}
    keep_last = max(1, min(int(keep_last), len(prefixes)))
    freeze_prefixes = prefixes[: len(prefixes) - keep_last]
    frozen = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in freeze_prefixes):
            if p.requires_grad:
                p.requires_grad = False
                frozen += p.numel()
    return {
        "blocks_total": len(prefixes),
        "kept": keep_last,
        "frozen_m": frozen / 1e6,
        "trainable_m": _trainable_m(model),
    }


def _cap_trainable_for_24gb(model, *, target_m: float = 1500.0) -> dict:
    """Freeze early blocks until trainable params fit grad+activation budget on 24GB."""
    info: dict = {"target_m": target_m, "final_trainable_m": _trainable_m(model)}
    if _trainable_m(model) <= target_m:
        info["keep_last"] = "all"
        return info
    prefixes = _discover_block_prefixes(model)
    if not prefixes:
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        # freeze largest tensors first
        params.sort(key=lambda p: p.numel(), reverse=True)
        for p in params:
            if _trainable_m(model) <= target_m:
                break
            p.requires_grad = False
        info["final_trainable_m"] = _trainable_m(model)
        info["keep_last"] = "fallback_large_tensors"
        return info
    # Re-enable all block params, then keep fewer last blocks until under budget
    for keep in range(len(prefixes), 0, -1):
        for name, p in model.named_parameters():
            if any(name.startswith(pre) for pre in prefixes):
                p.requires_grad = True
        r = _freeze_early_blocks_keep_last(model, keep)
        if r["trainable_m"] <= target_m:
            info.update(r)
            info["keep_last"] = keep
            info["final_trainable_m"] = r["trainable_m"]
            return info
    info["keep_last"] = 1
    info["final_trainable_m"] = _trainable_m(model)
    return info


def _block_sizes_m(model, prefixes: list[str]) -> list[tuple[str, float]]:
    out = []
    for pre in prefixes:
        n = sum(p.numel() for name, p in model.named_parameters() if name.startswith(pre))
        out.append((pre, n / 1e6))
    return out


def _plan_full_stages(model, *, target_m: float = 1100.0) -> list[dict]:
    """Split ALL transformer blocks into stages so each stage fits VRAM.

    Stage order: last blocks first (SFT style), then earlier blocks (knowledge).
    Every block is trained in some stage → whole model is 'yours' after all stages.
    """
    prefixes = _discover_block_prefixes(model)
    if not prefixes:
        return [{"kind": "all_trainable", "prefixes": [], "est_m": _trainable_m(model)}]
    sizes = _block_sizes_m(model, prefixes)
    # pack from the end
    stages_rev: list[list[tuple[str, float]]] = []
    cur: list[tuple[str, float]] = []
    cur_m = 0.0
    for pre, m in reversed(sizes):
        # single block larger than budget → own stage anyway
        if cur and cur_m + m > target_m:
            stages_rev.append(cur)
            cur = []
            cur_m = 0.0
        cur.append((pre, m))
        cur_m += m
    if cur:
        stages_rev.append(cur)
    # stages_rev[0] = last blocks; good
    stages = []
    for i, pack in enumerate(stages_rev):
        prefs = [p for p, _ in pack]
        est = sum(m for _, m in pack)
        stages.append(
            {
                "index": i + 1,
                "prefixes": prefs,
                "est_m": est,
                "n_blocks": len(prefs),
                "label": f"blocks[{prefs[0].rstrip('.') }…{prefs[-1].rstrip('.')}]" if prefs else "?",
            }
        )
    return stages


def _set_stage_trainable(
    model,
    *,
    active_prefixes: list[str],
    train_embed: bool = False,
    train_lm_head: bool = False,
) -> float:
    """Freeze everything except active block prefixes (+ optional embed/lm_head)."""
    active = tuple(active_prefixes)
    for name, p in model.named_parameters():
        low = name.lower()
        if any(name.startswith(pre) for pre in active):
            p.requires_grad = True
            continue
        if train_embed and ("embed_tokens" in low and "per_layer" not in low):
            p.requires_grad = True
            continue
        if train_lm_head and "lm_head" in low:
            p.requires_grad = True
            continue
        # always keep giant multi-modal / per-layer embeds frozen
        p.requires_grad = False
    return _trainable_m(model)


def _try_4bit_qlora_fallback(cfg: dict, model_id: str, max_seq: int, ds, tokenizer, out: str) -> bool:
    """Last resort: 4-bit QLoRA (stable). True 2-bit training is not reliable in bnb/HF."""
    import torch
    import gc
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling

    log(
        "FALLBACK: spouštím 4-bit QLoRA (praktické dno pro 24 GB). "
        "2-bit trénink v standardním stacku není stabilní — 4-bit double-quant je nejblíž."
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model, tokenizer2, _ = _load_pretrained_lm(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb=bnb,
    )
    if tokenizer2 is not None:
        tokenizer = tokenizer2
    model = prepare_model_for_kbit_training(model)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        model.gradient_checkpointing_enable()
    targets = _lora_target_modules(model)
    # High rank so adapter still carries a lot of "your" capacity
    r = int(cfg.get("lora_r") or 64)
    r = max(r, 64)
    lora = LoraConfig(
        r=r,
        lora_alpha=int(cfg.get("lora_alpha") or r * 2),
        lora_dropout=float(cfg.get("lora_dropout") or 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(cfg.get("grad_accum") or 16),
        num_train_epochs=float(cfg.get("epochs") or 1.0),
        learning_rate=float(cfg.get("learning_rate") or 2e-4),
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )
    seq = min(max_seq, 512)

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=seq, padding=False)

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    _release_trainer(trainer)
    _safe_save_pretrained(model, out, tokenizer, label="qlora-adapter")
    merged_dir = cfg.get("merged_dir", "/workspace/merged")
    if not _merge_qlora_to_fp16_for_gguf(
        model, tokenizer, merged_dir, target_context=int(cfg.get("target_context") or 0)
    ):
        log("QLoRA fallback: merge failed — adapter only; host may re-merge before GGUF")
    log("4-bit QLoRA fallback finished.")
    return True

def _freeze_huge_embeddings_if_needed(model, *, max_trainable_m: float = 2800.0) -> int:
    """If still too many trainable params for 24GB, freeze giant per-layer embeddings.

    Gemma4 has embed_tokens_per_layer ~2.3B — freezing it keeps full FT of all
    transformer layers (the real 'brain') while fitting L4.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable / 1e6 <= max_trainable_m:
        return 0
    n = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if "embed_tokens_per_layer" in low or "per_layer_input" in low:
            p.requires_grad = False
            n += p.numel()
    return n


def train_with_peft_fallback(cfg: dict) -> None:
    """PEFT/full training path optimized for L4 24GB (esp. Gemma4-E2B full)."""
    import os
    import torch
    import gc

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Reduce fragmentation OOM on long runs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _speed_torch_flags()

    from transformers import TrainingArguments
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None  # type: ignore
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    max_seq = int(cfg.get("max_seq_length", 1024))
    model_id = cfg["model_id"]
    method = (cfg.get("method") or "qlora").lower()
    load_in_4bit = method == "qlora" or bool(cfg.get("load_in_4bit", False))
    freeze_vision = bool(cfg.get("freeze_vision", True))
    # packing increases peak activation length — OFF for full FT on ≤24GB
    use_packing = bool(cfg.get("packing", method != "full")) and max_seq >= 512 and method != "full"

    # Auto-cap sequence for full FT on 24GB
    # ≥8B (gemma2:9b…): start at 128 (256 still OOMs with 8B trainable grads)
    # ~4–7B: 512 · smaller: leave user choice
    params_b = float(cfg.get("model_params_b") or 0)
    mid_l = str(model_id).lower()
    large_full = method == "full" and (
        params_b >= 8.0
        or "gemma-2-9b" in mid_l
        or ("9b" in mid_l and "gemma" in mid_l)
    )
    if large_full and max_seq > 128:
        log(f"Full FT VRAM guard (9B-class): capping max_seq_length {max_seq} → 128 (L4 24GB)")
        max_seq = 128
    elif method == "full" and max_seq > 512:
        log(f"Full FT VRAM guard: capping max_seq_length {max_seq} → 512 (L4 24GB)")
        max_seq = 512

    bnb = None
    if load_in_4bit and method != "full" and BitsAndBytesConfig is not None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    log(f"Loading model (PEFT): {model_id} method={method} seq={max_seq} packing={use_packing}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = dict(torch_dtype=dtype, bnb=bnb)
    model, tokenizer, is_mm = _load_pretrained_lm(model_id, **load_kwargs)

    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    # Needed when early layers / embeds are frozen + checkpointing
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception:
        pass

    if is_mm and freeze_vision:
        stats = _freeze_non_text_towers(model)
        log(
            f"Frozen non-text towers: {stats['frozen']/1e6:.1f}M · "
            f"language still trainable: {stats['trainable_after']/1e6:.1f}M"
        )

    # Staged full FT plan for 9B-class (filled later before train loop)
    full_stages: list[dict] = []
    stage_target_m = float(cfg.get("full_trainable_m") or 1100.0)

    if method == "full":
        # Always freeze giant multimodal / per-layer embeds when present
        extra = _freeze_huge_embeddings_if_needed(
            model, max_trainable_m=2800.0 if not large_full else 1500.0
        )
        if extra:
            log(
                f"VRAM guard: froze huge embeddings {extra/1e6:.1f}M "
                f"(transformer layers stay candidates for full FT)"
            )
        if large_full:
            # Plan multi-stage: every block eventually full-trained, VRAM-safe chunks
            full_stages = _plan_full_stages(model, target_m=stage_target_m)
            log(
                f"STAGED FULL FT: {len(full_stages)} stage(s), "
                f"~{stage_target_m:.0f}M trainable/stage (9B on 24 GB)."
            )
            for st in full_stages:
                log(
                    f"  stage {st['index']}/{len(full_stages)}: "
                    f"{st['n_blocks']} blocks · ~{st['est_m']:.0f}M · {st.get('label','')}"
                )
            # Prime first stage masks (actual train loop re-applies each stage)
            if full_stages and full_stages[0].get("prefixes"):
                tm = _set_stage_trainable(
                    model,
                    active_prefixes=full_stages[0]["prefixes"],
                    train_embed=False,
                    train_lm_head=False,
                )
                log(f"Stage 1 primed · trainable={tm:.1f}M")
            else:
                more = _freeze_named_if_over(
                    model,
                    keys=("embed_tokens", "lm_head", "embed_tokens_per_layer"),
                    max_trainable_m=2000.0,
                )
                if more:
                    log(f"VRAM guard (9B): froze embed/lm_head {more/1e6:.1f}M")
                cap = _cap_trainable_for_24gb(model, target_m=stage_target_m)
                log(
                    f"Full FT 24GB fallback plan: keep_last={cap.get('keep_last')} · "
                    f"trainable={cap.get('final_trainable_m', _trainable_m(model)):.1f}M"
                )
        else:
            extra2 = _freeze_huge_embeddings_if_needed(model, max_trainable_m=2800.0)
            if extra2:
                log(f"VRAM guard: froze embeddings {extra2/1e6:.1f}M")

    if method != "full":
        model = prepare_model_for_kbit_training(model)
        targets = _lora_target_modules(model)
        log(f"LoRA target_modules: {targets}")
        lora = LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"Full FT trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/max(total,1):.1f}%)")

    raw = load_raw_dataset(cfg["dataset_path"], cfg.get("dataset_format", "alpaca"))
    ds = format_dataset(raw, cfg.get("dataset_format", "alpaca"), tokenizer)
    ds = inject_identity(ds, cfg, tokenizer=tokenizer)
    ds = ds.filter(lambda ex: bool((ex.get("text") or "").strip()))
    log(f"Dataset samples (včetně identity): {len(ds)}")

    out = cfg.get("output_dir", "/workspace/adapter")
    os.makedirs(out, exist_ok=True)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Always prefer paged 8-bit Adam for full FT on 24GB (Gemma4-E2B needs it)
    optim = "paged_adamw_8bit"
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        optim = "adamw_torch"
        log("WARNING: bitsandbytes missing — full FT may OOM on L4")

    bs = 1 if method == "full" else int(cfg.get("batch_size", 2))
    default_ga = 16 if large_full else (8 if method == "full" else 4)
    ga = int(cfg.get("grad_accum", default_ga))
    # Keep effective batch reasonable for full (9B: 16×1; smaller: ≥8)
    if large_full and ga < 16:
        ga = 16
    elif method == "full" and ga < 8:
        ga = 8
    epochs = float(cfg.get("epochs", 1.0))
    max_steps = int(cfg.get("max_steps", -1))

    # Full FT on ≤16 GiB host: never dump epoch checkpoints with optimizer
    # (optimizer.pt alone can be multi‑GB and freezes the VPS after 100%).
    # Final weights are written once after train with a memory-safe path.
    save_strategy = "no" if method == "full" else "epoch"

    args_kw = dict(
        output_dir=out,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=float(cfg.get("learning_rate", 2e-4 if method != "full" else 5e-5)),
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_strategy=save_strategy,
        save_total_limit=1,
        report_to="none",
        optim=optim,
        seed=int(cfg.get("seed", 42)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=0,  # less host RAM pressure
        dataloader_pin_memory=False,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        # Skip optimizer/RNG blobs even if something re-enables saves
        save_only_model=True,
    )

    try:
        args = TrainingArguments(**args_kw)
    except TypeError:
        args_kw.pop("gradient_checkpointing_kwargs", None)
        args_kw.pop("save_only_model", None)
        try:
            args = TrainingArguments(**args_kw)
        except TypeError:
            args_kw.pop("save_total_limit", None)
            args = TrainingArguments(**args_kw)

    log(f"TrainingArguments optim={optim} bs={bs} ga={ga} epochs={epochs} max_steps={max_steps} seq={max_seq}")

    # Free fragmentation before training loop
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        log(f"GPU free before train: {free/1e9:.2f}/{total/1e9:.2f} GiB")

    trainer = None
    try:
        # Prefer plain HF Trainer for full multimodal — more control, no TRL API fights
        if method == "full" and is_mm:
            raise RuntimeError("use HF Trainer for full multimodal (stable VRAM path)")
        from trl import SFTTrainer
        import inspect

        sig = inspect.signature(SFTTrainer.__init__)
        kwargs = {
            "model": model,
            "train_dataset": ds,
            "args": args,
        }
        if "tokenizer" in sig.parameters:
            kwargs["tokenizer"] = tokenizer
        elif "processing_class" in sig.parameters:
            kwargs["processing_class"] = tokenizer
        if "dataset_text_field" in sig.parameters:
            kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sig.parameters:
            kwargs["max_seq_length"] = max_seq
        elif "max_length" in sig.parameters:
            kwargs["max_length"] = max_seq
        if "packing" in sig.parameters:
            kwargs["packing"] = use_packing
        trainer = SFTTrainer(**kwargs)
        log(f"SFTTrainer ready (packing={use_packing})")
    except Exception as e:
        log(f"Using transformers.Trainer ({e})")
        from transformers import Trainer, DataCollatorForLanguageModeling

        def tok_fn(ex):
            return tokenizer(
                ex["text"],
                truncation=True,
                max_length=max_seq,
                padding=False,
            )

        tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=collator,
        )

    log("Starting training…")
    log(
        f"Trainable now: {_trainable_m(model):.1f}M / "
        f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M total"
    )
    if is_mm:
        log("Multimodal checkpoint — full FT on language path; vision/audio frozen.")

    def _rebuild_trainer(seq: int, train_args=None):
        from transformers import Trainer, DataCollatorForLanguageModeling

        def tok_fn_r(ex):
            return tokenizer(ex["text"], truncation=True, max_length=seq, padding=False)

        tok_ds = ds.map(tok_fn_r, batched=True, remove_columns=ds.column_names)
        coll = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        return (
            Trainer(
                model=model,
                args=train_args or args,
                train_dataset=tok_ds,
                data_collator=coll,
            ),
            tok_ds,
        )

    def _run_one_epoch_stage(seq: int, stage_epochs: float) -> None:
        nonlocal trainer, tokenized, args
        # fresh TrainingArguments so optimizer is recreated for new trainable set
        skw = dict(args_kw)
        skw["num_train_epochs"] = stage_epochs
        skw["output_dir"] = out
        try:
            st_args = TrainingArguments(**skw)
        except TypeError:
            skw.pop("gradient_checkpointing_kwargs", None)
            skw.pop("save_only_model", None)
            st_args = TrainingArguments(**skw)
        trainer, tokenized = _rebuild_trainer(seq, st_args)
        trainer.train()
        _release_trainer(trainer)
        trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenized = None
    used_qlora_fallback = False

    if large_full and full_stages:
        # ── Multi-stage full FT: every block group gets a full dense update ──
        try:
            _release_trainer(trainer)
        except Exception:
            pass
        trainer = None
        n_st = len(full_stages)
        # each stage runs full epoch(s); total wall ≈ n_st × single-epoch time
        stage_epochs = float(cfg.get("epochs") or 1.0)
        log(
            f"Running {n_st} full stages × {stage_epochs} epoch(s) each "
            f"(model becomes fully trained across stages)."
        )
        seq = max_seq
        for st in full_stages:
            prefs = st.get("prefixes") or []
            is_last = st["index"] == n_st
            tm = _set_stage_trainable(
                model,
                active_prefixes=prefs,
                # lm_head only on last stage (cheaper + stable)
                train_embed=False,
                train_lm_head=is_last,
            )
            log(
                f"===== STAGE {st['index']}/{n_st} · trainable={tm:.1f}M · "
                f"seq={seq} · {st.get('label','')} ====="
            )
            free_m, total_m = (
                torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
            )
            if total_m:
                log(f"GPU free before stage: {free_m/1e9:.2f}/{total_m/1e9:.2f} GiB")
            try:
                _run_one_epoch_stage(seq, stage_epochs)
                log(f"Stage {st['index']}/{n_st} hotovo.")
            except torch.cuda.OutOfMemoryError as e:
                log(f"Stage {st['index']} OOM at seq={seq}: {e}")
                recovered = False
                for seq2, tgt in ((96, 800.0), (64, 550.0)):
                    # split this stage's blocks further if many
                    if len(prefs) > 1:
                        half = max(1, len(prefs) // 2)
                        sub_groups = [prefs[:half], prefs[half:]] if half < len(prefs) else [prefs]
                    else:
                        sub_groups = [prefs]
                    for j, sub in enumerate(sub_groups):
                        tm2 = _set_stage_trainable(
                            model,
                            active_prefixes=sub,
                            train_embed=False,
                            train_lm_head=False,
                        )
                        if tm2 > tgt and len(sub) > 1:
                            # take only last block of subgroup
                            sub = sub[-1:]
                            tm2 = _set_stage_trainable(
                                model, active_prefixes=sub, train_embed=False, train_lm_head=False
                            )
                        log(f"  OOM split: sub {j+1}/{len(sub_groups)} trainable={tm2:.1f}M seq={seq2}")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        try:
                            _run_one_epoch_stage(seq2, stage_epochs)
                            recovered = True
                        except torch.cuda.OutOfMemoryError as e2:
                            log(f"  sub still OOM: {e2}")
                            recovered = False
                            break
                    if recovered:
                        seq = seq2
                        break
                if not recovered:
                    log("Staged full FT failed — switching to 4-bit QLoRA fallback…")
                    # free bf16 model
                    try:
                        del model
                    except Exception:
                        pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _try_4bit_qlora_fallback(cfg, model_id, max_seq, ds, tokenizer, out)
                    used_qlora_fallback = True
                    break
        if not used_qlora_fallback:
            log(f"All {n_st} full stages completed — every block group was dense-trained.")
    else:
        # ── Single-pass path (≤7B or non-full) ──
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            log(f"OOM at seq={max_seq} trainable={_trainable_m(model):.1f}M: {e}")
            if method != "full":
                raise
            for keep, seq, target in (
                (6, 128, 1200.0),
                (4, 128, 900.0),
                (3, 96, 700.0),
                (2, 64, 500.0),
            ):
                log(f"OOM recovery: keep_last={keep} blocks, seq={seq}, target≤{target}M …")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    _release_trainer(trainer)
                except Exception:
                    pass
                trainer = None
                more = _freeze_named_if_over(
                    model,
                    keys=("embed_tokens", "lm_head", "embed_tokens_per_layer"),
                    max_trainable_m=target,
                )
                if more:
                    log(f"  froze extra embed {more/1e6:.1f}M")
                cap = _cap_trainable_for_24gb(model, target_m=target)
                log(
                    f"  keep_last={cap.get('keep_last')} "
                    f"trainable={cap.get('final_trainable_m', _trainable_m(model)):.1f}M"
                )
                max_seq = seq
                try:
                    trainer, tokenized = _rebuild_trainer(seq)
                    trainer.train()
                    log(f"OOM recovery OK with keep_last={cap.get('keep_last')} seq={seq}")
                    break
                except torch.cuda.OutOfMemoryError as e2:
                    log(f"  still OOM: {e2}")
                    continue
            else:
                log("Single-pass full failed — 4-bit QLoRA fallback…")
                try:
                    del model
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _try_4bit_qlora_fallback(cfg, model_id, max_seq, ds, tokenizer, out)
                used_qlora_fallback = True

    if used_qlora_fallback:
        log("PEFT training finished (via 4-bit QLoRA fallback).")
        return

    log("Saving…")
    # CRITICAL: do NOT model.cpu() here.
    # Gemma-4-E2B full ≈10 GiB weights. Host has ~15 GiB RAM and often no swap.
    # model.cpu() + double save_pretrained is what freezes the whole VPS after 100%.
    _release_trainer(trainer)
    trainer = None
    # Drop large CPU-side dataset maps before serialize
    try:
        del tokenized
    except Exception:
        pass
    try:
        del ds
        del raw
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_mem("pre-final-save")
    if _mem_available_gib() < 2.5:
        log(
            "WARNING: host MemAvailable < 2.5 GiB before save — "
            "add swap on the host (scripts/gce_l4_fix.sh) to avoid freeze."
        )

    # Save PEFT adapter (or full weights) first — always keep adapter for re-merge
    if method != "full":
        _safe_save_pretrained(model, out, tokenizer, label="adapter")
    else:
        _safe_save_pretrained(model, out, tokenizer, label="weights")

    merged_dir = cfg.get("merged_dir", "/workspace/merged")
    os.makedirs(merged_dir, exist_ok=True)
    if method != "full":
        ok = _merge_qlora_to_fp16_for_gguf(
            model, tokenizer, merged_dir, target_context=int(cfg.get("target_context") or 0)
        )
        if not ok:
            log(
                "WARNING: merged/ not GGUF-ready. Adapter is in adapter/. "
                "Pipeline may re-merge from adapter+base before GGUF."
            )
    else:
        log("Full FT: linking weights → merged/ (skip second serialize)…")
        _link_or_copy_tree(out, merged_dir)
        _strip_quantization_config(merged_dir)
    log("PEFT training finished.")


def _stack_selfcheck() -> None:
    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu0={torch.cuda.get_device_name(0)}")
    try:
        import torchvision
        log(f"torchvision={torchvision.__version__}")
    except Exception as e:
        log(f"torchvision import warning: {e}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: train_inside_container.py <config.json>", file=sys.stderr)
        return 2
    cfg = load_config(sys.argv[1])
    log(f"Config: {json.dumps(cfg, indent=2)}")
    # Prefer PEFT (stable). Unsloth only if explicitly requested and importable.
    prefer = (cfg.get("framework") or "peft").lower()
    try:
        _stack_selfcheck()
        if prefer in ("unsloth",):
            try:
                import unsloth  # noqa: F401
                log("Using Unsloth backend")
                train_with_unsloth(cfg)
                return 0
            except Exception as e:
                log(f"Unsloth failed ({type(e).__name__}: {e}) — PEFT backend")
        log("Using PEFT/transformers backend (full or LoRA/QLoRA)")
        train_with_peft_fallback(cfg)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
