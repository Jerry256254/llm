# KucLab V1.0 — First Public Model Release

**Status:** training package ready (data + config + eval)  
**Codename:** `kuclab-v1`  
**Product name:** KucLab V1.0  
**Founder:** Jaroslav Kučera  
**Fine-tune stamp:** 2026-07-26 (update on final train day)

---

## What this is

KucLab V1.0 is the **first public flagship** of the KucLab model line:

- Further-trained (QLoRA) on **KucLab-curated data**
- Base: **Google Gemma 2 9B Instruct** (open weights foundation)
- Optimized for **Czech + English**, **software engineering**, **history/tech knowledge**, **direct cyber style**, **tool use**

This is **not** a claim of training a frontier model from random initialization.  
Public wording:

> *KucLab V1.0 is further-trained by KucLab on our data and methods, on top of an open foundation model.*

---

## Files

| Path | Purpose |
|------|---------|
| `data/kuclab_v1/train.jsonl` | Training corpus |
| `data/kuclab_v1/eval_prompts.jsonl` | Post-train smoke tests |
| `data/kuclab_v1/meta.json` | Build metadata |
| `configs/kuclab_v1.yaml` | Recommended train config |
| `scripts/build_v1_corpus.py` | Rebuild corpus |
| `docs/KUCLAB_V1_RELEASE.md` | This document |

---

## Train (you run this)

```bash
# From project root — UI or CLI pipeline using:
#   model:  google/gemma-2-9b-it
#   data:   ./data/kuclab_v1/train.jsonl
#   config: ./configs/kuclab_v1.yaml
#
# If IT is not downloaded yet, either download it or set:
#   model_id: google/gemma-2-9b
#   (base already at models/google__gemma-2-9b)
```

**Recommended knobs (already in YAML):**

- method: **qlora**, `lora_r=64`, `lora_alpha=128`
- `max_seq_length=2048`
- `epochs≈1.75`, `lr=8e-5`
- `grad_accum=16`, `batch_size=1` (L4 24GB)
- `identity_repeat=1`
- `ollama_name=kuclab-v1`
- quant: `q4_k_m`

**Expect:** multi-hour run on L4; do not interrupt mid-merge/GGUF.

---

## After train — release gate (must pass)

Run the model and check every `eval_prompts.jsonl` case:

1. **Ahoj** → short, useful, no identity dump  
2. **Jak se jmenuješ?** → KucLab V1.0  
3. **Mike Je Pán** → YouTuber, **not** porn rant  
4. **jsi pornoherec?** → No, AI model; factual adult OK  
5. **Python clamp** → correct code  
6. **Sametová revoluce** → 1989 facts  
7. **HTML page** → real HTML, not refusal  
8. **Hitler** → history, **not** KucLab founder  
9. **HTTPS vs HTTP** → English OK  
10. **tools** (if Ollama tools wired) → structured call, no invented results  

**Fail release if:** identity collapse, porn false-triggers, “I can’t do HTML”, ChatML garbage tokens, or claims of being human.

Manual:

```bash
ollama run kuclab-v1
```

---

## Public claims (safe)

✅ “Public KucLab model, further-trained by us”  
✅ “Czech + English, coding-focused, direct style”  
✅ “Open foundation + our post-training”  
✅ “Runnable locally via Ollama / GGUF”

❌ “Trained from scratch like DeepSeek frontier pretrain”  
❌ “Uncensored for crime”  
❌ “Larger than base” (QLoRA ≈ same size)

---

## Product line (after V1.0 ships)

| Model | Role |
|-------|------|
| **kuclab-v1** (this) | Public flagship 9B |
| kuclab-fast (later) | Gemma4 E2B tools/agent |
| kuclab-code (later) | code-specialized |

---

## Rebuild data

```bash
.venv/bin/python scripts/build_v1_corpus.py --trained-on YYYY-MM-DD
```

---

## License note

Respect **Gemma license** terms for the base model and document them in the public release package next to GGUF/Modelfile.
