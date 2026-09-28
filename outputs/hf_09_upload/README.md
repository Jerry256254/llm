---
license: apache-2.0
library: gguf
pipeline_tag: text-generation
language:
- cs
- en
tags:
- qwen3_5
- qwen
- qwen3.5
- czech
- stem
- physics
- chemistry
- biology
- mathematics
- programming
- lora
- distillation
- distilled
- gguf
- conversational
base_model: Qwen/Qwen3.5-9B
---

# KucLab Hertz 0.9

A Czech/English STEM + programming assistant built by [KucLab](https://kuclab.org) on top of **Qwen/Qwen3.5-9B**. Tuned 0.8: same API-teacher data, surgically cleaned (absurd premises and identity leaks removed, depth kept).

## What this is

- **Base:** Qwen/Qwen3.5-9B (~9B params, Apache 2.0)
- **Method:** QLoRA, r=16 / alpha=32, merged to bf16 then quantized
- **Training data:** 4700 rows — tuned 0.8 corpus (182 absurd/poison rows removed, 0 duplicates), 309 terminology rows, 28 answer-first examples, identity rows (Hertz 0.9)
- **Context:** 32768 tokens (`num_ctx`)
- **Format:** GGUF q4_k_m (~5.3GB)

## Quickstart (Ollama)

```bash
curl -O https://huggingface.co/KucLab/kuclab-hertz-0.9/resolve/main/Modelfile
ollama create kuclab-hertz-0.9 -f Modelfile
ollama run kuclab-hertz-0.9
```

## Benchmarks

Same harness throughout (240 MMLU-Pro STEM + 206 CZ terminology questions).

**MMLU-Pro STEM**

| | Hertz 0.7F | Hertz 0.8 | **Hertz 0.9** |
|---|---|---|---|
| Biology | 83.3% | 86.7% | **85.0%** |
| Chemistry | 78.3% | 83.3% | **83.3%** |
| Math | 96.7% | 95.0% | **93.3%** |
| Physics | 75.0% | 90.0% | **85.0%** |
| **Total** | **83.3%** | **88.8%** | **86.7%** |

**Czech terminology**

| | Hertz 0.7F | Hertz 0.8 | **Hertz 0.9** |
|---|---|---|---|
| CS→EN | 88.3% | 87.4% | **86.4%** |
| EN→CS | 82.5% | 81.6% | **87.4%** |
| **Total** | **85.4%** | **84.5%** | **86.9%** |

## Honest status

- ✅ Best Czech of the line (86.9%, EN→CS 87.4%)
- ⚠️ STEM 86.7% below 0.8's 88.8% — cleaning cost some physics depth
- ✅ KucLab Hertz 0.9 identity, no founder named

## License

Apache 2.0, inherited from Qwen/Qwen3.5-9B.

## Credits

- Base: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- Teachers: DeepSeek-flash + Gemini 3.8 Flash via API; [KucLab](https://kuclab.org)
