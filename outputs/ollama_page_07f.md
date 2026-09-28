# Ollama page for KucLab/hertz-0.7f — paste into Edit on https://ollama.com/KucLab/hertz-0.7f

## Cover image
Upload file: `/home/admin/llm/outputs/bench_card_07f.png`
(Page → Edit/Settings → cover image → upload.)

## Description (paste as-is)

**KucLab Hertz 0.7F** — Czech/English STEM assistant (9B, Q4_K_M, Apache-2.0).

Distilled from a 30B reasoning teacher into Qwen3.5-9B. Measured on the same
240 MMLU-Pro STEM + 206 CZ terminology questions for every model:

- MMLU-Pro STEM: **83.3%** (chemistry 61.7% → 78.3%, math 90.0% → 96.7%)
- CZ terminology: **85.4%** (hard EN→CS direction: 65.0% → 82.5%)
- Standard GSM8K: **87.8%** · thinking tokens −47% vs base

```bash
ollama run KucLab/hertz-0.7f
```

Full model card, Modelfile and benchmarks:
https://huggingface.co/KucLab/kuclab-hertz-0.7f

Tip: append `"think": false` to API requests for instant short answers
without the reasoning trace.
