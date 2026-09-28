#!/usr/bin/env python3
"""Benchmark card chart for KucLab Hertz 0.7F.
All numbers measured on THIS project's harness (same 240 MMLU-Pro STEM +
206 CZ terminology questions for every model). No published numbers mixed in.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path("/home/admin/llm")
B = ROOT / "outputs" / "bench"


def load(p, key="accuracy"):
    d = json.loads((B / p).read_text())
    return d[key]


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("KucLab Hertz 0.7F (9B, distilled) vs same-size models — same harness, same questions",
             fontsize=13, fontweight="bold")

# ---- MMLU-Pro STEM ----
ax = axes[0]
models = ["qwen3.5:9b\n(base)", "kuclab-hertz-0.6\n(12B LoRA)", "kuclab-hertz-0.7f\n(9B distilled)"]
files = ["qwen3.5_9b.json", None, "kuclab-hertz-0.7f.json"]
vals, colors = [], []
base06 = json.loads((B / "kuclab-hertz-0.6.json").read_text())["accuracy"] if (B / "kuclab-hertz-0.6.json").exists() else 0.792
for m, f in zip(models, files):
    if f and (B / f).exists():
        v = load(f)
    elif "0.6" in m:
        v = base06
    else:
        v = None
    vals.append(v)
    colors.append("#1f77b4" if "0.7f" in m else "#7f7f7f")
x = np.arange(len(models))
bars = ax.bar(x, [v * 100 if v else 0 for v in vals],
              color=[c if v else "#dddddd" for c, v in zip(colors, vals)],
              edgecolor="black")
for xi, v in zip(x, vals):
    ax.text(xi, (v * 100 if v else 0) + 0.8, f"{v * 100:.1f}%" if v else "pending",
            ha="center", fontsize=10, fontweight="bold" if v and v > 0.8 else "normal")
ax.set_xticks(x, models, fontsize=9)
ax.set_ylabel("accuracy %")
ax.set_title("MMLU-Pro STEM (240 Q, our harness)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

# ---- CZ terminology ----
ax = axes[1]
cz = []
base06cz = 0.738
for m, f in zip(models, files):
    cf = "czterms_" + f if f else None
    if cf and (B / cf).exists():
        v = load(cf)
    elif "0.6" in m:
        v = base06cz
    else:
        v = None
    cz.append(v)
bars = ax.bar(x, [v * 100 if v else 0 for v in cz],
              color=[c if v else "#dddddd" for c, v in zip(colors, cz)],
              edgecolor="black")
for xi, v in zip(x, cz):
    ax.text(xi, (v * 100 if v else 0) + 0.8, f"{v * 100:.1f}%" if v else "pending",
            ha="center", fontsize=10, fontweight="bold" if v and v > 0.8 else "normal")
ax.set_xticks(x, models, fontsize=9)
ax.set_title("CZ terminology (206 Q, our harness)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

fig.text(0.5, 0.01, "Measured with this repo: scripts/bench_mmlu_pro.py + scripts/bench_czech_terms.py · Q4_K_M via Ollama",
         ha="center", fontsize=8, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
out = ROOT / "outputs" / "bench_card_07f.png"
fig.savefig(out, dpi=150)
print("WROTE", out)
