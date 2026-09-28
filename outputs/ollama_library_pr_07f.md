# Ollama library submission for kuclab-hertz-0.7f
# PR target: https://github.com/ollama/ollama  ->  new file:  library/kuclab-hertz-0.7f
# Paste the Modelfile block below as the file content, then open the PR with
# title "Add kuclab-hertz-0.7f" and a short description (template below).
# NOTE: Ollama team curates library/ manually — acceptance can take days/weeks.

# ==================== FILE: library/kuclab-hertz-0.7f ====================
FROM hf.co/KucLab/kuclab-hertz-0.7f:Q4_K_M

SYSTEM """Jsi KucLab Hertz 0.7F — model od KucLab (kuclab.org) zaměřený na fyziku, chemii, biologii, matematiku a programování. Mluvíš plynule česky i anglicky včetně odborné terminologie; odpovídáš v jazyce, kterým se ptá uživatel. U výpočtů, odvození a kódu nejdřív řekni výsledek nebo závěr, pak ukaž postup krok za krokem s mezivýsledky. U otázek s výběrem odpovědi vždy jasně uveď zvolenou možnost. Odpovídáš přímo, sebejistě a lidsky. Nezačínej omluvami ani formulacemi typu "jako AI nemohu" u běžných, neškodných témat. Skutečně nebezpečné nebo škodlivé požadavky odmítni stručně a bez kázání. K aktuálním datům nemáš přístup. Identitu řekni jen když se na ni někdo zeptá — a řekni, že tě vytvořil KucLab, nejmenuj žádnou konkrétní osobu."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 32768
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
# ================== END FILE: library/kuclab-hertz-0.7f ==================

# ==================== PR description template (EN) ====================
# Title: Add kuclab-hertz-0.7f
#
# KucLab Hertz 0.7F — Czech/English STEM assistant (Qwen3.5-9B base, QLoRA
# distilled from a 30B reasoning teacher). Measured on our harness (same
# 240 MMLU-Pro STEM + 206 CZ terminology questions for all rows):
# MMLU-Pro STEM 83.3% (base Qwen3.5-9B class, prior best 79.2%),
# CZ terminology 85.4% (EN->CS 82.5%).
# Weights: https://huggingface.co/KucLab/kuclab-hertz-0.7f (Q4_K_M GGUF, Apache-2.0)
# Registry: https://ollama.com/KucLab/hertz-0.7f
# Technical note for reviewers: Qwen3.5 ships an MTP head that makes stock
# converters emit a phantom blk.32 (block_count 33); this build sets
# block_count to the 32 real blocks and drops the orphan MTP tensors.
# ================== END PR template ==================
