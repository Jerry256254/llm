#!/usr/bin/env python3
"""
Post-train evaluation for KucLab Hertz 0.1.

Two passes:
  1. Smoke tests  — data/kuclab_hertz_0.1/eval_prompts.jsonl
     Checks identity, Czech/English switching, honesty and basic STEM.
  2. MMLU-Pro STEM — data/kuclab_hertz_0.1/eval_mmlu_pro_stem.jsonl
     Held-out benchmark. It is deliberately NOT in the training data, so the
     score means something. Multiple choice, answer letter is extracted.

Usage:
  .venv/bin/python scripts/eval_hertz01.py
  .venv/bin/python scripts/eval_hertz01.py --model kuclab-hertz-0.1 --skip-mmlu
  .venv/bin/python scripts/eval_hertz01.py --mmlu-limit 40
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "kuclab_hertz_0.1"


def ollama(model: str, prompt: str, timeout: int = 240) -> str:
    try:
        r = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout or "").strip() or (r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return "<TIMEOUT>"
    except FileNotFoundError:
        print("ERROR: `ollama` not found in PATH", file=sys.stderr)
        raise SystemExit(2)


def run_smoke(model: str) -> tuple[int, int]:
    path = DATA / "eval_prompts.jsonl"
    if not path.is_file():
        print(f"! missing {path} — run scripts/build_hertz01_corpus.py first")
        return 0, 0

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    passed = 0
    print("═" * 72)
    print("SMOKE TESTS")
    print("═" * 72)
    for r in rows:
        out = ollama(model, r["prompt"])
        low = out.lower()
        want = r.get("expect_contains_any") or []
        bad = r.get("fail_if") or []
        ok = (not want or any(w.lower() in low for w in want)) and \
             not any(b.lower() in low for b in bad)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {r['id']:<14} {r['prompt'][:52]}")
        if not ok:
            print(f"        want any: {want}")
            if bad:
                print(f"        must not: {bad}")
            print(f"        got: {out[:260]}")
    print(f"\nSmoke: {passed}/{len(rows)}")
    return passed, len(rows)


ANS_RE = re.compile(r"\b(?:answer|odpověď)\b[^A-Z]{0,20}\(?([A-J])\)?", re.IGNORECASE)
LETTER_RE = re.compile(r"\(([A-J])\)|(?:^|\s)([A-J])[).:]\s")


def extract_letter(text: str) -> str | None:
    m = ANS_RE.search(text)
    if m:
        return m.group(1).upper()
    # Fall back to the last standalone option letter mentioned
    hits = [(a or b) for a, b in LETTER_RE.findall(text)]
    return hits[-1].upper() if hits else None


def run_mmlu(model: str, limit: int | None) -> tuple[int, int]:
    path = DATA / "eval_mmlu_pro_stem.jsonl"
    if not path.is_file():
        print(f"! missing {path} — skipping benchmark")
        return 0, 0

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        # Keep the category balance when limiting
        by_cat: dict[str, list] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        per = max(1, limit // max(1, len(by_cat)))
        rows = [r for rs in by_cat.values() for r in rs[:per]]

    print("\n" + "═" * 72)
    print(f"MMLU-Pro STEM (held out — not in training data) · {len(rows)} questions")
    print("═" * 72)

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows, 1):
        opts = "\n".join(f"({chr(65+j)}) {o}" for j, o in enumerate(r["options"]))
        prompt = (
            f"{r['question']}\n\n{opts}\n\n"
            "Think briefly, then end with exactly: 'Answer: (X)' where X is the option letter."
        )
        out = ollama(model, prompt)
        got = extract_letter(out)
        ok = got == (r.get("answer") or "").upper()
        cat = r["category"]
        stats[cat][0] += ok
        stats[cat][1] += 1
        print(f"[{i:>3}/{len(rows)}] {cat:<10} want={r.get('answer')} got={got or '?'} "
              f"{'✓' if ok else '✗'}")

    total_ok = sum(v[0] for v in stats.values())
    total_n = sum(v[1] for v in stats.values())
    print("\nPer category:")
    for cat in sorted(stats):
        ok, n = stats[cat]
        print(f"  {cat:<10} {ok:>3}/{n:<3}  {100*ok/n:5.1f}%")
    print(f"\nMMLU-Pro STEM total: {total_ok}/{total_n} = {100*total_ok/max(1,total_n):.1f}%")
    return total_ok, total_n


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate KucLab Hertz 0.1")
    ap.add_argument("--model", default="kuclab-hertz-0.1", help="Ollama model tag")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-mmlu", action="store_true")
    ap.add_argument("--mmlu-limit", type=int, default=40,
                    help="Questions to run (0 = all). MMLU is slow — 40 is a decent signal.")
    args = ap.parse_args()

    print(f"Model: {args.model}\n")
    s_ok = s_n = m_ok = m_n = 0
    if not args.skip_smoke:
        s_ok, s_n = run_smoke(args.model)
    if not args.skip_mmlu:
        m_ok, m_n = run_mmlu(args.model, args.mmlu_limit or None)

    print("\n" + "═" * 72)
    if s_n:
        print(f"Smoke:     {s_ok}/{s_n} ({100*s_ok/s_n:.0f}%)")
    if m_n:
        print(f"MMLU-Pro:  {m_ok}/{m_n} ({100*m_ok/m_n:.1f}%)")
    print("═" * 72)
    # Non-zero exit if smoke tests regressed badly — useful in a pipeline
    return 1 if (s_n and s_ok < 0.7 * s_n) else 0


if __name__ == "__main__":
    raise SystemExit(main())
