#!/usr/bin/env python3
"""
Czech scientific terminology benchmark.

The base Qwen3.5-9B scores ~89% on MMLU-Pro STEM — it knows the science. What it
does NOT reliably know is Czech scientific vocabulary: it rendered "standard
deviation" as "směrrovný odchylna" (a non-word) instead of "směrodatná odchylka".

That gap is what KucLab Hertz is actually for, so it needs its own number.

Both directions are tested:
  EN -> CS   the failure mode we observed (model invents Czech words)
  CS -> EN   sanity check in the easy direction

Grading is exact-ish substring matching after diacritic-insensitive
normalisation, so it is automatic and reproducible.

Usage:
  .venv/bin/python scripts/bench_czech_terms.py --model qwen3.5:9b
  .venv/bin/python scripts/bench_czech_terms.py --compare qwen3.5:9b kuclab-hertz-0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT_DIR = ROOT / "outputs" / "bench"


def norm(s: str) -> str:
    """Lowercase, strip diacritics — so 'smerodatna' matches 'směrodatná'."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def load_terms() -> list[tuple[str, str, str, str]]:
    """HELD-OUT terms only — these never enter the training data.

    Testing on terms we trained on would measure recall of a word list, not
    whether the model actually learned Czech scientific language.
    """
    from cz_terms import eval_terms
    return eval_terms()


def build_cases(terms) -> list[dict]:
    cases = []
    for cs, en, dom, _defn in terms:
        cases.append({
            "id": f"en2cs:{en}", "dir": "EN→CS", "domain": dom,
            "prompt": f"Jak se česky odborně řekne „{en}“? Odpověz jen tím českým termínem, nic víc.",
            "expect": cs,
        })
        cases.append({
            "id": f"cs2en:{cs}", "dir": "CS→EN", "domain": dom,
            "prompt": f"What is the English scientific term for the Czech „{cs}“? Answer with the term only.",
            "expect": en,
        })
    return cases


def run(model: str, workers: int, num_predict: int = 512) -> dict:
    from bench_mmlu_pro import generate

    cases = build_cases(load_terms())
    print(f"\n{'═'*70}\n{model} · {len(cases)} terminology questions\n{'═'*70}", flush=True)

    t0 = time.time()
    done = [0]

    def one(c: dict) -> dict:
        # 96 tokens was far too few. Qwen3.5 is a thinking model: with a system
        # prompt it opens with "Thinking Process: 1. Analyze the Request..." and
        # the budget ran out long before the actual term appeared. Every such row
        # scored as wrong, which made a working configuration look catastrophic
        # (15.5%). Grading is substring-based, so the term just has to appear
        # somewhere — give the model room to get there.
        out = generate(model, c["prompt"], num_predict)
        ok = norm(c["expect"]) in norm(out)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]}/{len(cases)}  {time.time()-t0:.0f}s", flush=True)
        return {**c, "got": out.strip()[:160], "ok": ok}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, cases))

    by_dir: dict[str, list[int]] = {}
    for r in results:
        d = by_dir.setdefault(r["dir"], [0, 0])
        d[0] += r["ok"]
        d[1] += 1

    total_ok = sum(v[0] for v in by_dir.values())
    total_n = sum(v[1] for v in by_dir.values())

    print(f"\n{model}")
    for d in sorted(by_dir):
        ok, n = by_dir[d]
        print(f"  {d:<8} {ok:>3}/{n:<3}  {100*ok/n:5.1f}%")
    print(f"  {'TOTAL':<8} {total_ok:>3}/{total_n:<3}  {100*total_ok/max(1,total_n):5.1f}%")

    wrong = [r for r in results if not r["ok"] and r["dir"] == "EN→CS"]
    if wrong:
        print(f"\n  Nejhorší selhání (EN→CS), prvních 12:")
        for r in wrong[:12]:
            print(f"    {r['expect']:<26} → {r['got'][:70]!r}")

    summary = {
        "model": model, "n": total_n, "correct": total_ok,
        "accuracy": total_ok / max(1, total_n),
        "per_direction": {k: {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]}
                          for k, v in sorted(by_dir.items())},
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"czterms_{model.replace(':', '_').replace('/', '_')}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {path}")
    return summary


def compare(models: list[str]) -> None:
    loaded = []
    for m in models:
        p = OUT_DIR / f"czterms_{m.replace(':', '_').replace('/', '_')}.json"
        if not p.is_file():
            print(f"! no results for {m} — run the benchmark first")
            return
        loaded.append(json.loads(p.read_text(encoding="utf-8")))
    w = max(len(m) for m in models) + 2
    dirs = sorted({d for s in loaded for d in s["per_direction"]})
    print("\n" + "═" * 70)
    print(f"{'':<10}" + "".join(f"{m:<{w}}" for m in models))
    print("═" * 70)
    for d in dirs:
        line = f"{d:<10}"
        for s in loaded:
            pd = s["per_direction"].get(d)
            line += f"{(f'{100*pd["accuracy"]:.1f}%' if pd else '—'):<{w}}"
        print(line)
    print("─" * 70)
    line = f"{'TOTAL':<10}"
    for s in loaded:
        line += f"{f'{100*s["accuracy"]:.1f}%':<{w}}"
    print(line)
    if len(loaded) == 2:
        d = 100 * (loaded[1]["accuracy"] - loaded[0]["accuracy"])
        print(f"\nRozdíl: {d:+.1f} pp")
    print("═" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description="Czech scientific terminology benchmark")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=512,
                    help="Token budget. Too low scores thinking-model preambles as wrong.")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "TUNED"))
    args = ap.parse_args()
    if args.compare:
        compare(list(args.compare))
        return 0
    run(args.model, args.workers, args.num_predict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
