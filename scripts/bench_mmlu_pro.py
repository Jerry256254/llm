#!/usr/bin/env python3
"""
Benchmark any Ollama model on the held-out MMLU-Pro STEM set.

Used to establish the BASELINE (base Qwen3.5-9B) that KucLab Hertz must beat,
and afterwards to score the fine-tune on exactly the same questions.

Uses the Ollama HTTP API directly (not `ollama run`) so we can pin temperature
to 0, cap the output length and run a few requests concurrently — otherwise a
239-question sweep takes hours.

Results are written to outputs/bench/<model>.json so runs are comparable.

Usage:
  .venv/bin/python scripts/bench_mmlu_pro.py --model qwen3.5:9b
  .venv/bin/python scripts/bench_mmlu_pro.py --model kuclab-hertz-0.1 --limit 100
  .venv/bin/python scripts/bench_mmlu_pro.py --compare qwen3.5:9b kuclab-hertz-0.1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "kuclab_hertz_0.1" / "eval_mmlu_pro_stem.jsonl"
OUT_DIR = ROOT / "outputs" / "bench"
API = "http://127.0.0.1:11434/api/generate"


def _text_of(d: dict) -> str:
    """Return the model's text wherever Ollama put it.

    Thinking models split output across two fields and which one is populated
    depends on the Modelfile, not on the model's ability:

      base qwen3.5:9b  (RENDERER/PARSER qwen3.5) -> text lands in `thinking`
                                                    unless the request sets
                                                    "think": false
      our fine-tuned export (plain TEMPLATE)     -> text lands in `response`,
                                                    and "think": false makes it
                                                    return an empty string

    Reading only one field silently scores a working model as 0%. Both of those
    traps were hit during this project, in opposite directions. Take whatever is
    non-empty instead of assuming a shape.
    """
    resp = (d.get("content") or d.get("response") or "").strip()
    if not resp:
        resp = (d.get("thinking") or "").strip()
    # A merged/re-quantized export was observed to leak a raw <|im_stop|>
    # token as literal text instead of it being recognised as a stop marker —
    # strip it so it doesn't get mistaken for content or break letter parsing.
    for junk in ("<|im_stop|>", "<|im_end|>", "<|endoftext|>"):
        resp = resp.replace(junk, "").strip()
    return resp


CHAT_API = "http://127.0.0.1:11434/api/chat"


def generate(model: str, prompt: str, num_predict: int, timeout: int = 1200) -> str:
    """Ask the model via /api/chat (message-based), not /api/generate.

    /api/generate with a bare prompt string was unreliable on Qwen3.5-style
    RENDERER/PARSER exports: `ollama run` (which uses /api/chat under the hood)
    reliably answered "17×23?" correctly, while raw /api/generate calls to the
    identical model returned completely empty response AND thinking fields for
    the same question. Whatever templating /api/generate does with a bare
    string is not equivalent to the message-based path these models expect.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": num_predict,
            # 4096 was tried and is too tight in combination with num_predict:
            # a verified hard question hit done_reason=length at exactly
            # num_predict=2048 tokens of reasoning without ever reaching an
            # answer. Give real headroom for both the prompt and a long
            # reasoning trace.
            "num_ctx": 6144,
        },
    }).encode()
    req = urllib.request.Request(CHAT_API, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return _text_of(d.get("message") or {})
    except Exception as e:
        return f"<ERR {type(e).__name__}>"


# NOTE: no re.IGNORECASE on the captured group. With it, [A-J] also matches
# lowercase, so "The answer is D" captured the "i" of "is" and scored as (I).
# Models that used the requested "Answer: (X)" form were unaffected while models
# that wrote prose were penalised — which measured format compliance, not
# knowledge, and understated one model by ~70 points.
ANSWER_RE = re.compile(
    r"(?:[Aa]nswer|[Oo]dpov[ěe][ďd])\s*(?:is|je)?\s*[:\-–]?\s*\(?\s*([A-J])\b"
)
PAREN_RE = re.compile(r"\(([A-J])\)")


def extract_letter(text: str) -> str | None:
    """Only trust an explicit 'Answer: X', or a lone (X) at the very end.

    Scraping any '(E)' out of the reasoning text is worse than useless: on a
    truncated answer it returns whatever option the model happened to discuss,
    which silently turns 'ran out of tokens' into 'got it wrong'. Return None
    instead so the caller can re-ask.
    """
    hits = ANSWER_RE.findall(text)
    if hits:
        return hits[-1].upper()
    tail = text.strip()[-120:]
    hits = PAREN_RE.findall(tail)
    if hits:
        return hits[-1].upper()
    m = re.search(r"\b([A-J])\b[.\s]*$", tail)
    return m.group(1).upper() if m else None


def selftest_extractor() -> int:
    """Guard the grader itself.

    Every measurement bug in this project came from trusting the harness without
    checking it: a 96-token cap that scored thinking preambles as wrong, reading
    only `response` when the text was in `thinking`, and an IGNORECASE class that
    read "the answer is D" as (I). Run this before believing any score.
    """
    cases = [
        ("The answer is D.", "D"),
        ("Answer: (E)", "E"),
        ("Answer: C", "C"),
        ("... so the answer is (B).", "B"),
        ("The answer to your question is A", "A"),
        ("Odpověď: F", "F"),
        ("blah blah (G) blah\n\nAnswer: (H)", "H"),
        ("I think option (J) is right. Answer: J", "J"),
        ("Reasoning about (A) and (B)... The answer is C.", "C"),
        ("no letter here at all", None),
    ]
    bad = 0
    for text, want in cases:
        got = extract_letter(text)
        if got != want:
            bad += 1
            print(f"  FAIL want={want} got={got} | {text[:60]!r}")
    print(f"extractor self-test: {len(cases)-bad}/{len(cases)} passed")
    return bad


def build_prompt(r: dict, answer_only: bool = False) -> str:
    opts = "\n".join(f"({chr(65+j)}) {o}" for j, o in enumerate(r["options"]))
    if answer_only:
        return (
            f"{r['question']}\n\n{opts}\n\n"
            f"Reply with ONLY the letter of the correct option, in the form: Answer: X"
        )
    return (
        f"Answer this multiple-choice question.\n\n"
        f"{r['question']}\n\n{opts}\n\n"
        f"Reason step by step, but keep it under 200 words. "
        f"Then finish with exactly this line and nothing after it:\n"
        f"Answer: (X)\n"
        f"where X is the letter of the correct option."
    )


def load_rows(limit: int | None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not limit:
        return rows
    # MMLU-Pro's own ordering is not randomised (source-grouped), so taking the
    # first N per category is a biased sample — it can systematically pick an
    # easier or harder cluster and make --limit runs incomparable to a full run.
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(seed)
    for v in by_cat.values():
        rng.shuffle(v)
    per = max(1, limit // max(1, len(by_cat)))
    return [r for rs in by_cat.values() for r in rs[:per]]


def run(model: str, limit: int | None, workers: int, num_predict: int) -> dict:
    rows = load_rows(limit)
    print(f"\n{'═'*70}\n{model} · {len(rows)} questions · {workers} workers\n{'═'*70}", flush=True)

    t0 = time.time()
    done = [0]

    def one(r: dict) -> dict:
        out = generate(model, build_prompt(r), num_predict)
        got = extract_letter(out)
        retried = False
        if got is None:
            # Ran out of budget before committing to a letter. Re-ask for the
            # letter alone rather than guessing from the truncated reasoning.
            #
            # 24 tokens was nowhere near enough: this is a thinking model that
            # opens a new reasoning chain even for "reply with ONLY the letter"
            # ("The user wants to identify which molecule..."). Verified with a
            # hard MMLU-Pro question that it still hadn't produced a letter after
            # 60 tokens of preamble. Give it real room.
            retried = True
            out2 = generate(model, build_prompt(r, answer_only=True), 400)
            got = extract_letter(out2)
            out = out + "\n\n[RETRY] " + out2
        ok = got == (r.get("answer") or "").upper()
        done[0] += 1
        if done[0] % 10 == 0:
            el = time.time() - t0
            print(f"  {done[0]}/{len(rows)}  {el:.0f}s  ({el/done[0]:.1f}s/q)", flush=True)
        # Keep head AND tail — a head-only slice silently hid every retry
        # (appended after the primary output) once the primary output alone
        # exceeded the slice length, which is exactly the case for long
        # reasoning traces. That is how the retry failures went unnoticed.
        raw_saved = out if len(out) <= 1200 else out[:600] + " …[snip]… " + out[-600:]
        return {"id": r["id"], "category": r["category"], "want": r.get("answer"),
                "got": got, "ok": ok, "retried": retried, "raw": raw_saved}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, rows))

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unparsed = 0
    retried = 0
    for res in results:
        stats[res["category"]][0] += res["ok"]
        stats[res["category"]][1] += 1
        unparsed += res["got"] is None
        retried += res.get("retried", False)

    total_ok = sum(v[0] for v in stats.values())
    total_n = sum(v[1] for v in stats.values())
    elapsed = time.time() - t0

    print(f"\n{model}")
    for cat in sorted(stats):
        ok, n = stats[cat]
        print(f"  {cat:<10} {ok:>3}/{n:<3}  {100*ok/n:5.1f}%")
    print(f"  {'TOTAL':<10} {total_ok:>3}/{total_n:<3}  {100*total_ok/max(1,total_n):5.1f}%")
    if retried:
        print(f"  ({retried} needed an answer-only re-ask after running out of tokens)")
    if unparsed:
        print(f"  ! {unparsed} answers still unparseable (counted as wrong)")
    print(f"  {elapsed:.0f}s total, {elapsed/max(1,total_n):.1f}s/question")

    summary = {
        "model": model,
        "n": total_n,
        "correct": total_ok,
        "accuracy": total_ok / max(1, total_n),
        "unparsed": unparsed,
        "retried": retried,
        "per_category": {k: {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]}
                         for k, v in sorted(stats.items())},
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{model.replace(':', '_').replace('/', '_')}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {path}")
    return summary


def compare(models: list[str]) -> None:
    loaded = []
    for m in models:
        p = OUT_DIR / f"{m.replace(':', '_').replace('/', '_')}.json"
        if not p.is_file():
            print(f"! no results for {m} — run the benchmark first")
            return
        loaded.append(json.loads(p.read_text(encoding="utf-8")))

    cats = sorted({c for s in loaded for c in s["per_category"]})
    w = max(len(m) for m in models) + 2
    print("\n" + "═" * 70)
    print(f"{'':<12}" + "".join(f"{m:<{w}}" for m in models))
    print("═" * 70)
    for c in cats:
        line = f"{c:<12}"
        for s in loaded:
            pc = s["per_category"].get(c)
            line += f"{(f'{100*pc["accuracy"]:.1f}%' if pc else '—'):<{w}}"
        print(line)
    line = f"{'TOTAL':<12}"
    for s in loaded:
        line += f"{f'{100*s["accuracy"]:.1f}%':<{w}}"
    print("─" * 70)
    print(line)
    if len(loaded) == 2:
        d = 100 * (loaded[1]["accuracy"] - loaded[0]["accuracy"])
        verdict = "zlepšení" if d > 0 else ("beze změny" if d == 0 else "ZHORŠENÍ")
        print(f"\nRozdíl: {d:+.1f} pp — {verdict}")
    print("═" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description="MMLU-Pro STEM benchmark via Ollama")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=2048,
                    help="Token budget. Too low silently turns truncation into wrong answers.")
    ap.add_argument("--selftest", action="store_true",
                    help="check the answer extractor and exit")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "TUNED"),
                    help="print a side-by-side table from saved results")
    args = ap.parse_args()

    if args.selftest:
        return 1 if selftest_extractor() else 0
    if args.compare:
        compare(list(args.compare))
        return 0
    if selftest_extractor():
        print("Refusing to benchmark with a broken grader.")
        return 2
    if not EVAL.is_file():
        print(f"ERROR: missing {EVAL}")
        return 2
    run(args.model, args.limit or None, args.workers, args.num_predict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
