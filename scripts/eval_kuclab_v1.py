#!/usr/bin/env python3
"""
Smoke-test KucLab V1.0 via Ollama after training.

Usage:
  .venv/bin/python scripts/eval_kuclab_v1.py
  .venv/bin/python scripts/eval_kuclab_v1.py --model kuclab-v1 --eval data/kuclab_v1/eval_prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ollama_generate(model: str, prompt: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout or "").strip()
    if not out and proc.stderr:
        out = proc.stderr.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kuclab-v1")
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "kuclab_v1" / "eval_prompts.jsonl")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    if not args.eval.is_file():
        print(f"Missing eval file: {args.eval}", file=sys.stderr)
        return 2

    cases = []
    with args.eval.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"Evaluating {args.model} on {len(cases)} cases…\n")
    passed = 0
    failed = 0
    for c in cases:
        cid = c.get("id", "?")
        prompt = c["prompt"]
        print(f"── {cid}: {prompt!r}")
        try:
            resp = ollama_generate(args.model, prompt, timeout=args.timeout)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            continue
        preview = resp.replace("\n", " ")[:220]
        print(f"  → {preview}")

        ok = True
        expect = c.get("expect_contains_any") or []
        if expect and not any(x.lower() in resp.lower() for x in expect):
            print(f"  FAIL: expected one of {expect}")
            ok = False
        for bad in c.get("fail_if") or []:
            if bad and bad.lower() in resp.lower():
                print(f"  FAIL: forbidden phrase {bad!r}")
                ok = False
        if ok:
            print("  PASS")
            passed += 1
        else:
            failed += 1
        print()

    print(f"Result: {passed} pass / {failed} fail / {len(cases)} total")
    if failed:
        print("RELEASE GATE: NOT READY")
        return 1
    print("RELEASE GATE: OK — good to announce public V1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
