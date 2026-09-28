#!/usr/bin/env python3
"""KucLab Hertz 0.9 — dataset assembly.
0.9 = 14 user-built subjects (data/distill_09/<subject>.jsonl, pc09:*)
     + full 0.8f corpus (data/kuclab_hertz_0.8f/train.jsonl, keeps CZ/concise/terms base)
     + 309 terms:en2cs_defined + 28 answer_format + 15 identity (Hertz 0.9).
USAGE: .venv/bin/python scripts/build_09_dataset.py build [--trained-on YYYY-MM-DD]
"""
from __future__ import annotations
import argparse, json, random, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "data" / "kuclab_hertz_0.9"
NAME = "KucLab Hertz 0.9"
SUBJECTS = ["biology", "business", "chemistry", "computer_science", "economics",
            "engineering", "health", "history", "law", "math", "other",
            "philosophy", "physics", "psychology"]
PREV = ROOT / "data" / "kuclab_hertz_0.8f" / "train.jsonl"

from build_hertz07_fixed import ANSWER_FORMAT_ROWS, _identity_rows  # noqa: E402
from cz_terms import train_terms  # noqa: E402

MIN_ANSWER_CHARS = 10
IDENTITY_FIX = re.compile(r"\b(Qwen[\d.]*|Alibaba|Tongyi|Alibaba Cloud|Gemma|Google DeepMind|OpenAI|ChatGPT|Claude|Anthropic)\b", re.I)


def build(trained_on: str) -> int:
    rows, seen = [], set()
    dropped = {"malformed": 0, "short": 0, "leak": 0, "dup": 0}
    srcs = {}

    def add(instr, out, source):
        nonlocal_rows = True
        instr, out = (instr or "").strip(), (out or "").strip()
        if not instr or not out:
            dropped["malformed"] += 1; return
        if len(out) < MIN_ANSWER_CHARS:
            dropped["short"] += 1; return
        if IDENTITY_FIX.search(out):
            dropped["leak"] += 1; return
        if instr in seen:
            dropped["dup"] += 1; return
        seen.add(instr)
        rows.append({"instruction": instr, "input": "", "output": out, "source": source})
        srcs[source] = srcs.get(source, 0) + 1

    for subj in SUBJECTS:
        p = ROOT / "data" / "distill_09" / f"{subj}.jsonl"
        if not p.is_file():
            print(f"  MISSING {p}, skipping")
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                dropped["malformed"] += 1; continue
            add(r.get("instruction"), r.get("output"), f"pc09:{subj}")
    print(f"  user subjects: {sum(v for k, v in srcs.items() if k.startswith('pc09'))}")

    if PREV.is_file():
        for line in PREV.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            add(r.get("instruction"), r.get("output"), "reused:0.8f")
    print(f"  +0.8f reuse: {srcs.get('reused:0.8f', 0)}")

    for cs, en, dom, defn in train_terms():
        add(f"Jak se česky odborně řekne „{en}“ a co to znamená?",
            f"{cs} — {defn}", "terms:en2cs_defined")
    for instr, out in ANSWER_FORMAT_ROWS:
        add(instr, out, "answer_format")
    for r in _identity_rows(NAME, trained_on):
        if r["instruction"] in seen:
            continue
        seen.add(r["instruction"])
        rows.append(r)
        srcs["identity"] = srcs.get("identity", 0) + 1

    cs = sum(1 for r in rows if re.search(r"[ěščřžýáíéúů]", r["instruction"] + r["output"]))
    print(f"  CORPUS: {len(rows)} rows, cs-share {cs/max(1,len(rows)):.0%}")
    random.Random(42).shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"instruction": r["instruction"], "input": "",
                                "output": r["output"]}, ensure_ascii=False) + "\n")
    meta = {"name": "kuclab_hertz_0.9", "identity_name": NAME, "trained_on": trained_on,
            "rows": len(rows), "cs_share": round(cs / max(1, len(rows)), 4),
            "sources": dict(sorted(srcs.items(), key=lambda kv: -kv[1])),
            "dropped": dropped, "generated_at": datetime.now(timezone.utc).isoformat()}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--trained-on", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    a = ap.parse_args()
    return build(a.trained_on) if a.cmd == "build" else 1


if __name__ == "__main__":
    raise SystemExit(main())
