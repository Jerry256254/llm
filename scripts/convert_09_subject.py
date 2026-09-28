#!/usr/bin/env python3
"""Convert user-supplied 0.9 subject files (rich schema) to train schema.
Input:  llm/0.9/<subject>.jsonl with question/answer/steps/choices/text
Output: data/distill_09/<subject>.jsonl with instruction/input/output.
MCQ outputs get a canonical trailing 'Answer: (X)' line.
Usage: .venv/bin/python scripts/convert_09_subject.py biology
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "distill_09"


def convert(subject: str) -> int:
    src = ROOT / "0.9" / f"{subject}.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        t = (r.get("text") or "").strip()
        if not q or not t:
            print(f"  SKIP empty: {r.get('id')}")
            continue
        if r.get("kind") == "multiple_choice":
            m = re.search(r"Answer:\s*\(?([A-D])\)?", t)
            if m and not re.search(r"Answer:\s*\([A-D]\)\s*$", t):
                t = t + f"\nAnswer: ({m.group(1)})"
        instr = q
        if r.get("choices"):
            ch = r["choices"]
            instr += " " + " ".join(f"({k}) {v}" for k, v in ch.items())
        out.append({"instruction": instr, "input": "", "output": t,
                    "source": f"pc09:{subject}"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{subject}.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{subject}: {len(out)}/{len(rows)} converted -> data/distill_09/{subject}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(convert(sys.argv[1]))
