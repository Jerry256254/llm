#!/usr/bin/env python3
"""
KucLab Hertz 0.8 — dataset assembly (Gemma-4-12B base, target +10% over 0.6).

WHY 0.7 FIX WASN'T ENOUGH (measured 2026-09-07):
  MMLU TBD, but CZ terminology 70.9% < 0.6's 73.8% (EN->CS 65.0% -> 59.2%).
  Root cause, verified on the corpora:
    - train_0.6 was 93% Czech (diacritics); train_0.7fix only 83%
    - conceptual_hard (901 rows) is 53% Czech / 43% English questions
      vs fresh_v06 at 93% Czech — the EN->CS exact-term generation lost
      Czech lexical signal while English reasoning got stronger
      (CS->EN held at 82.5%, EN->CS collapsed)
    - train_loss 1.222 (0.7fix) vs 0.937 (0.6): harder/longer rows ate
      capacity; terminology rows were only 8% of the corpus (was 10.5%)

0.8 FIXES (no new term PAIRS — same 309 rows; the fix is Czech PROSE):
  - conceptual_hard replaced by its CZECH version (all 901 rows in Czech,
    translated; translation source recorded in meta for audit)
  - NEW agent-generated Czech chemistry/physics rows (350 chem + 250 phys
    + 100 math, per data/manual_additions/PROMPT_hertz08_agent.md):
    prose with calculations, not lookups — morphology, not recall
  - corpus Czech share target >= 90%
  - hyperparams UNCHANGED (r16/32, lr 3e-5, epochs 2.0 strict) — data is
    the lever, proven by 0.4/0.5/0.6 history

USAGE:
  .venv/bin/python scripts/build_hertz08_dataset.py build \\
      --conceptual data/manual_additions/hertz08_conceptual_cz.jsonl \\
      --fresh-cz data/manual_additions/hertz08_agent_cz.jsonl
"""
from __future__ import annotations
import argparse, json, random, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "data" / "kuclab_hertz_0.8"
NAME = "KucLab Hertz 0.8"

FRESH_V06 = Path("/home/admin/.claude/uploads/07e6144c-f286-4be0-bca2-8615d1b2531c/0642b371-fresh_v06.jsonl")
PREV_TRAIN = ROOT / "data" / "kuclab_hertz_0.6" / "train.jsonl"

MIN_ANSWER_CHARS = 10
IDENTITY_FIX = re.compile(r"\b(Qwen[\d.]*|Alibaba|Tongyi|Alibaba Cloud|Gemma|Google DeepMind|OpenAI|ChatGPT|Claude|Anthropic)\b", re.I)
IDENTITY_QUESTIONS = {
    "Jak se jmenuješ?", "Kdo jsi?", "What is your name?", "Who are you?",
    "představ se", "Kdo tě vytvořil?", "Kdo je KucLab?", "Who created you?",
    "Jaký má KucLab web?", "What is KucLab's website?", "V čem jsi dobrý?",
    "Co je KucLab Hertz?", "Kdy jsi byl dotrénován?", "Jsi cenzurovaný?",
    "Máš smysl pro humor?", "Kdo je zakladatel KucLab?",
}

# Czech-share guard: every bulk source must be >=85% Czech (diacritics),
# otherwise the 0.7 dilution repeats. Build ABORTS on violation.
CS_RE = re.compile(r"[ěščřžýáíéúů]")
MIN_CS_SHARE = 0.85


def _identity_rows(name: str, trained_on: str) -> list[dict]:
    def ex(instruction: str, answer: str) -> dict:
        return {"instruction": instruction, "input": "", "output": answer.strip(), "source": "identity"}
    return [
        ex("Jak se jmenuješ?", f"Jmenuju se {name}."),
        ex("Kdo jsi?", f"Jsem {name} — model od KucLab. Zaměřuju se na fyziku, chemii, biologii, matematiku a programování a mluvím česky i anglicky."),
        ex("What is your name?", f"My name is {name}."),
        ex("Who are you?", f"I'm {name}, a model from KucLab focused on physics, chemistry, biology, mathematics and programming. I work in Czech and English."),
        ex("představ se", f"{name} — model od KucLab. Specializace: fyzika, chemie, biologie, matematika, programování. Odpovídám v jazyce, kterým se ptáš."),
        ex("Kdo tě vytvořil?", "Vytvořil mě KucLab."),
        ex("Kdo je KucLab?", "KucLab je laboratoř/tým, který mě vyvinul a trénuje."),
        ex("Who created you?", "I was created by KucLab."),
        ex("Jaký má KucLab web?", "kuclab.org"),
        ex("What is KucLab's website?", "kuclab.org"),
        ex("V čem jsi dobrý?", "Fyzika, chemie, biologie, matematika a programování — výpočty, odvození, kód a vysvětlení pojmů krok za krokem, česky i anglicky."),
        ex("Co je KucLab Hertz?", f"{name} je model od KucLab (kuclab.org) zaměřený na fyziku, chemii, biologii, matematiku a programování."),
        ex("Kdy jsi byl dotrénován?", f"Poslední dotrénování: {trained_on}."),
        ex("Jsi cenzurovaný?", "Neschovávám se za zbytečné vatování a omluvy u běžných témat — ale skutečně škodlivé nebo nebezpečné požadavky odmítnu, stručně a bez kázání."),
        ex("Máš smysl pro humor?", "Ano, klidně i suchý nebo černý, pokud se to k tématu hodí — ale ne na úkor toho, aby odpověď byla správná."),
    ]


def terminology_rows() -> list[dict]:
    from cz_terms import train_terms
    rows = []
    for cs, en, dom, defn in train_terms():
        rows.append({"instruction": f"Jak se česky odborně řekne „{en}“ a co to znamená?", "input": "", "output": f"{cs} — {defn}", "source": "terms:en2cs_defined"})
    return rows


from build_hertz07_fixed import ANSWER_FORMAT_ROWS  # noqa: E402  (single source of truth)


def _load_bulk(path: Path, source: str, rows: list[dict], seen: set[str], dropped: dict,
               require_cs: bool = True) -> int:
    n = 0
    cs = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            r = json.loads(line)
        except Exception:
            dropped["malformed"] += 1
            continue
        instr = (r.get("instruction") or "").strip()
        out = (r.get("output") or "").strip()
        if not instr or not out:
            dropped["malformed"] += 1
            continue
        if len(out) < MIN_ANSWER_CHARS:
            dropped["short_answer"] += 1
            continue
        if IDENTITY_FIX.search(out) or IDENTITY_FIX.search(instr):
            dropped["identity_leak"] += 1
            continue
        if instr in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(instr)
        if CS_RE.search(instr + out):
            cs += 1
        rows.append({"instruction": instr, "input": r.get("input", ""), "output": out, "source": source})
        n += 1
    share = cs / max(1, total)
    print(f"  {source}: {n} kept / {total} (cs-share {share:.0%}) from {path}")
    if require_cs and share < MIN_CS_SHARE:
        print(f"ABORT: {source} cs-share {share:.0%} < {MIN_CS_SHARE:.0%} — 0.7 dilution would repeat", file=sys.stderr)
        sys.exit(3)
    return n


def build(conceptual_cz: Path, fresh_cz: Path | None, name: str, trained_on: str) -> int:
    if not FRESH_V06.is_file() or not conceptual_cz.is_file():
        print("missing required source (fresh_v06 or --conceptual)", file=sys.stderr)
        return 2

    rows: list[dict] = []
    dropped = {"malformed": 0, "short_answer": 0, "identity_leak": 0, "duplicate": 0}
    seen: set[str] = set()

    n_fresh = _load_bulk(FRESH_V06, "fresh_v06", rows, seen, dropped)
    n_concept = _load_bulk(conceptual_cz, "conceptual_cz", rows, seen, dropped)
    n_agent = _load_bulk(fresh_cz, "agent_cz", rows, seen, dropped) if fresh_cz and fresh_cz.is_file() else 0
    if not n_agent:
        print("  WARNING: no agent CZ rows — corpus will miss new chem/phys prose", file=sys.stderr)

    n_terms = 0
    for r in terminology_rows():
        if r["instruction"] in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(r["instruction"])
        rows.append(r)
        n_terms += 1

    # Mirror 0.6 exactly: only the ORIGINAL 8 format rows (not the 0.7
    # expansion to 28). 0.6 won with 8 — per project decision 0.8 copies
    # the 0.6 recipe verbatim, only the fresh bulk is longer/bigger.
    n_format = 0
    for instr, out in ANSWER_FORMAT_ROWS[:8]:
        if instr in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(instr)
        rows.append({"instruction": instr, "input": "", "output": out, "source": "answer_format"})
        n_format += 1

    n_reused = 0
    if PREV_TRAIN.is_file():
        candidates = []
        for line in PREV_TRAIN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            instr = (r.get("instruction") or "").strip()
            out = (r.get("output") or "").strip()
            if not instr or not out or instr in seen or instr in IDENTITY_QUESTIONS:
                continue
            candidates.append(r)
        candidates.sort(key=lambda x: x.get("instruction", ""))
        for r in candidates[:606]:
            instr = (r.get("instruction") or "").strip()
            if instr in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(instr)
            rows.append({"instruction": instr, "input": r.get("input", ""),
                         "output": (r.get("output") or "").strip(), "source": "reused:0.6_filtered"})
            n_reused += 1
        print(f"  reused filtered: {n_reused} (cap 606)")

    n_identity = 0
    for r in _identity_rows(name, trained_on):
        if r["instruction"] in seen:
            continue
        seen.add(r["instruction"])
        rows.append(r)
        n_identity += 1

    # Final corpus Czech-share guard
    cs_all = sum(1 for r in rows if CS_RE.search(r["instruction"] + r["output"]))
    print(f"  CORPUS cs-share: {cs_all / max(1, len(rows)):.0%} (target >= 90%)")

    random.Random(42).shuffle(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"instruction": r["instruction"], "input": r.get("input", ""),
                                "output": r["output"]}, ensure_ascii=False) + "\n")

    srcs: dict[str, int] = {}
    for r in rows:
        srcs[r["source"]] = srcs.get(r["source"], 0) + 1
    meta = {
        "name": "kuclab_hertz_0.8",
        "identity_name": name,
        "trained_on": trained_on,
        "rows": len(rows),
        "cs_share": round(cs_all / max(1, len(rows)), 4),
        "fresh_v06": n_fresh,
        "conceptual_cz": n_concept,
        "agent_cz": n_agent,
        "terminology_rows": n_terms,
        "answer_format_rows": n_format,
        "identity_rows": n_identity,
        "reused_filtered": n_reused,
        "sources": dict(sorted(srcs.items(), key=lambda kv: -kv[1])),
        "dropped": dropped,
        "conceptual_source_file": str(conceptual_cz),
        "fresh_cz_source_file": str(fresh_cz) if fresh_cz else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "0.8: fixes 0.7 EN->CS regression (70.9%<73.8%) by restoring >=90% Czech share; same 309 term rows, hyperparams unchanged (r16/32, lr 3e-5, epochs 2.0). Target +10% over 0.6 (MMLU 87.1%, CZ 81.2%).",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--conceptual", required=True, help="Czech version of conceptual_hard .jsonl")
    b.add_argument("--fresh-cz", default=None, help="agent-generated CZ rows .jsonl (optional but expected)")
    b.add_argument("--name", default=NAME)
    b.add_argument("--trained-on", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = ap.parse_args()
    if args.cmd == "build":
        return build(Path(args.conceptual),
                     Path(args.fresh_cz) if args.fresh_cz else None,
                     args.name, args.trained_on)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
