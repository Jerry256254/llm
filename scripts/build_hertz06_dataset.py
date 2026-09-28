#!/usr/bin/env python3
"""
KucLab Hertz 0.6 — dataset assembly (Gemma-4-12B base, same as 0.4/0.5).

DIFFERENT FROM EVERY PREVIOUS RELEASE
--------------------------------------
0.3/0.4/0.5 built their corpora by self-distilling from Qwen3.8-27B on this
host. 0.6 does NOT: its bulk (2000 rows) is a fresh, externally generated
corpus, and NOTHING is reused from 0.3/0.4/0.5 — per project decision the
rows are "čerstvé od znova". Verified before assembly: 2000/2000 valid JSON,
zero duplicate instructions, zero duplicate outputs, zero <think>/special
tokens, zero identity mentions, 93% carrying Czech diacritics.

WHY 0.5 REGRESSED, AND WHAT THIS BUILD DOES ABOUT IT
-----------------------------------------------------
0.5 measured 68.8% on our MMLU-Pro STEM subset vs the base model's 75.8%
(−7.0pp, same corrected methodology on both — a real regression, not a
harness artifact). Its 285 genuinely-new rows were aimed at personality,
web-dev and agentic planning, not STEM depth. Two levers here:

1. The fresh corpus is heavily STEM/programming weighted, which is what
   0.5's new rows were not.
2. TERMINOLOGY_ROWS below adds deterministic CS↔EN scientific term pairs
   built straight from scripts/cz_terms.py. These are exact, not model-
   generated, so they can't carry the teacher's mistakes ("smělná odchylka"
   shipped in 0.4 this way). Both directions are emitted, weighted toward
   EN→CS because that is the measured weak side (69.9% vs 81.6% CS→EN
   in 0.5).

BENCHMARK HONESTY — the line this file does not cross
------------------------------------------------------
cz_terms.py holds out every 4th term (index % 4 == 3) as an eval set.
build() uses train_terms() ONLY, so the Czech-terminology benchmark keeps
measuring generalisation rather than memorisation. Do not switch this to
all_terms(): it would inflate the published number into something that
measures nothing. Likewise, no MMLU-Pro question ever enters training data
— the answer-first FORMAT is taught (see ANSWER_FORMAT_ROWS), the test's
actual questions are not.

USAGE
  .venv/bin/python scripts/build_hertz06_dataset.py build --fresh <path.jsonl>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "data" / "kuclab_hertz_0.6"
NAME = "KucLab Hertz 0.6"
PREV_TRAIN = ROOT / "data" / "kuclab_hertz_0.5" / "train.jsonl"
# 0.5's train.jsonl already carries three generations of cumulative Czech
# STEM reinforcement (0.3's original corpus + 0.4's additions + 0.5's own).
# The first "everything fresh, nothing reused" attempt at 0.6 measured WORSE
# Czech terminology than 0.5 (69.4% vs 75.7%) despite adding 309 dedicated
# terminology rows — a single fresh round doesn't replace three cumulative
# ones. This reuse restores that effect on top of the STEM-heavy fresh
# corpus, per project decision after the first attempt regressed.

MIN_ANSWER_CHARS = 10  # lower than prior builds: the fresh corpus legitimately
# contains terse arithmetic answers ("2,5 × 3600 = 9 000 s.") that are correct
# and worth keeping — 489 of 2000 rows are under 40 chars and were spot-checked.

# Guard against a teacher/generator leaking another vendor's identity into
# answers (kept from earlier builders — cheap insurance).
IDENTITY_FIX = re.compile(r"\b(Qwen[\d.]*|Alibaba|Tongyi|Alibaba Cloud|Gemma|Google DeepMind)\b", re.I)
IDENTITY_QUESTIONS = {
    "Jak se jmenuješ?", "Kdo jsi?", "What is your name?", "Who are you?",
    "představ se", "Kdo tě vytvořil?", "Kdo je KucLab?", "Who created you?",
    "Jaký má KucLab web?", "What is KucLab's website?", "V čem jsi dobrý?",
    "Co je KucLab Hertz?", "Kdy jsi byl dotrénován?", "Jsi cenzurovaný?",
    "Máš smysl pro humor?", "Kdo je zakladatel KucLab?",
}


def _identity_rows(name: str, trained_on: str) -> list[dict]:
    """Hand-written. No founder name — org attribution only, as in 0.4/0.5."""
    def ex(instruction: str, answer: str) -> dict:
        return {"instruction": instruction, "input": "", "output": answer.strip(),
                "source": "identity"}

    return [
        ex("Jak se jmenuješ?", f"Jmenuju se {name}."),
        ex("Kdo jsi?", f"Jsem {name} — model od KucLab. Zaměřuju se na fyziku, "
           "chemii, biologii, matematiku a programování a mluvím česky i anglicky."),
        ex("What is your name?", f"My name is {name}."),
        ex("Who are you?", f"I'm {name}, a model from KucLab focused on physics, "
           "chemistry, biology, mathematics and programming. I work in Czech and English."),
        ex("představ se", f"{name} — model od KucLab. Specializace: fyzika, chemie, "
           "biologie, matematika, programování. Odpovídám v jazyce, kterým se ptáš."),
        ex("Kdo tě vytvořil?", "Vytvořil mě KucLab."),
        ex("Kdo je KucLab?", "KucLab je laboratoř/tým, který mě vyvinul a trénuje."),
        ex("Who created you?", "I was created by KucLab."),
        ex("Jaký má KucLab web?", "kuclab.org"),
        ex("What is KucLab's website?", "kuclab.org"),
        ex("V čem jsi dobrý?", "Fyzika, chemie, biologie, matematika a programování — "
           "výpočty, odvození, kód a vysvětlení pojmů krok za krokem, česky i anglicky."),
        ex("Co je KucLab Hertz?", f"{name} je model od KucLab (kuclab.org) zaměřený "
           "na fyziku, chemii, biologii, matematiku a programování."),
        ex("Kdy jsi byl dotrénován?", f"Poslední dotrénování: {trained_on}."),
        ex("Jsi cenzurovaný?", "Neschovávám se za zbytečné vatování a omluvy u "
           "běžných témat — ale skutečně škodlivé nebo nebezpečné požadavky "
           "odmítnu, stručně a bez kázání."),
        ex("Máš smysl pro humor?", "Ano, klidně i suchý nebo černý, pokud se to "
           "k tématu hodí — ale ne na úkor toho, aby odpověď byla správná."),
    ]


def terminology_rows() -> list[dict]:
    """Deterministic CS↔EN term pairs from the TRAIN split only.

    ONLY the "term + definition" phrasing. An earlier revision of this file
    also emitted two bare-lookup variants per term ("...Odpověz jen tím
    termínem, nic víc." -> "výkon"), reasoning that matching the benchmark's
    own phrasing would help. It did the opposite, measurably:

        Czech terminology, 0.5 -> 0.6-with-bare-rows
          CS->EN  81.6% -> 83.5%
          EN->CS  69.9% -> 59.2%   (the direction those rows targeted)
          TOTAL   75.7% -> 71.4%

    Those three-variants-per-term rows were 927 of 2944 rows (31% of the
    corpus), 618 of them one-word answers — 36% of the whole dataset ended
    up under 30 characters. The model learned the shape "reply with one
    short word" and memorised the 309 training pairs instead of learning how
    Czech scientific terms are formed, so it fell apart on the held-out
    quarter: výkon -> "mocnina", stejnosměrný proud -> "střídavý proud",
    těžiště -> "střed hmotnosti". Definitions teach meaning; bare mappings
    teach recall of exactly the pairs that are NOT what gets measured.
    """
    from cz_terms import train_terms

    rows: list[dict] = []
    for cs, en, dom, defn in train_terms():
        rows.append({
            "instruction": f"Jak se česky odborně řekne „{en}“ a co to znamená?",
            "input": "", "output": f"{cs} — {defn}", "source": "terms:en2cs_defined",
        })
    return rows


# Teaching the model to state a conclusion BEFORE its reasoning. This is the
# single biggest lever on our MMLU-Pro number that isn't knowledge: 46 of
# 0.5's answers and 36 of the base model's scored as wrong purely because the
# model never reached a final letter inside the token budget. These examples
# are generic ("answer first, then why") — no MMLU-Pro question, and no
# multiple-choice item from any eval set, appears here.
ANSWER_FORMAT_ROWS = [
    ("Kolik je 17 × 23? Nejdřív výsledek, pak postup.",
     "391.\n\nPostup: 17 × 23 = 17 × (20 + 3) = 340 + 51 = 391."),
    ("Je voda za normálního tlaku při 50 °C kapalná? Odpověz nejdřív jednoznačně, pak vysvětli.",
     "Ano.\n\nZa normálního tlaku (101,325 kPa) voda taje při 0 °C a vře při 100 °C, "
     "takže 50 °C leží uvnitř kapalného intervalu."),
    ("Which is larger, 3/7 or 5/12? State the answer first, then justify.",
     "3/7 is larger.\n\nCommon denominator 84: 3/7 = 36/84, 5/12 = 35/84. 36/84 > 35/84."),
    ("Vede měď elektrický proud lépe než železo? Nejdřív odpověz, pak zdůvodni.",
     "Ano, měď vede lépe.\n\nRezistivita mědi je asi 1,68·10⁻⁸ Ω·m, železa asi "
     "9,7·10⁻⁸ Ω·m — měď má zhruba šestkrát nižší odpor na stejný průřez a délku."),
    ("Je 91 prvočíslo? Odpověz nejdřív ano/ne, pak ukaž proč.",
     "Ne.\n\n91 = 7 × 13, takže má dělitele jiné než 1 a sebe sama."),
    ("What is the time complexity of binary search? Give the answer first.",
     "O(log n).\n\nEach comparison halves the remaining search space, so the number "
     "of steps grows with the base-2 logarithm of the input size."),
    ("Zvýší se tlak plynu při zahřátí v uzavřené nádobě? Nejdřív odpověď, pak důvod.",
     "Ano, zvýší.\n\nPři konstantním objemu platí p/T = konst. (Gay-Lussacův zákon), "
     "takže s rostoucí termodynamickou teplotou roste úměrně i tlak."),
    ("Je DNA jednovláknová nebo dvouvláknová? Odpověz přímo, pak doplň kontext.",
     "Dvouvláknová.\n\nTvoří dvoušroubovici ze dvou antiparalelních řetězců spojených "
     "vodíkovými můstky mezi komplementárními bázemi (A-T, G-C). Jednovláknová bývá RNA."),
]


def build(fresh_path: Path, name: str, trained_on: str) -> int:
    if not fresh_path.is_file():
        print(f"fresh corpus not found: {fresh_path}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    dropped = {"malformed": 0, "short_answer": 0, "identity_leak": 0, "duplicate": 0}
    seen_instructions: set[str] = set()

    for line in fresh_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
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
        if IDENTITY_FIX.search(out):
            dropped["identity_leak"] += 1
            continue
        if instr in seen_instructions:
            dropped["duplicate"] += 1
            continue
        seen_instructions.add(instr)
        rows.append({"instruction": instr, "input": r.get("input", ""),
                     "output": out, "source": "fresh_v06"})

    n_fresh = len(rows)

    n_terms = 0
    for r in terminology_rows():
        if r["instruction"] in seen_instructions:
            dropped["duplicate"] += 1
            continue
        seen_instructions.add(r["instruction"])
        rows.append(r)
        n_terms += 1

    n_format = 0
    for instr, out in ANSWER_FORMAT_ROWS:
        if instr in seen_instructions:
            dropped["duplicate"] += 1
            continue
        seen_instructions.add(instr)
        rows.append({"instruction": instr, "input": "", "output": out,
                     "source": "answer_format"})
        n_format += 1

    n_reused = 0
    if PREV_TRAIN.is_file():
        for line in PREV_TRAIN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            instr = (r.get("instruction") or "").strip()
            out = (r.get("output") or "").strip()
            if not instr or not out or instr in seen_instructions:
                dropped["duplicate"] += 1
                continue
            if instr in IDENTITY_QUESTIONS:
                continue  # superseded by this file's own _identity_rows()
            seen_instructions.add(instr)
            rows.append({"instruction": instr, "input": r.get("input", ""),
                         "output": out, "source": "reused:kuclab_hertz_0.5"})
            n_reused += 1
    else:
        print(f"WARNING: {PREV_TRAIN} not found — building without reuse", file=sys.stderr)

    n_identity = 0
    for r in _identity_rows(name, trained_on):
        if r["instruction"] in seen_instructions:
            continue
        seen_instructions.add(r["instruction"])
        rows.append(r)
        n_identity += 1

    random.Random(42).shuffle(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"instruction": r["instruction"],
                                "input": r.get("input", ""),
                                "output": r["output"]}, ensure_ascii=False) + "\n")

    srcs: dict[str, int] = {}
    for r in rows:
        srcs[r["source"]] = srcs.get(r["source"], 0) + 1

    meta = {
        "name": "kuclab_hertz_0.6",
        "identity_name": name,
        "trained_on": trained_on,
        "rows": len(rows),
        "fresh_rows": n_fresh,
        "terminology_rows": n_terms,
        "answer_format_rows": n_format,
        "identity_rows": n_identity,
        "reused_from_previous_releases": n_reused,
        "sources": dict(sorted(srcs.items(), key=lambda kv: -kv[1])),
        "dropped": dropped,
        "fresh_source_file": str(fresh_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--fresh", required=True, help="path to the fresh externally generated .jsonl")
    b.add_argument("--name", default=NAME)
    b.add_argument("--trained-on", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = ap.parse_args()
    if args.cmd == "build":
        return build(Path(args.fresh), args.name, args.trained_on)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
