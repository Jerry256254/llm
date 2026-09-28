#!/usr/bin/env python3
"""
KucLab Hertz 0.7 — FIXED dataset assembly (Gemma-4-12B base).

PROBLEM WITH FAILED 0.7 (4481 rows):
  - reused entire 0.6 train (2600 rows) => 58% old data, only 32% fresh (1463)
  - fresh was subagent 1463 filtered, avg quality lower than fresh_v06 (1994, proven 79.2%)
  - identity ballooned to 71 + 10 personality = 1.8% STEM dilution
  - train ran with epochs 3.0 (not 2.0 per yaml) => train_loss 0.885 overfit, bench 63-68% variance, best 78.8% <0.6
  - CZ terms 69.9% vs 73.8% (0.6) — reuse bloat did NOT help

FIX STRATEGY FOR +5% OVER 0.6:
  Target: MMLU 79.2% -> 84.2% (+5pp), CZ 73.8% -> 78.8% (+5pp)
  Weak spots in 0.6: chemistry 61.7%, physics 73.3% (vs bio 91.7%, math 90%)
  So fresh must be chemistry/physics heavy, hard multi-step.

  Dataset composition (aim ~3800 rows, ~75% fresh):
   - 1994 fresh_v06 (proven, external) — keep entirely
   - 901 hertz07_retry_conceptual (hard conceptual, avg 626 chars) — chemistry/physics weighted
   - 309 terminology en2cs_defined (train split only, no bare lookups)
   - 28 answer_format (full 0.7 set, strongest MMLU lever: 46->18 unparseable)
   - 15 identity (minimal, as in 0.6, not 71 — avoid dilution)
   - 606 reuse from 0.5-equivalent via 0.6 filtered (preserve cumulative Czech without 2600 bloat)
   = ~3853 rows total, fresh 2895 (75%)

  Hyperparams: KEEP 0.6 proven (r16 alpha32 lr3e-5 epochs2.0 batch1 grad8) — data is lever, not rank.
  Epochs strictly 2.0 (not 3.0).

USAGE:
  .venv/bin/python scripts/build_hertz07_fixed.py build
"""
from __future__ import annotations
import argparse, json, random, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "data" / "kuclab_hertz_0.7"
NAME = "KucLab Hertz 0.7"

FRESH_V06 = Path("/home/admin/.claude/uploads/07e6144c-f286-4be0-bca2-8615d1b2531c/0642b371-fresh_v06.jsonl")
CONCEPTUAL = ROOT / "data" / "manual_additions" / "hertz07_retry_conceptual.jsonl"
PREV_TRAIN = ROOT / "data" / "kuclab_hertz_0.6" / "train.jsonl"  # for 606 reuse subset

MIN_ANSWER_CHARS = 10
IDENTITY_FIX = re.compile(r"\b(Qwen[\d.]*|Alibaba|Tongyi|Alibaba Cloud|Gemma|Google DeepMind)\b", re.I)
IDENTITY_QUESTIONS = {
    "Jak se jmenuješ?", "Kdo jsi?", "What is your name?", "Who are you?",
    "představ se", "Kdo tě vytvořil?", "Kdo je KucLab?", "Who created you?",
    "Jaký má KucLab web?", "What is KucLab's website?", "V čem jsi dobrý?",
    "Co je KucLab Hertz?", "Kdy jsi byl dotrénován?", "Jsi cenzurovaný?",
    "Máš smysl pro humor?", "Kdo je zakladatel KucLab?",
}

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

ANSWER_FORMAT_ROWS = [
    ("Kolik je 17 × 23? Nejdřív výsledek, pak postup.", "391.\n\nPostup: 17 × 23 = 17 × (20 + 3) = 340 + 51 = 391."),
    ("Je voda za normálního tlaku při 50 °C kapalná? Odpověz nejdřív jednoznačně, pak vysvětli.", "Ano.\n\nZa normálního tlaku (101,325 kPa) voda taje při 0 °C a vře při 100 °C, takže 50 °C leží uvnitř kapalného intervalu."),
    ("Which is larger, 3/7 or 5/12? State the answer first, then justify.", "3/7 is larger.\n\nCommon denominator 84: 3/7 = 36/84, 5/12 = 35/84. 36/84 > 35/84."),
    ("Vede měď elektrický proud lépe než železo? Nejdřív odpověz, pak zdůvodni.", "Ano, měď vede lépe.\n\nRezistivita mědi je asi 1,68·10⁻⁸ Ω·m, železa asi 9,7·10⁻⁸ Ω·m — měď má zhruba šestkrát nižší odpor na stejný průřez a délku."),
    ("Je 91 prvočíslo? Odpověz nejdřív ano/ne, pak ukaž proč.", "Ne.\n\n91 = 7 × 13, takže má dělitele jiné než 1 a sebe sama."),
    ("What is the time complexity of binary search? Give the answer first.", "O(log n).\n\nEach comparison halves the remaining search space, so the number of steps grows with the base-2 logarithm of the input size."),
    ("Zvýší se tlak plynu při zahřátí v uzavřené nádobě? Nejdřív odpověď, pak důvod.", "Ano, zvýší.\n\nPři konstantním objemu platí p/T = konst. (Gay-Lussacův zákon), takže s rostoucí termodynamickou teplotou roste úměrně i tlak."),
    ("Je DNA jednovláknová nebo dvouvláknová? Odpověz přímo, pak doplň kontext.", "Dvouvláknová.\n\nTvoří dvoušroubovici ze dvou antiparalelních řetězců spojených vodíkovými můstky mezi komplementárními bázemi (A-T, G-C). Jednovláknová bývá RNA."),
    ("Je 144 druhá mocnina celého čísla? Odpověz ano/ne hned, pak zdůvodni.", "Ano.\n\n12² = 144, takže 144 je druhou mocninou celého čísla 12."),
    ("Kolik je odmocnina ze 169? Výsledek první.", "13.\n\n13 × 13 = 169, takže √169 = 13."),
    ("Vyber správnou možnost: rychlost světla ve vakuu je (A) 3·10⁵ km/s (B) 3·10⁸ km/s (C) 3·10⁸ m/s. Odpověz písmenem hned.", "(C).\n\nRychlost světla ve vakuu je přibližně 2,998·10⁸ m/s ≈ 3·10⁸ m/s, což v km/s odpovídá 3·10⁵ km/s — proto je (A) číselně stejné, ale ve špatných jednotkách, a (C) je správně formulovaná odpověď."),
    ("Which has a higher boiling point at standard pressure, ethanol or water? Answer first.", "Water.\n\nWater boils at 100 °C, ethanol at about 78 °C — the difference comes mainly from stronger hydrogen bonding per molecule in water."),
    ("Je 0,1 + 0,2 v desetinných číslech přesně 0,3? Odpověz ano/ne, pak vysvětli.", "Ne, ne úplně přesně ve dvojkové soustavě.\n\nPočítač reprezentuje 0,1 a 0,2 binárně jako nekonečné periodické zlomky, takže součet vychází jako 0,30000000000000004, ne přesně 0,3 — matematicky ano, ve float aritmetice ne."),
    ("Vede se teplo v kovu hlavně vedením, prouděním nebo zářením? Odpověz přímo.", "Vedením (kondukcí).\n\nV kovu přenášejí energii hlavně volné elektrony a kmity mřížky, což je mnohem rychlejší mechanismus než proudění nebo záření při běžných teplotách."),
    ("Je funkce f(x) = x² sudá, lichá, nebo ani jedno? Odpověz hned, pak dokaž.", "Sudá.\n\nf(-x) = (-x)² = x² = f(x) pro všechna x, což je přesně definice sudé funkce."),
    ("What's the output of `print(2 ** 10)` in Python? Give the number first.", "1024.\n\n2 raised to the 10th power is 2·2·2·2·2·2·2·2·2·2 = 1024."),
    ("Má had (ve smyslu plaz) plíce? Odpověz ano/ne rovnou.", "Ano.\n\nHadi jsou plazi a dýchají plícemi (obvykle jednou funkční, protáhlou pravou plící); žábry ani kožní dýchání jako hlavní mechanismus nemají."),
    ("Kolik gramů má 1 mol vody? Výsledek první, pak výpočet.", "18 g.\n\nMolární hmotnost vody H₂O = 2×1 + 16 = 18 g/mol, takže 1 mol váží 18 g."),
    ("Roste, nebo klesá entropie vesmíru jako celku v čase? Odpověz přímo.", "Roste.\n\nDruhý termodynamický zákon říká, že celková entropie izolované soustavy (a vesmír jako celek je z definice izolovaný) v čase neklesá."),
    ("Is a tomato botanically a fruit or a vegetable? Answer directly first.", "Botanically, a fruit.\n\nA fruit develops from the ovary of a flowering plant and contains seeds — a tomato meets that definition, even though it's used as a vegetable in cooking."),
    ("Vyřeš: log₂(8) = ? Nejdřív číslo.", "3.\n\n2³ = 8, takže log₂(8) = 3."),
    ("Je Slunce hvězda hlavní posloupnosti? Odpověz ano/ne hned.", "Ano.\n\nSlunce je žlutý trpaslík typu G2V, momentálně slučuje vodík na helium v jádru — to je definiční znak hvězdy hlavní posloupnosti."),
    ("Which grows faster asymptotically, n log n or n²? State the answer first.", "n² grows faster.\n\nFor large n, n² dominates n log n because the ratio n²/(n log n) = n/log n increases without bound."),
    ("Obsahuje vzduch víc dusíku nebo kyslíku (objemově)? Odpověz přímo.", "Víc dusíku.\n\nSuchý vzduch je přibližně 78 % N₂ a jen asi 21 % O₂ objemově."),
    ("Je nula sudé číslo? Odpověz ano/ne rovnou.", "Ano.\n\nNula je dělitelná dvěma beze zbytku (0 ÷ 2 = 0), což splňuje definici sudého čísla."),
    ("Does increasing pressure on a gas at constant temperature increase or decrease its volume? Answer first.", "Decrease.\n\nBoyle's law: pV = constant at fixed temperature, so volume is inversely proportional to pressure."),
    ("Má měsíc vlastní atmosféru srovnatelnou se Zemí? Odpověz ano/ne, pak zdůvodni.", "Ne.\n\nMěsíc má jen extrémně řídkou exosféru (miliardykrát řidší než zemská atmosféra) — jeho gravitace je příliš slabá na to, aby udržela hustou atmosféru."),
    ("Vyber správně: je 17 prvočíslo? Odpověz ano/ne hned.", "Ano.\n\n17 nemá žádného celočíselného dělitele mezi 2 a √17 ≈ 4,1 kromě sebe a jedné, takže je prvočíslo."),
]

def build(name: str, trained_on: str) -> int:
    for p in [FRESH_V06, CONCEPTUAL]:
        if not p.is_file():
            print(f"missing fresh source: {p}", file=sys.stderr)
            return 2

    rows: list[dict] = []
    dropped = {"malformed": 0, "short_answer": 0, "identity_leak": 0, "duplicate": 0}
    seen: set[str] = set()

    def add_file(path: Path, source: str):
        nonlocal rows
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line: continue
            try:
                r=json.loads(line)
            except: 
                dropped["malformed"]+=1; continue
            instr=(r.get("instruction") or "").strip()
            out=(r.get("output") or "").strip()
            if not instr or not out:
                dropped["malformed"]+=1; continue
            if len(out) < MIN_ANSWER_CHARS:
                dropped["short_answer"]+=1; continue
            if IDENTITY_FIX.search(out):
                dropped["identity_leak"]+=1; continue
            if instr in seen:
                dropped["duplicate"]+=1; continue
            seen.add(instr)
            rows.append({"instruction": instr, "input": r.get("input",""), "output": out, "source": source})
            n+=1
        print(f"  {source}: {n} kept from {path}")
        return n

    n_fresh_v06 = add_file(FRESH_V06, "fresh_v06")
    n_concept = add_file(CONCEPTUAL, "conceptual_hard")

    n_terms=0
    for r in terminology_rows():
        if r["instruction"] in seen:
            dropped["duplicate"]+=1; continue
        seen.add(r["instruction"])
        rows.append(r); n_terms+=1

    n_format=0
    for instr,out in ANSWER_FORMAT_ROWS:
        if instr in seen:
            dropped["duplicate"]+=1; continue
        seen.add(instr)
        rows.append({"instruction": instr, "input": "", "output": out, "source": "answer_format"})
        n_format+=1

    n_reused=0
    if PREV_TRAIN.is_file():
        # Limit reuse to preserve 75% fresh ratio — take at most 606 oldest good rows
        # (0.6's own reuse from 0.5). Filter to keep only those marked as reused in 0.6 meta?
        # Simpler: sample 606 rows from 0.6 that are NOT fresh_v06 duplicates (which we already deduped)
        # and NOT identity questions.
        candidates=[]
        for line in PREV_TRAIN.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try: r=json.loads(line)
            except: continue
            instr=(r.get("instruction") or "").strip()
            out=(r.get("output") or "").strip()
            if not instr or not out or instr in seen or instr in IDENTITY_QUESTIONS:
                continue
            candidates.append(r)
        # Take first 606 (original 0.5 heritage is early in shuffled file? Shuffle random 42, but we just take 606)
        # To keep determinism, sort by instruction
        candidates.sort(key=lambda x: x.get("instruction",""))
        for r in candidates[:606]:
            instr=(r.get("instruction") or "").strip()
            if instr in seen: 
                dropped["duplicate"]+=1; continue
            seen.add(instr)
            rows.append({"instruction": instr, "input": r.get("input",""), "output": (r.get("output") or "").strip(), "source": "reused:0.6_filtered"})
            n_reused+=1
        print(f"  reused filtered: {n_reused} / {len(candidates)} candidates (cap 606)")

    n_identity=0
    for r in _identity_rows(name, trained_on):
        if r["instruction"] in seen: continue
        seen.add(r["instruction"])
        rows.append(r); n_identity+=1

    random.Random(42).shuffle(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"instruction": r["instruction"], "input": r.get("input",""), "output": r["output"]}, ensure_ascii=False)+"\n")

    srcs={}
    for r in rows: srcs[r["source"]] = srcs.get(r["source"],0)+1
    meta={
        "name": "kuclab_hertz_0.7_fixed",
        "identity_name": name,
        "trained_on": trained_on,
        "rows": len(rows),
        "fresh_v06": n_fresh_v06,
        "conceptual_hard": n_concept,
        "terminology_rows": n_terms,
        "answer_format_rows": n_format,
        "identity_rows": n_identity,
        "reused_filtered": n_reused,
        "sources": dict(sorted(srcs.items(), key=lambda kv: -kv[1])),
        "dropped": dropped,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Fixed 0.7: replaces failed 4481-row 32% fresh with ~3850 rows 75% fresh, epochs 2.0, identity 15, r16. Target +5pp over 0.6 (84.2% MMLU, 78.8% CZ)"
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd", required=True)
    b=sub.add_parser("build")
    b.add_argument("--name", default=NAME)
    b.add_argument("--trained-on", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args=ap.parse_args()
    if args.cmd=="build":
        return build(args.name, args.trained_on)
    return 1

if __name__=="__main__":
    raise SystemExit(main())
