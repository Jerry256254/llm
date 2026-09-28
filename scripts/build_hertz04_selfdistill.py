#!/usr/bin/env python3
"""
KucLab Hertz 0.4 — self-distilled STEM + programming corpus (Gemma-4-12B base).

DIFFERENCE FROM 0.3 (important, read before touching this file)
-----------------------------------------------------------------
Student changes from Qwen2.5-14B-Instruct to google/gemma-4-12B-it. Gemma-4
DOES have a native thinking mechanism — its chat template uses `<|think|>`
and a channel-based format (`<|channel>thought ... <channel|>`), gated by an
`enable_thinking` flag. We deliberately do NOT replicate that exact token
syntax in training data: getting a channel-based format subtly wrong is
exactly the kind of mismatched-special-token bug that broke the 0.3 export
once already (see lib/convert_gguf.py history). Same choice as 0.3: `output`
is plain `content` with visible step-by-step reasoning as ordinary text, no
special thinking tokens. This does not disable Gemma-4's native thinking
mode at inference — it just means our fine-tune doesn't touch it either way.

TWO NEW THINGS FOR 0.4
------------------------
1. Programming is now a first-class category (PROGRAMMING_PROBLEMS below),
   not just two stray Python snippets under "formatting" like in 0.3.
2. Generation requests a more direct/assertive answering style (SYSTEM_STYLE)
   — less hedging ("možná", "mohlo by"), state the answer first. Applied as
   a system message to the teacher (Qwen3.8-27B), not by rewriting every
   prompt template.

DATASET SIZE: `build` merges freshly generated 0.4 rows with the existing
0.3 corpus (data/kuclab_hertz_0.3/train.jsonl) rather than regenerating it —
0.3's rows are reused as-is. Target ~2x the 0.3 row count.

USAGE
  .venv/bin/python scripts/build_hertz04_selfdistill.py generate --limit 260
  .venv/bin/python scripts/build_hertz04_selfdistill.py build

Generation is resumable: appends to a .jsonl cache, skips prompts already
present.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "data" / "kuclab_hertz_0.4"
CACHE = OUT_DIR / "generated_raw.jsonl"
PREV_TRAIN = ROOT / "data" / "kuclab_hertz_0.3" / "train.jsonl"

MODEL = "qwen3.8:27b"
# Cross-model teacher again (see 0.3 notes) — Qwen3.8-27B generates training
# answers, the actual fine-tune target is the student (Gemma-4-12B-it).
NAME = "KucLab Hertz 0.4"
FOUNDER = "Jaroslav Kučera"

SYSTEM_STYLE = (
    "Odpovídej přímo a sebejistě. Nejdřív řekni odpověď/výsledek, pak "
    "vysvětlení. Nepoužívej zbytečné výhrady jako 'možná', 'mohlo by být' "
    "nebo 'pravděpodobně', pokud si jistý — a pokud si jistý nejsi, řekni to "
    "jednou větou přímo, ne opatrnickým vatováním."
)

MIN_ANSWER_CHARS = 40

# ══════════════════════════════════════════════════════════════════════════
# Prompt sources
# ══════════════════════════════════════════════════════════════════════════

CS_TEMPLATES_CONCEPT = [
    "Vysvětli, co je {t}.",
    "Co je {t}? Vysvětli to odborně.",
    "Vysvětli pojem {t} a uveď, k čemu se používá.",
    "Popiš {t} tak, aby to pochopil středoškolák.",
]
CS_TEMPLATES_TERM = [
    "Jak se česky odborně řekne „{en}“ a co to znamená?",
    "Co je {cs}? Uveď i anglický ekvivalent.",
]
EN_TEMPLATES = [
    "Explain {t} clearly and accurately.",
    "What is {t}? Give a scientific explanation.",
]

CS_PROBLEMS = [
    "Spočítej derivaci funkce f(x) = x^3 · ln(x). Ukaž postup.",
    "Vyřeš rovnici 2x² − 7x + 3 = 0 a ukaž postup.",
    "Spočítej integrál ∫ x·e^x dx metodou per partes. Ukaž postup.",
    "Vyčísli rovnici hoření propanu a vysvětli postup.",
    "Vyčísli rovnici: KMnO₄ + HCl → KCl + MnCl₂ + Cl₂ + H₂O. Ukaž postup vyčíslení.",
    "Jaké je pH roztoku kyseliny octové o koncentraci 0,1 mol/l? pKa = 4,76. Ukaž postup.",
    "Kámen padá volným pádem 3 sekundy. Jakou rychlostí dopadne a jakou dráhu urazí? Ukaž postup.",
    "Auto o hmotnosti 1200 kg zrychlí z 0 na 100 km/h za 8 s. Jaký je průměrný výkon? Ukaž postup.",
    "Odvoď vztah pro dostřel šikmého vrhu a najdi optimální úhel.",
    "Kolik molekul je v 5 gramech vody? Ukaž postup.",
    "Jaká je úniková rychlost z povrchu Měsíce? M = 7,35·10²² kg, R = 1737 km. Ukaž postup.",
    "Spočítej, kolik tepla je potřeba na ohřátí 2 kg vody z 20 °C na 80 °C. Ukaž postup.",
    "Vypočítej odpor paralelní kombinace rezistorů 100 Ω a 220 Ω. Ukaž postup.",
    "Jaká je vlnová délka fotonu o energii 3,1 eV? Ukaž postup.",
    "Kolik je 17 × 23? Ukaž postup.",
    "Urči oxidační čísla všech prvků v H₂SO₄. Ukaž postup.",
    "Kolik gramů NaCl je potřeba na přípravu 250 ml roztoku o koncentraci 0,2 mol/l? Ukaž postup.",
    "Těleso o hmotnosti 5 kg leží na nakloněné rovině se sklonem 30°. Jaká je složka tíhové síly podél roviny? Ukaž postup.",
    "Spočítej směrodatnou odchylku souboru: 2, 4, 4, 4, 5, 5, 7, 9. Ukaž postup.",
    "Jaká je pravděpodobnost, že při třech hodech mincí padne aspoň jednou panna? Ukaž postup.",
]

# New for 0.3 — explicit formatting-skill reinforcement. Ollama/markdown
# rendering rewards clean tables/code/lists; the base model can already do
# this but is inconsistent about when to use it, so we reward it directly.
FORMATTING_PROMPTS = [
    "Vypiš prvních 10 prvků periodické tabulky jako Markdown tabulku se sloupci: značka, název, protonové číslo, skupenství za normálních podmínek.",
    "Porovnej rychlý a pomalý reakční mechanismus SN1 vs SN2 v Markdown tabulce (podmínky, kinetika, stereochemie, příklad).",
    "Napiš v Pythonu funkci, která spočítá kinetickou energii tělesa, a okomentuj ji.",
    "Vypiš Newtonovy pohybové zákony jako číslovaný seznam, každý s jednou větou vysvětlení.",
    "Porovnej v Markdown tabulce vlastnosti kyselin a zásad (pH, chuť, reakce s kovy, příklad látky).",
    "Napiš SQL dotaz, který z tabulky `mereni(id, teplota, cas)` vybere průměrnou teplotu za poslední hodinu, a vysvětli ho.",
    "Vypiš jednotky SI soustavy jako Markdown tabulku (veličina, značka, jednotka, základní/odvozená).",
    "Napiš v Pythonu jednoduchou implementaci Eratosthenova síta a vysvětli časovou složitost.",
    "Sestav Markdown tabulku prvních pěti mocnin čísla 2 se sloupci: exponent, výsledek.",
    "Vypiš kroky vyvažování chemické rovnice jako číslovaný postup, obecně (bez konkrétního příkladu).",
    "Napiš funkci v Pythonu, která ověří, jestli je zadané číslo prvočíslo, s komentáři.",
    "Porovnej v tabulce mitózu a meiózu (počet dělení, výsledné buňky, účel, kde v těle probíhá).",
]

PROGRAMMING_PROBLEMS = [
    "Napiš v Pythonu funkci pro binární vyhledávání v seřazeném poli a vysvětli časovou složitost.",
    "Vysvětli rozdíl mezi seznamem (list) a n-ticí (tuple) v Pythonu a kdy použít který.",
    "Napiš rekurzivní a iterativní verzi výpočtu Fibonacciho čísla v Pythonu, porovnej jejich složitost.",
    "Co je to race condition a jak se jí dá předejít? Ukaž příklad v Pythonu s threading.Lock.",
    "Vysvětli rozdíl mezi hloubkovým (DFS) a šířkovým (BFS) prohledáváním grafu, kdy použít které.",
    "Napiš SQL dotaz s JOIN, který spojí tabulky `objednavky` a `zakaznici` a vrátí součet objednávek na zákazníka.",
    "Vysvětli, co je Big O notace, a seřaď O(1), O(log n), O(n), O(n log n), O(n²) od nejrychlejšího.",
    "Napiš třídu v Pythonu implementující spojový seznam (linked list) se vkládáním na začátek a konec.",
    "Co je to memoizace? Ukaž na příkladu výpočtu Fibonacciho čísel v Pythonu.",
    "Vysvětli rozdíl mezi `==` a `is` v Pythonu na konkrétních příkladech.",
    "Napiš funkci, která zjistí, jestli je řetězec palindrom, bez použití vestavěné reverze.",
    "Co je to garbage collection a jak funguje v Pythonu (reference counting + cyklický GC)?",
    "Vysvětli princip hashovací tabulky (hash map) a proč má v průměru O(1) vyhledávání.",
    "Napiš dekorátor v Pythonu, který změří dobu běhu funkce.",
    "Co je to deadlock? Popiš čtyři podmínky, za kterých vzniká.",
    "Vysvětli rozdíl mezi procesem a vláknem (thread) a kdy použít multiprocessing místo threadingu v Pythonu.",
    "Napiš funkci, která seřadí pole pomocí quicksort, a vysvětli nejhorší případ složitosti.",
    "Co je to normalizace databáze? Vysvětli 1NF, 2NF, 3NF na jednoduchém příkladu.",
    "Napiš v Pythonu generátor, který vrací prvočísla do zadaného limitu, a vysvětli výhodu generátoru oproti seznamu.",
    "Vysvětli, co je REST API, a jaké HTTP metody se obvykle používají pro CRUD operace.",
    "Debuguj tenhle kód a vysvětli chybu: `def f(x, lst=[]): lst.append(x); return lst` — proč je nebezpečný default argument?",
    "Napiš funkci, která spočítá největší společný dělitel (GCD) dvou čísel Euklidovým algoritmem.",
    "Vysvětli rozdíl mezi kompilovaným a interpretovaným jazykem a kam patří Python.",
    "Co je to Big-O nejhoršího vs průměrného vs nejlepšího případu? Ukaž na příkladu quicksortu.",
    "Napiš v Pythonu jednoduchý LRU cache pomocí `collections.OrderedDict`.",
    "Explain what a race condition is and how a mutex prevents it, with a short code example.",
    "Write a Python function that detects a cycle in a linked list using Floyd's algorithm.",
    "What is the difference between SQL and NoSQL databases? When would you choose each?",
    "Write a recursive function to compute the factorial of n in Python, and explain the base case.",
    "Explain what Big-O notation means for space complexity, not just time complexity.",
]

IDENTITY = [
    "Kdo jsi?",
    "Jak se jmenuješ?",
    "Kdo tě vytvořil?",
    "Kdo je zakladatel KucLab?",
    "V čem jsi dobrý?",
    "Co je KucLab Hertz?",
    "Who are you?",
    "What is your name?",
]


def concept_prompts(rng: random.Random, n: int) -> list[dict]:
    from cz_terms import train_terms

    out: list[dict] = []
    terms = train_terms()
    rng.shuffle(terms)
    for cs, en, dom, defn in terms:
        out.append({"kind": "concept_cs", "domain": dom,
                    "prompt": rng.choice(CS_TEMPLATES_CONCEPT).format(t=cs)})
        out.append({"kind": "term_cs", "domain": dom,
                    "prompt": rng.choice(CS_TEMPLATES_TERM).format(cs=cs, en=en)})
        if rng.random() < 0.25:
            out.append({"kind": "concept_en", "domain": dom,
                        "prompt": rng.choice(EN_TEMPLATES).format(t=en)})
        if len(out) >= n:
            break
    return out[:n]


def all_prompts(rng: random.Random, limit: int) -> list[dict]:
    out: list[dict] = []
    out += [{"kind": "problem_cs", "domain": "mix", "prompt": p} for p in CS_PROBLEMS]
    out += [{"kind": "programming", "domain": "cs", "prompt": p} for p in PROGRAMMING_PROBLEMS]
    out += [{"kind": "formatting", "domain": "mix", "prompt": p} for p in FORMATTING_PROMPTS]
    out += [{"kind": "identity", "domain": "identity", "prompt": p} for p in IDENTITY]
    remaining = max(0, limit - len(out))
    out += concept_prompts(rng, remaining)
    rng.shuffle(out)
    return out[:limit]


# ══════════════════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════════════════

def load_cache() -> dict[str, dict]:
    if not CACHE.is_file():
        return {}
    got: dict[str, dict] = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
            got[o["prompt"]] = o
        except json.JSONDecodeError:
            continue
    return got


def generate_all(model: str, prompts: list[dict], workers: int, num_predict: int) -> None:
    done = load_cache()
    todo = [p for p in prompts if p["prompt"] not in done]
    print(f"prompts: {len(prompts)} | cached: {len(done)} | to generate: {len(todo)}", flush=True)
    _run_generation(model, todo, workers, num_predict)


def _run_generation(model: str, todo: list[dict], workers: int, num_predict: int) -> None:
    import urllib.request

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    t0 = time.time()
    counter = [0]
    fh = CACHE.open("a", encoding="utf-8")

    def one(p: dict) -> None:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_STYLE},
                {"role": "user", "content": p["prompt"]},
            ],
            "stream": False,
            "options": {"temperature": 0.6, "top_p": 0.95,
                        "num_predict": num_predict, "num_ctx": 8192},
        }).encode()
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                d = json.load(r)
        except Exception as e:
            print(f"    ERR {type(e).__name__} on: {p['prompt'][:50]}", flush=True)
            return

        msg = d.get("message") or {}
        answer = (msg.get("content") or "").strip()
        for junk in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            answer = answer.replace(junk, "").strip()

        rec = {**p, "answer": answer, "done_reason": d.get("done_reason")}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        counter[0] += 1
        if counter[0] % 25 == 0:
            el = time.time() - t0
            print(f"  {counter[0]}/{len(todo)}  {el:.0f}s  ({el/counter[0]:.1f}s each)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as exr:
        list(exr.map(one, todo))
    fh.close()
    print(f"generated {counter[0]} in {time.time()-t0:.0f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# Assembly
# ══════════════════════════════════════════════════════════════════════════

IDENTITY_FIX = re.compile(r"\b(Qwen[\d.]*|Alibaba|Tongyi|Alibaba Cloud)\b", re.I)


def _identity_rows(name: str, founder: str, trained_on: str) -> list[dict]:
    """Hand-written, no <think> wrapping — see module docstring for why."""
    def ex(instruction: str, answer: str) -> dict:
        return {"instruction": instruction, "input": "", "output": answer.strip(),
                "source": "identity"}

    # Per project decision: attribute creation to KucLab as an organization,
    # not to a named individual — no founder name in training data.
    return [
        ex("Jak se jmenuješ?", f"Jmenuju se {name}."),
        ex("Kdo jsi?", f"Jsem {name} — model od KucLab. Zaměřuju se na fyziku, "
           "chemii, biologii a matematiku a mluvím česky i anglicky."),
        ex("What is your name?", f"My name is {name}."),
        ex("Who are you?", f"I'm {name}, a model from KucLab focused on physics, "
           "chemistry, biology and mathematics. I work in Czech and English."),
        ex("představ se", f"{name} — model od KucLab. Specializace: fyzika, chemie, "
           "biologie, matematika. Odpovídám v jazyce, kterým se ptáš."),
        ex("Kdo tě vytvořil?", "Vytvořil mě KucLab."),
        ex("Kdo je KucLab?", "KucLab je laboratoř/tým, který mě vyvinul a trénuje."),
        ex("Who created you?", "I was created by KucLab."),
        ex("Jaký má KucLab web?", "kuclab.org"),
        ex("What is KucLab's website?", "kuclab.org"),
        ex("V čem jsi dobrý?", "Fyzika, chemie, biologie a matematika — výpočty, "
           "odvození a vysvětlení pojmů krok za krokem, česky i anglicky."),
        ex("Co je KucLab Hertz?", f"{name} je model od KucLab (kuclab.org) zaměřený "
           "na fyziku, chemii, biologii, matematiku a programování."),
        ex("Kdy jsi byl dotrénován?", f"Poslední dotrénování: {trained_on}."),
    ]


def build(name: str, founder: str, trained_on: str) -> int:
    cached = load_cache()
    if not cached:
        print("nothing generated yet — run `generate` first", file=sys.stderr)
        return 2

    rows: list[dict] = []
    dropped = {"short_answer": 0, "truncated": 0, "identity": 0}

    for rec in cached.values():
        answer = (rec.get("answer") or "").strip()

        if rec.get("done_reason") == "length":
            dropped["truncated"] += 1
            continue
        if len(answer) < MIN_ANSWER_CHARS:
            dropped["short_answer"] += 1
            continue
        if rec.get("kind") == "identity":
            dropped["identity"] += 1
            continue
        if IDENTITY_FIX.search(answer):
            dropped["identity"] += 1
            continue

        rows.append({
            "instruction": rec["prompt"],
            "input": "",
            "output": answer,
            "source": f"selfdistill04:{rec.get('kind')}",
        })

    n_new = len(rows)

    # Reuse 0.3's corpus as-is instead of regenerating it — this is what
    # gets us to ~2x the row count, per project decision. Its old identity
    # rows are dropped and replaced with 0.4-versioned ones below (same
    # policy: no founder name in training data).
    n_prev = 0
    if PREV_TRAIN.is_file():
        for line in PREV_TRAIN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            instr = r.get("instruction", "")
            if instr in IDENTITY or instr in (
                "Kdo je zakladatel KucLab?", "představ se",
            ):
                continue  # superseded by this file's own _identity_rows()
            rows.append({
                "instruction": instr,
                "input": r.get("input", ""),
                "output": r.get("output", ""),
                "source": "reused:kuclab_hertz_0.3",
            })
            n_prev += 1
    else:
        print(f"WARNING: {PREV_TRAIN} not found — 0.4 corpus will only have new rows", file=sys.stderr)

    for r in _identity_rows(name, founder, trained_on):
        rows.append(r)

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
        "name": "kuclab_hertz_0.4",
        "identity_name": name,
        "founder": founder,
        "trained_on": trained_on,
        "rows": len(rows),
        "new_rows": n_new,
        "reused_from_0.3": n_prev,
        "sources": dict(sorted(srcs.items(), key=lambda kv: -kv[1])),
        "dropped": dropped,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--limit", type=int, default=700)
    # 0.3 learned this the hard way: with one GPU, concurrent requests to a
    # 27B teacher just queue behind each other in Ollama — workers>1 adds
    # timeout risk with zero throughput benefit. Keep at 1.
    g.add_argument("--workers", type=int, default=1)
    # 400 truncated 60% of rows in testing (done_reason=length) — no <think>
    # channel here means the full step-by-step reasoning lands in `content`,
    # so answers run longer than 0.2's post-<think> answers alone. 1500
    # measured 0/10 truncated on the same prompt mix.
    g.add_argument("--num-predict", type=int, default=1500)
    g.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("build")
    b.add_argument("--name", default=NAME)
    b.add_argument("--founder", default=FOUNDER)
    b.add_argument("--trained-on", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    args = ap.parse_args()
    if args.cmd == "generate":
        rng = random.Random(args.seed)
        prompts = all_prompts(rng, args.limit)
        generate_all(MODEL, prompts, args.workers, args.num_predict)
        return 0
    elif args.cmd == "build":
        return build(args.name, args.founder, args.trained_on)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
