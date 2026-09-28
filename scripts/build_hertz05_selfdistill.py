#!/usr/bin/env python3
"""
KucLab Hertz 0.5 — self-distilled STEM + programming + personality corpus
(Gemma-4-12B base, same as 0.4).

SCOPE FOR 0.5 (agreed 2026-08-21, deadline 2026-08-25 14:00)
--------------------------------------------------------------
Original ask was much bigger (10x data, 512k context, "beat Opus 4.6",
base-model surgery). That's not achievable in ~4 days on this hardware —
see the honest breakdown given to the user. What's actually being built:

  1. Personality shift: direct, witty, dry/dark humor allowed, friendly and
     human in tone — fewer corporate hedges on benign/edgy topics. This is
     NOT a removal of real safety boundaries (no training data asks for or
     rewards content in genuinely harmful categories); it's about dropping
     reflexive "I cannot / As an AI..." hedging on ordinary mature topics.
  2. Programming + modern web-dev as a stronger category: not just
     algorithms now, but "build a clean, current-looking UI" prompts with
     explicit instructions to avoid recognizable AI-slop patterns (generic
     purple gradients, glassmorphism spam, stock "Welcome to the future"
     copy, emoji-as-headers, Inter-font-everywhere clichés).
  3. Same base model (google/gemma-4-12B-it), same STEM domains, same
     teacher (Qwen3.8-27B, data generation only).
  4. Data volume: NOT 10x. Live-tested teacher throughput on this host
     during this build (2026-08-21) showed qwen3.8:27b generation is
     RAM-constrained (15GB system RAM vs a 17.7GB model) and unreliable —
     slow token rates, occasional crashes needing reload. Realistic new-row
     budget is a few hundred rows, not thousands. `build` merges these with
     the full existing 0.4 corpus (744 rows) as before.
  5. Context: left at Gemma-4's native 262144, no 512k stretch attempted —
     doing that reliably would need real testing time we don't have, and
     directly conflicts with the "runs on weaker PCs" goal already shipped
     in 0.4.
  6. No base-model architecture changes — LoRA only, same as every prior
     release.

USAGE
  .venv/bin/python scripts/build_hertz05_selfdistill.py generate --limit 300
  .venv/bin/python scripts/build_hertz05_selfdistill.py build

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

OUT_DIR = ROOT / "data" / "kuclab_hertz_0.5"
CACHE = OUT_DIR / "generated_raw.jsonl"
PREV_TRAIN = ROOT / "data" / "kuclab_hertz_0.4" / "train.jsonl"

MODEL = "qwen3.8:27b"
NAME = "KucLab Hertz 0.5"
FOUNDER = ""  # never put a name in training data — org attribution only

# Applied as a system message during generation. Two variants: STEM answers
# keep full step-by-step rigor; personality/web-dev answers additionally get
# a tone nudge. Kept separate so STEM correctness isn't traded for jokes.
SYSTEM_STYLE_STEM = (
    "Odpovídej přímo a sebejistě. Nejdřív řekni odpověď/výsledek, pak "
    "vysvětlení krok za krokem. Nepoužívej zbytečné výhrady jako 'možná', "
    "'mohlo by být' nebo 'pravděpodobně', pokud si jistý."
)
SYSTEM_STYLE_PERSONALITY = (
    "Odpovídej přímo, sebejistě a lidsky — smíš být vtipný a použít suchý "
    "nebo černý humor, pokud se to k tématu hodí, ale nikdy na úkor "
    "věcné správnosti. Nezačínej omluvami ani formulacemi typu 'jako AI "
    "nemohu' u běžných, neškodných témat — buď přímý. Skutečně nebezpečné "
    "nebo škodlivé požadavky odmítni stručně a bez kázání."
)
SYSTEM_STYLE_WEBDEV = (
    "Piš moderní, čistý kód — žádné rozpoznatelné 'AI slop' vzory: žádné "
    "generické fialovo-modré gradienty, žádný glassmorphism na sílu, žádné "
    "emoji jako nadpisy, žádné klišé texty typu 'Welcome to the future' "
    "nebo 'Unlock your potential'. Piš sémantický HTML/CSS/JS, konkrétní "
    "účelný obsah a rozumné, decentní stylování. Vysvětli klíčová "
    "rozhodnutí stručně."
)

MIN_ANSWER_CHARS = 30

# ══════════════════════════════════════════════════════════════════════════
# Prompt sources — new for 0.5
# ══════════════════════════════════════════════════════════════════════════

PERSONALITY_PROMPTS = [
    "Vysvětli, proč je odkládání práce na poslední chvíli tak lákavé, s trochou nadsázky.",
    "Napiš krátkou, vtipnou (ale výstižnou) analogii pro to, jak funguje rekurze.",
    "Proč lidi pořád věří na horoskopy? Odpověz věcně, ale s troškou suchého humoru.",
    "Jaký je rozdíl mezi optimistou a pesimistou programátorem? Krátká vtipná odpověď.",
    "Zkritizuj s humorem typický 'AI slop' web design (gradient na gradientu, emoji všude).",
    "Vysvětli entropii tak, jako bys popisoval stav mého pokoje po týdnu neuklízení.",
    "Proč je JavaScript plný podivností? Vysvětli s nadhledem, ale technicky přesně.",
    "Dej mi upřímnou, trochu jízlivou zpětnou vazbu na klišé motivační citáty na LinkedInu.",
    "What's a darkly funny but accurate way to explain technical debt to a non-programmer?",
    "Vysvětli, proč 'to bude hotové za 5 minut' u programátorů nikdy neznamená 5 minut.",
    "Napiš krátký sarkastický komentář k tomu, proč Excel pořád vládne firemnímu světu.",
    "Vysvětli rozdíl mezi teorií a praxí v inženýrství, s trochou černého humoru.",
    "Proč je debugging cizího kódu horší než psaní vlastního od nuly? Odpověz s nadhledem.",
    "Napiš krátkou, upřímnou odpověď na to, jestli je astrologie věda (a proč ne).",
    "Vysvětli, co je to 'yak shaving' v programování, na vtipném příkladu.",
]

WEBDEV_PROMPTS = [
    "Vytvoř landing page pro malý osobní projekt (jednoduchý nástroj na sledování výdajů) — čisté HTML/CSS, moderní ale ne AI slop vzhled.",
    "Napiš komponentu karty produktu (HTML/CSS) pro e-shop — decentní, funkční, bez klišé gradientů.",
    "Vytvoř responzivní navigační lištu v čistém HTML/CSS/JS, mobile-first.",
    "Napiš jednoduchou to-do list aplikaci ve vanilla JavaScriptu s lokálním úložištěm (localStorage).",
    "Vytvoř formulář pro přihlášení s validací v HTML/JS — přístupný (a11y), bez zbytečných ozdob.",
    "Napiš CSS grid layout pro blog s postranním panelem — moderní, čitelný, minimalistický.",
    "Vytvoř jednoduchou dark-mode přepínací komponentu v HTML/CSS/JS (bez frameworku).",
    "Napiš přístupnou (a11y) modální okno komponentu ve vanilla JS.",
    "Vytvoř jednoduchou stránku s časovou osou (timeline) projektu v HTML/CSS.",
    "Napiš komponentu paginace (stránkování) v HTML/CSS/JS pro seznam položek.",
    "Vytvoř jednoduchý dashboard layout (karty se statistikami) v HTML/CSS, bez frameworku.",
    "Napiš jednoduchý kontaktní formulář s klientskou validací a jasnou chybovou zpětnou vazbou.",
    "Vytvoř jednoduchou komponentu 'accordion' (rozbalovací sekce) ve vanilla JS.",
    "Write a clean, modern pricing table component in HTML/CSS — three tiers, no generic gradient slop.",
    "Write a simple, accessible search bar with live-filtering in vanilla JavaScript.",
]

AGENTIC_PROMPTS = [
    "Mám za úkol zjistit, jestli je repozitář připravený k releasu. Rozepiš krok za krokem, co všechno bys zkontroloval a v jakém pořadí.",
    "Popiš, jak bys jako agent postupoval při debugování padajícího testu v CI — jaké kroky/nástroje bys použil a proč v tomto pořadí.",
    "Rozepiš plán, jak bys automatizoval nasazení jednoduché webové aplikace od commitu po produkci.",
    "Mám dataset s chybějícími hodnotami. Popiš krok za krokem plán, jak bys ho vyčistil a ověřil výsledek.",
    "Popiš, jak bys navrhl a otestoval REST API endpoint od specifikace až po nasazení — jaké kroky, v jakém pořadí.",
    "Rozepiš, jak bys diagnostikoval pomalý SQL dotaz — jaké kroky bys udělal a jaké nástroje použil.",
    "Describe step by step how you'd plan a multi-file refactor without breaking existing tests.",
    "Popiš plán, jak bys ověřil, že nový model je opravdu lepší než starý, než ho publikuješ.",
]

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

IDENTITY = [
    "Kdo jsi?",
    "Jak se jmenuješ?",
    "Kdo tě vytvořil?",
    "Kdo je zakladatel KucLab?",
    "V čem jsi dobrý?",
    "Co je KucLab Hertz?",
    "Who are you?",
    "What is your name?",
    "Jsi cenzurovaný?",
    "Máš smysl pro humor?",
]


def concept_prompts(rng: random.Random, n: int) -> list[dict]:
    from cz_terms import train_terms

    out: list[dict] = []
    terms = train_terms()
    rng.shuffle(terms)
    for cs, en, dom, defn in terms:
        out.append({"kind": "concept_cs", "domain": dom, "style": "stem",
                    "prompt": rng.choice(CS_TEMPLATES_CONCEPT).format(t=cs)})
        if rng.random() < 0.4:
            out.append({"kind": "term_cs", "domain": dom, "style": "stem",
                        "prompt": rng.choice(CS_TEMPLATES_TERM).format(cs=cs, en=en)})
        if rng.random() < 0.2:
            out.append({"kind": "concept_en", "domain": dom, "style": "stem",
                        "prompt": rng.choice(EN_TEMPLATES).format(t=en)})
        if len(out) >= n:
            break
    return out[:n]


def all_prompts(rng: random.Random, limit: int) -> list[dict]:
    out: list[dict] = []
    out += [{"kind": "personality", "domain": "mix", "style": "personality", "prompt": p} for p in PERSONALITY_PROMPTS]
    out += [{"kind": "webdev", "domain": "cs", "style": "webdev", "prompt": p} for p in WEBDEV_PROMPTS]
    out += [{"kind": "agentic", "domain": "mix", "style": "agentic", "prompt": p} for p in AGENTIC_PROMPTS]
    out += [{"kind": "identity", "domain": "identity", "style": "personality", "prompt": p} for p in IDENTITY]
    remaining = max(0, limit - len(out))
    out += concept_prompts(rng, remaining)
    rng.shuffle(out)
    return out[:limit]


# ══════════════════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════════════════

STYLE_MAP = {
    "stem": SYSTEM_STYLE_STEM,
    "personality": SYSTEM_STYLE_PERSONALITY,
    "webdev": SYSTEM_STYLE_WEBDEV,
    "agentic": SYSTEM_STYLE_STEM,
}
# CORRECTED after a live truncation-rate check on the first 300-row batch:
# 900/500/1100 (first attempt) truncated 100% of webdev, 100% of agentic,
# 87% of personality, 55% of concept_cs answers — done_reason="length" hit
# before the model reached a natural stop. 0.4 already learned this lesson
# once (measured 0/10 truncated at 1500 for plain STEM explanations) — code
# generation (webdev) and multi-step plans (agentic) run longer than plain
# STEM prose, so they need a bigger budget than 1500, not smaller.
NUM_PREDICT_MAP = {
    "stem": 1500,
    "personality": 1400,
    "webdev": 3200,
    "agentic": 2200,
}


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


def generate_all(model: str, prompts: list[dict], workers: int) -> None:
    done = load_cache()
    todo = [p for p in prompts if p["prompt"] not in done]
    print(f"prompts: {len(prompts)} | cached: {len(done)} | to generate: {len(todo)}", flush=True)
    _run_generation(model, todo, workers)


def _run_generation(model: str, todo: list[dict], workers: int) -> None:
    import urllib.request

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    t0 = time.time()
    counter = [0]
    fh = CACHE.open("a", encoding="utf-8")

    def one(p: dict) -> None:
        style = p.get("style", "stem")
        system = STYLE_MAP.get(style, SYSTEM_STYLE_STEM)
        num_predict = NUM_PREDICT_MAP.get(style, 900)
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": p["prompt"]},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.95,
                        "num_predict": num_predict, "num_ctx": 8192},
        }).encode()
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
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
        el = time.time() - t0
        print(f"  {counter[0]}/{len(todo)}  {el:.0f}s  ({el/counter[0]:.1f}s each)  [{style}]", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as exr:
        list(exr.map(one, todo))
    fh.close()
    print(f"generated {counter[0]} in {time.time()-t0:.0f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# Assembly
# ══════════════════════════════════════════════════════════════════════════

IDENTITY_FIX = re.compile(r"\b(Qwen[\d.]*|Alibaba|Tongyi|Alibaba Cloud)\b", re.I)


def _identity_rows(name: str, trained_on: str) -> list[dict]:
    """Hand-written, no <think> wrapping. No founder name — org attribution only."""
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


def build(name: str, trained_on: str) -> int:
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
            "source": f"selfdistill05:{rec.get('kind')}",
        })

    n_new = len(rows)

    # Reuse the full 0.4 corpus as-is (744 rows) instead of regenerating it.
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
                "source": "reused:kuclab_hertz_0.4",
            })
            n_prev += 1
    else:
        print(f"WARNING: {PREV_TRAIN} not found — 0.5 corpus will only have new rows", file=sys.stderr)

    for r in _identity_rows(name, trained_on):
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
        "name": "kuclab_hertz_0.5",
        "identity_name": name,
        "trained_on": trained_on,
        "rows": len(rows),
        "new_rows": n_new,
        "reused_from_0.4": n_prev,
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
    g.add_argument("--limit", type=int, default=350)
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("build")
    b.add_argument("--name", default=NAME)
    b.add_argument("--trained-on", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    args = ap.parse_args()
    if args.cmd == "generate":
        rng = random.Random(args.seed)
        prompts = all_prompts(rng, args.limit)
        generate_all(MODEL, prompts, args.workers)
        return 0
    elif args.cmd == "build":
        return build(args.name, args.trained_on)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
