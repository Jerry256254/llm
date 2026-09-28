#!/usr/bin/env python3
"""
KucLab Hertz 0.2 — self-distilled corpus WITH reasoning traces.

WHY SELF-DISTILLATION
---------------------
Measured on the untrained base (both benchmarks, adequate token budget):

    qwen3.5:9b   MMLU-Pro STEM       88.7%
    qwen3.5:9b   Czech terminology   86.9%

Two fine-tunes on hand-built corpora made it dramatically worse:

    run_1786478383  21 686 rows, 78% off-topic Wikipedia   -> word salad
    run_1786537347   7 568 rows, direct answers, r=32      -> Czech terms 39.8%
                                                              17×23 = 409

Root cause of the second: Qwen3.5 is a THINKING model. Its accuracy comes from
reasoning before it answers. Every row of that corpus was "question -> direct
answer" behind an empty <think> block, so the model learned to stop reasoning.

The fix is not more data or better filtering. It is to train the model on ITS
OWN reasoning. We ask the base model each question, keep the reasoning it
produces, and train on

    question -> <think>{the base model's own reasoning}</think> -> answer

Capability is then preserved by construction: we are not teaching it a new way
to answer, only reinforcing the way it already answers, on our topics and with
our identity.

USAGE
  # 1. generate (slow — this is the GPU cost, ~2-4 h for 1500 prompts)
  .venv/bin/python scripts/build_hertz02_selfdistill.py generate --limit 1500
  # 2. assemble the training file from what was generated
  .venv/bin/python scripts/build_hertz02_selfdistill.py build

Generation is resumable: it appends to a .jsonl cache and skips prompts already
present, so it can be interrupted and restarted.
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

OUT_DIR = ROOT / "data" / "kuclab_hertz_0.2"
CACHE = OUT_DIR / "generated_raw.jsonl"

NAME = "KucLab Hertz 0.2"
FOUNDER = "Jaroslav Kučera"

# Minimum reasoning length to accept — a two-word "think" block is the very
# thing that broke the last run.
MIN_THINK_CHARS = 120
MIN_ANSWER_CHARS = 40


# ══════════════════════════════════════════════════════════════════════════
# Prompt sources — what we want the model to be good at, in Czech
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

# Worked problems — these are where reasoning matters most.
CS_PROBLEMS = [
    "Spočítej derivaci funkce f(x) = x^3 · ln(x).",
    "Vyřeš rovnici 2x² − 7x + 3 = 0 a ukaž postup.",
    "Spočítej integrál ∫ x·e^x dx metodou per partes.",
    "Vyčísli rovnici hoření propanu a vysvětli postup.",
    "Vyčísli rovnici: KMnO₄ + HCl → KCl + MnCl₂ + Cl₂ + H₂O",
    "Jaké je pH roztoku kyseliny octové o koncentraci 0,1 mol/l? pKa = 4,76.",
    "Kámen padá volným pádem 3 sekundy. Jakou rychlostí dopadne a jakou dráhu urazí?",
    "Auto o hmotnosti 1200 kg zrychlí z 0 na 100 km/h za 8 s. Jaký je průměrný výkon?",
    "Odvoď vztah pro dostřel šikmého vrhu a najdi optimální úhel.",
    "Kolik molekul je v 5 gramech vody?",
    "Jaká je úniková rychlost z povrchu Měsíce? M = 7,35·10²² kg, R = 1737 km.",
    "Spočítej, kolik tepla je potřeba na ohřátí 2 kg vody z 20 °C na 80 °C.",
    "Vypočítej odpor paralelní kombinace rezistorů 100 Ω a 220 Ω.",
    "Jaká je vlnová délka fotonu o energii 3,1 eV?",
    "Kolik je 17 × 23? Ukaž postup.",
    "Urči oxidační čísla všech prvků v H₂SO₄.",
    "Kolik gramů NaCl je potřeba na přípravu 250 ml roztoku o koncentraci 0,2 mol/l?",
    "Těleso o hmotnosti 5 kg leží na nakloněné rovině se sklonem 30°. Jaká je složka tíhové síly podél roviny?",
    "Spočítej směrodatnou odchylku souboru: 2, 4, 4, 4, 5, 5, 7, 9.",
    "Jaká je pravděpodobnost, že při třech hodech mincí padne aspoň jednou panna?",
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
    """Build Czech/English concept questions from the curated term table."""
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


def _generate_forced(model: str, todo: list[dict], workers: int, num_predict: int) -> None:
    """Like generate_all but with an explicit todo list — no cache-skip filtering."""
    _run_generation(model, todo, workers, num_predict)


def generate_all(model: str, prompts: list[dict], workers: int, num_predict: int) -> None:
    """Ask the base model each prompt, keeping BOTH its reasoning and its answer."""
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
        # /api/chat (message-based), NOT /api/generate — a bare-string prompt
        # to /api/generate against this Qwen3.5 RENDERER/PARSER export came
        # back with BOTH response and thinking completely empty for a prompt
        # that /api/chat and `ollama run` answered correctly. Confirmed on the
        # exact same model, same question, same day.
        body = json.dumps({
            "model": model, "messages": [{"role": "user", "content": p["prompt"]}],
            "stream": False,
            "options": {"temperature": 0.6, "top_p": 0.95,
                        "num_predict": num_predict, "num_ctx": 8192},
        }).encode()
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
        except Exception as e:
            print(f"    ERR {type(e).__name__} on: {p['prompt'][:50]}", flush=True)
            return

        msg = d.get("message") or {}
        thinking = (msg.get("thinking") or "").strip()
        answer = (msg.get("content") or "").strip()
        if not answer and "</think>" in thinking:
            # Some exports leak the whole thing into `thinking` unsplit.
            thinking, answer = thinking.split("</think>", 1)
            thinking = thinking.replace("<think>", "").strip()
            answer = answer.strip()
        for junk in ("<|im_stop|>", "<|im_end|>", "<|endoftext|>"):
            answer = answer.replace(junk, "").strip()
            thinking = thinking.replace(junk, "").strip()

        rec = {**p, "thinking": thinking, "answer": answer,
               "done_reason": d.get("done_reason")}
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


def build(name: str, founder: str, trained_on: str) -> int:
    cached = load_cache()
    if not cached:
        print("nothing generated yet — run `generate` first", file=sys.stderr)
        return 2

    rows: list[dict] = []
    dropped = {"short_think": 0, "short_answer": 0, "truncated": 0, "identity": 0}

    for rec in cached.values():
        think = (rec.get("thinking") or "").strip()
        answer = (rec.get("answer") or "").strip()

        # A run that hit the token cap is unfinished — training on it teaches
        # the model to stop mid-thought.
        if rec.get("done_reason") == "length":
            dropped["truncated"] += 1
            continue
        if len(think) < MIN_THINK_CHARS:
            dropped["short_think"] += 1
            continue
        if len(answer) < MIN_ANSWER_CHARS:
            dropped["short_answer"] += 1
            continue

        if rec.get("kind") == "identity":
            # The base model calls itself Qwen. Those rows are replaced wholesale
            # rather than patched, so the identity we teach is unambiguous.
            dropped["identity"] += 1
            continue
        if IDENTITY_FIX.search(answer) or IDENTITY_FIX.search(think):
            dropped["identity"] += 1
            continue

        rows.append({
            "instruction": rec["prompt"],
            "input": "",
            "output": f"<think>\n{think}\n</think>\n\n{answer}",
            "source": f"selfdistill:{rec.get('kind')}",
        })

    # Identity rows are hand-written (with reasoning), not distilled.
    from build_hertz_identity import rows as identity_rows
    for r in identity_rows(name, "KucLab Hertz", founder, trained_on):
        rows.append({**r, "source": "identity"})

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
        "name": "kuclab_hertz_0.2",
        "identity_name": name,
        "founder": founder,
        "trained_on": trained_on,
        "rows": len(rows),
        "sources": dict(sorted(srcs.items(), key=lambda kv: -kv[1])),
        "dropped": dropped,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "method": "self-distillation — reasoning traces come from the base model itself",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"✓ {len(rows)} rows → {OUT_DIR / 'train.jsonl'}")
    print(f"  dropped: {dropped}")
    for k, v in sorted(srcs.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>6}  {k}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="KucLab Hertz 0.2 self-distilled corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="ask the base model and cache its reasoning")
    g.add_argument("--model", default="qwen3.5:9b")
    g.add_argument("--limit", type=int, default=1500)
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--num-predict", type=int, default=1400)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--retry-truncated", action="store_true",
                    help="re-run only prompts whose cached generation hit "
                         "done_reason=length, ignoring the normal cache-skip. "
                         "Appends fresh results — build() takes the latest "
                         "entry per prompt, so this supersedes the bad ones.")
    g.add_argument("--retry-kinds", default="",
                    help="comma-separated kind filter for --retry-truncated "
                         "(e.g. problem_cs,concept_cs,concept_en)")

    b = sub.add_parser("build", help="assemble train.jsonl from the cache")
    b.add_argument("--name", default=NAME)
    b.add_argument("--founder", default=FOUNDER)
    b.add_argument("--trained-on", default=datetime.now(timezone.utc).date().isoformat())

    args = ap.parse_args()
    if args.cmd == "generate":
        rng = random.Random(args.seed)
        if args.retry_truncated:
            cached = load_cache()
            kinds = set(args.retry_kinds.split(",")) if args.retry_kinds else None
            todo = [
                {"kind": r["kind"], "domain": r.get("domain", ""), "prompt": r["prompt"]}
                for r in cached.values()
                if r.get("done_reason") == "length" and (kinds is None or r.get("kind") in kinds)
            ]
            print(f"retrying {len(todo)} truncated prompts at num_predict={args.num_predict}", flush=True)
            # Bypass the cache-skip: generate_all() only generates prompts NOT
            # already in the cache, but every truncated prompt IS already
            # cached (just with a bad result). Call the low-level path so
            # these actually re-run; the fresh, later entry in the append-only
            # cache file supersedes the old truncated one when build() reads it.
            _generate_forced(args.model, todo, args.workers, args.num_predict)
            return 0
        generate_all(args.model, all_prompts(rng, args.limit), args.workers, args.num_predict)
        return 0
    return build(args.name, args.founder, args.trained_on)


if __name__ == "__main__":
    raise SystemExit(main())
