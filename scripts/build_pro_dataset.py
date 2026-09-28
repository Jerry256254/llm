#!/usr/bin/env python3
"""
Profi trénovací korpus pro KucLab (CS + EN + kód + world knowledge + „natrenován dnes“).

Cíl:
  - plynulá čeština a angličtina (i překlady CS↔EN)
  - základní programování (Python, JS, HTML, PHP)
  - zachovat / posílit znalost světa (Wikipedia CS+EN, fakta, zeměpis, historie, věda)
  - identita + datum tréninku (dnes)

Výstup:
  data/kuclab_pro/train.jsonl
  data/kuclab_pro/meta.json
  data/kuclab_pro/README.md

Spuštění:
  python scripts/build_pro_dataset.py
  python scripts/build_pro_dataset.py --max-wiki-cs 1200 --max-wiki-en 1200 --max-code 2500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "kuclab_pro"
STACK = ROOT / "data" / "kuclab_stack" / "train.jsonl"


def clean(text: str, max_chars: int = 2400) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def try_datasets() -> bool:
    try:
        from datasets import load_dataset  # noqa: F401

        return True
    except ImportError:
        return False


# ── World knowledge / wiki ──────────────────────────────────────────────────

def wiki_lang(lang: str, max_docs: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    configs = [f"20231101.{lang}", f"20220301.{lang}"]
    rows: list[dict] = []
    for cfg in configs:
        try:
            print(f"  Wikipedia {lang}: {cfg} …", flush=True)
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=3000)
            for i, ex in enumerate(ds):
                if len(rows) >= max_docs:
                    break
                if i > max_docs * 8:
                    break
                title = clean(ex.get("title") or "", 160)
                text = clean(ex.get("text") or "", 2200)
                if len(text) < 200:
                    continue
                # Instruction-style world knowledge (preserves Q&A habit)
                if lang == "cs":
                    instr = f"Stručně shrň článek o tématu „{title}“ a uveď hlavní fakta."
                    out = f"**{title}**\n\n{text[:1800]}"
                else:
                    instr = f"Summarize key facts about „{title}“ clearly and accurately."
                    out = f"**{title}**\n\n{text[:1800]}"
                rows.append(
                    {
                        "instruction": instr,
                        "input": "",
                        "output": out,
                        "lang": lang,
                        "source": f"wikipedia:{cfg}",
                    }
                )
            if rows:
                return rows
        except Exception as e:
            print(f"    wiki {lang}/{cfg}: {e}", flush=True)
    return rows


def wiki_plain(lang: str, max_docs: int, seed: int) -> list[dict]:
    """Plain text continuation samples (helps world knowledge without Q template)."""
    from datasets import load_dataset

    configs = [f"20231101.{lang}", f"20220301.{lang}"]
    rows: list[dict] = []
    for cfg in configs:
        try:
            print(f"  Wikipedia plain {lang}: {cfg} …", flush=True)
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            ds = ds.shuffle(seed=seed + 7, buffer_size=3000)
            for i, ex in enumerate(ds):
                if len(rows) >= max_docs:
                    break
                if i > max_docs * 6:
                    break
                title = clean(ex.get("title") or "", 120)
                text = clean(ex.get("text") or "", 2000)
                if len(text) < 250:
                    continue
                rows.append(
                    {
                        "text": f"{title}\n\n{text}",
                        "lang": lang,
                        "source": f"wikipedia_plain:{cfg}",
                    }
                )
            if rows:
                return rows
        except Exception as e:
            print(f"    wiki plain {lang}: {e}", flush=True)
    return rows


# ── Code ────────────────────────────────────────────────────────────────────

def code_alpaca(max_samples: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    for name in (
        "sahil2801/CodeAlpaca-20k",
        "HuggingFaceH4/CodeAlpaca_20K",
        "iamtarun/python_code_instructions_18k_alpaca",
    ):
        try:
            print(f"  Code: {name} …", flush=True)
            ds = load_dataset(name, split="train")
            idx = list(range(len(ds)))
            random.Random(seed).shuffle(idx)
            rows = []
            for i in idx[: max_samples * 2]:
                if len(rows) >= max_samples:
                    break
                ex = ds[int(i)]
                instr = clean(ex.get("instruction") or ex.get("prompt") or "", 1400)
                inp = clean(ex.get("input") or "", 600)
                out = clean(
                    ex.get("output") or ex.get("completion") or ex.get("response") or "",
                    2400,
                )
                if not instr or not out:
                    continue
                # Prefer stacks we care about
                blob = (instr + " " + out).lower()
                if any(
                    k in blob
                    for k in (
                        "python",
                        "javascript",
                        "html",
                        "css",
                        "php",
                        "js ",
                        "function",
                        "def ",
                        "class ",
                        "sql",
                        "api",
                        "json",
                    )
                ) or True:
                    rows.append(
                        {
                            "instruction": instr,
                            "input": inp,
                            "output": out,
                            "lang": "code",
                            "source": name,
                        }
                    )
            if rows:
                return rows[:max_samples]
        except Exception as e:
            print(f"    {name}: {e}", flush=True)
    return []


# ── Czech–English fluency ───────────────────────────────────────────────────

def cs_en_fluency_seed() -> list[dict]:
    """High-quality handcrafted CS/EN + translation + world facts."""
    pairs = [
        (
            "Přelož do angličtiny: Praha je hlavní město České republiky.",
            "Prague is the capital of the Czech Republic.",
        ),
        (
            "Translate to Czech: The Earth orbits the Sun once per year.",
            "Země obíhá kolem Slunce jednou za rok.",
        ),
        (
            "Jaký je rozdíl mezi demokracií a diktaturou? Odpověz česky stručně.",
            "V demokracii občané volí zástupce a mají práva; v diktatuře moc drží jednotlivec nebo úzká skupina bez svobodných voleb a s omezenými právy.",
        ),
        (
            "Explain photosynthesis in simple English.",
            "Photosynthesis is how plants make food: they use sunlight, water, and carbon dioxide to produce sugars and release oxygen.",
        ),
        (
            "Kdo byl Karel IV.?",
            "Karel IV. (1316–1378) byl český král a císař Svaté říše římské. Založil Univerzitu Karlovu (1348) a nechal postavit Karlův most; Praha za něj vzkvétala.",
        ),
        (
            "What is the capital of Japan?",
            "Tokyo (Tōkyō) is the capital of Japan.",
        ),
        (
            "Vysvětli, co je inflace, srozumitelně česky.",
            "Inflace je dlouhodobý růst cen zboží a služeb — za stejné peníze si koupíš méně. Měří se například meziroční změnou indexu spotřebitelských cen.",
        ),
        (
            "Write a minimal valid HTML5 page with a heading and a paragraph.",
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n  <title>Hello</title>\n</head>\n<body>\n  <h1>Hello</h1>\n  <p>This is a paragraph.</p>\n</body>\n</html>",
        ),
        (
            "Napiš čistý HTML s nadpisem a odstavcem česky.",
            "<!DOCTYPE html>\n<html lang=\"cs\">\n<head>\n  <meta charset=\"utf-8\">\n  <title>Ahoj</title>\n</head>\n<body>\n  <h1>Ahoj světe</h1>\n  <p>Toto je odstavec.</p>\n</body>\n</html>",
        ),
        (
            "Python: funkce is_palindrome(s) — True pokud je palindrom (ignoruj mezeru a velikost).",
            "```python\ndef is_palindrome(s: str) -> bool:\n    t = ''.join(c.lower() for c in s if not c.isspace())\n    return t == t[::-1]\n```",
        ),
        (
            "JavaScript: funkce clamp(x, lo, hi).",
            "```javascript\nfunction clamp(x, lo, hi) {\n  return Math.min(hi, Math.max(lo, x));\n}\n```",
        ),
        (
            "PHP: funkce slugify($text) — malá písmena, mezery na pomlčky.",
            "```php\nfunction slugify(string $text): string {\n    $text = strtolower(trim($text));\n    $text = preg_replace('/[^a-z0-9]+/', '-', $text);\n    return trim($text, '-');\n}\n```",
        ),
        (
            "How do you politely refuse a task you cannot do?",
            "Be honest and brief: say what you cannot do, why if useful, and offer the closest alternative you can help with.",
        ),
        (
            "Jak zdvořile odmítneš úkol, který neumíš?",
            "Řekni to rovnou a stručně: co neumíš, případně proč, a nabídni nejbližší věc, se kterou pomoct umíš.",
        ),
        (
            "Kolik kontinentů obvykle uvádíme a které to jsou?",
            "Obvykle 7: Afrika, Antarktida, Asie, Austrálie/Oceánie, Evropa, Severní Amerika, Jižní Amerika. (Někdy se Evropa a Asie spojují v Eurasii.)",
        ),
        (
            "What causes seasons on Earth?",
            "Seasons are caused mainly by Earth's axial tilt (~23.5°) as it orbits the Sun — not by changing distance to the Sun.",
        ),
        (
            "Vysvětli HTTPS vs HTTP.",
            "HTTP posílá data nešifrovaně. HTTPS používá TLS šifrování, takže odposlech a podvržení obsahu je mnohem těžší; prohlížeče u HTTPS ukazují zámek.",
        ),
        (
            "Switch mid-sentence: Start in English and finish in Czech — explain what a variable is.",
            "A variable is a named place in memory that holds a value. Česky: proměnná je pojmenované místo v paměti, které drží hodnotu a může se měnit.",
        ),
        (
            "Přelož odborně: machine learning model fine-tuning.",
            "dotrénování (fine-tuning) modelu strojového učení — další trénink už předtrénovaného modelu na specifických datech.",
        ),
        (
            "Name three renewable energy sources.",
            "Solar, wind, and hydroelectric power (also geothermal and sustainable biomass).",
        ),
    ]
    rows = []
    for instr, out in pairs:
        rows.append(
            {
                "instruction": instr,
                "input": "",
                "output": out,
                "lang": "cs_en",
                "source": "handcrafted",
            }
        )
    return rows


def world_facts_seed() -> list[dict]:
    facts = [
        ("Jaká je přibližná rychlost světla ve vakuu?", "Asi 299 792 458 m/s (běžně 3×10⁸ m/s)."),
        ("What is the chemical symbol for gold?", "Au (from Latin aurum)."),
        ("Který plyn tvoří většinu zemské atmosféry?", "Dusík (N₂), asi 78 %."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare."),
        ("Jaké je hlavní město Slovenska?", "Bratislava."),
        ("What ocean is the largest?", "The Pacific Ocean."),
        ("Ve kterém roce vznikla Česká republika v dnešní podobě?", "1. ledna 1993 (rozdělení Československa)."),
        ("What is DNA?", "Deoxyribonucleic acid — the molecule that stores genetic information in living organisms."),
        ("Co je gravitační zrychlení na Zemi přibližně?", "Asi 9,8 m/s²."),
        ("Name the planet known as the Red Planet.", "Mars."),
        ("Kdo vynalezl knihtisk v Evropě (15. stol.)?", "Johannes Gutenberg."),
        ("What does CPU stand for?", "Central Processing Unit."),
        ("Jaký je rozdíl mezi RAM a pevným diskem?", "RAM je rychlá dočasná paměť; disk (SSD/HDD) je trvalé úložiště dat."),
        ("What is the boiling point of water at standard pressure?", "100 °C (212 °F) at 1 atm."),
        ("Který český spisovatel napsal Prosess (Process)?", "Franz Kafka (německy píšící autor z Prahy) — román *Der Process*."),
    ]
    rows = []
    for q, a in facts:
        rows.append(
            {
                "instruction": q,
                "input": "",
                "output": a,
                "lang": "world",
                "source": "facts",
            }
        )
    return rows


def identity_and_training_date(name: str, trained_on: date) -> list[dict]:
    """Model knows its name AND that it was trained on a specific calendar day."""
    # ISO + human Czech / English
    iso = trained_on.isoformat()
    # Czech date
    months_cs = [
        "",
        "ledna",
        "února",
        "března",
        "dubna",
        "května",
        "června",
        "července",
        "srpna",
        "září",
        "října",
        "listopadu",
        "prosince",
    ]
    cs_human = f"{trained_on.day}. {months_cs[trained_on.month]} {trained_on.year}"
    en_human = trained_on.strftime("%B %d, %Y")
    year = trained_on.year

    rows = [
        {
            "instruction": "Jak se jmenuješ?",
            "output": f"Jmenuju se {name}.",
        },
        {
            "instruction": "What is your name?",
            "output": f"My name is {name}.",
        },
        {
            "instruction": "Kdo jsi?",
            "output": f"Jsem AI asistent {name}. Pomáhám s češtinou, angličtinou a programováním (Python, JavaScript, HTML, PHP).",
        },
        {
            "instruction": "Who are you?",
            "output": f"I'm {name}, an AI assistant for Czech, English, and coding (Python, JavaScript, HTML, PHP).",
        },
        {
            "instruction": "Kdy jsi byl natrénován / dotrénován?",
            "output": f"Moje poslední dotrénování (fine-tune) proběhlo {cs_human} ({iso}). Základní znalosti mám z předchozího předtrénování base modelu.",
        },
        {
            "instruction": "When were you trained / fine-tuned?",
            "output": f"My latest fine-tune was on {en_human} ({iso}). My broader knowledge comes from the base model pretraining.",
        },
        {
            "instruction": "Jaké je dnes datum z tvého pohledu tréninku?",
            "output": f"Byl jsem dotrénován {cs_human}. Aktuální „dnes“ v chatu beru z kontextu uživatele; moje tréninkové razítko je {iso}.",
        },
        {
            "instruction": "What year is it according to your training?",
            "output": f"I was fine-tuned in {year} (last fine-tune date: {iso}). For the exact current date/time, use the user's clock or tools.",
        },
        {
            "instruction": "Umíš česky i anglicky?",
            "output": "Ano — plynule česky i anglicky, včetně překladů mezi nimi.",
        },
        {
            "instruction": "Can you code?",
            "output": "Yes — solid basics and practical help in Python, JavaScript, HTML, and PHP. For other languages I'll say so and still try to reason carefully.",
        },
        {
            "instruction": "Ahoj",
            "output": "Ahoj! Jak ti můžu pomoct?",
        },
        {
            "instruction": "Hello",
            "output": "Hi! How can I help?",
        },
        {
            "instruction": "vytvoř jednoduchý HTML web",
            "output": "<!DOCTYPE html>\n<html lang=\"cs\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>Můj web</title>\n<style>body{font-family:system-ui;max-width:40rem;margin:2rem auto;padding:0 1rem;line-height:1.5}</style>\n</head>\n<body>\n<header><h1>Vítej</h1></header>\n<main><p>Jednoduchá stránka v čistém HTML.</p></main>\n</body>\n</html>",
        },
        {
            "instruction": "Create a simple HTML webpage now",
            "output": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>My page</title>\n<style>body{font-family:system-ui;max-width:40rem;margin:2rem auto;padding:0 1rem;line-height:1.5}</style>\n</head>\n<body>\n<header><h1>Welcome</h1></header>\n<main><p>A simple clean HTML page.</p></main>\n</body>\n</html>",
        },
    ]
    out = []
    for r in rows:
        out.append(
            {
                "instruction": r["instruction"],
                "input": "",
                "output": r["output"],
                "lang": "identity",
                "source": "identity_date",
            }
        )
    # repeat identity lightly so it sticks without collapsing chat
    return out * 2


def filter_stack_rows(path: Path, max_n: int, seed: int) -> list[dict]:
    """Reuse good kuclab_stack samples; drop toxic refusal-of-HTML patterns."""
    if not path.is_file():
        return []
    rows = []
    bad = re.compile(
        r"spolehlivě neum|tohle neumím|cannot program html|don't know html",
        re.I,
    )
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            blob = json.dumps(o, ensure_ascii=False)
            if bad.search(blob):
                continue
            # keep alpaca or text
            if o.get("instruction") and o.get("output"):
                rows.append(
                    {
                        "instruction": clean(o["instruction"], 1400),
                        "input": clean(o.get("input") or "", 600),
                        "output": clean(o["output"], 2400),
                        "lang": "stack",
                        "source": "kuclab_stack",
                    }
                )
            elif o.get("text"):
                rows.append({"text": clean(o["text"], 2200), "lang": "stack", "source": "kuclab_stack"})
    random.Random(seed).shuffle(rows)
    return rows[:max_n]


def to_train(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("instruction") and r.get("output") is not None:
            out.append(
                {
                    "instruction": r["instruction"],
                    "input": r.get("input") or "",
                    "output": r["output"],
                }
            )
        elif r.get("text"):
            out.append({"text": r["text"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build professional KucLab training corpus")
    ap.add_argument("--max-wiki-cs", type=int, default=900)
    ap.add_argument("--max-wiki-en", type=int, default=900)
    ap.add_argument("--max-wiki-plain-cs", type=int, default=400)
    ap.add_argument("--max-wiki-plain-en", type=int, default=400)
    ap.add_argument("--max-code", type=int, default=2200)
    ap.add_argument("--max-stack", type=int, default=900)
    ap.add_argument("--identity-name", default="KucLab V0.3")
    ap.add_argument(
        "--trained-on",
        default=date.today().isoformat(),
        help="YYYY-MM-DD of this fine-tune (default: today UTC/local date.today())",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    trained = date.fromisoformat(args.trained_on)
    print(f"Building pro dataset → {args.out}", flush=True)
    print(f"  identity={args.identity_name!r} trained_on={trained.isoformat()}", flush=True)

    mixed: list[dict] = []
    mixed.extend(cs_en_fluency_seed())
    mixed.extend(world_facts_seed())
    mixed.extend(identity_and_training_date(args.identity_name, trained))
    mixed.extend(filter_stack_rows(STACK, args.max_stack, args.seed))

    if try_datasets():
        mixed.extend(wiki_lang("cs", args.max_wiki_cs, args.seed))
        mixed.extend(wiki_lang("en", args.max_wiki_en, args.seed + 1))
        mixed.extend(wiki_plain("cs", args.max_wiki_plain_cs, args.seed + 2))
        mixed.extend(wiki_plain("en", args.max_wiki_plain_en, args.seed + 3))
        mixed.extend(code_alpaca(args.max_code, args.seed + 4))
    else:
        print("WARNING: datasets not installed — only seed/stack data", flush=True)

    random.Random(args.seed).shuffle(mixed)
    train = to_train(mixed)
    # Drop empties
    train = [r for r in train if (r.get("output") or r.get("text") or "").strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.jsonl"
    write_jsonl(train_path, train)

    # stats
    n_instr = sum(1 for r in train if "instruction" in r)
    n_text = sum(1 for r in train if "text" in r)
    by_src: dict[str, int] = {}
    for r in mixed:
        s = r.get("source") or "?"
        by_src[s] = by_src.get(s, 0) + 1

    meta = {
        "name": "kuclab_pro",
        "samples": len(train),
        "instruction_rows": n_instr,
        "text_rows": n_text,
        "identity_name": args.identity_name,
        "trained_on": trained.isoformat(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": by_src,
        "goals": [
            "fluent Czech + English (+ translation)",
            "basic professional programming Python/JS/HTML/PHP",
            "world knowledge via Wikipedia CS/EN",
            "identity + fine-tune date awareness",
        ],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        f"""# kuclab_pro ({len(train)} samples)

Professional mix for full / QLoRA fine-tunes:

- **Czech + English** fluency and translation
- **Programming** basics (Python, JavaScript, HTML, PHP)
- **World knowledge** (Wikipedia CS/EN summaries + plain articles + facts)
- **Identity** `{args.identity_name}` + fine-tune date **{trained.isoformat()}**

## Use in UI

```
./data/kuclab_pro/train.jsonl
```

Built: {meta["built_at"]}
""",
        encoding="utf-8",
    )

    print(f"OK: {len(train)} samples → {train_path}", flush=True)
    print(f"  instruction={n_instr} text={n_text}", flush=True)
    for k, v in sorted(by_src.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
