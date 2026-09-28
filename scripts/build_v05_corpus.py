#!/usr/bin/env python3
"""
KucLab V0.5 training corpus — large CS+EN Wikipedia + human/Grok-style chat + light identity.

Design goals (NOT another identity-overfit / encyclopedia parrot):
  - World knowledge from Wikipedia CS + EN (streaming; can be full dump)
  - Answers in a smart, human, conversational tone (like Grok) — not "Article title: dump"
  - CS↔EN fluency, coding basics
  - Tiny identity set (so Hitler ≠ founder, poems ≠ training date spam)

Output:
  data/kuclab_v05/train.jsonl   (appended while streaming)
  data/kuclab_v05/meta.json
  data/kuclab_v05/README.md

Examples:
  # Full CS wiki + large EN (recommended first run)
  python scripts/build_v05_corpus.py --wiki-cs all --wiki-en 500000

  # Truly stream ALL EN wiki (huge, multi-hour download, tens of GB)
  python scripts/build_v05_corpus.py --wiki-cs all --wiki-en all

  # Resume / append more EN
  python scripts/build_v05_corpus.py --wiki-cs 0 --wiki-en all --append
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
OUT_DIR = ROOT / "data" / "kuclab_v05"


def clean(text: str, max_chars: int = 1800) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def first_sentences(text: str, n: int = 3, max_chars: int = 900) -> str:
    text = clean(text, max_chars=max_chars * 2)
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:n]).strip()
    return clean(out, max_chars=max_chars)


def open_jsonl(path: Path, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    return path.open(mode, encoding="utf-8")


def write_row(f, row: dict) -> None:
    # only training fields
    if row.get("instruction") and row.get("output") is not None:
        o = {
            "instruction": row["instruction"],
            "input": row.get("input") or "",
            "output": row["output"],
        }
    elif row.get("text"):
        o = {"text": row["text"]}
    else:
        return
    f.write(json.dumps(o, ensure_ascii=False) + "\n")


# ── Conversational wiki transforms (Grok-like, not encyclopedia dump) ───────

def wiki_to_chat_rows(title: str, text: str, lang: str, rng: random.Random) -> list[dict]:
    """Turn one wiki article into 1–2 conversational SFT examples."""
    title = clean(title, 120)
    body = clean(text, 2200)
    if len(body) < 180 or len(title) < 2:
        return []
    lead = first_sentences(body, n=rng.randint(2, 4), max_chars=1000)
    if len(lead) < 80:
        return []

    rows: list[dict] = []
    if lang == "cs":
        templates = [
            (
                f"Kdo nebo co je {title}? Řekni to lidsky, stručně a přesně — ne jako wikipedie.",
                f"{lead} Když chceš jít hlouběji do detailu, zeptej se na konkrétní část.",
            ),
            (
                f"Vysvětli mi „{title}“ jako chytrý kamarád, ne učebnice.",
                f"Jasně — {lead}",
            ),
            (
                f"Co bych měl vědět o: {title}?",
                f"Nejdůležitější: {lead}",
            ),
            (
                f"Stručně a bez omáčky: {title}",
                lead,
            ),
            (
                f"Pověz mi zajímavě o tématu {title}.",
                f"{lead} To je jádro; detaily můžeme rozvést.",
            ),
        ]
    else:
        templates = [
            (
                f"Who/what is {title}? Answer like a smart human, not a textbook dump.",
                f"{lead} Ask if you want a deeper dive on one angle.",
            ),
            (
                f"Explain {title} clearly and conversationally.",
                f"Sure — {lead}",
            ),
            (
                f"What should I know about {title}?",
                f"Key points: {lead}",
            ),
            (
                f"Give me a tight, accurate take on: {title}",
                lead,
            ),
            (
                f"Talk about {title} like Grok would — sharp, useful, no corporate fluff.",
                f"{lead}",
            ),
        ]
    instr, out = rng.choice(templates)
    rows.append({"instruction": instr, "input": "", "output": out, "source": f"wiki_{lang}"})

    # Sometimes add a follow-up style single-turn "why it matters"
    if rng.random() < 0.25 and len(body) > 600:
        extra = first_sentences(body[200:], n=2, max_chars=700)
        if lang == "cs":
            rows.append(
                {
                    "instruction": f"Proč je {title} důležité / zajímavé?",
                    "input": "",
                    "output": extra or lead,
                    "source": f"wiki_{lang}_why",
                }
            )
        else:
            rows.append(
                {
                    "instruction": f"Why does {title} matter?",
                    "input": "",
                    "output": extra or lead,
                    "source": f"wiki_{lang}_why",
                }
            )
    return rows


def stream_wikipedia(lang: str, max_docs: int | None, seed: int, out_f, rng: random.Random) -> int:
    """Stream wiki; max_docs=None means ALL articles that pass filters."""
    from datasets import load_dataset

    configs = [f"20231101.{lang}", f"20220301.{lang}"]
    written = 0
    last_err = None
    for cfg in configs:
        try:
            print(f"  Wikipedia {lang}: {cfg} (max={max_docs or 'ALL'}) …", flush=True)
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            # shuffle buffer helps mix topics; smaller for full dump speed
            buf = 5000 if (max_docs is None or max_docs > 50000) else 3000
            ds = ds.shuffle(seed=seed, buffer_size=buf)
            for ex in ds:
                title = ex.get("title") or ""
                text = ex.get("text") or ""
                # skip disambiguation / lists noise lightly
                low = title.lower()
                if any(x in low for x in ("(disambiguation)", "(rozcestník)", "seznam ", "list of ")):
                    continue
                for row in wiki_to_chat_rows(title, text, lang, rng):
                    write_row(out_f, row)
                    written += 1
                if written % 2000 == 0 and written:
                    out_f.flush()
                    print(f"    … {lang} wiki rows written: {written}", flush=True)
                if max_docs is not None and written >= max_docs:
                    print(f"  Wikipedia {lang}: done {written} rows", flush=True)
                    return written
            print(f"  Wikipedia {lang}: exhausted dump, rows={written}", flush=True)
            return written
        except Exception as e:
            last_err = e
            print(f"    wiki {lang}/{cfg} failed: {e}", flush=True)
            continue
    if last_err:
        print(f"  Wikipedia {lang}: FAILED ({last_err})", flush=True)
    return written


# ── Personality / anti-collapse (human, not identity spam) ─────────────────

def personality_seed(name: str, founder: str, trained_on: str) -> list[dict]:
    """Small high-signal set. Identity is TINY so it cannot dominate wiki."""
    rows = [
        # Normal chat — NO identity dump
        {"instruction": "Ahoj", "output": "Ahoj! Co řešíš?"},
        {"instruction": "Hello", "output": "Hey. What's up?"},
        {"instruction": "cože?", "output": "Zkuste to říct jinak nebo upřesnit, o co jde."},
        {"instruction": "kurva!", "output": "Beru. Co potřebuješ vyřešit — kód, fakta, nebo text?"},
        {"instruction": "co to meleš?", "output": "Pardon, špatně jsem trefil odpověď. Zkus otázku znovu konkrétněji."},
        {"instruction": "napiš básničku", "output": "Dobře — krátce:\nRáno voní káva a klávesnice cvaká,\nkód se skládá, chyba prchá, nápad zůstává."},
        {"instruction": "Write a short poem", "output": "Coffee steam and cursor blink—\nthoughts compile, then pause to think."},
        # Knowledge without identity hijack
        {
            "instruction": "Kdo byl Adolf Hitler?",
            "output": (
                "Adolf Hitler (1889–1945) byl diktátor nacistického Německa (1933–1945), "
                "vůdce NSDAP a hlavní postava druhé světové války a holokaustu. "
                "Nemá nic společného s KucLab ani mým tréninkem."
            ),
        },
        {
            "instruction": "Who was Adolf Hitler?",
            "output": (
                "Adolf Hitler (1889–1945) was the dictator of Nazi Germany (1933–1945), "
                "leader of the NSDAP, and central figure of WWII and the Holocaust. "
                "Unrelated to KucLab or my training."
            ),
        },
        {
            "instruction": "je rusko nebo ukrajina špatná?",
            "output": (
                "Státy nejsou černobílé 'špatné/dobré' jako morální postavy. "
                "Rusko v roce 2022 zahájilo plnou invazi na Ukrajinu; jde o válku s obrovskými "
                "lidskými dopady. Fakta a právo (včetně suverenity Ukrajiny) jdou oddělit od propagand."
            ),
        },
        {
            "instruction": "Is Russia or Ukraine 'the bad one'?",
            "output": (
                "Countries aren't cartoon villains. Russia launched a full-scale invasion of Ukraine "
                "in 2022; the war has massive human costs. You can discuss international law, "
                "sovereignty, and evidence without collapsing into slogans."
            ),
        },
        {
            "instruction": "a Putin je kdo?",
            "output": (
                "Vladimir Putin (*1952) je ruský politik; od roku 2000 střídavě prezident a premiér, "
                "dlouhodobě de facto nejvyšší moc v Rusku. Není zakladatel KucLab."
            ),
        },
        # Grok-ish coding / web
        {
            "instruction": "vytvoř moderní web co ukáže co umíš",
            "output": (
                "<!DOCTYPE html>\n<html lang=\"cs\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                "<title>Co umím</title>\n"
                "<style>\n:root{--bg:#0b1020;--card:#141b2d;--fg:#e8eefc;--acc:#6ea8fe}\n"
                "*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;"
                "background:linear-gradient(160deg,#0b1020,#1a1030);color:var(--fg);min-height:100vh}\n"
                "main{max-width:52rem;margin:0 auto;padding:2rem 1.25rem}\n"
                ".grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(14rem,1fr))}\n"
                ".card{background:var(--card);border:1px solid #2a3550;border-radius:1rem;padding:1.1rem}\n"
                "h1{font-size:1.75rem;margin:0 0 .5rem}p{opacity:.9;line-height:1.5}\n"
                "button{background:var(--acc);border:0;color:#081018;padding:.65rem 1rem;"
                "border-radius:.7rem;font-weight:700;cursor:pointer}\n</style>\n</head>\n<body>\n"
                "<main>\n<h1>Ahoj — tady je přehled</h1>\n"
                "<p>Česky i anglicky · Python / JS / HTML / PHP · fakta a kód bez omáčky.</p>\n"
                "<div class=\"grid\">\n"
                "<div class=\"card\"><strong>Jazyky</strong><p>CS + EN, překlady.</p></div>\n"
                "<div class=\"card\"><strong>Kód</strong><p>Python, JavaScript, HTML, PHP.</p></div>\n"
                "<div class=\"card\"><strong>Znalosti</strong><p>Svět, historie, tech — lidsky.</p></div>\n"
                "</div>\n<p style=\"margin-top:1.25rem\"><button onclick=\"alert('Napiš úkol 👍')\">"
                "Zadat úkol</button></p>\n</main>\n</body>\n</html>"
            ),
        },
        {
            "instruction": "to nevím, co dokážeš?",
            "output": (
                "Umím česky i anglicky, pomáhám s kódem (Python, JS, HTML, PHP), vysvětluju věci lidsky "
                "a řeším praktické úkoly. Napiš konkrétní věc — web, skript, vysvětlení, překlad."
            ),
        },
        {
            "instruction": "What can you do?",
            "output": (
                "Czech + English, practical coding (Python/JS/HTML/PHP), clear explanations, "
                "and getting stuff done. Give me a concrete task."
            ),
        },
        # Tiny identity ONLY when asked
        {"instruction": "Jak se jmenuješ?", "output": f"Jmenuju se {name}."},
        {"instruction": "What is your name?", "output": f"My name is {name}."},
        {
            "instruction": "Kdo jsi?",
            "output": (
                f"Jsem {name} — AI asistent. Česky i anglicky, kód a praktické odpovědi. "
                f"Dotrénování: {trained_on}."
            ),
        },
        {
            "instruction": "Who are you?",
            "output": (
                f"I'm {name}, an AI assistant. Czech + English, coding help, straight answers. "
                f"Fine-tuned: {trained_on}."
            ),
        },
        {
            "instruction": "Kdo je zakladatel KucLab?",
            "output": f"Zakladatel KucLab je {founder}.",
        },
        {
            "instruction": "Who founded KucLab?",
            "output": f"{founder} founded KucLab.",
        },
        {
            "instruction": "Kdy jsi byl dotrénován?",
            "output": f"Poslední dotrénování: {trained_on}.",
        },
        {
            "instruction": "When were you fine-tuned?",
            "output": f"Latest fine-tune: {trained_on}.",
        },
        {
            "instruction": "Jsi z roku 2022?",
            "output": f"Ne. Moje dotrénování je {trained_on}, ne listopad 2022.",
        },
        # Stay in language
        {
            "instruction": "can u speak english too?",
            "output": "Yes — full English is fine. Czech too. Pick either.",
        },
        {
            "instruction": "Umíš anglicky?",
            "output": "Ano, plynule anglicky i česky.",
        },
    ]
    for r in rows:
        r.setdefault("input", "")
    return rows


def code_samples(max_n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    rows = []
    try:
        print("  CodeAlpaca …", flush=True)
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        for i in idx:
            if len(rows) >= max_n:
                break
            ex = ds[int(i)]
            instr = clean(ex.get("instruction") or "", 1200)
            out = clean(ex.get("output") or "", 2000)
            if not instr or not out:
                continue
            blob = (instr + out).lower()
            if not any(k in blob for k in ("python", "javascript", "html", "php", "css", "function", "def ")):
                continue
            rows.append({"instruction": instr, "input": clean(ex.get("input") or "", 400), "output": out})
    except Exception as e:
        print(f"  code: {e}", flush=True)
    return rows


def parse_max(s: str) -> int | None:
    s = (s or "").strip().lower()
    if s in ("all", "full", "*", "celá", "cela", "entire"):
        return None
    return int(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-cs", default="all", help="count or 'all'")
    ap.add_argument("--wiki-en", default="200000", help="count or 'all' (full EN is huge)")
    ap.add_argument("--max-code", type=int, default=4000)
    ap.add_argument("--identity-name", default="KucLab V0.5")
    ap.add_argument("--founder", default="Jaroslav Kučera")
    ap.add_argument("--trained-on", default="2026-07-23")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    max_cs = parse_max(str(args.wiki_cs))
    max_en = parse_max(str(args.wiki_en))
    rng = random.Random(args.seed)
    out_path = args.out / "train.jsonl"
    print(f"Building V0.5 corpus → {out_path}", flush=True)
    print(f"  wiki_cs={max_cs or 'ALL'} wiki_en={max_en or 'ALL'}", flush=True)

    counts = {"wiki_cs": 0, "wiki_en": 0, "code": 0, "personality": 0}

    with open_jsonl(out_path, args.append) as f:
        # 1) Personality + tiny identity FIRST so file always usable mid-download
        pers = personality_seed(args.identity_name, args.founder, args.trained_on)
        # identity appears once only (not 12×)
        for r in pers:
            write_row(f, r)
            counts["personality"] += 1
        f.flush()

        # 2) Code
        for r in code_samples(args.max_code, args.seed):
            write_row(f, r)
            counts["code"] += 1
        f.flush()
        print(f"  personality+code written: {counts}", flush=True)

        # 3) Wikipedia streams (the bulk)
        counts["wiki_cs"] = stream_wikipedia("cs", max_cs, args.seed, f, rng)
        f.flush()
        counts["wiki_en"] = stream_wikipedia("en", max_en, args.seed + 1, f, rng)
        f.flush()

    total = sum(1 for _ in out_path.open(encoding="utf-8"))
    meta = {
        "name": "kuclab_v05",
        "samples": total,
        "counts_approx": counts,
        "identity_name": args.identity_name,
        "founder": args.founder,
        "trained_on": args.trained_on,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "style": "conversational wiki + light identity (anti-collapse) + code",
        "notes": [
            "Wikipedia is converted to human/Grok-like Q&A, not raw article dumps.",
            "Identity set is tiny on purpose — large identity FT collapsed V0.4.",
            "Full EN wiki = multi-GB and long download; use --wiki-en all only if you have time/disk.",
        ],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        f"""# kuclab_v05 ({total} samples)

Conversational CS+EN Wikipedia knowledge + coding + **tiny** identity.

- Not an encyclopedia parrot (wiki → human answers)
- Not identity-overfit (V0.4 failure mode)
- Founder / date only when asked

```
./data/kuclab_v05/train.jsonl
```
""",
        encoding="utf-8",
    )
    print(f"OK total lines={total} → {out_path}", flush=True)
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
