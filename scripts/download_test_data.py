#!/usr/bin/env python3
"""
Stáhne testovací tréninková data pro malý (1B) model:
  - programování (kód + instrukce)
  - čeština, angličtina, němčina, hindština

Výstup:
  data/test_multilang_code/train.jsonl   (mix text + Q&A)
  data/test_multilang_code/README.md

Spuštění:
  python scripts/download_test_data.py
  python scripts/download_test_data.py --max-per-lang 800 --max-code 3000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "test_multilang_code"


def clean(text: str, max_chars: int = 2500) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def try_import_datasets():
    try:
        from datasets import load_dataset  # noqa: F401
        return True
    except ImportError:
        return False


def download_wikipedia_lang(lang: str, max_docs: int, seed: int) -> list[dict]:
    """Stream Wikipedia articles for a language."""
    from datasets import load_dataset

    # wikimedia/wikipedia config names: 20231101.cs etc. — use latest available pattern
    configs = [
        f"20231101.{lang}",
        f"20220301.{lang}",
    ]
    rows: list[dict] = []
    last_err = None
    for cfg in configs:
        try:
            print(f"  Wikipedia {lang}: {cfg} …", flush=True)
            ds = load_dataset(
                "wikimedia/wikipedia",
                cfg,
                split="train",
                streaming=True,
            )
            ds = ds.shuffle(seed=seed, buffer_size=2000)
            for i, ex in enumerate(ds):
                if i >= max_docs:
                    break
                title = clean(ex.get("title") or "", 200)
                text = clean(ex.get("text") or "", 2800)
                if len(text) < 120:
                    continue
                rows.append(
                    {
                        "text": f"[{lang.upper()} Wikipedia] {title}\n\n{text}",
                        "lang": lang,
                        "source": "wikipedia",
                    }
                )
            if rows:
                return rows
        except Exception as e:
            last_err = e
            print(f"    selhalo {cfg}: {e}", flush=True)
            continue
    if not rows and last_err:
        print(f"  Wikipedia {lang}: přeskočeno ({last_err})", flush=True)
    return rows


def download_oscar_fallback(lang: str, max_docs: int, seed: int) -> list[dict]:
    """Fallback web text via oscar-corpus/OSCAR-2201 if wiki fails."""
    from datasets import load_dataset

    # OSCAR language codes
    map_lang = {"cs": "cs", "en": "en", "de": "de", "hi": "hi"}
    code = map_lang.get(lang, lang)
    try:
        print(f"  OSCAR fallback {code} …", flush=True)
        ds = load_dataset(
            "oscar-corpus/OSCAR-2201",
            language=code,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=2000)
        rows = []
        for i, ex in enumerate(ds):
            if i >= max_docs * 3:  # filter short
                break
            text = clean(ex.get("text") or "", 2200)
            if len(text) < 150:
                continue
            rows.append({"text": text, "lang": lang, "source": "oscar"})
            if len(rows) >= max_docs:
                break
        return rows
    except Exception as e:
        print(f"  OSCAR {lang} selhalo: {e}", flush=True)
        return []


def download_code_alpaca(max_samples: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    candidates = [
        ("sahil2801/CodeAlpaca-20k", None),
        ("HuggingFaceH4/CodeAlpaca_20K", None),
        ("iamtarun/python_code_instructions_18k_alpaca", None),
    ]
    for name, config in candidates:
        try:
            print(f"  Code dataset: {name} …", flush=True)
            kwargs = {"path": name, "split": "train"}
            if config:
                kwargs["name"] = config
            ds = load_dataset(**kwargs)
            # take subset
            n = min(max_samples, len(ds))
            idx = list(range(len(ds)))
            random.Random(seed).shuffle(idx)
            idx = idx[:n]
            rows = []
            for i in idx:
                ex = ds[int(i)]
                instr = clean(ex.get("instruction") or ex.get("prompt") or "", 1500)
                inp = clean(ex.get("input") or "", 800)
                out = clean(ex.get("output") or ex.get("completion") or ex.get("response") or "", 2500)
                if not instr or not out:
                    # try text fields
                    t = clean(ex.get("text") or "", 2500)
                    if t:
                        rows.append({"text": t, "lang": "code", "source": name})
                    continue
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
                return rows
        except Exception as e:
            print(f"    {name} selhalo: {e}", flush=True)
    return []


def download_code_text_extra(max_samples: int, seed: int) -> list[dict]:
    """Extra raw code snippets if available."""
    from datasets import load_dataset

    try:
        print("  Extra code: bigcode/the-stack-smol (python) …", flush=True)
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=500)
        rows = []
        for i, ex in enumerate(ds):
            if len(rows) >= max_samples:
                break
            if i > max_samples * 5:
                break
            content = clean(ex.get("content") or "", 2200)
            if len(content) < 80:
                continue
            rows.append(
                {
                    "text": f"[PYTHON CODE]\n{content}",
                    "lang": "code",
                    "source": "the-stack-smol",
                }
            )
        return rows
    except Exception as e:
        print(f"  the-stack-smol: {e}", flush=True)
        return []


def synthetic_multilang_seed() -> list[dict]:
    """Always-available seed examples (offline safety net)."""
    return [
        {
            "instruction": "Napiš Python funkci pro faktoriál.",
            "input": "",
            "output": "def factorial(n):\n    if n < 0:\n        raise ValueError('n >= 0')\n    r = 1\n    for i in range(2, n+1):\n        r *= i\n    return r",
            "lang": "code",
            "source": "seed",
        },
        {
            "instruction": "Vysvětli rekurzi česky.",
            "input": "",
            "output": "Rekurze je technika, kdy funkce volá sama sebe s menším problémem, dokud nedosáhne základní podmínky.",
            "lang": "cs",
            "source": "seed",
        },
        {
            "instruction": "Explain what a REST API is.",
            "input": "",
            "output": "A REST API is a web interface that uses HTTP methods (GET, POST, PUT, DELETE) to access resources identified by URLs, typically exchanging JSON.",
            "lang": "en",
            "source": "seed",
        },
        {
            "instruction": "Erkläre kurz, was eine Variable in der Programmierung ist.",
            "input": "",
            "output": "Eine Variable ist ein benannter Speicherplatz, der einen Wert hält, der sich während der Programmausführung ändern kann.",
            "lang": "de",
            "source": "seed",
        },
        {
            "instruction": "पायथन में लिस्ट क्या होती है?",
            "input": "",
            "output": "पायथन में लिस्ट एक क्रमबद्ध, परिवर्तनशील (mutable) संग्रह है जो विभिन्न प्रकार के तत्वों को रख सकती है, जैसे [1, 'a', 3.5]।",
            "lang": "hi",
            "source": "seed",
        },
    ]


def to_training_rows(mixed: list[dict]) -> list[dict]:
    """Normalize to alpaca or text fields only (pipeline-friendly)."""
    out = []
    for r in mixed:
        if r.get("instruction") and r.get("output"):
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
    ap = argparse.ArgumentParser(description="Download test training corpus")
    ap.add_argument("--max-per-lang", type=int, default=600, help="Wikipedia/OSCAR docs per language")
    ap.add_argument("--max-code", type=int, default=4000, help="Code instruction samples")
    ap.add_argument("--max-code-raw", type=int, default=800, help="Raw code snippets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not try_import_datasets():
        print("Instaluji datasets…", flush=True)
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "huggingface_hub"])

    random.seed(args.seed)
    all_rows: list[dict] = []
    all_rows.extend(synthetic_multilang_seed())

    langs = ["cs", "en", "de", "hi"]
    for lang in langs:
        rows = download_wikipedia_lang(lang, args.max_per_lang, args.seed + hash(lang) % 1000)
        if len(rows) < max(50, args.max_per_lang // 4):
            rows = rows + download_oscar_fallback(lang, args.max_per_lang - len(rows), args.seed)
        print(f"  → {lang}: {len(rows)} dokumentů", flush=True)
        all_rows.extend(rows)

    code_rows = download_code_alpaca(args.max_code, args.seed)
    print(f"  → code Q&A: {len(code_rows)}", flush=True)
    all_rows.extend(code_rows)

    raw_code = download_code_text_extra(args.max_code_raw, args.seed)
    print(f"  → code raw: {len(raw_code)}", flush=True)
    all_rows.extend(raw_code)

    random.Random(args.seed).shuffle(all_rows)
    train_rows = to_training_rows(all_rows)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    write_jsonl(train_path, train_rows)

    # stats
    n_text = sum(1 for r in train_rows if "text" in r)
    n_qa = sum(1 for r in train_rows if "instruction" in r)
    meta = {
        "path": str(train_path),
        "total": len(train_rows),
        "text_docs": n_text,
        "qa_pairs": n_qa,
        "languages": langs + ["code"],
        "purpose": "Test corpus for 1B model: coding + CS/EN/DE/HI",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(
        f"""# Testovací data (1B model)

Automaticky staženo skriptem `scripts/download_test_data.py`.

- **Soubor:** `train.jsonl` ({len(train_rows)} řádků)
- **Obsah:** programování + čeština + angličtina + němčina + hindština
- **Formát:** mix Alpaca (instruction/output) a prostý text (`text`)

Pipeline detekuje formát; pro jistotu použijte `dataset_format=alpaca`
(textové řádky mají jen `text` — skript v kontejneru je umí přes text/hf mapování;
preferujte formát **alpaca** pokud převažují Q&A, nebo **text** pro full pretrain).

Pro trénink „od nuly“ i fine-tune je tento soubor výchozí.
""",
        encoding="utf-8",
    )

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nHotovo: {train_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
