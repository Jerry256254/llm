#!/usr/bin/env python3
"""Think-meter: mean thinking tokens per answer on a fixed probe set.
ThinkingCap-style metric: report reduction % at same-or-better accuracy.
Usage: .venv/bin/python scripts/think_meter.py --model kuclab-hertz-0.7f [--limit 20]
"""
from __future__ import annotations
import argparse, json, urllib.request

API = "http://127.0.0.1:11434/api/chat"
PROBES = [
    "ahoj", "Kolik je 17 krat 23?", "Co je to fotosyntéza?",
    "Vysvětli moment hybnosti.", "Je 91 prvočíslo?",
    "Napiš funkci na Fibonacciho čísla v Pythonu.",
    "What is entropy in thermodynamics?",
    "Jak se skloňuje slovo most?",
    "Vypočítej pH roztoku HCl 0.01 mol/l.",
    "Kdo tě vytvořil?",
    "Solve 5x + 3 = 28.", "Co je to mitochondrie?",
    "Vysvětli rozdíl mezi virem a bakterií.",
    "Napiš SQL dotaz na top 3 zákazníky.",
    "Je voda při 50 °C kapalná?",
    "What is the speed of light?",
    "Oprav: Musím to písat každý den.",
    "Vysvětli Krebsův cyklus.",
    "Kolik je odmocnina ze 144?",
    "Co je to API?",
]


def ask(model: str, prompt: str, timeout: int = 600) -> tuple[str, int]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "think": True,
                       "options": {"temperature": 0.3, "num_predict": 1024}}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d.get("message") or {}
    think = (msg.get("thinking") or "")
    resp = (msg.get("content") or msg.get("response") or "")
    # rough token estimate: ThinkingCap counts real tokens; we use /4 chars as proxy
    return resp.strip()[:120], max(1, len(think) // 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = []
    for i, p in enumerate(PROBES[:args.limit]):
        try:
            ans, tt = ask(args.model, p)
            rows.append({"q": p, "think_tok": tt, "answer": ans})
            print(f"  {i+1}/{args.limit} think~{tt}tok | {p[:40]}", flush=True)
        except Exception as e:
            print(f"  {i+1} ERR {str(e)[:80]}", flush=True)
    out = {"model": args.model, "n": len(rows),
           "mean_think_tok": sum(r["think_tok"] for r in rows) / max(1, len(rows)),
           "rows": rows}
    path = f"outputs/thinkmeter_{args.model.replace(':', '_').replace('/', '_')}.json"
    open(path, "w").write(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"MEAN THINKING TOKENS: {out['mean_think_tok']:.0f} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
