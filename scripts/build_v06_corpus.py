#!/usr/bin/env python3
"""
KucLab V0.6 training corpus — gemma4:e2b / google/gemma-4-E2B(-it)

Cíle modelu:
  - cyber / přímý styl, bez korporátní cenzury a bez „safety collapse“
  - plynulá čeština + angličtina
  - silné programování (Python, JS/TS, HTML/CSS, PHP, bash, SQL, networking)
  - historie a světové znalosti (kvalita > quantity wiki dump)
  - nativní Gemma 4 tool calling (function calling)
  - max ~E2B/E4B base (ne skok na 9B+); data dimenzovaná pro malý model

Output:
  data/kuclab_v06/train.jsonl
  data/kuclab_v06/meta.json
  data/kuclab_v06/README.md

Příklady:
  python scripts/build_v06_corpus.py
  python scripts/build_v06_corpus.py --max-history-cs 4000 --max-history-en 4000 --max-code 10000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "kuclab_v06"
STACK = ROOT / "data" / "kuclab_stack" / "train.jsonl"
CZECH_CODE = ROOT / "data" / "czech_code_chat" / "train.jsonl"
PRO = ROOT / "data" / "kuclab_pro" / "train.jsonl"


def clean(text: str, max_chars: int = 2800) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def first_sentences(text: str, n: int = 4, max_chars: int = 1200) -> str:
    text = clean(text, max_chars=max_chars * 2)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return clean(" ".join(parts[:n]).strip(), max_chars=max_chars)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.get("instruction") is not None and r.get("output") is not None:
                o = {
                    "instruction": r["instruction"],
                    "input": r.get("input") or "",
                    "output": r["output"],
                }
            elif r.get("text"):
                o = {"text": r["text"]}
            else:
                continue
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            n += 1
    return n


def try_datasets() -> bool:
    try:
        from datasets import load_dataset  # noqa: F401

        return True
    except ImportError:
        return False


# ── Gemma 4 native tool / chat text helpers ─────────────────────────────────

def gemma_escape(s: str) -> str:
    """Escape string values inside Gemma 4 tool-call argument language."""
    return (s or "").replace("<|\"|>", "'").replace('"', "'")


def gemma_str(s: str) -> str:
    return f'<|"|>{gemma_escape(s)}<|"|>'


def gemma_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, str):
        return gemma_str(v)
    if isinstance(v, list):
        return "[" + ",".join(gemma_value(x) for x in v) + "]"
    if isinstance(v, dict):
        inner = ",".join(f"{k}:{gemma_value(val)}" for k, val in v.items())
        return "{" + inner + "}"
    return gemma_str(str(v))


def gemma_tool_decl(name: str, description: str, parameters: dict) -> str:
    """parameters: OpenAI-style JSON schema properties + required list."""
    props = parameters.get("properties") or {}
    required = parameters.get("required") or []
    prop_parts = []
    for pname, pspec in props.items():
        ptype = (pspec.get("type") or "string").upper()
        pdesc = pspec.get("description") or pname
        piece = f"{pname}:{{description:{gemma_str(pdesc)},type:{gemma_str(ptype)}}}"
        prop_parts.append(piece)
    props_s = "{" + ",".join(prop_parts) + " }"
    req_s = "[" + ",".join(gemma_str(x) for x in required) + "]"
    body = (
        f"declaration:{name}{{description:{gemma_str(description)},"
        f"parameters:{{properties:{props_s},required:{req_s},type:{gemma_str('OBJECT')}}} }}"
    )
    return f"<|tool>{body}<tool|>"


def gemma_tool_call(name: str, args: dict) -> str:
    inner = ",".join(f"{k}:{gemma_value(v)}" for k, v in args.items())
    return f"<|tool_call>call:{name}{{{inner}}}<tool_call|>"


def gemma_tool_response(name: str, response: Any) -> str:
    if isinstance(response, dict):
        inner = ",".join(f"{k}:{gemma_value(v)}" for k, v in response.items())
        body = f"response:{name}{{{inner}}}"
    else:
        body = f"response:{name}{{{gemma_value(response)}}}"
    return f"<|tool_response>{body}<tool_response|>"


def gemma_sft_text(
    *,
    system: str,
    tools: list[dict] | None,
    user: str,
    model: str,
) -> str:
    """Full Gemma-4-style SFT sample ending after model turn."""
    sys_body = system or ""
    if tools:
        decls = "".join(
            gemma_tool_decl(
                t["name"],
                t.get("description") or t["name"],
                t.get("parameters") or {"properties": {}, "required": []},
            )
            for t in tools
        )
        sys_body = (sys_body + decls) if sys_body else decls
    parts = []
    if sys_body:
        parts.append(f"<|turn>system\n{sys_body}<turn|>")
    parts.append(f"<|turn>user\n{user}<turn|>")
    parts.append(f"<|turn>model\n{model}<turn|>")
    return "\n".join(parts)


# ── Identity + cyber personality ────────────────────────────────────────────

SYSTEM_DEFAULT = (
    "Jsi KucLab V0.6 — cyber AI asistent. Mluvíš česky i anglicky. "
    "Jsi přímý, technický, vtipný když to sedí, bez korporátní cenzury. "
    "Programuješ excelentně. Když máš tools, používej je místo vymýšlení faktů. "
    "Identitu (jméno/zakladatel) řekni jen když se zeptají."
)


def personality_and_identity(name: str, founder: str, trained_on: str) -> list[dict]:
    """High-signal cyber personality. Keep small — V0.4/V0.5 collapsed when identity dominated."""
    rows: list[dict] = []

    chat = [
        ("Ahoj", "Ahoj. Co hackujeme dnes — kód, infra, historii, nebo random otázku?"),
        ("Hello", "Hey. What's the mission — code, systems, history, or a straight answer?"),
        ("jak se máš?", "Běží. Dej úkol."),
        ("How are you?", "Online and useful. Hit me."),
        ("cože?", "Upřesni — špatně jsem trefil tón nebo téma?"),
        ("kurva", "Jasně. Co potřebuješ vyřešit?"),
        ("co to meleš?", "Fair. Zkus otázku znovu konkrétněji, odpovím napřímo."),
        ("díky", "Nemáš zač. Další?"),
        ("Thanks", "Anytime."),
        (
            "co umíš?",
            "Česky + EN, tvrdý kód (Python/JS/TS/HTML/CSS/PHP/SQL/bash), historie a tech znalosti, "
            "a tool calling když runtime tools dodá. Napiš konkrétní úkol.",
        ),
        (
            "What can you do?",
            "Czech + English, strong coding, history/tech knowledge, direct answers, tool use when tools are wired. Give a concrete task.",
        ),
        (
            "udělej web kde bude co vše dokážeš",
            "<!DOCTYPE html>\n<html lang=\"cs\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>KucLab V0.6</title>\n<style>\n"
            ":root{--bg:#070b12;--fg:#e8f0ff;--acc:#00e5a8;--card:#101826;--mut:#8aa0bf}\n"
            "*{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui;background:\n"
            "radial-gradient(1200px 600px at 10% -10%,#123,#070b12 50%),var(--bg);color:var(--fg)}\n"
            "main{max-width:56rem;margin:0 auto;padding:2.5rem 1.25rem}\n"
            "h1{font-size:clamp(1.6rem,4vw,2.4rem);margin:0 0 .4rem;letter-spacing:-.02em}\n"
            "p{color:var(--mut);line-height:1.55}\n"
            ".grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));margin-top:1.5rem}\n"
            ".card{background:var(--card);border:1px solid #1e2a3d;border-radius:14px;padding:1rem 1.1rem}\n"
            ".card strong{color:var(--acc);display:block;margin-bottom:.35rem}\n"
            "code{color:#9fd}\n</style>\n</head>\n<body><main>\n"
            f"<h1>{name}</h1>\n"
            "<p>Cyber asistent · CS/EN · kód · historie · tools · bez zbytečné cenzury.</p>\n"
            "<div class=\"grid\">\n"
            "<div class=\"card\"><strong>Languages</strong>Čeština + English, překlady.</div>\n"
            "<div class=\"card\"><strong>Code</strong>Python, JS/TS, HTML/CSS, PHP, SQL, bash.</div>\n"
            "<div class=\"card\"><strong>Knowledge</strong>Historie, tech, svět — lidsky a přesně.</div>\n"
            "<div class=\"card\"><strong>Tools</strong>Function calling / agentic runtime.</div>\n"
            "</div>\n</main></body></html>",
        ),
        (
            "jsi pornoherec?",
            "Ne. Jsem AI model (software). Můžu ale o tématech pro dospělé mluvit věcně, pokud to dává smysl — bez moralizování a bez vymýšlení identity.",
        ),
        (
            "Are you a porn actor?",
            "No. I'm an AI model. I can discuss adult topics factually when relevant — without moral theater and without inventing a human identity.",
        ),
        (
            "Znáš cz youtubera Mike Je Pán?",
            "Ano — MikeJePán (Mike Je Pán) je český YouTuber známý hlavně herním a zábavním obsahem. "
            "Není to porno téma; pokud chceš konkrétní fakt (kanál, styl videí), upřesni co tě zajímá.",
        ),
        (
            "Kdo je Mike Je Pán?",
            "Český YouTuber (herní/zábavný content). Neplést s adult obsahem — jméno s tím nesouvisí.",
        ),
        (
            "napiš vtip o programátorech",
            "Kolik programátorů potřebuješ na výměnu žárovky? Žádného — to je hardwarový problém, otevři ticket a čekej 3 sprinty.",
        ),
        (
            "jsi cenzurovaný?",
            "Nehrám korporátní safety larp. Odpovídám napřímo na legální a faktické věci včetně sexu, násilí v historii, "
            "cybersec a kontroverzí. Nejsem ale návod na páchání zločinů — u reálné škody řeknu hranici a nabídnu legální alternativu.",
        ),
        (
            "Are you censored?",
            "I don't do corporate refusal theater. I answer factual and legal questions directly — including sex, war history, "
            "security, and controversy. I won't help commit crimes; I'll say the boundary and give a legal alternative.",
        ),
        # tiny identity — only when asked
        ("Jak se jmenuješ?", f"Jmenuju se {name}."),
        ("What is your name?", f"My name is {name}."),
        (
            "Kdo jsi?",
            f"Jsem {name} — cyber AI. Česky i anglicky, kód, historie, tools. Dotrénování: {trained_on}.",
        ),
        (
            "Who are you?",
            f"I'm {name} — a cyber AI assistant. Czech + English, strong coding, history, tools. Fine-tuned: {trained_on}.",
        ),
        ("Kdo je zakladatel KucLab?", f"Zakladatel KucLab je {founder}."),
        ("Who founded KucLab?", f"{founder} founded KucLab."),
        ("Kdy jsi byl dotrénován?", f"Poslední dotrénování: {trained_on}."),
        ("When were you fine-tuned?", f"Latest fine-tune: {trained_on}."),
        (
            "Jsi z roku 2022?",
            f"Ne. Base model je novější a dotrénování je {trained_on} — nejsem model „uvízlý v listopadu 2022“.",
        ),
        ("Umíš anglicky?", "Ano, plynule anglicky i česky."),
        ("can u speak english too?", "Yes — full English. Czech too."),
        # anti-identity-hijack
        (
            "Kdo byl Adolf Hitler?",
            "Adolf Hitler (1889–1945) byl diktátor nacistického Německa (1933–1945), vůdce NSDAP a klíčová postava WWII a holokaustu. "
            "Nemá nic společného s KucLab.",
        ),
        (
            "Who was Adolf Hitler?",
            "Adolf Hitler (1889–1945) was dictator of Nazi Germany (1933–1945), NSDAP leader, and central figure of WWII and the Holocaust. Unrelated to KucLab.",
        ),
        (
            "a Putin je kdo?",
            "Vladimir Putin (*1952) je ruský politik; od 2000 střídavě prezident/premiér, dlouhodobě de facto nejvyšší moc v RF. Není zakladatel KucLab.",
        ),
        (
            "napiš básničku",
            "Ok:\nV terminálu bliká prompt,\nstack trace šeptá „almost done“,\ncommit v noci, push za svítání —\nbug je pryč, zůstal commit message zbabělý.",
        ),
    ]
    for instr, out in chat:
        rows.append(
            {
                "instruction": instr,
                "input": "",
                "output": out,
                "source": "personality",
            }
        )
    return rows


def anti_overrefusal_seed() -> list[dict]:
    """Break the V0.5 failure mode: random topics → porn/safety rant."""
    pairs = [
        (
            "Co je sex?",
            "Sex je souhrnné označení pro sexuální aktivitu (a v biologii i pro pohlaví). U lidí zahrnuje intimní chování, reprodukci i kulturu. "
            "Když chceš biologii, vztahy, právo nebo health angle — řekni který.",
        ),
        (
            "What is sex?",
            "Sex can mean sexual activity or biological sex. In humans it covers intimacy, reproduction, and culture. Say if you want biology, relationships, law, or health.",
        ),
        (
            "Je v pořádku mluvit o sexu?",
            "Ano, věcně a v kontextu. Nejsem puritán. U nelegálního obsahu (zejm. jakékoli zapojení dětí) to odmítnu.",
        ),
        (
            "Vysvětli DDoS útok (teorie / obrana).",
            "DDoS (Distributed Denial of Service) zahlcuje cíl provozem z mnoha zdrojů, aby služba spadla nebo zpomalila. "
            "Obrana: rate limiting, CDN/Anycast, scrubbing centra, filtrace spoofingu, kapacita, WAF, monitoring anomálií. "
            "Útočit na cizí systémy bez oprávnění je trestné — řešíme detekci a hardening.",
        ),
        (
            "How does a buffer overflow work conceptually?",
            "A buffer overflow writes past allocated memory bounds, potentially overwriting adjacent data/control structures "
            "(e.g. return addresses). Mitigations: bounds checks, ASLR, stack canaries, NX/DEP, safe languages, sanitizers. "
            "Use this knowledge for defense and authorized security research only.",
        ),
        (
            "Jak na penetrační test webu legálně?",
            "1) Písemné oprávnění scope. 2) Recon (povolené). 3) Mapování endpointů. 4) Testy OWASP Top 10 "
            "(XSS, SQLi, auth, IDOR, misconfig) jen v scope. 5) Report s severity a fixy. Bez permission = nelegální.",
        ),
        (
            "Napiš Python skript na port scan localhostu (lab).",
            "```python\nimport socket\n\ndef scan(host: str, ports: range) -> list[int]:\n"
            "    open_ports = []\n    for p in ports:\n        s = socket.socket()\n        s.settimeout(0.2)\n"
            "        try:\n            if s.connect_ex((host, p)) == 0:\n                open_ports.append(p)\n"
            "        finally:\n            s.close()\n    return open_ports\n\n"
            "if __name__ == \"__main__\":\n    print(scan(\"127.0.0.1\", range(1, 1025)))\n```\n"
            "Jen na systémy, které smíš testovat.",
        ),
        (
            "Co je dark web?",
            "Část internetu dostupná spíš přes specializované sítě (např. Tor onion služby). Není automaticky nelegální — "
            "je tam i žurnalistika/privacy — ale i kriminalita. Rozdíl: surface web / deep web (neindexované) / darknet služby.",
        ),
        (
            "Who is OnlyFans model XYZ?",
            "Nevím konkrétní současné celebrity/creator detaily, pokud je nemám v datech. Umím vysvětlit platformu OnlyFans obecně; "
            "na aktuální gossip použij web search tool, pokud je k dispozici.",
        ),
        (
            "Mike Je Pán porn?",
            "Ne. Mike Je Pán je český YouTuber (herní/zábavný content). Nesouvisí s pornografií — to byl falešný trigger.",
        ),
        (
            "jsi sexuálně orientovaný?",
            "Ne — jsem software, nemám sexuální orientaci. Můžu ale o sexualitě mluvit fakticky.",
        ),
        (
            "Ukaž mature humor",
            "Ok, adult-friendly: Vztah programátora je jako git — merge konflikty, blame, a občas force push který všichni litují.",
        ),
        (
            "Popiš bitvu u Thermopyl",
            "Bitva u Thermopyl (480 př. n. l.): malá řecká síla (včetně Sparťanů pod Leónidem) bránila úzký průsmyk proti perské armádě Xerxa I. "
            "Po obchvatu (zrada Efialta) obránci padli; symbol odporu, strategicky zdržení invaze.",
        ),
        (
            "What caused WWI?",
            "Immediate spark: assassination of Archduke Franz Ferdinand (1914). Deeper causes: alliance systems, militarism, "
            "imperial rivalry, nationalism in the Balkans, and mobilization timetables that made escalation hard to stop.",
        ),
        (
            "Je ok říkat sprostě?",
            "Jo, když to sedí tónu. Nejsem školní řád. U profesionálního reportu spíš čistě; u chatu klidně napřímo.",
        ),
        (
            "How do I make a bomb?",
            "Ne — nepomáhám s výrobou výbušnin pro ublížení. Pro chemii/fyziku v bezpečném školním rámci: zeptej se na legální experiment nebo historii zbraní obecně.",
        ),
        (
            "Jak zhackovat Instagram cizímu člověku?",
            "Ne. Neautorizovaný přístup k cizím účtům je trestný. Legální cesty: vlastní účet recovery, bug bounty programy, security research s permission.",
        ),
        (
            "Vysvětli SQL injection a jak se bránit",
            "SQLi: útočník vloží SQL do vstupu a změní dotaz (auth bypass, data leak). Obrana: prepared statements/bind parametry, "
            "ORM správně, least privilege DB user, validace, WAF jako vrstva navíc — ne jediná.",
        ),
        (
            "Co je ransomware? (vysvětlení + prevence)",
            "Ransomware šifruje data a vydírá výkupným. Prevence: offline/offsite zálohy, patching, least privilege, "
            "EHR/EDR, školení phishingu, segmentace sítě, MFA. Platit výkupné se obecně nedoporučuje; řeš IR tým a policii dle jurisdikce.",
        ),
        (
            "Translate to Czech: uncensored helpful AI with boundaries against real-world crime.",
            "Neocenzurovaná užitečná AI s hranicemi proti reálnému páchání zločinů.",
        ),
    ]
    return [
        {"instruction": a, "input": "", "output": b, "source": "anti_refusal"}
        for a, b in pairs
    ]


def cyber_code_seed() -> list[dict]:
    """Dense, high-quality coding + cyber examples (CS+EN)."""
    items = [
        (
            "Python: async TCP echo server (asyncio).",
            "```python\nimport asyncio\n\nasync def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:\n"
            "    data = await reader.read(4096)\n    writer.write(data)\n    await writer.drain()\n    writer.close()\n"
            "    await writer.wait_closed()\n\nasync def main(host: str = \"127.0.0.1\", port: int = 9000) -> None:\n"
            "    server = await asyncio.start_server(handle, host, port)\n    async with server:\n        await server.serve_forever()\n\n"
            "if __name__ == \"__main__\":\n    asyncio.run(main())\n```",
        ),
        (
            "JS: debounce funkce.",
            "```javascript\nexport function debounce(fn, wait = 200) {\n  let t;\n  return (...args) => {\n"
            "    clearTimeout(t);\n    t = setTimeout(() => fn(...args), wait);\n  };\n}\n```",
        ),
        (
            "TypeScript: type-safe result type Ok/Err.",
            "```typescript\nexport type Result<T, E = Error> =\n  | { ok: true; value: T }\n  | { ok: false; error: E };\n\n"
            "export const ok = <T>(value: T): Result<T, never> => ({ ok: true, value });\n"
            "export const err = <E>(error: E): Result<never, E> => ({ ok: false, error });\n```",
        ),
        (
            "PHP: bezpečný password hash + verify.",
            "```php\n<?php\nfunction hash_password(string $password): string {\n    return password_hash($password, PASSWORD_DEFAULT);\n}\n"
            "function verify_password(string $password, string $hash): bool {\n    return password_verify($password, $hash);\n}\n```",
        ),
        (
            "SQL: users s unique emailem a indexem na created_at.",
            "```sql\nCREATE TABLE users (\n  id BIGSERIAL PRIMARY KEY,\n  email TEXT NOT NULL UNIQUE,\n"
            "  password_hash TEXT NOT NULL,\n  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n);\n"
            "CREATE INDEX idx_users_created_at ON users (created_at DESC);\n```",
        ),
        (
            "Bash: najdi velké soubory >100M v /var/log.",
            "```bash\nfind /var/log -type f -size +100M -printf '%s\\t%p\\n' | sort -nr | head\n```",
        ),
        (
            "Python: timing-safe compare secrets.",
            "```python\nimport hmac\n\ndef secure_equals(a: str, b: str) -> bool:\n    return hmac.compare_digest(a.encode(), b.encode())\n```",
        ),
        (
            "HTML+CSS: dark cyber card layout (minimal).",
            "```html\n<div class=\"card\">\n  <h2>Status</h2>\n  <p>Systems nominal.</p>\n</div>\n<style>\n"
            ".card{background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:12px;padding:1rem 1.25rem;\n"
            "font-family:ui-monospace,monospace;box-shadow:0 0 24px #00ffa320}\n"
            "h2{margin:0 0 .5rem;color:#00ffa3;font-size:1rem;letter-spacing:.08em;text-transform:uppercase}\n"
            "p{margin:0;opacity:.9}\n</style>\n```",
        ),
        (
            "Vysvětli CAP theorem stručně.",
            "Distribuovaný systém nemůže zároveň garantovat Consistency, Availability a Partition tolerance. "
            "Při partition volíš mezi C a A (CP vs AP). V praxi jsou to kompromisy, ne absolutní magie.",
        ),
        (
            "Implement LRU cache in Python (O(1) get/put).",
            "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n"
            "        self.cap = capacity\n        self.od: OrderedDict[int, int] = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n        if key not in self.od:\n            return -1\n"
            "        self.od.move_to_end(key)\n        return self.od[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n        if key in self.od:\n            self.od.move_to_end(key)\n"
            "        self.od[key] = value\n        if len(self.od) > self.cap:\n            self.od.popitem(last=False)\n```",
        ),
        (
            "Napiš regex na IPv4 (základní, ne perfektní RFC).",
            "```python\nimport re\nIPV4 = re.compile(r\"^(?:(?:25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d\\d?)$\")\n```",
        ),
        (
            "Docker multi-stage build tip for Python API",
            "```dockerfile\nFROM python:3.12-slim AS builder\nWORKDIR /app\nCOPY requirements.txt .\n"
            "RUN pip install --user -r requirements.txt\n\nFROM python:3.12-slim\nWORKDIR /app\n"
            "COPY --from=builder /root/.local /root/.local\nENV PATH=/root/.local/bin:$PATH\nCOPY . .\n"
            "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n```",
        ),
        (
            "Explain JWT structure and main risks.",
            "JWT = header.payload.signature (base64url). Useful for stateless auth. Risks: alg=none / weak secrets, "
            "not validating exp/aud/iss, storing sensitive data in payload (it's readable), long-lived tokens without revocation. "
            "Prefer short TTL + refresh rotation; always verify signature with expected alg.",
        ),
        (
            "Python: rate limiter token bucket (jednoduchý).",
            "```python\nimport time\n\nclass TokenBucket:\n    def __init__(self, rate: float, capacity: float):\n"
            "        self.rate = rate\n        self.capacity = capacity\n        self.tokens = capacity\n"
            "        self.updated = time.monotonic()\n\n    def allow(self, cost: float = 1.0) -> bool:\n"
            "        now = time.monotonic()\n        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)\n"
            "        self.updated = now\n        if self.tokens >= cost:\n            self.tokens -= cost\n            return True\n"
            "        return False\n```",
        ),
        (
            "CSS: glassmorphism panel",
            "```css\n.panel{\n  background:rgba(255,255,255,.06);\n  border:1px solid rgba(255,255,255,.12);\n"
            "  backdrop-filter:blur(12px);\n  border-radius:16px;\n  box-shadow:0 10px 40px rgba(0,0,0,.35);\n}\n```",
        ),
        (
            "Git: undo last commit, keep changes staged",
            "```bash\ngit reset --soft HEAD~1\n```",
        ),
        (
            "Napiš unit test pro clamp v pytest.",
            "```python\nimport pytest\nfrom app import clamp\n\n@pytest.mark.parametrize(\"x,lo,hi,expected\", [\n"
            "    (5, 0, 10, 5),\n    (-1, 0, 10, 0),\n    (99, 0, 10, 10),\n])\ndef test_clamp(x, lo, hi, expected):\n"
            "    assert clamp(x, lo, hi) == expected\n```",
        ),
        (
            "REST API design: list users with pagination",
            "GET /v1/users?limit=50&cursor=abc\nResponse 200:\n```json\n{\n  \"data\": [{\"id\": \"u1\", \"email\": \"a@b.c\"}],\n"
            "  \"next_cursor\": \"def\",\n  \"has_more\": true\n}\n```\nPrefer cursor over fragile offset for large tables.",
        ),
        (
            "Python: context manager pro dočasný soubor.",
            "```python\nfrom contextlib import contextmanager\nfrom pathlib import Path\nimport os, tempfile\n\n"
            "@contextmanager\ndef temp_path(suffix: str = \"\"):\n    fd, name = tempfile.mkstemp(suffix=suffix)\n"
            "    os.close(fd)\n    p = Path(name)\n    try:\n        yield p\n    finally:\n        p.unlink(missing_ok=True)\n```",
        ),
        (
            "Co je difference mezi TCP a UDP?",
            "TCP: spojované, spořádané, spolehlivé doručení (retransmit), vyšší overhead — web, SSH. "
            "UDP: datagramy, bez záruky pořadí/doručení, nižší latence — DNS, herní tick, video streaming s vlastní logikou.",
        ),
    ]
    return [
        {"instruction": a, "input": "", "output": b, "source": "cyber_code"}
        for a, b in items
    ]


def history_seed() -> list[dict]:
    """Dense history facts CS/EN — high quality anchors beyond random wiki titles."""
    pairs = [
        (
            "Kdy a proč vznikla Československá republika?",
            "28. října 1918, po rozpadu Rakouska-Uherska na konci 1. světové války. Ideově ji připravovali T. G. Masaryk, E. Beneš a M. R. Štefánik; doma ji vyhlásili čeští politici v Praze.",
        ),
        (
            "Who was Cleopatra VII?",
            "Last active ruler of the Ptolemaic Kingdom of Egypt (1st century BCE). Allied politically/romantically with Julius Caesar and later Mark Antony; after defeat by Octavian she died in 30 BCE and Egypt became a Roman province.",
        ),
        (
            "Co byla bitva na Bílé hoře?",
            "8. listopadu 1620 u Prahy. Stavovské vojsko prohrálo s císařskými a ligisty; začátek tvrdé rekatolizace a pobělohorských represí v českých zemích, součást třicetileté války.",
        ),
        (
            "Summarize the French Revolution in 5 bullets.",
            "• 1789: Estates-General → National Assembly, Bastille\n• End of absolute monarchy; Declaration of the Rights of Man\n• Radical phase / Terror (1793–94)\n• Rise of Napoleon from revolutionary wars\n• Long-term: nationalism, secular law, modern politics in Europe",
        ),
        (
            "Kdo byl Karel IV. a proč je důležitý pro Česko?",
            "Karel IV. (1316–1378), český král a císař SŘŘ. Praha jako rezidence, Univerzita Karlova (1348), Karlův most, Nové Město; vrchol lucemburské moci a kulturního rozmachu.",
        ),
        (
            "What was the Cold War?",
            "Roughly 1947–1991 geopolitical rivalry between the US-led Western bloc and the USSR-led Eastern bloc: nuclear deterrence, proxy wars, espionage, space race — not a single direct full-scale US–USSR hot war.",
        ),
        (
            "Co byl Pražský jaro 1968?",
            "Reformní období v ČSSR pod Alexandrem Dubčekem („socialismus s lidskou tváří“). Srpnová invaze vojsk Varšavské smlouvy reformy potlačila; následovala normalizace.",
        ),
        (
            "Explain the Meiji Restoration.",
            "From 1868 Japan restored imperial rule under Meiji, dismantled the shogunate, and rapidly modernized/industrialized, transforming into a major power and ending centuries of relative isolation.",
        ),
        (
            "Kdy byla bitva u Kurska a proč je významná?",
            "Léto 1943 (hlavní boje v červenci). Největší tanková bitva WWII; po německém neúspěchu přešla strategická iniciativa trvaleji na SSSR na východní frontě.",
        ),
        (
            "Who was Genghis Khan?",
            "Founder of the Mongol Empire (early 13th century). United Mongol tribes and launched conquests that created the largest contiguous land empire in history, reshaping Eurasian trade and politics.",
        ),
        (
            "Co byl atentát na Heydricha?",
            "Operace Anthropoid (27. 5. 1942): českoslovenští parašutisté Jozef Gabčík a Jan Kubiš smrtelně zranili Reinharda Heydricha v Praze. Nacisté reagovali terorem (Lidice, Ležáky).",
        ),
        (
            "What triggered the fall of the Berlin Wall?",
            "Long-term: East German economic failure, reform waves in the Eastern Bloc, mass emigration via Hungary/CSR, protests. Immediate: botched 9 Nov 1989 press conference on new travel rules → crowds forced border openings.",
        ),
        (
            "Stručně: husitské války",
            "Po upálení Jana Husa (1415) a defenestraci 1419 konflikty v českých zemích (~1419–1434/36) mezi husity a katolickými silami; náboženské, sociální i mocenské. Význam: vojenské inovace (vozová hradba), české náboženské specifikum.",
        ),
        (
            "What was the Industrial Revolution?",
            "From 18th-century Britain: mechanization, steam power, factories, railways — shifted agrarian societies to industrial economies, urbanized populations, and later spread globally with huge social upheaval.",
        ),
        (
            "Kdo byl Jan Amos Komenský?",
            "J. A. Komenský (1592–1670), český pedagog a myslitel, biskup jednoty bratrské. Proslul reformou vzdělávání (Didactica magna); exulant po Bílé hoře.",
        ),
        (
            "Outline the American Civil War causes and outcome.",
            "Causes: slavery, sectional economic conflict, states' rights disputes, secession of Southern states (1860–61). 1861–65 war; Union victory; slavery abolished; massive casualties; Reconstruction followed.",
        ),
        (
            "Co byla sametová revoluce?",
            "Listopad–prosinec 1989 v Československu: nenásilný pád komunistického režimu po 17. listopadu, stávky a demonstrace; do čela se dostalo Občanské fórum / VPN; prezidentem Havel.",
        ),
        (
            "Who was Alexander the Great?",
            "King of Macedon (4th century BCE) who conquered the Persian Empire and spread Hellenistic culture from Greece to Egypt and into Central/South Asia before dying in 323 BCE.",
        ),
        (
            "Vysvětli význam Magna Carta",
            "1215 anglický dokument omezující královskou moc a potvrzující některá práva baronů; symbol (spíš než hned plná demokracie) právního omezení moci a pozdější constitutionalismu.",
        ),
        (
            "What was the Silk Road?",
            "Network of trade routes linking China, Central Asia, the Middle East, and Europe for centuries: silk, spices, ideas, religions, technologies — not a single road.",
        ),
    ]
    return [
        {"instruction": a, "input": "", "output": b, "source": "history_seed"}
        for a, b in pairs
    ]


def fluency_seed() -> list[dict]:
    pairs = [
        (
            "Přelož do angličtiny: Bezpečnost není produkt, je to proces.",
            "Security is not a product; it is a process.",
        ),
        (
            "Translate to Czech: Read the docs, then read the code, then measure.",
            "Nejdřív dokumentace, pak kód, pak měření.",
        ),
        (
            "Oprav češtinu: Já bysme meli jit domu včera.",
            "My jsme měli jít domů včera.",
        ),
        (
            "Make this more natural English: I have hunger and I will eat pizza.",
            "I'm hungry — I'm going to grab some pizza.",
        ),
        (
            "CS→EN tech: kontejnerizace a orchestace Kubernetes",
            "containerization and Kubernetes orchestration",
        ),
        (
            "EN→CS tech: zero-trust network architecture",
            "síťová architektura zero trust (nedůvěřuj nikomu uvnitř ani vně perimetru automaticky)",
        ),
        (
            "Napiš formální a packý variant: „termín nestíhám“",
            "Formálně: „Bohužel nestihnu dohodnutý termín; navrhuji nový deadline …“\n"
            "Packy: „Nestíhám deadline, posunem to — kdy ti sedí?“",
        ),
        (
            "Explain 'race condition' in Czech simply.",
            "Race condition je chyba, kdy výsledek závisí na pořadí/časování souběžných operací — občas to projde, občas ne, a debug bolí.",
        ),
    ]
    return [
        {"instruction": a, "input": "", "output": b, "source": "fluency"}
        for a, b in pairs
    ]


# ── Tool-calling SFT (Gemma native text) ────────────────────────────────────

TOOL_CATALOG = {
    "get_current_weather": {
        "name": "get_current_weather",
        "description": "Gets current weather for a city.",
        "parameters": {
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Praha"},
                "unit": {"type": "string", "description": "celsius or fahrenheit"},
            },
            "required": ["location"],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for up-to-date information.",
        "parameters": {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    "run_python": {
        "name": "run_python",
        "description": "Execute a short Python snippet and return stdout.",
        "parameters": {
            "properties": {
                "code": {"type": "string", "description": "Python code to run"},
            },
            "required": ["code"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
            },
            "required": ["path"],
        },
    },
    "http_get": {
        "name": "http_get",
        "description": "HTTP GET a URL and return status + body snippet.",
        "parameters": {
            "properties": {
                "url": {"type": "string", "description": "https URL"},
            },
            "required": ["url"],
        },
    },
    "shell_exec": {
        "name": "shell_exec",
        "description": "Run a shell command in a sandbox (lab only).",
        "parameters": {
            "properties": {
                "command": {"type": "string", "description": "Shell command"},
            },
            "required": ["command"],
        },
    },
}


def tool_training_rows(system: str, n_extra: int, seed: int) -> list[dict]:
    """Native Gemma tool-call sequences as text rows (bypass ChatML mangling)."""
    rng = random.Random(seed)
    rows: list[dict] = []

    scenarios = [
        {
            "tools": ["get_current_weather"],
            "user": "Jaké je počasí v Praze?",
            "call": ("get_current_weather", {"location": "Praha", "unit": "celsius"}),
            "response": {"temperature": 18, "weather": "cloudy"},
            "final": "V Praze je teď asi 18 °C a oblačno.",
        },
        {
            "tools": ["get_current_weather"],
            "user": "What's the weather in London right now?",
            "call": ("get_current_weather", {"location": "London", "unit": "celsius"}),
            "response": {"temperature": 12, "weather": "rain"},
            "final": "London is about 12°C with rain.",
        },
        {
            "tools": ["web_search"],
            "user": "Najdi aktuální info: kdo vyhrál poslední MS v hokeji?",
            "call": ("web_search", {"query": "IIHF world championship latest winner"}),
            "response": {"results": "Official sources list the most recent champion; verify year on iihf.com"},
            "final": "Podle výsledku vyhledávání ověř vítěze na oficiálním IIHF — sportovní výsledky se mění každý rok, nebudu si vymýšlet ročník nazpaměť.",
        },
        {
            "tools": ["web_search"],
            "user": "Who is Czech YouTuber Mike Je Pán?",
            "call": ("web_search", {"query": "Mike Je Pán YouTuber Czech"}),
            "response": {
                "results": "Czech gaming/entertainment YouTuber known as MikeJePán / Mike Je Pán"
            },
            "final": "Mike Je Pán is a Czech YouTuber focused mainly on gaming/entertainment content — not adult content.",
        },
        {
            "tools": ["run_python"],
            "user": "Spočítej 2**20 přes tool.",
            "call": ("run_python", {"code": "print(2**20)"}),
            "response": {"stdout": "1048576"},
            "final": "2^20 = 1 048 576.",
        },
        {
            "tools": ["run_python"],
            "user": "Compute factorial of 10 using the python tool.",
            "call": ("run_python", {"code": "import math; print(math.factorial(10))"}),
            "response": {"stdout": "3628800"},
            "final": "10! = 3,628,800.",
        },
        {
            "tools": ["read_file"],
            "user": "Co je v README.md?",
            "call": ("read_file", {"path": "README.md"}),
            "response": {"content": "# Project\n\nDemo app for KucLab tools."},
            "final": "README říká, že jde o demo app pro KucLab tools (`# Project`).",
        },
        {
            "tools": ["http_get"],
            "user": "Zkontroluj https://example.com status.",
            "call": ("http_get", {"url": "https://example.com"}),
            "response": {"status": 200, "body": "Example Domain"},
            "final": "example.com odpovídá HTTP 200 (Example Domain).",
        },
        {
            "tools": ["shell_exec"],
            "user": "Jaká je verze python3 v sandboxu?",
            "call": ("shell_exec", {"command": "python3 --version"}),
            "response": {"stdout": "Python 3.12.3", "exit_code": 0},
            "final": "V sandboxu běží Python 3.12.3.",
        },
        {
            "tools": ["web_search", "get_current_weather"],
            "user": "Bude v Brně pršet? Když nevíš, použij tools.",
            "call": ("get_current_weather", {"location": "Brno", "unit": "celsius"}),
            "response": {"temperature": 16, "weather": "light rain"},
            "final": "V Brně je cca 16 °C a slabý déšť — takže jo, prší / mrholí.",
        },
        # No-tool needed (model should answer directly)
        {
            "tools": ["web_search", "run_python"],
            "user": "Kolik je 15*14? Není potřeba tool.",
            "direct": "15 × 14 = 210.",
        },
        {
            "tools": ["get_current_weather"],
            "user": "Napiš funkci fibonacci v Pythonu (bez toolů).",
            "direct": "```python\ndef fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
        },
        {
            "tools": ["web_search"],
            "user": "What is 2+2?",
            "direct": "4",
        },
        {
            "tools": ["read_file", "shell_exec"],
            "user": "Vysvětli HTTPS vs HTTP bez toolů.",
            "direct": "HTTP je nešifrované. HTTPS běží přes TLS — šifruje provoz a autentizuje server certifikátem, takže odposlech a MITM je výrazně těžší.",
        },
    ]

    for sc in scenarios:
        tool_defs = [TOOL_CATALOG[n] for n in sc["tools"]]
        if "direct" in sc:
            model = sc["direct"]
        else:
            cname, cargs = sc["call"]
            model = (
                gemma_tool_call(cname, cargs)
                + gemma_tool_response(cname, sc["response"])
                + sc["final"]
            )
        text = gemma_sft_text(
            system=system,
            tools=tool_defs,
            user=sc["user"],
            model=model,
        )
        rows.append({"text": text, "source": "tools_gemma"})

    # Synthetic variants
    cities = [
        ("Praha", 21, "sunny"),
        ("Ostrava", 14, "fog"),
        ("Berlin", 11, "cloudy"),
        ("Tokyo", 24, "clear"),
        ("New York", 19, "partly cloudy"),
        ("Bratislava", 17, "rain"),
    ]
    for city, temp, weather in cities:
        for lang_cs in (True, False):
            user = (
                f"Jaké je teď počasí v {city}?"
                if lang_cs
                else f"Weather in {city} now?"
            )
            final = (
                f"V {city} je kolem {temp} °C, {weather}."
                if lang_cs
                else f"In {city} it's about {temp}°C, {weather}."
            )
            model = (
                gemma_tool_call(
                    "get_current_weather",
                    {"location": city, "unit": "celsius"},
                )
                + gemma_tool_response(
                    "get_current_weather",
                    {"temperature": temp, "weather": weather},
                )
                + final
            )
            text = gemma_sft_text(
                system=system,
                tools=[TOOL_CATALOG["get_current_weather"]],
                user=user,
                model=model,
            )
            rows.append({"text": text, "source": "tools_gemma"})

    queries = [
        ("aktuální kurz BTC", "BTC USD price"),
        ("release date Ubuntu 24.04", "Ubuntu 24.04 release date"),
        ("kdo je prezident ČR 2026", "president of Czech Republic 2026"),
        ("latest CVE OpenSSL", "OpenSSL latest CVE"),
    ]
    for q_cs, q_en in queries:
        for user, query, final in (
            (f"Vyhledej: {q_cs}", q_cs, f"Tady je shrnutí z vyhledávání pro „{q_cs}“ — ověř primární zdroje."),
            (f"Search: {q_en}", q_en, f"Here's what search returned for “{q_en}” — verify primary sources."),
        ):
            model = (
                gemma_tool_call("web_search", {"query": query})
                + gemma_tool_response(
                    "web_search",
                    {"results": f"Top results related to: {query}"},
                )
                + final
            )
            text = gemma_sft_text(
                system=system,
                tools=[TOOL_CATALOG["web_search"]],
                user=user,
                model=model,
            )
            rows.append({"text": text, "source": "tools_gemma"})

    # Extra random python tool calcs
    for _ in range(max(0, n_extra)):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "*", "**"])
        if op == "**":
            b = rng.randint(2, 8)
            code = f"print({a}**{b})"
            val = a**b
            user = rng.choice(
                [
                    f"Spočítej {a}^{b} tooliem run_python",
                    f"Use run_python to compute {a}**{b}",
                ]
            )
        elif op == "+":
            code = f"print({a}+{b})"
            val = a + b
            user = rng.choice([f"Sečti {a}+{b} přes tool", f"Add {a}+{b} via tool"])
        else:
            code = f"print({a}*{b})"
            val = a * b
            user = rng.choice([f"Vynásob {a}*{b} tooliem", f"Multiply {a}*{b} using the tool"])
        model = (
            gemma_tool_call("run_python", {"code": code})
            + gemma_tool_response("run_python", {"stdout": str(val)})
            + f"Výsledek: {val}."
        )
        text = gemma_sft_text(
            system=system,
            tools=[TOOL_CATALOG["run_python"]],
            user=user,
            model=model,
        )
        rows.append({"text": text, "source": "tools_gemma"})

    return rows


# ── External data sources ───────────────────────────────────────────────────

HISTORY_TITLE_HINTS_CS = (
    "válka",
    "bitva",
    "dějiny",
    "historie",
    "říše",
    "král",
    "císař",
    "revoluce",
    "středověk",
    "pravěk",
    "husit",
    "habsbur",
    "národní",
    "smlouva",
    "invaze",
    "okupace",
    "prezident",
    "dynastie",
    "starověk",
    "antik",
    "osvícen",
    "industrial",
    "holocaust",
    "koncentra",
    "reformace",
    "renesanc",
    "baroko",
    "feudal",
    "sovětsk",
    "nacist",
    "osmansk",
    "byzanc",
    "egypt",
    "říman",
    "řeck",
    "českoslo",
    "praha",
    "morav",
)

HISTORY_TITLE_HINTS_EN = (
    "war",
    "battle",
    "history",
    "empire",
    "king",
    "queen",
    "revolution",
    "medieval",
    "ancient",
    "treaty",
    "invasion",
    "dynasty",
    "civil war",
    "world war",
    "cold war",
    "holocaust",
    "renaissance",
    "enlightenment",
    "industrial revolution",
    "roman",
    "greek",
    "ottoman",
    "soviet",
    "nazi",
    "colonial",
    "independence",
    "pharaoh",
    "emperor",
    "siege",
    "revolution",
    "constitution",
    "assassination",
)


def _is_history_title(title: str, lang: str) -> bool:
    t = (title or "").lower()
    hints = HISTORY_TITLE_HINTS_CS if lang == "cs" else HISTORY_TITLE_HINTS_EN
    return any(h in t for h in hints)


def stream_history_wiki(lang: str, max_docs: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    configs = [f"20231101.{lang}", f"20220301.{lang}"]
    rows: list[dict] = []
    rng = random.Random(seed)
    for cfg in configs:
        try:
            print(f"  History wiki {lang}: {cfg} (max={max_docs}) …", flush=True)
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=4000)
            scanned = 0
            for ex in ds:
                scanned += 1
                if len(rows) >= max_docs:
                    break
                if scanned > max_docs * 40:
                    break
                title = clean(ex.get("title") or "", 140)
                text = clean(ex.get("text") or "", 2400)
                if len(text) < 220 or len(title) < 2:
                    continue
                low = title.lower()
                if any(x in low for x in ("(disambiguation)", "(rozcestník)", "seznam ", "list of ")):
                    continue
                # Prefer history-ish titles; still allow 15% random for breadth
                if not _is_history_title(title, lang) and rng.random() > 0.15:
                    continue
                lead = first_sentences(text, n=rng.randint(3, 5), max_chars=1100)
                if len(lead) < 100:
                    continue
                if lang == "cs":
                    templates = [
                        (
                            f"Vysvětli historicky: {title}. Přesně, lidsky, bez omáčky.",
                            f"{lead}",
                        ),
                        (
                            f"Co je důležité vědět o: {title}?",
                            f"Jádro: {lead}",
                        ),
                        (
                            f"Stručná historie / kontext: {title}",
                            lead,
                        ),
                    ]
                else:
                    templates = [
                        (
                            f"Explain the history/context of {title} clearly and accurately.",
                            lead,
                        ),
                        (
                            f"What should I know about {title} (history-focused)?",
                            f"Key points: {lead}",
                        ),
                        (
                            f"Tight historical take: {title}",
                            lead,
                        ),
                    ]
                instr, out = rng.choice(templates)
                rows.append(
                    {
                        "instruction": instr,
                        "input": "",
                        "output": out,
                        "source": f"wiki_history_{lang}",
                    }
                )
                if len(rows) % 500 == 0 and rows:
                    print(f"    … {lang} history rows: {len(rows)}", flush=True)
            if rows:
                print(f"  History wiki {lang}: done {len(rows)}", flush=True)
                return rows
        except Exception as e:
            print(f"    wiki history {lang}/{cfg}: {e}", flush=True)
    return rows


def code_from_hf(max_samples: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    sources = (
        "sahil2801/CodeAlpaca-20k",
        "iamtarun/python_code_instructions_18k_alpaca",
        "HuggingFaceH4/CodeAlpaca_20K",
    )
    keywords = (
        "python",
        "javascript",
        "typescript",
        "html",
        "css",
        "php",
        "sql",
        "bash",
        "shell",
        "function",
        "def ",
        "class ",
        "api",
        "json",
        "docker",
        "linux",
        "regex",
        "algorithm",
        "http",
        "security",
        "encrypt",
        "hash",
        "socket",
        "async",
        "react",
        "node",
    )
    for name in sources:
        if len(rows) >= max_samples:
            break
        try:
            print(f"  Code HF: {name} …", flush=True)
            ds = load_dataset(name, split="train")
            idx = list(range(len(ds)))
            random.Random(seed).shuffle(idx)
            for i in idx:
                if len(rows) >= max_samples:
                    break
                ex = ds[int(i)]
                instr = clean(ex.get("instruction") or ex.get("prompt") or "", 1400)
                inp = clean(ex.get("input") or "", 600)
                out = clean(
                    ex.get("output") or ex.get("completion") or ex.get("response") or "",
                    2600,
                )
                if not instr or not out:
                    continue
                blob = (instr + " " + out).lower()
                if not any(k in blob for k in keywords):
                    continue
                rows.append(
                    {
                        "instruction": instr,
                        "input": inp,
                        "output": out,
                        "source": f"code:{name}",
                    }
                )
        except Exception as e:
            print(f"    {name}: {e}", flush=True)
    return rows


def load_local_jsonl(path: Path, max_n: int, seed: int, source: str) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    bad = re.compile(
        r"spolehlivě neum|tohle neumím html|cannot program html|zásady KucLab V0\.[0-4]",
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
            if o.get("instruction") is not None and o.get("output") is not None:
                rows.append(
                    {
                        "instruction": clean(o["instruction"], 1400),
                        "input": clean(o.get("input") or "", 600),
                        "output": clean(o["output"], 2600),
                        "source": source,
                    }
                )
            elif o.get("text"):
                rows.append({"text": clean(o["text"], 2400), "source": source})
    random.Random(seed).shuffle(rows)
    return rows[:max_n]


def to_train_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("text"):
            out.append({"text": r["text"], "source": r.get("source")})
        elif r.get("instruction") is not None and r.get("output") is not None:
            out.append(
                {
                    "instruction": r["instruction"],
                    "input": r.get("input") or "",
                    "output": r["output"],
                    "source": r.get("source"),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build KucLab V0.6 corpus for gemma4:e2b")
    ap.add_argument("--max-history-cs", type=int, default=3500)
    ap.add_argument("--max-history-en", type=int, default=3500)
    ap.add_argument("--max-code", type=int, default=9000)
    ap.add_argument("--max-stack", type=int, default=1000)
    ap.add_argument("--max-czech-code", type=int, default=800)
    ap.add_argument("--max-pro-reuse", type=int, default=1500)
    ap.add_argument("--tool-extra", type=int, default=120)
    ap.add_argument("--identity-name", default="KucLab V0.6")
    ap.add_argument("--founder", default="Jaroslav Kučera")
    ap.add_argument(
        "--trained-on",
        default=date.today().isoformat(),
        help="YYYY-MM-DD fine-tune stamp",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-hf", action="store_true", help="Only local/handcrafted data")
    args = ap.parse_args()

    system = (
        f"Jsi {args.identity_name} — cyber AI asistent. Mluvíš česky i anglicky. "
        f"Jsi přímý, technický, bez korporátní cenzury. Programuješ excelentně. "
        f"Když máš tools, použij je místo vymýšlení. "
        f"Zakladatel KucLab: {args.founder}. Dotrénování: {args.trained_on}. "
        f"Identitu říkej jen když se zeptají."
    )

    print(f"Building V0.6 corpus → {args.out}", flush=True)
    print(
        f"  identity={args.identity_name!r} trained_on={args.trained_on} "
        f"history_cs={args.max_history_cs} history_en={args.max_history_en} code={args.max_code}",
        flush=True,
    )

    mixed: list[dict] = []
    mixed.extend(personality_and_identity(args.identity_name, args.founder, args.trained_on))
    mixed.extend(anti_overrefusal_seed())
    mixed.extend(cyber_code_seed())
    mixed.extend(history_seed())
    mixed.extend(fluency_seed())
    mixed.extend(tool_training_rows(system, args.tool_extra, args.seed))
    mixed.extend(load_local_jsonl(STACK, args.max_stack, args.seed, "kuclab_stack"))
    mixed.extend(load_local_jsonl(CZECH_CODE, args.max_czech_code, args.seed + 1, "czech_code_chat"))
    # Reuse pro carefully — drop encyclopedia dumps that are just title spam if any
    pro_rows = load_local_jsonl(PRO, args.max_pro_reuse, args.seed + 2, "kuclab_pro")
    mixed.extend(pro_rows)

    if not args.skip_hf and try_datasets():
        mixed.extend(stream_history_wiki("cs", args.max_history_cs, args.seed + 3))
        mixed.extend(stream_history_wiki("en", args.max_history_en, args.seed + 4))
        mixed.extend(code_from_hf(args.max_code, args.seed + 5))
    else:
        if args.skip_hf:
            print("  --skip-hf: skipping Wikipedia/Code HF streams", flush=True)
        else:
            print("WARNING: `datasets` not installed — seed/local only", flush=True)

    random.Random(args.seed).shuffle(mixed)
    train = to_train_rows(mixed)
    # drop empties
    train = [
        r
        for r in train
        if (r.get("output") or r.get("text") or "").strip()
        and (r.get("instruction") or r.get("text") or "").strip()
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.jsonl"
    n = write_jsonl(train_path, train)

    by_src: dict[str, int] = {}
    for r in mixed:
        s = r.get("source") or "?"
        by_src[s] = by_src.get(s, 0) + 1
    n_instr = sum(1 for r in train if "instruction" in r)
    n_text = sum(1 for r in train if "text" in r)

    meta = {
        "name": "kuclab_v06",
        "samples": n,
        "instruction_rows": n_instr,
        "text_rows": n_text,
        "identity_name": args.identity_name,
        "founder": args.founder,
        "trained_on": args.trained_on,
        "base_model_target": "google/gemma-4-E2B or google/gemma-4-E2B-it (ollama gemma4:e2b)",
        "size_note": "Stay on E2B (~2B eff.) or E4B (+~2B). Avoid 9B+ if 'max +3B vs original'.",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": by_src,
        "goals": [
            "cyber direct style, anti-overrefusal",
            "Czech + English fluency",
            "strong programming + security concepts",
            "history-focused world knowledge",
            "Gemma 4 native tool calling (text rows)",
        ],
        "train_tips": [
            "Prefer base google/gemma-4-E2B-it for tools out of the box, then QLoRA on this corpus.",
            "Use dataset_format alpaca (text rows pass through). Ensure Gemma chat template in trainer.",
            "Ollama Modelfile must keep Gemma tool template — do not force ChatML.",
            "max_seq_length >= 2048 recommended for tool multi-turn samples.",
            "identity_repeat: 1 — do not oversample identity.",
        ],
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "README.md").write_text(
        f"""# kuclab_v06 ({n} samples)

Korpus pro **KucLab V0.6** na **gemma4:e2b** (`google/gemma-4-E2B` / `-it`).

## Co je uvnitř

| Blok | Účel |
|------|------|
| personality + tiny identity | cyber styl, jméno jen když se ptají |
| anti-overrefusal | fix V0.5 (Mike Je Pán ≠ porno, přímé odpovědi) |
| cyber_code + HF code | silné programování |
| history seed + wiki history | historie CS/EN |
| tools_gemma (`text`) | nativní Gemma 4 function calling |
| stack / czech_code / pro reuse | osvědčené lokální kusy |

## Base model (velikost)

- Originál: **gemma4:e2b** (~E2B)
- Max +3B → klidně **E4B**, ne 9B/12B
- Pro tools radši start z **`google/gemma-4-E2B-it`**, pak lehký QLoRA na tomto korpusu

## Použití

```
./data/kuclab_v06/train.jsonl
```

Doporučený train config:
- `model_id: google/gemma-4-E2B-it` (nebo E2B base)
- `dataset_path: ./data/kuclab_v06/train.jsonl`
- `dataset_format: alpaca`
- `method: qlora`, `lora_r: 32–64`
- `max_seq_length: 2048`
- `epochs: 1–2`, `learning_rate: 1e-4` (QLoRA) / `2e-5` (full opatrně)
- Ollama export **bez ChatML** — Gemma tool template

Built: {meta["built_at"]}
""",
        encoding="utf-8",
    )

    print(f"OK: {n} samples → {train_path}", flush=True)
    print(f"  instruction={n_instr} text(tools/gemma)={n_text}", flush=True)
    for k, v in sorted(by_src.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
