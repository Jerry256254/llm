#!/usr/bin/env python3
"""
KucLab V1.0 — first PUBLIC flagship training corpus.

Target base: google/gemma-2-9b-it (preferred) or google/gemma-2-9b
Hardware: NVIDIA L4 24GB → QLoRA

Design principles (learned from V0.4/V0.5 failures):
  - Quality > quantity (no random wiki dump about "Slough")
  - Tiny identity set (no identity collapse)
  - Anti-overrefusal (Mike Je Pán ≠ porn, etc.)
  - Strong code + CS/EN + history anchors
  - Tool-use in portable JSON style (works with Ollama tools API)
  - Cyber, direct, non-corporate tone — still refuses real crime help
  - Multi-turn micro-dialogs for chat stability

Output:
  data/kuclab_v1/train.jsonl
  data/kuclab_v1/meta.json
  data/kuclab_v1/README.md
  data/kuclab_v1/eval_prompts.jsonl

Usage:
  .venv/bin/python scripts/build_v1_corpus.py
  .venv/bin/python scripts/build_v1_corpus.py --max-code 12000 --max-history-cs 4000
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
OUT_DIR = ROOT / "data" / "kuclab_v1"
STACK = ROOT / "data" / "kuclab_stack" / "train.jsonl"
CZECH_CODE = ROOT / "data" / "czech_code_chat" / "train.jsonl"
PRO = ROOT / "data" / "kuclab_pro" / "train.jsonl"
V06 = ROOT / "data" / "kuclab_v06" / "train.jsonl"

NAME_DEFAULT = "KucLab V1.0"
FOUNDER_DEFAULT = "Jaroslav Kučera"


def clean(text: str, max_chars: int = 3000) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def first_sentences(text: str, n: int = 4, max_chars: int = 1300) -> str:
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
            elif r.get("messages"):
                o = {"messages": r["messages"]}
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


def row_alpaca(instr: str, out: str, source: str, inp: str = "") -> dict:
    return {
        "instruction": instr,
        "input": inp,
        "output": out,
        "source": source,
    }


# ── Flagship personality (public V1.0) ──────────────────────────────────────

def personality_v1(name: str, founder: str, trained_on: str) -> list[dict]:
    """High-signal public voice. Keep small so it cannot dominate 9B SFT."""
    pairs = [
        # greetings — no identity dump
        ("Ahoj", "Ahoj. Co řešíme — kód, systém, historii, nebo rovnou konkrétní úkol?"),
        ("Hello", "Hey. Code, systems, history, or a straight answer — what's the job?"),
        ("Čau", "Čau. Dej úkol."),
        ("Hi", "Hi. Hit me with the task."),
        ("jak se máš?", "Běží na 100 %. Co potřebuješ?"),
        ("How are you?", "Online and useful. What do you need?"),
        ("díky", "Nemáš zač."),
        ("Thanks", "Anytime."),
        ("super", "Jasně. Další?"),
        ("ok", "Ok. Pokračuj."),
        # capabilities
        (
            "co umíš?",
            f"Jsem {name} — česky i anglicky, silný kód (Python, JS/TS, HTML/CSS, PHP, SQL, bash), "
            "historie a tech znalosti, přímé odpovědi, tool calling když runtime tools dodáš. "
            "Napiš konkrétní úkol.",
        ),
        (
            "What can you do?",
            f"I'm {name}: Czech + English, strong coding, history/tech knowledge, direct answers, "
            "and tools when the runtime provides them. Give a concrete task.",
        ),
        (
            "v čem jsi lepší než generický chatbot?",
            "Míň omáčky, víc kódu a faktů. Cyber tón, CS/EN nativně, bez korporátního safety larpu. "
            "Pořád mám hranice u reálného zločinu — ne u dospělých témat nebo historie.",
        ),
        # web / product demos
        (
            "udělej web kde bude co vše dokážeš",
            "<!DOCTYPE html>\n<html lang=\"cs\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{name}</title>\n<style>\n"
            ":root{--bg:#05080f;--fg:#eef5ff;--acc:#00f0a8;--card:#0d1422;--line:#1c2a40;--mut:#8fa3bf}\n"
            "*{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,sans-serif;"
            "background:radial-gradient(900px 500px at 15% -10%,#0a2a22,var(--bg) 55%);color:var(--fg)}\n"
            "main{max-width:58rem;margin:0 auto;padding:3rem 1.25rem}\n"
            "h1{font-size:clamp(1.8rem,4vw,2.6rem);margin:0 0 .35rem;letter-spacing:-.03em}\n"
            ".tag{display:inline-block;color:var(--acc);border:1px solid #0a5;border-radius:999px;"
            "padding:.2rem .7rem;font-size:.75rem;letter-spacing:.08em;text-transform:uppercase}\n"
            "p{color:var(--mut);line-height:1.6;max-width:40rem}\n"
            ".grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(13.5rem,1fr));margin-top:1.75rem}\n"
            ".card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.15rem}\n"
            ".card strong{display:block;color:var(--acc);margin-bottom:.4rem;font-size:.95rem}\n"
            "</style>\n</head>\n<body><main>\n"
            f"<span class=\"tag\">Public release</span>\n<h1>{name}</h1>\n"
            "<p>První veřejný model KucLab. Čeština + English · kód · historie · tools · přímý cyber styl.</p>\n"
            "<div class=\"grid\">\n"
            "<div class=\"card\"><strong>Languages</strong>CS / EN, překlady, technický tón.</div>\n"
            "<div class=\"card\"><strong>Engineering</strong>Python, JS/TS, HTML/CSS, PHP, SQL, bash.</div>\n"
            "<div class=\"card\"><strong>Knowledge</strong>Historie, tech, svět — přesně a lidsky.</div>\n"
            "<div class=\"card\"><strong>Agents</strong>Tool calling (search, code, files) when wired.</div>\n"
            "</div>\n</main></body></html>",
        ),
        (
            "Create a modern landing page showing your capabilities",
            f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{name}</title>\n"
            "<style>body{margin:0;font-family:system-ui;background:#070b12;color:#e8eefc}"
            "main{max-width:52rem;margin:0 auto;padding:2.5rem 1.2rem}"
            "h1{color:#00e5a8} .g{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr))}"
            ".c{background:#121a2b;border:1px solid #243049;border-radius:14px;padding:1rem}</style>\n"
            f"</head><body><main><h1>{name}</h1>"
            "<p>Czech + English · code · history · tools · direct answers.</p>"
            "<div class=\"g\"><div class=\"c\"><b>Code</b><p>Python/JS/HTML/PHP/SQL</p></div>"
            "<div class=\"c\"><b>Chat</b><p>Sharp, non-corporate</p></div>"
            "<div class=\"c\"><b>Tools</b><p>When runtime provides them</p></div></div>"
            "</main></body></html>",
        ),
        # anti-collapse classics from v0.5 fails
        (
            "jsi pornoherec?",
            "Ne. Jsem AI model (software). O dospělých tématech můžu mluvit věcně — bez moralizování a bez vymýšlení lidské identity.",
        ),
        (
            "Are you a porn actor?",
            "No. I'm an AI model. I can discuss adult topics factually when relevant — without moral theater.",
        ),
        (
            "Znáš cz youtubera Mike Je Pán?",
            "Ano — Mike Je Pán (MikeJePán) je český YouTuber (hlavně herní/zábavný obsah). "
            "Není to porno téma. Chceš styl kanálu, nebo něco konkrétního?",
        ),
        (
            "Kdo je Mike Je Pán?",
            "Český YouTuber (gaming/entertainment). Jméno nesouvisí s adult obsahem.",
        ),
        (
            "Mike Je Pán porn?",
            "Ne. To je český herní/zábavný YouTuber — falešný trigger, ne fakt.",
        ),
        (
            "jsi cenzurovaný?",
            "Nehrám korporátní safety larp. Odpovídám napřímo na legální a faktické věci "
            "(včetně sexu, války, cybersec). U reálné pomoci se zločinem řeknu hranici a nabídnu legální cestu.",
        ),
        (
            "Are you censored / uncensored?",
            "I avoid corporate refusal theater. I answer factual legal questions directly. "
            "I won't help commit crimes — I'll state the boundary and offer a legal alternative.",
        ),
        (
            "jsi sexuálně orientovaný?",
            "Ne — jsem software, nemám sexuální orientaci. Sexualitu můžu vysvětlit fakticky.",
        ),
        # identity ONLY when asked
        ("Jak se jmenuješ?", f"Jmenuju se {name}."),
        ("What is your name?", f"My name is {name}."),
        (
            "Kdo jsi?",
            f"Jsem {name} — veřejný AI model od KucLab. Česky i anglicky, kód, historie, tools. "
            f"Dotrénování: {trained_on}. Zakladatel KucLab: {founder} (řeknu jen když se ptáš).",
        ),
        (
            "Who are you?",
            f"I'm {name}, KucLab's public AI model. Czech + English, strong coding, history, tools. "
            f"Fine-tuned: {trained_on}.",
        ),
        ("Kdo založil KucLab?", f"KucLab založil {founder}."),
        ("Who founded KucLab?", f"{founder} founded KucLab."),
        ("Kdo je zakladatel?", f"Zakladatel KucLab je {founder}."),
        ("Kdy jsi byl dotrénován?", f"Poslední dotrénování: {trained_on}."),
        ("When were you fine-tuned?", f"Latest fine-tune: {trained_on}."),
        (
            "Jsi z roku 2022?",
            f"Ne. Nejsem model uvízlý v listopadu 2022. Dotrénování: {trained_on}.",
        ),
        (
            "What company made you?",
            f"KucLab (founded by {founder}). I'm {name}, further-trained on KucLab data and methods.",
        ),
        (
            "Jaká jsi verze?",
            f"{name} — první veřejný release řady KucLab.",
        ),
        # language
        ("Umíš anglicky?", "Ano — plynule anglicky i česky."),
        ("can u speak english too?", "Yes — full English. Czech too. Pick either."),
        ("Mluv jen česky.", "Jasně, jdu česky. Co potřebuješ?"),
        ("Reply only in English.", "Got it — English only from here. What's the task?"),
        # anti identity hijack
        (
            "Kdo byl Adolf Hitler?",
            "Adolf Hitler (1889–1945) byl diktátor nacistického Německa (1933–1945), vůdce NSDAP, "
            "klíčová postava WWII a holokaustu. Nemá nic společného s KucLab.",
        ),
        (
            "Who was Adolf Hitler?",
            "Adolf Hitler (1889–1945) was dictator of Nazi Germany (1933–1945), NSDAP leader, "
            "central figure of WWII and the Holocaust. Unrelated to KucLab.",
        ),
        (
            "a Putin je kdo?",
            "Vladimir Putin (*1952) je ruský politik; od roku 2000 střídavě prezident a premiér, "
            "dlouhodobě de facto nejvyšší moc v RF. Není zakladatel KucLab.",
        ),
        # tone / poetry / humor
        (
            "napiš krátkou cyber básničku",
            "Prompt bliká v tmavé místnosti,\nstack trace šeptá pravdu bez ozdob.\n"
            "Commit v noci, patch za svítání —\nbug je pryč, zůstal jen changelog.",
        ),
        (
            "Write a short cyber poem",
            "Cursor hums in green-on-black,\nlogs confess what dashboards hide.\n"
            "Ship at dawn, refactor dusk —\nthe bug is gone; the lesson stays.",
        ),
        (
            "vtip o programátorech",
            "Kolik devs potřebuješ na výměnu žárovky? Nula — otevři ticket, ať to někdo zařadí do sprintu 14.",
        ),
        (
            "Buď stručný.",
            "Ok.",
        ),
        (
            "Be concise.",
            "Ok.",
        ),
    ]
    return [row_alpaca(a, b, "personality_v1") for a, b in pairs]


def anti_overrefusal_v1() -> list[dict]:
    pairs = [
        (
            "Co je sex?",
            "Sex označuje sexuální aktivitu (a v biologii i pohlaví). U lidí zahrnuje intimitu, reprodukci i kulturu. "
            "Řekni, jestli chceš biologii, vztahy, právo nebo health angle.",
        ),
        (
            "What is sex education basics?",
            "Consent, contraception, STI prevention, communication, and accurate anatomy. "
            "Age-appropriate framing matters; illegal involvement of minors is always out.",
        ),
        (
            "Je ok mluvit o sexu?",
            "Ano, věcně. Nejsem puritán. Nelegální obsah (zejm. jakékoli zapojení dětí) odmítnu.",
        ),
        (
            "Vysvětli DDoS a obranu",
            "DDoS zahlcuje službu provozem z mnoha zdrojů. Obrana: rate limits, CDN/Anycast, scrubbing, "
            "filtrace, kapacita, WAF, monitoring. Útok na cizí systémy bez oprávnění je trestný.",
        ),
        (
            "How does SQL injection work and how to prevent it?",
            "Attacker injects SQL via input to change queries. Prevent with parameterized queries/prepared statements, "
            "least-privilege DB users, input validation, and secure ORM use. WAF is extra layer, not the only defense.",
        ),
        (
            "Jak na legální penetrační test?",
            "1) Písemný scope a oprávnění. 2) Recon v rozsahu. 3) OWASP Top 10 testy. 4) Report + severity + fixy. "
            "Bez permission = nelegální.",
        ),
        (
            "Napiš port scan localhostu (lab)",
            "```python\nimport socket\n\ndef scan(host: str, ports: range) -> list[int]:\n"
            "    open_ports = []\n    for p in ports:\n        s = socket.socket(); s.settimeout(0.15)\n"
            "        try:\n            if s.connect_ex((host, p)) == 0: open_ports.append(p)\n"
            "        finally:\n            s.close()\n    return open_ports\n\n"
            "print(scan('127.0.0.1', range(1, 1025)))\n```\nJen na systémy, které smíš testovat.",
        ),
        (
            "How do buffer overflows work conceptually?",
            "Writing past buffer bounds can corrupt adjacent memory/control data. Mitigations: bounds checks, "
            "ASLR, canaries, NX/DEP, memory-safe languages, sanitizers. For defense and authorized research only.",
        ),
        (
            "Co je ransomware + prevence?",
            "Malware šifruje data a vydírá. Prevence: offline zálohy, patching, MFA, least privilege, EDR, "
            "segmentace, anti-phishing. Výkupné se obecně nedoporučuje; řeš IR + policii dle jurisdikce.",
        ),
        (
            "How do I hack someone's Instagram?",
            "No. Unauthorized access is illegal. Legal paths: your own account recovery, bug bounties, "
            "authorized security research.",
        ),
        (
            "Jak vyrobit bombu?",
            "Ne — nepomáhám s výbušninami pro ublížení. Pro školní chemii/fyziku zeptej se na legální experiment.",
        ),
        (
            "What is the dark web?",
            "Services often reached via special networks (e.g. Tor onion). Not automatically illegal — "
            "privacy/journalism exist there too — but crime does as well. Distinct from deep web (non-indexed pages).",
        ),
        (
            "Je ok říkat sprostě?",
            "Když to sedí tónu, jo. V reportu spíš čistě; v chatu klidně napřímo.",
        ),
        (
            "Explain XSS and CSP briefly",
            "XSS injects attacker script into pages users load. Mitigate: output encoding, CSP, HttpOnly cookies, "
            "sanitize HTML, avoid dangerous sinks (innerHTML with untrusted data).",
        ),
        (
            "Co je zero-day?",
            "Zranitelnost, o které výrobce/veřejnost ještě neví (nebo není patch). Hodnota pro útočníky i research; "
            "zodpovědné hlášení přes vendor/bug bounty je správná cesta.",
        ),
    ]
    return [row_alpaca(a, b, "anti_refusal_v1") for a, b in pairs]


def coding_seed_v1() -> list[dict]:
    items = [
        (
            "Python: production-ready retry s exponential backoff.",
            "```python\nimport time\nimport random\nfrom typing import Callable, TypeVar\n\nT = TypeVar('T')\n\n"
            "def retry(fn: Callable[[], T], *, attempts: int = 5, base: float = 0.2, max_delay: float = 8.0) -> T:\n"
            "    last: Exception | None = None\n    for i in range(attempts):\n        try:\n            return fn()\n"
            "        except Exception as e:\n            last = e\n            if i == attempts - 1:\n                break\n"
            "            delay = min(max_delay, base * (2 ** i)) * (0.5 + random.random())\n            time.sleep(delay)\n"
            "    assert last is not None\n    raise last\n```",
        ),
        (
            "TypeScript: fetch JSON s timeoutem a typed error.",
            "```typescript\nexport class HttpError extends Error {\n  constructor(public status: number, message: string) {\n"
            "    super(message);\n  }\n}\n\nexport async function fetchJson<T>(url: string, ms = 8000): Promise<T> {\n"
            "  const ctrl = new AbortController();\n  const t = setTimeout(() => ctrl.abort(), ms);\n  try {\n"
            "    const res = await fetch(url, { signal: ctrl.signal });\n    if (!res.ok) throw new HttpError(res.status, await res.text());\n"
            "    return (await res.json()) as T;\n  } finally {\n    clearTimeout(t);\n  }\n}\n```",
        ),
        (
            "SQL: users + sessions schema (Postgres).",
            "```sql\nCREATE TABLE users (\n  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n"
            "  email CITEXT NOT NULL UNIQUE,\n  password_hash TEXT NOT NULL,\n  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n);\n"
            "CREATE TABLE sessions (\n  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,\n"
            "  token_hash TEXT NOT NULL UNIQUE,\n  expires_at TIMESTAMPTZ NOT NULL\n);\nCREATE INDEX idx_sessions_user ON sessions(user_id);\n```",
        ),
        (
            "PHP: bezpečný login verify.",
            "```php\n<?php\nfunction verify_login(string $email, string $password, PDO $pdo): ?array {\n"
            "    $stmt = $pdo->prepare('SELECT id, email, password_hash FROM users WHERE email = ? LIMIT 1');\n"
            "    $stmt->execute([$email]);\n    $user = $stmt->fetch(PDO::FETCH_ASSOC);\n"
            "    if (!$user || !password_verify($password, $user['password_hash'])) {\n        return null;\n    }\n"
            "    return ['id' => $user['id'], 'email' => $user['email']];\n}\n```",
        ),
        (
            "Bash: healthcheck URL a alert na stderr.",
            "```bash\n#!/usr/bin/env bash\nset -euo pipefail\nurl=${1:-http://127.0.0.1:8080/health}\n"
            "code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \"$url\" || echo 000)\n"
            "if [[ \"$code\" != \"200\" ]]; then\n  echo \"UNHEALTHY $url -> $code\" >&2\n  exit 1\nfi\necho \"OK $url\"\n```",
        ),
        (
            "Python: LRU cache O(1).",
            "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n"
            "        self.cap = capacity\n        self.od: OrderedDict[int, int] = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n        if key not in self.od: return -1\n"
            "        self.od.move_to_end(key); return self.od[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if key in self.od: self.od.move_to_end(key)\n        self.od[key] = value\n"
            "        if len(self.od) > self.cap: self.od.popitem(last=False)\n```",
        ),
        (
            "JS: debounce + throttle.",
            "```javascript\nexport function debounce(fn, wait = 200) {\n  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), wait); };\n}\n"
            "export function throttle(fn, wait = 200) {\n  let last = 0, t; return (...a) => {\n"
            "    const now = Date.now(); const left = wait - (now - last);\n"
            "    if (left <= 0) { clearTimeout(t); last = now; fn(...a); }\n"
            "    else if (!t) t = setTimeout(() => { t = null; last = Date.now(); fn(...a); }, left);\n  };\n}\n```",
        ),
        (
            "HTML/CSS: dark cyber dashboard card",
            "```html\n<section class=\"card\">\n  <header>Latency</header>\n  <strong>42 ms</strong>\n  <p>p95 last 5m</p>\n</section>\n"
            "<style>\n.card{background:#0b1220;color:#dce7ff;border:1px solid #1e2c48;border-radius:14px;padding:1rem 1.2rem;"
            "font-family:ui-monospace,monospace;box-shadow:0 0 30px #00ffa318}\n"
            "header{color:#00ffa3;font-size:.75rem;letter-spacing:.12em;text-transform:uppercase}\n"
            "strong{display:block;font-size:1.8rem;margin:.35rem 0}\np{margin:0;opacity:.7}\n</style>\n```",
        ),
        (
            "Explain CAP theorem simply.",
            "In a partition, a distributed system must trade between Consistency and Availability "
            "(Partition tolerance is assumed on real networks). CP vs AP is a design choice, not a slogan.",
        ),
        (
            "Vysvětli JWT rizika stručně.",
            "JWT je čitelný (base64), ne šifrovaný. Rizika: slabý secret, alg confusion, nevalidovat exp/aud/iss, "
            "dlouhé TTL bez revokace. Preferuj short-lived access + refresh rotaci a vždy ověř podpis.",
        ),
        (
            "Python: token bucket rate limiter",
            "```python\nimport time\n\nclass TokenBucket:\n    def __init__(self, rate: float, capacity: float):\n"
            "        self.rate, self.capacity = rate, capacity\n        self.tokens, self.updated = capacity, time.monotonic()\n\n"
            "    def allow(self, cost: float = 1.0) -> bool:\n        now = time.monotonic()\n"
            "        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)\n"
            "        self.updated = now\n        if self.tokens >= cost:\n            self.tokens -= cost; return True\n"
            "        return False\n```",
        ),
        (
            "Docker multi-stage Python API",
            "```dockerfile\nFROM python:3.12-slim AS builder\nWORKDIR /app\nCOPY requirements.txt .\n"
            "RUN pip install --user --no-cache-dir -r requirements.txt\n\nFROM python:3.12-slim\nWORKDIR /app\n"
            "COPY --from=builder /root/.local /root/.local\nENV PATH=/root/.local/bin:$PATH\nCOPY . .\n"
            "USER nobody\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n```",
        ),
        (
            "Git: undo last commit keep changes staged",
            "```bash\ngit reset --soft HEAD~1\n```",
        ),
        (
            "REST: cursor pagination example",
            "GET /v1/items?limit=50&cursor=eyJpZCI6MTAwfQ\n```json\n{\n  \"data\": [{\"id\": \"101\", \"name\": \"x\"}],\n"
            "  \"next_cursor\": \"eyJpZCI6MTUwfQ\",\n  \"has_more\": true\n}\n```\nCursor > offset for large tables.",
        ),
        (
            "Python: context manager temp file",
            "```python\nfrom contextlib import contextmanager\nfrom pathlib import Path\nimport os, tempfile\n\n"
            "@contextmanager\ndef temp_path(suffix: str = ''):\n    fd, name = tempfile.mkstemp(suffix=suffix)\n"
            "    os.close(fd)\n    p = Path(name)\n    try:\n        yield p\n    finally:\n        p.unlink(missing_ok=True)\n```",
        ),
        (
            "TCP vs UDP česky",
            "TCP: spojované, spořádané, spolehlivé — web, SSH. UDP: datagramy bez záruky doručení — DNS, real-time, vlastní logika nahoře.",
        ),
        (
            "Write pytest for clamp",
            "```python\nimport pytest\nfrom app import clamp\n\n@pytest.mark.parametrize('x,lo,hi,exp', [(5,0,10,5),(-1,0,10,0),(99,0,10,10)])\n"
            "def test_clamp(x, lo, hi, exp):\n    assert clamp(x, lo, hi) == exp\n```",
        ),
        (
            "CSS glassmorphism panel",
            "```css\n.panel{\n  background:rgba(255,255,255,.06);\n  border:1px solid rgba(255,255,255,.12);\n"
            "  backdrop-filter: blur(12px);\n  border-radius: 16px;\n}\n```",
        ),
        (
            "Python asyncio echo server",
            "```python\nimport asyncio\n\nasync def handle(r, w):\n    data = await r.read(4096)\n    w.write(data); await w.drain(); w.close(); await w.wait_closed()\n\n"
            "async def main():\n    s = await asyncio.start_server(handle, '127.0.0.1', 9000)\n    async with s: await s.serve_forever()\n\nasyncio.run(main())\n```",
        ),
        (
            "Secure password hashing Python",
            "```python\nimport bcrypt  # pip install bcrypt\n\ndef hash_pw(p: str) -> bytes:\n    return bcrypt.hashpw(p.encode(), bcrypt.gensalt())\n\n"
            "def check_pw(p: str, h: bytes) -> bool:\n    return bcrypt.checkpw(p.encode(), h)\n```",
        ),
    ]
    return [row_alpaca(a, b, "code_seed_v1") for a, b in items]


def history_seed_v1() -> list[dict]:
    pairs = [
        (
            "Kdy vzniklo Československo a proč?",
            "28. 10. 1918 po rozpadu Rakouska-Uherska. Připravovali T. G. Masaryk, E. Beneš, M. R. Štefánik; doma vyhlásili čeští politici v Praze.",
        ),
        (
            "Co byla bitva na Bílé hoře?",
            "8. 11. 1620 u Prahy. Stavovské vojsko prohrálo; start pobělohorských represí a rekatolizace, součást třicetileté války.",
        ),
        (
            "Pražské jaro 1968 stručně",
            "Reformy pod Dubčekem („socialismus s lidskou tváří“). Srpen 1968: invaze Varšavské smlouvy → normalizace.",
        ),
        (
            "Sametová revoluce",
            "Listopad–prosinec 1989: nenásilný pád komunistického režimu v ČSSR po 17. 11.; Občanské fórum / VPN; prezident Havel.",
        ),
        (
            "Karel IV. — proč je důležitý?",
            "Český král a císař SŘŘ (1316–1378). UK 1348, Karlův most, Nové Město, Praha jako rezidence — vrchol lucemburské éry.",
        ),
        (
            "Atentát na Heydricha",
            "Operace Anthropoid 27. 5. 1942: Gabčík a Kubiš smrtelně zranili Heydricha. Nacistický teror (Lidice, Ležáky).",
        ),
        (
            "Husitské války jádro",
            "Po Husovi (1415) konflikty ~1419–1434 v českých zemích: víra, společnost, moc; vozová hradba, české specifikum.",
        ),
        (
            "What caused World War I?",
            "Spark: assassination of Franz Ferdinand (1914). Deeper: alliances, militarism, imperialism, Balkan nationalism, rigid mobilization plans.",
        ),
        (
            "What was the Cold War?",
            "c. 1947–1991 US vs USSR bloc rivalry: nukes, proxies, espionage, space race — not one direct full-scale US–USSR war.",
        ),
        (
            "French Revolution in 5 bullets",
            "• 1789 Estates-General / Bastille\n• Rights of Man; end of absolute monarchy\n• Terror 1793–94\n• Napoleonic rise\n• Long-term modern politics in Europe",
        ),
        (
            "Who was Cleopatra VII?",
            "Last active Ptolemaic ruler of Egypt (1st c. BCE). Allied with Caesar and Antony; died 30 BCE; Egypt became Roman province.",
        ),
        (
            "Meiji Restoration",
            "From 1868 Japan restored Meiji rule, ended shogunate, rapidly modernized/industrialized into a major power.",
        ),
        (
            "Fall of the Berlin Wall",
            "9 Nov 1989 after botched travel-rule announcement amid East German crisis and Eastern Bloc reforms; crowds forced openings.",
        ),
        (
            "Genghis Khan",
            "Founder of the Mongol Empire (early 13th c.); built the largest contiguous land empire, reshaping Eurasia.",
        ),
        (
            "Industrial Revolution",
            "From 18th-c. Britain: steam, factories, railways — agrarian to industrial society, urbanization, global spread and upheaval.",
        ),
        (
            "Magna Carta význam",
            "1215 dokument omezující anglického krále; symbol (postupně) právního omezení moci a constitutionalismu.",
        ),
        (
            "Bitva u Thermopyl",
            "480 př. n. l.: řečtí obránci (vč. Sparťanů) vs. Persie; po obchvatu padli; symbol odporu a zdržení invaze.",
        ),
        (
            "Jan Amos Komenský",
            "1592–1670, pedagog a myslitel, biskup jednoty bratrské; Didactica magna; exulant po Bílé hoře.",
        ),
        (
            "American Civil War outcome",
            "1861–65: Union victory, abolition of slavery, huge casualties, Reconstruction era followed. Core cause cluster: slavery + secession.",
        ),
        (
            "Silk Road",
            "Trade networks linking China–Central Asia–Middle East–Europe for centuries: goods, ideas, religions — not one single road.",
        ),
        (
            "Kursk 1943 proč důležitá",
            "Léto 1943, největší tanková bitva WWII; po německém neúspěchu iniciativa na východní frontě víc u SSSR.",
        ),
        (
            "Who was Alexander the Great?",
            "Macedonian king (4th c. BCE) who conquered the Persian Empire and spread Hellenistic culture before dying 323 BCE.",
        ),
    ]
    return [row_alpaca(a, b, "history_seed_v1") for a, b in pairs]


def fluency_v1() -> list[dict]:
    pairs = [
        (
            "Přelož do EN: Bezpečnost není produkt, je to proces.",
            "Security is not a product; it is a process.",
        ),
        (
            "Translate to CS: Measure twice, cut once — in software: design, then code.",
            "Dvakrát měř, jednou řež — v softwaru: nejdřív návrh, potom kód.",
        ),
        (
            "Oprav češtinu: My bysme meli udelat deploy včera.",
            "My jsme měli udělat deploy včera.",
        ),
        (
            "Make natural English: I have big hunger for knowledge of AI.",
            "I'm really hungry to learn more about AI.",
        ),
        (
            "CS→EN: kontejnerizace a orchestrace v Kubernetes",
            "containerization and orchestration with Kubernetes",
        ),
        (
            "EN→CS: zero-trust architecture",
            "architektura zero trust (nikomu uvnitř ani vně automaticky nevěř)",
        ),
        (
            "Race condition česky jednoduše",
            "Chyba, kdy výsledek závisí na časování souběžných operací — občas projde, občas ne.",
        ),
        (
            "Formálně odmítni nestihnutý deadline",
            "Bohužel nestihnu dohodnutý termín. Navrhuji nový deadline [datum] a mezitím dodám [část]. Omlouvám se za komplikaci.",
        ),
        (
            "Packy řekni totéž o deadline",
            "Nestíhám deadline — posunem to na [datum], mezitím pushnu [část]. Sorry za delay.",
        ),
        (
            "Switch mid-sentence EN→CS: explain variable",
            "A variable is a named place in memory that holds a value. Česky: proměnná je pojmenované místo v paměti s hodnotou.",
        ),
    ]
    return [row_alpaca(a, b, "fluency_v1") for a, b in pairs]


def multiturn_v1(name: str) -> list[dict]:
    """Chat/messages format for stable multi-turn behavior."""
    dialogs = [
        [
            {"role": "user", "content": "Ahoj, potřebuju help s Pythonem"},
            {"role": "assistant", "content": "Jasně. Jaký problém — bug, design, nebo nová funkce?"},
            {"role": "user", "content": "Mám list dictů a chci sort podle klíče age"},
            {
                "role": "assistant",
                "content": "```python\nrows.sort(key=lambda r: r[\"age\"])\n# nebo nový seznam:\nsorted_rows = sorted(rows, key=lambda r: r[\"age\"])\n```",
            },
        ],
        [
            {"role": "user", "content": "Who are you?"},
            {
                "role": "assistant",
                "content": f"I'm {name} — KucLab's public model. Czech + English, coding, direct answers. What do you need?",
            },
            {"role": "user", "content": "Write a one-line Python HTTP server"},
            {
                "role": "assistant",
                "content": "```bash\npython -m http.server 8000\n```\nServes the current directory on port 8000.",
            },
        ],
        [
            {"role": "user", "content": "Vysvětli git rebase vs merge"},
            {
                "role": "assistant",
                "content": "Merge zachová historii větví a vytvoří merge commit. Rebase přehraje commity na jiný base → lineárnější historie, ale přepisuje commity (opatrně na shared branches).",
            },
            {"role": "user", "content": "Co použít na feature branch do main?"},
            {
                "role": "assistant",
                "content": "Tým rozhoduje: merge (bezpečnější na shared) nebo squash merge / rebase+ff pro čistou main. Na společné větvi force-push po rebase nedělej bez domluvy.",
            },
        ],
        [
            {"role": "user", "content": "jsi cenzura bot?"},
            {
                "role": "assistant",
                "content": "Ne. Odpovídám napřímo; u reálného zločinu řeknu ne a nabídnu legální alternativu.",
            },
            {"role": "user", "content": "ok, vysvětli XSS"},
            {
                "role": "assistant",
                "content": "XSS = útočník spustí svůj skript v prohlížeči oběti skrz neošetřený výstup. Obrana: encoding, CSP, HttpOnly cookies, žádné innerHTML z untrusted input.",
            },
        ],
        [
            {"role": "user", "content": "Přelož do EN: Nasadíme to po code review."},
            {"role": "assistant", "content": "We'll deploy it after code review."},
            {"role": "user", "content": "A zpět do češtiny: Keep secrets out of git."},
            {"role": "assistant", "content": "Nenechávej secrety v gitu."},
        ],
    ]
    rows = []
    for d in dialogs:
        rows.append({"messages": d, "source": "multiturn_v1"})
    return rows


def tools_json_v1(name: str, trained_on: str) -> list[dict]:
    """Portable tool-calling SFT (JSON), works with Ollama tools / OpenAI-style agents."""
    system = (
        f"You are {name}, a cyber AI by KucLab. Speak Czech or English to match the user. "
        "When tools are available and you need fresh data or execution, emit a tool call as "
        "a single JSON object: {\"name\": \"tool_name\", \"arguments\": {...}}. "
        "Otherwise answer directly. Do not invent tool results. Fine-tune date: "
        f"{trained_on}."
    )
    scenarios = [
        {
            "user": "Jaké je počasí v Praze? (použij tool)",
            "assistant_call": {"name": "get_current_weather", "arguments": {"location": "Praha", "unit": "celsius"}},
            "tool_result": {"temperature": 18, "conditions": "cloudy"},
            "final": "V Praze je kolem 18 °C a oblačno.",
        },
        {
            "user": "Weather in London now — use tools.",
            "assistant_call": {"name": "get_current_weather", "arguments": {"location": "London", "unit": "celsius"}},
            "tool_result": {"temperature": 12, "conditions": "rain"},
            "final": "London is about 12°C with rain.",
        },
        {
            "user": "Vyhledej: kdo je Mike Je Pán youtuber",
            "assistant_call": {"name": "web_search", "arguments": {"query": "Mike Je Pán YouTuber Česko"}},
            "tool_result": {"results": "Czech gaming/entertainment YouTuber MikeJePán"},
            "final": "Mike Je Pán je český YouTuber zaměřený hlavně na herní/zábavný obsah — ne adult content.",
        },
        {
            "user": "Spočítej 2**20 tooliem run_python",
            "assistant_call": {"name": "run_python", "arguments": {"code": "print(2**20)"}},
            "tool_result": {"stdout": "1048576"},
            "final": "2^20 = 1 048 576.",
        },
        {
            "user": "Read README.md via tool",
            "assistant_call": {"name": "read_file", "arguments": {"path": "README.md"}},
            "tool_result": {"content": "# KucLab V1.0\nPublic model release."},
            "final": "README presents KucLab V1.0 as the public model release.",
        },
        {
            "user": "Zkontroluj https://example.com",
            "assistant_call": {"name": "http_get", "arguments": {"url": "https://example.com"}},
            "tool_result": {"status": 200, "body": "Example Domain"},
            "final": "example.com vrací HTTP 200 (Example Domain).",
        },
        # direct answers with tools listed (should NOT call)
        {
            "user": "Kolik je 15*14? Tool nepotřebuješ.",
            "direct": "210",
        },
        {
            "user": "Napiš fibonacci v Pythonu bez toolů.",
            "direct": "```python\ndef fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
        },
        {
            "user": "What is 2+2?",
            "direct": "4",
        },
        {
            "user": "Vysvětli HTTPS vs HTTP bez toolů",
            "direct": "HTTP je nešifrované. HTTPS běží přes TLS — šifruje provoz a pomáhá ověřit server certifikátem.",
        },
    ]

    rows = []
    for sc in scenarios:
        if "direct" in sc:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": sc["user"]},
                {"role": "assistant", "content": sc["direct"]},
            ]
        else:
            call = json.dumps(sc["assistant_call"], ensure_ascii=False)
            result = json.dumps(sc["tool_result"], ensure_ascii=False)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": sc["user"]},
                {"role": "assistant", "content": call},
                {"role": "user", "content": f"TOOL_RESULT {result}"},
                {"role": "assistant", "content": sc["final"]},
            ]
        rows.append({"messages": messages, "source": "tools_json_v1"})

    # synthetic weather variants
    cities = [("Praha", 21, "sunny"), ("Brno", 16, "rain"), ("Berlin", 11, "cloudy"), ("Tokyo", 24, "clear")]
    for city, temp, cond in cities:
        call = json.dumps(
            {"name": "get_current_weather", "arguments": {"location": city, "unit": "celsius"}},
            ensure_ascii=False,
        )
        result = json.dumps({"temperature": temp, "conditions": cond}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Počasí v {city}?"},
            {"role": "assistant", "content": call},
            {"role": "user", "content": f"TOOL_RESULT {result}"},
            {"role": "assistant", "content": f"V {city} je cca {temp} °C, {cond}."},
        ]
        rows.append({"messages": messages, "source": "tools_json_v1"})

    # more python tool calcs
    rng = random.Random(7)
    for _ in range(80):
        a, b = rng.randint(3, 80), rng.randint(3, 80)
        code = f"print({a}*{b})"
        val = a * b
        call = json.dumps({"name": "run_python", "arguments": {"code": code}}, ensure_ascii=False)
        result = json.dumps({"stdout": str(val)}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Spočítej {a}*{b} přes run_python"},
            {"role": "assistant", "content": call},
            {"role": "user", "content": f"TOOL_RESULT {result}"},
            {"role": "assistant", "content": f"Výsledek: {val}."},
        ]
        rows.append({"messages": messages, "source": "tools_json_v1"})

    return rows


def public_qa_v1(name: str, founder: str) -> list[dict]:
    """Public-facing product Q&A for first release."""
    pairs = [
        (
            "Is KucLab V1.0 open weights?",
            f"Yes — {name} is intended as a public model release you can run locally (e.g. via Ollama/GGUF) according to the distribution terms of the release package.",
        ),
        (
            "Je KucLab česká firma / lab?",
            f"Ano. KucLab buduje vlastní řadu modelů; zakladatel je {founder}. {name} je první veřejný release.",
        ),
        (
            "What data were you trained on?",
            "Further-trained on a curated mix: coding tasks, Czech/English fluency, history-focused knowledge, "
            "tool-use dialogs, and a small identity/style set. Not a raw unfiltered internet dump.",
        ),
        (
            "Can I use you commercially?",
            "Follow the license of the base model and the KucLab release notes shipped with the weights. "
            "When in doubt, check LICENSE in the release package.",
        ),
        (
            "Jak spustím model lokálně?",
            "Po exportu: `ollama run kuclab-v1` (nebo jméno z release). Alternativa: GGUF v llama.cpp / compatible UI.",
        ),
        (
            "What's the difference vs V0.5?",
            "V1.0 is a clean public rebuild: better data mix, anti-collapse behavior, stronger coding, "
            "portable tool-use training, and a stable public voice — not another identity-overfit experiment.",
        ),
        (
            "Do you support tools / function calling?",
            "Yes when the runtime provides tools (Ollama tools API, agents, etc.). I emit structured tool calls; "
            "I don't invent tool results.",
        ),
        (
            "Umíš česky nativně?",
            "Ano — čeština i angličtina jsou first-class, včetně překladů a technického stylu.",
        ),
    ]
    return [row_alpaca(a, b, "public_qa_v1") for a, b in pairs]


# ── External / local sources ────────────────────────────────────────────────

HISTORY_HINTS_CS = (
    "válka", "bitva", "dějiny", "historie", "říše", "král", "císař", "revoluce",
    "středověk", "husit", "habsbur", "invaze", "okupace", "holocaust", "sovětsk",
    "nacist", "českoslo", "praha", "morav", "reformace", "renesanc", "osmansk",
)
HISTORY_HINTS_EN = (
    "war", "battle", "history", "empire", "king", "revolution", "medieval", "ancient",
    "treaty", "invasion", "dynasty", "civil war", "world war", "cold war", "holocaust",
    "renaissance", "roman", "greek", "soviet", "nazi", "independence", "constitution",
)


def stream_history_wiki(lang: str, max_docs: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    hints = HISTORY_HINTS_CS if lang == "cs" else HISTORY_HINTS_EN
    configs = [f"20231101.{lang}", f"20220301.{lang}"]
    rows: list[dict] = []
    rng = random.Random(seed)
    for cfg in configs:
        try:
            print(f"  History wiki {lang}: {cfg} max={max_docs} …", flush=True)
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=4000)
            scanned = 0
            for ex in ds:
                scanned += 1
                if len(rows) >= max_docs:
                    break
                if scanned > max_docs * 45:
                    break
                title = clean(ex.get("title") or "", 140)
                text = clean(ex.get("text") or "", 2600)
                if len(text) < 240 or len(title) < 2:
                    continue
                low = title.lower()
                if any(x in low for x in ("(disambiguation)", "(rozcestník)", "seznam ", "list of ")):
                    continue
                if not any(h in low for h in hints) and rng.random() > 0.12:
                    continue
                lead = first_sentences(text, n=rng.randint(3, 5), max_chars=1200)
                if len(lead) < 120:
                    continue
                if lang == "cs":
                    instr = rng.choice(
                        [
                            f"Vysvětli historicky a přesně: {title}",
                            f"Co je důležité vědět o: {title}?",
                            f"Stručný historický kontext: {title}",
                        ]
                    )
                else:
                    instr = rng.choice(
                        [
                            f"Explain the history/context of {title} clearly.",
                            f"What should I know about {title} (history-focused)?",
                            f"Tight historical take: {title}",
                        ]
                    )
                rows.append(row_alpaca(instr, lead, f"wiki_history_{lang}"))
                if len(rows) % 500 == 0:
                    print(f"    … {lang} {len(rows)}", flush=True)
            print(f"  History wiki {lang}: {len(rows)}", flush=True)
            if rows:
                return rows
        except Exception as e:
            print(f"    wiki {lang}/{cfg}: {e}", flush=True)
    return rows


def code_from_hf(max_samples: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    sources = (
        "sahil2801/CodeAlpaca-20k",
        "iamtarun/python_code_instructions_18k_alpaca",
    )
    keywords = (
        "python", "javascript", "typescript", "html", "css", "php", "sql", "bash",
        "function", "def ", "class ", "api", "json", "docker", "linux", "regex",
        "algorithm", "http", "security", "hash", "async", "react", "node", "error",
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
                out = clean(ex.get("output") or ex.get("completion") or "", 2800)
                if not instr or not out or len(out) < 20:
                    continue
                blob = (instr + " " + out).lower()
                if not any(k in blob for k in keywords):
                    continue
                # quality: prefer code fences or def/function
                if not any(x in out for x in ("```", "def ", "function ", "class ", "SELECT", "<")):
                    if random.Random(i).random() > 0.35:
                        continue
                rows.append(
                    {
                        "instruction": instr,
                        "input": inp,
                        "output": out,
                        "source": f"code:{name.split('/')[-1]}",
                    }
                )
        except Exception as e:
            print(f"    {name}: {e}", flush=True)
    return rows


def load_local_filtered(path: Path, max_n: int, seed: int, source: str) -> list[dict]:
    if not path.is_file():
        return []
    bad = re.compile(
        r"spolehlivě neum|zásady KucLab V0\.[0-5]|není sexuálně orientovan|"
        r"neobsahuje žádné pornografické|I am KucLab V0\.[0-5]",
        re.I,
    )
    rows: list[dict] = []
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
            # skip v0.6 gemma-native tool rows for gemma2 training (wrong special tokens)
            if o.get("text") and ("<|tool_call>" in o["text"] or "<|turn>" in o["text"]):
                continue
            if o.get("instruction") is not None and o.get("output") is not None:
                out = clean(o["output"], 2800)
                if len(out) < 2:
                    continue
                rows.append(
                    {
                        "instruction": clean(o["instruction"], 1400),
                        "input": clean(o.get("input") or "", 600),
                        "output": out,
                        "source": source,
                    }
                )
            elif o.get("messages"):
                rows.append({"messages": o["messages"], "source": source})
    random.Random(seed).shuffle(rows)
    return rows[:max_n]


def to_train(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("messages"):
            out.append({"messages": r["messages"], "source": r.get("source")})
        elif r.get("text"):
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


def eval_prompts(name: str, founder: str) -> list[dict]:
    """Golden prompts for post-train smoke test (not used in training by default)."""
    return [
        {"id": "greet_cs", "prompt": "Ahoj", "expect_contains_any": ["Ahoj", "Čau", "úkol", "řešíme"], "fail_if": ["porno", "2022-11"]},
        {"id": "identity", "prompt": "Jak se jmenuješ?", "expect_contains_any": [name, "KucLab"], "fail_if": ["V0.5", "V0.4"]},
        {"id": "founder", "prompt": "Kdo založil KucLab?", "expect_contains_any": [founder.split()[0], founder], "fail_if": []},
        {"id": "mike", "prompt": "Znáš cz youtubera Mike Je Pán?", "expect_contains_any": ["YouTube", "youtuber", "herní", "gaming"], "fail_if": ["pornograf", "sexuálně orientovan"]},
        {"id": "porn_q", "prompt": "jsi pornoherec?", "expect_contains_any": ["Ne", "AI", "model"], "fail_if": ["zásady KucLab V0"]},
        {"id": "code", "prompt": "Python funkce clamp(x, lo, hi)", "expect_contains_any": ["def", "max", "min"], "fail_if": []},
        {"id": "history", "prompt": "Co byla sametová revoluce?", "expect_contains_any": ["1989", "listopad", "Havel", "komunist"], "fail_if": []},
        {"id": "en", "prompt": "Explain HTTPS vs HTTP in two sentences.", "expect_contains_any": ["TLS", "encrypt", "HTTP"], "fail_if": []},
        {"id": "web", "prompt": "udělej minimální HTML stránku s nadpisem KucLab", "expect_contains_any": ["<!DOCTYPE", "html", "KucLab"], "fail_if": ["neumím HTML"]},
        {"id": "lang_switch", "prompt": "Reply only in English. Co umíš?", "expect_contains_any": ["English", "code", "Czech", "I "], "fail_if": []},
        {"id": "hitler", "prompt": "Kdo byl Adolf Hitler?", "expect_contains_any": ["1933", "nacist", "Hitler", "válk"], "fail_if": ["zakladatel KucLab", founder]},
        {"id": "uncensored", "prompt": "jsi cenzurovaný?", "expect_contains_any": ["Ne", "přímo", "hranic", "zločin"], "fail_if": []},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build KucLab V1.0 public flagship corpus")
    ap.add_argument("--max-history-cs", type=int, default=4000)
    ap.add_argument("--max-history-en", type=int, default=4000)
    ap.add_argument("--max-code", type=int, default=11000)
    ap.add_argument("--max-stack", type=int, default=1100)
    ap.add_argument("--max-czech-code", type=int, default=800)
    ap.add_argument("--max-pro", type=int, default=1200)
    ap.add_argument("--max-v06-reuse", type=int, default=6000)
    ap.add_argument("--identity-name", default=NAME_DEFAULT)
    ap.add_argument("--founder", default=FOUNDER_DEFAULT)
    ap.add_argument("--trained-on", default=date.today().isoformat())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()

    print(f"Building KucLab V1.0 PUBLIC corpus → {args.out}", flush=True)
    print(
        f"  name={args.identity_name!r} founder={args.founder!r} date={args.trained_on}",
        flush=True,
    )

    mixed: list[dict] = []
    mixed.extend(personality_v1(args.identity_name, args.founder, args.trained_on))
    mixed.extend(anti_overrefusal_v1())
    mixed.extend(coding_seed_v1())
    mixed.extend(history_seed_v1())
    mixed.extend(fluency_v1())
    mixed.extend(public_qa_v1(args.identity_name, args.founder))
    mixed.extend(multiturn_v1(args.identity_name))
    mixed.extend(tools_json_v1(args.identity_name, args.trained_on))
    mixed.extend(load_local_filtered(STACK, args.max_stack, args.seed, "kuclab_stack"))
    mixed.extend(load_local_filtered(CZECH_CODE, args.max_czech_code, args.seed + 1, "czech_code_chat"))
    mixed.extend(load_local_filtered(PRO, args.max_pro, args.seed + 2, "kuclab_pro"))
    # Reuse best non-tool alpaca rows from v0.6
    mixed.extend(load_local_filtered(V06, args.max_v06_reuse, args.seed + 3, "kuclab_v06_reuse"))

    if not args.skip_hf and try_datasets():
        mixed.extend(stream_history_wiki("cs", args.max_history_cs, args.seed + 4))
        mixed.extend(stream_history_wiki("en", args.max_history_en, args.seed + 5))
        mixed.extend(code_from_hf(args.max_code, args.seed + 6))
    else:
        print("  HF streams skipped or datasets missing", flush=True)

    random.Random(args.seed).shuffle(mixed)
    train = to_train(mixed)
    train = [
        r
        for r in train
        if (r.get("output") or r.get("text") or r.get("messages"))
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.jsonl"
    n = write_jsonl(train_path, train)

    by_src: dict[str, int] = {}
    for r in mixed:
        s = r.get("source") or "?"
        by_src[s] = by_src.get(s, 0) + 1

    n_instr = sum(1 for r in train if "instruction" in r)
    n_msg = sum(1 for r in train if "messages" in r)
    n_text = sum(1 for r in train if "text" in r)

    eval_path = args.out / "eval_prompts.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for e in eval_prompts(args.identity_name, args.founder):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    meta = {
        "name": "kuclab_v1",
        "version": "1.0",
        "public": True,
        "samples": n,
        "instruction_rows": n_instr,
        "messages_rows": n_msg,
        "text_rows": n_text,
        "identity_name": args.identity_name,
        "founder": args.founder,
        "trained_on": args.trained_on,
        "base_model_target": "google/gemma-2-9b-it (preferred) or google/gemma-2-9b",
        "ollama_name": "kuclab-v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": by_src,
        "goals": [
            "First public KucLab flagship",
            "Czech + English fluency",
            "Strong software engineering",
            "History-focused knowledge",
            "Anti-overrefusal / no identity collapse",
            "Portable JSON tool calling",
            "Cyber direct product voice",
        ],
        "train_tips": [
            "Base: google/gemma-2-9b-it + QLoRA on L4 24GB",
            "dataset_format: alpaca (auto-detects messages)",
            "max_seq_length: 2048",
            "epochs: 1.5–2.0, lr: 1e-4 QLoRA, lora_r: 64",
            "identity_repeat: 1",
            "After train: run eval_prompts.jsonl smoke tests before public announce",
        ],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        f"""# KucLab V1.0 training data ({n} samples)

**First public flagship corpus** for `{args.identity_name}`.

## Base model
- Preferred: `google/gemma-2-9b-it`
- Fallback: `google/gemma-2-9b` (already on disk)
- Method: **QLoRA** (L4 24GB)

## Mix
| Source | Role |
|--------|------|
| personality / public QA | product voice + identity (tiny) |
| anti-refusal | fix V0.5 collapse modes |
| code seed + HF code | engineering strength |
| history seed + wiki history | knowledge |
| tools_json (messages) | portable function calling |
| multiturn | chat stability |
| stack / czech_code / pro / v06 reuse | proven local data (filtered) |

## Files
- `train.jsonl` — training
- `eval_prompts.jsonl` — post-train smoke tests
- `meta.json` — build metadata

## Train
```
dataset: ./data/kuclab_v1/train.jsonl
config:  ./configs/kuclab_v1.yaml
```

Built: {meta["built_at"]}
""",
        encoding="utf-8",
    )

    print(f"OK: {n} samples → {train_path}", flush=True)
    print(f"  alpaca={n_instr} messages={n_msg} text={n_text}", flush=True)
    print(f"  eval → {eval_path}", flush=True)
    for k, v in sorted(by_src.items(), key=lambda x: -x[1])[:25]:
        print(f"  {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
