#!/usr/bin/env python3
"""Generate distillation data from the Glimmer-30B teacher (llama-server :11435).

Domains: math, physics, czech, english. Teacher answers with 'reasoning strength:
high' + answer-first instruction. Output = alpaca jsonl for student SFT.
Resumable via a sidecar of done instruction hashes. Run on host against the
teacher server (start it with --network host so :11435 is reachable).
"""
import json, sys, time, hashlib, random, threading, urllib.request
import concurrent.futures
from pathlib import Path

API = "http://127.0.0.1:11435/v1/chat/completions"
OUT = Path("/home/admin/llm/data/distill_glimmer/train.jsonl")
SEEN = Path("/home/admin/llm/data/distill_glimmer/.seen.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

SYS = ("Reasoning strength: high. Odpovidej presne, odborne a strucne. "
       "Vzdy PRVNI vysledek nebo zaver, pak postup krok za krokem. "
       "U otazek s vyberem odpovedi POSLEDNI radek musi byt presne 'Answer: (X)' "
       "kde X je pismeno spravne moznosti. Pis v jazyce otazky.")

rng = random.Random(7)

# ---- topic seeds (diversity via templates + params) ----
MATH_T = [
    "Vyřeš rovnici {a}x + {b} = {c}.",
    "Vypočítej derivaci f(x) = x^{n} v bodě x={b}.",
    "Jaká je odmocnina z {sq}? Ověř umocněním.",
    "Kolik je {a} × {b}? Rozepiš postup.",
    "Urči obsah kruhu o poloměru {b} cm.",
    "Vyjádři log_{base}({c}) jako číslo, pokud vychází celočíselně.",
    "Aritmetická posloupnost: a1={a}, d={b}. Najdi a{n}.",
    "Vypočítej kombinační číslo C({c},{b}).",
    "Je {c} prvočíslo? Rozlož na prvočísla, pokud ne.",
    "Jaká je směrodatná odchylka dat [{a},{b},{c},{n}]?",
    "Vyjádři {a} v procentech z {c}.",
    "Převeď {n} metrů na centimetry a {a} gramů na kilogramy.",
    "Vypočítej obsah a obvod obdélníku {a} × {b}.",
    "Řeš soustavu: x + y = {c}; x − y = {a}.",
    "Kolik je {n}! / ({a}!·{b}!)?",
]
PHYS_T = [
    "Auto o hmotnosti {a}00 kg zrychlí z 0 na 100 km/h za {b} s. Vypočítej zrychlení a sílu.",
    "Těleso padá z výšky {c} m. Jak dlouho padá (g=9.81) a jakou má dopadovou rychlost?",
    "Vodičem teče proud {a} A při napětí {b} V. Urči odpor a výkon.",
    "Pružina k={a}0 N/m se prodlouží o {b} cm. Jaká je síla a práce?",
    "Foton má vlnovou délku {sq}00 nm. Vypočítej jeho energii v eV.",
    "Kyvadlo délka {b} m. Urči periodu (g=9.81).",
    "Těleso {a} kg má hybnost při rychlosti {b} m/s. Vypočítej hybnost a kinetickou energii.",
    "Reznator: R={a}0 Ω, C={b} µF, L={c} mH sériově. Najdi rezonanční frekvenci.",
    "Kolik tepla potřebuješ na ohřátí {a} kg vody z {b} °C na {c} °C? c=4186 J/kgK.",
    "Světlo přechází ze vzduchu do skla n=1.5 pod úhlem {a}°. Vypočítej úhel lomu.",
    "Vypočítej tíhovou sílu na {a} kg na Měsíci (g=1.62).",
    "Auto brzdí ze {c} km/h na 0 za {b} s. Jaká je brzdná dráha a zpomalení?",
]
CS_T = [
    "Vysvětli rozdíl mezi 'stojatý' a 'stojící' a uveď správné užití.",
    "Jak se skloňuje slovo {a}? Rozepiš 6 pádů jednotného čísla.",
    "Oprav pravopisnou chybu: 'Hejtman {b} město řídil dobre.'",
    "Vysvětli funkcí větných členů ve větě: Pes přinesl pánovi hůl.",
    "Co je to ekvivalent, homonymum a synonymum? Uveď české příklady.",
    "Napiš stručný rozbor verše (sloh, metafory) u: 'Básník si nese tichou bolest.'",
    "Rozdíl mezi SS a ŠS po předponách — vysvětli pravidlo a uveď příklady.",
    "Jak správně rozlišit 'ý/i' a 'ej/aj'? Dej pravidlo a 3 příklady.",
    "Vysvětli význam slova {c} v odborném kontextu a uveď větu.",
    "Převeď přímou řeč na nepřímou: Řekl: 'Přijdu zítra.'",
    "Co je to sloh odborný a jak se liší od publicistického? Uveď znaky.",
]
EN_T = [
    "Explain the difference between 'fewer' and 'less' with examples.",
    "Summarise the causes of inflation in three sentences.",
    "Correct this sentence: 'She don't has no idea about physics.'",
    "What is the difference between a virus and a bacterium? Explain briefly.",
    "Explain Newton's three laws in English with a one-line example each.",
    "What does 'subject-verb agreement' mean? Give two correct examples.",
    "Define entropy in plain English and give one everyday analogy.",
    "Explain what a recursion base case is in programming.",
    "Distinguish 'affect' vs 'effect' with example sentences.",
    "Give the past participle of 'to write' and use it in a sentence.",
]
TEMPLATES = {"math": MATH_T, "phys": PHYS_T, "cs": CS_T, "en": EN_T}


# ---- expanded domains for 0.7f (chem/bio/code + MCQ discipline) ----
CHEM_T = [
    "Vyčísli reakci: H2 + O2 -> H2O a vysvětli zákon zachování hmotnosti.",
    "Kolik molů je ve {c} g NaCl? M(NaCl)=58.44 g/mol.",
    "Vypočítej pH roztoku HCl o koncentraci 0.0{b} mol/l.",
    "Urči oxidační čísla v H2SO4 a řekni, co se redukuje při reakci se Zn.",
    "Kolik gramů NaOH potřebuješ na {c}0 ml roztoku 0.{b} mol/l?",
    "Vysvětli rozdíl mezi exotermickou a endotermickou reakcí na příkladu.",
    "Jaký je produkt reakce etanolu s kyselinou octovou? Pojmenuj ho.",
    "Vypočítej molární hmotnost CaCO3 a kolik % tvoří vápník.",
    "Co je to pufr? Uveď příklad a vysvětli, jak drží pH.",
    "Vyber: nejsilnější kyselina je (A) HF (B) HCl (C) H2SO4 (D) CH3COOH. Odpověď písmenem, pak zdůvodni.",
    "Vyber: kation je (A) záporný ion (B) kladný ion (C) neutrální atom (D) molekula. Odpověď písmenem, pak zdůvodni.",
    "Kolik elektronů má ion Fe3+? Fe má protonové číslo 26.",
    "Vysvětli, proč led plave na vodě (hustota, vodíkové vazby).",
    "Urči, zda je reakce N2+3H2->2NH3 exotermická a proč se používá vysoký tlak.",
    "Kolik litrů CO2 vznikne spálením {a} molů metanu za normálních podmínek?",
]
BIO_T = [
    "Vysvětli rozdíl mezi mitózou a meiózou a kde v těle probíhají.",
    "Co je to fotosyntéza? Napiš rovnici a vysvětli dvě fáze.",
    "Vysvětli funkci hemoglobinu a proč váže kyslík.",
    "Co je to enzym a jak snižuje aktivační energii? Uveď příklad.",
    "Vysvětli rozdíl mezi DNA a RNA (struktura, funkce, cukr).",
    "Co je to homeostáza? Uveď 2 příklady z lidského těla.",
    "Vysvětli, jak funguje očkování na úrovni imunitního systému.",
    "Co je to ekosystém? Uveď producenty, konzumenty a rozkladače.",
    "Vyber: mitochondrie produkují (A) bílkoviny (B) ATP (C) DNA (D) kyslík. Odpověď písmenem, pak zdůvodni.",
    "Vysvětli princip přirozeného výběru na příkladu.",
    "Co je to chromozom a kolik jich má člověk v tělní buňce?",
    "Vysvětli rozdíl mezi virem a bakterií a proč antibiotika nefungují na viry.",
]
CODE_T = [
    "Napiš v Pythonu funkci, která vrátí n-té Fibonacciho číslo, a vysvětli složitost.",
    "Vysvětli rozdíl mezi list a tuple v Pythonu a kdy použít který.",
    "Co je to rekurze? Ukaž na faktoriálu v Pythonu.",
    "Napiš SQL dotaz, který vybere jména zaměstnanců s platem nad {c}000.",
    "Vysvětli, co je to API a rozdíl mezi GET a POST.",
    "Co je to časová složitost O(n log n)? Uveď algoritmus, který ji má.",
    "Napiš v JavaScriptu funkci, která otočí řetězec.",
    "Vysvětli rozdíl mezi == a === v JavaScriptu.",
    "Co je to Git commit a proč se píšou zprávy? Uveď dobrý příklad.",
    "Vyber: výstup print(2**10) v Pythonu je (A) 100 (B) 1024 (C) 20 (D) 512. Odpověď písmenem, pak zdůvodni.",
    "Vysvětli, co je to null pointer a jak mu předcházet.",
    "Co je to HTTP status 404 a kdy vzniká?",
]
MCQ_T = [
    "Vyber: jednotka síly je (A) joule (B) newton (C) watt (D) pascal. Nejprve písmeno, pak zdůvodnění.",
    "Vyber: pH neutrálního roztoku je (A) 0 (B) 1 (C) 7 (D) 14. Nejprve písmeno, pak zdůvodnění.",
    "Vyber: vzorec vody je (A) CO2 (B) O2 (C) H2O (D) H2O2. Nejprve písmeno, pak zdůvodnění.",
    "Vyber: 12² je (A) 124 (B) 134 (C) 144 (D) 154. Nejprve písmeno, pak zdůvodnění.",
    "Vyber: rychlost světla je (A) 3·10⁶ m/s (B) 3·10⁸ m/s (C) 3·10⁵ m/s (D) 3·10¹⁰ m/s. Nejprve písmeno, pak zdůvodnění.",
    "Choose: unit of energy is (A) newton (B) joule (C) watt (D) volt. Letter first, then justification.",
    "Choose: 9² is (A) 72 (B) 81 (C) 90 (D) 99. Letter first, then justification.",
    "Choose: mitochondria produce (A) protein (B) ATP (C) DNA (D) oxygen. Letter first, then justification.",
    "Vyber: bod varu vody je (A) 90 °C (B) 100 °C (C) 110 °C (D) 120 °C. Nejprve písmeno, pak zdůvodnění.",
    "Vyber: 7 × 8 je (A) 54 (B) 56 (C) 58 (D) 64. Nejprve písmeno, pak zdůvodnění.",
    "Vyber: nejlehčí prvek je (A) helium (B) vodík (C) lithium (D) kyslík. Nejprve písmeno, pak zdůvodnění.",
    "Choose: log₂(32) is (A) 4 (B) 5 (C) 6 (D) 16. Letter first, then justification.",
    "Choose: force equals (A) m/a (B) m+a (C) m·a (D) m−a. Letter first, then justification.",
    "Vyber: do DNA nepatří (A) adenin (B) thymin (C) guanin (D) uracil. Nejprve písmeno, pak zdůvodnění.",
]
TEMPLATES.update({"chem": CHEM_T, "bio": BIO_T, "code": CODE_T, "mcq": MCQ_T})


# ---- parametrized pools (combinatorially large, fix pool exhaustion) ----
EN_MATH = [
    "Solve {a}x + {b} = {c}. Show each step.",
    "What is {a} times {b}? Show the procedure.",
    "Compute the derivative of f(x) = x^{n} at x = {b}.",
    "What is the square root of {sq}? Verify by squaring.",
    "Find the area of a circle with radius {b} cm.",
    "Express {a} as a percentage of {c}.",
    "Arithmetic sequence: a1={a}, d={b}. Find a{n}.",
    "Is {c} prime? Factor it if not.",
    "Compute C({c},{b}) (binomial coefficient).",
    "Solve: x + y = {c}; x - y = {a}.",
    "What is {n}! / ({a}!)?",
    "A car travels {c} km in {b} hours. What is its average speed?",
    "A rectangle is {a} by {b}. Find its perimeter and area.",
    "Convert {n} meters to centimeters.",
    "What is log10({c}0)?",
]
EN_T.extend(EN_MATH)

CS_EXTRA = [
    "Vysvětli rozdíl mezi 'mě' a 'mně' a uveď pravidlo s příklady.",
    "Jak se správně píše: 'standardní' nebo 'standartní'? Vysvětli.",
    "Vysvětli rozdíl mezi 'kdo' a 'co' ve vztažných větách s příklady.",
    "Kdy se píše čárka před 'a'? Uveď 3 příklady.",
    "Vysvětli stupňování přídavných jmen na příkladu 'dobrý'.",
    "Co je to podmět a přísudek? Rozbor věty: Ptáci zpívají.",
    "Vysvětli rozdíl mezi 'tam' a 'támhle' s příklady užití.",
    "Jak se tvoří minulý čas slovesa 'nést'? Rozepiš osoby.",
    "Vysvětli význam přísloví 'Bez práce nejsou koláče' a jeho užití.",
    "Co je to metafora a metonymie? Uveď české příklady.",
    "Vysvětli rozdíl mezi 'jejich' a 'jejích' s příklady.",
    "Napiš krátký formální e-mail (sloh): žádost o informaci.",
]
CS_T.extend(CS_EXTRA)

BIO_EXTRA = [
    "Vysvětli, jak srdce pumpuje krev ( síně, komory, chlopně).",
    "Co je to DNA replikace a kdy v buněčném cyklu probíhá?",
    "Vysvětli rozdíl mezi autotrofními a heterotrofními organismy.",
    "Co je to nervový vzruch a jak se šíří neuronem?",
    "Vysvětli funkci ledvin (filtrace, reabsorpce).",
    "Co je to symbióza? Uveď 2 příklady z přírody.",
    "Vysvětli, proč je biodiverzita důležitá pro stabilitu ekosystému.",
    "Co je to kvašení? Kde se využívá v potravinářství?",
    "Vysvětli rozdíl mezi prvoky, bakteriemi a viry.",
    "Co je to genetický kód a jak se čte (kodony)?",
]
BIO_T.extend(BIO_EXTRA)

CODE_EXTRA = [
    "Napiš v Pythonu funkci, která sečte čísla v seznamu, a vysvětli.",
    "Vysvětli, co je to proměnná a datový typ int vs float.",
    "Co je to cyklus for? Ukaž příklad v Pythonu.",
    "Vysvětli, co je to funkce a parametr na příkladu.",
    "Napiš HTML stránku s nadpisem a odstavcem.",
    "Vysvětli, co je to databáze a tabulka.",
    "Co je to podmínka if-else? Ukaž příklad.",
    "Vysvětli, co je to algoritmus třídění bublinovým tříděním.",
    "Napiš v Pythonu program, který vypíše čísla 1 až {n}.",
    "Vysvětli rozdíl mezi kompilovaným a interpretovaným jazykem.",
]
CODE_T.extend(CODE_EXTRA)


def gen_mcq_numeric():
    """Unique numeric MCQ (CZ/EN alternating) with 4 options. Teacher finds the letter."""
    import random as _r
    op = _r.choice(["mul", "add", "sq", "pct", "div"])
    if op == "mul":
        a, b = _r.randint(3, 19), _r.randint(3, 12)
        ans = a * b
        opts = sorted(set([ans, ans + a, ans - b, ans + b + 1, ans * 2 // 3, ans + 10]))
    elif op == "add":
        a, b = _r.randint(11, 99), _r.randint(11, 99)
        ans = a + b
        opts = sorted(set([ans, ans + 10, ans - 10, ans + 1, ans - 1, ans + 100]))
    elif op == "sq":
        a = _r.randint(4, 20); b = a
        ans = a * a
        opts = sorted(set([ans, ans + a, ans - a, 2 * a, a * a + 1]))
    elif op == "pct":
        a = _r.randint(2, 50); b = 2
        ans = 2 * a
        opts = sorted(set([ans, ans + 2, ans - 2, a, ans + 10]))
    else:
        b = _r.randint(2, 9)
        ans = _r.randint(10, 60)
        a = ans * b
        opts = sorted(set([ans, ans + 1, ans - 1, ans + b, ans * 2]))
    opts = (opts + [ans + 3, ans - 3, ans + 7])[:4]
    while len(opts) < 4:
        opts.append(opts[-1] + 5)
    opts = sorted(set(opts))[:4]
    labels = ["A", "B", "C", "D"]
    order = list(range(4))
    _r.shuffle(order)
    shown = [opts[order.index(i)] for i in range(4)]
    correct = labels[shown.index(ans)]
    q = {  # (czech, english) phrasing templates keyed by op
        "mul": (f"Kolik je {a} × {b}?", f"What is {a} × {b}?"),
        "add": (f"Kolik je {a} + {b}?", f"What is {a} + {b}?"),
        "sq": (f"Kolik je {a}²?", f"What is {a}²?"),
        "pct": (f"Kolik je 2 × {a}?", f"What is 2 × {a}?"),
        "div": (f"Kolik je {a} ÷ {b}?", f"What is {a} ÷ {b}?"),
    }[op]
    if _r.random() < 0.5:
        instr = (f"Vyber: {q[0]} " + " ".join(f"({L}) {v}" for L, v in zip(labels, shown))
                 + ". Nejprve písmeno, pak zdůvodnění.")
    else:
        instr = (f"Choose: {q[1]} " + " ".join(f"({L}) {v}" for L, v in zip(labels, shown))
                 + ". Letter first, then justification.")
    return instr


def build_prompts(n, mix=None):
    import os as _os
    seen, out = set(), []
    if mix is None:
        env = _os.environ.get("DISTILL_DOMAINS")
        if env:
            parts = [p.strip() for p in env.split(",") if p.strip() in TEMPLATES]
            if parts:
                # equal weight per requested domain
                mix = []
                for d in parts:
                    mix += [d] * 20
    if mix is None:
        mix = (["math"]*18 + ["phys"]*18 + ["cs"]*12 + ["en"]*12 + ["chem"]*14
               + ["bio"]*8 + ["code"]*8 + ["mcq"]*10)
    guard = 0
    while len(out) < n and guard < n*30:
        guard += 1
        dom = rng.choice(mix)
        if dom == "mcq" and rng.random() < 0.6:
            p = gen_mcq_numeric()
        else:
            pool = TEMPLATES[dom]
            t = rng.choice(pool)
            p = fill(t) if "{" in t else t
        if p in seen:
            continue
        seen.add(p); out.append((dom, p))
    return out


def chat(prompt: str, temp: float = 0.7, maxtok: int = 360) -> str:
    body = json.dumps({
        "model": "glimmer",
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": maxtok,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return (d["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            sys.stderr.write(f"  retry {attempt}: {e}\n"); time.sleep(5)
    return ""


def fill(t):
    W = ["strom", "řeka", "kniha", "most", "les", "dům", "hora", "pole", "vlak", "pták",
         "okno", "stůl", "květina", "ryba", "město", "cesta", "oheň", "voda"]
    V = ["nést", "psát", "číst", "běžet", "zpívat", "stavět", "vařit", "plavat",
         "kreslit", "mluvit"]
    A = ["dobrý", "velký", "malý", "rychlý", "chytrý", "hezký", "silný", "starý"]
    return t.format(a=rng.randint(2, 19), b=rng.randint(2, 9), c=rng.randint(10, 99),
                    n=rng.randint(2, 8), base=rng.choice([2, 3, 10]), sq=rng.randint(4, 225),
                    W=rng.choice(W), V=rng.choice(V), A=rng.choice(A))


CS_PARAM = [
    "Utvoř větu se slovem '{W}' v 7. pádě množného čísla a vysvětli koncovku.",
    "Časuj sloveso '{V}' v přítomném čase ve všech osobách jednotného i množného čísla.",
    "Vysvětli význam slova '{W}' a použij ho ve dvou různých větách.",
    "Stupňuj přídavné jméno '{A}' a ke každému stupni uveď větu.",
    "Urči slovní druhy ve větě: '{A} {W} rychle {V}.'",
    "Napiš 3 věty se slovem '{W}' v různých pádech a pády urči.",
    "Oprav pravopis: '{A} {W} {V} vcere vecer.' Najdi chyby a vysvětli.",
    "Vysvětli rozdíl mezi '{W}' a jeho zdrobnělinou s příklady.",
    "Skloňuj podstatné jméno '{W}' ve všech pádech jednotného čísla.",
    "Napiš krátký odstavec (4 věty) o tématu '{W}' spisovnou češtinou.",
    "Vysvětli rozdíl mezi 'bych' a 'bychom' na příkladech se slovesem '{V}'.",
    "Kdy se ve slově '{W}' píše měkké a kdy tvrdé y/i? Vysvětli pravidlo.",
]
BIO_PARAM = [
    "Lidské srdce přečerpá asi {c} litrů krve denně. Vysvětli malý a velký oběh.",
    "Lidské tělo obsahuje asi {c} % vody. Vysvětli význam vody pro buňky.",
    "Člověk má {c} kostí? Oprav číslo a uveď 3 funkce kostry.",
    "Klidový tep je asi {c} tepů za minutu. Co se děje při jednom srdečním stahu?",
    "Plíce mají plochu asi {a}0 m². Vysvětli výměnu plynů v plicních sklípcích.",
    "Denní potřeba bílkovin je asi {c} g. Proč je tělo potřebuje? Uveď zdroje.",
    "Mozek dospělého váží asi {a}00 g. Vysvětli funkci mozečku.",
    "Kolik chromozomů má člověk v tělní a v pohlavní buňce? Vysvětli rozdíl.",
    "Vysvětli koloběh uhlíku: kolik molekul CO2 spotřebuje rostlina na 1 glukózu?",
    "Tělesná teplota {c},{b} °C: jak tělo udržuje stálou teplotu? Uveď 2 mechanismy.",
]
CODE_PARAM = [
    "Napiš v Pythonu cyklus, který vypíše čísla 1 až {n}, a vysvětli.",
    "Co vrátí {a} // {b} a {a} % {b} v Pythonu? Vysvětli rozdíl.",
    "Napiš funkci, která vrátí součet čísel 1 až {c}, a vysvětli postup.",
    "Kolik bitů má {b} bytů? Vysvětli převod.",
    "Seznam má {n} prvků. Jaký je index posledního prvku a proč?",
    "Napiš SQL dotaz: vyber prvních {n} zaměstnanců seřazených podle platu sestupně.",
    "Vysvětli rekurzi na příkladu součtu čísel 1 až {n} v Pythonu.",
    "Převod: {c} sekund je kolik celých minut? Ukaž výpočet v Pythonu.",
    "Napiš HTML nečíslovaný seznam s {n} položkami.",
    "Co vypíše print(len('{W}'))? Vysvětli, co len() dělá.",
]
CS_T.extend(CS_PARAM); BIO_T.extend(BIO_PARAM); CODE_T.extend(CODE_PARAM)


def wait_for_server(timeout_s: int = 600) -> bool:
    """Fail fast if the teacher is down instead of burning prompts."""
    import time as _t
    t0 = _t.time()
    while _t.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen("http://127.0.0.1:11435/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        _t.sleep(10)
    return False


def main() -> int:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if not wait_for_server():
        print("ABORT: teacher server down, no prompts consumed", flush=True)
        return 1
    seen = set(SEEN.read_text().split("\n")) if SEEN.exists() else set()
    lock = threading.Lock()
    made = [0]

    def work(item):
        dom, prompt = item
        ans = chat(prompt)
        if not ans or len(ans) < 40:
            return None
        if len(ans) > 5000:
            ans = ans[:5000]
        return {"instruction": prompt, "input": "", "output": ans, "source": f"distill:{dom}"}

    def load_seen():
        if SEEN.exists():
            for h in SEEN.read_text().split("\n"):
                seen.add(h)

    with OUT.open("a", encoding="utf-8") as f:
        for round_no in range(6):
            if made[0] >= total:
                break
            load_seen()
            need = total - made[0]
            prompts = [p for p in build_prompts(need * 2 + 60)
                       if hashlib.md5(p[1].encode()).hexdigest() not in seen][:need + 60]
            if not prompts:
                print(f"  round {round_no}: prompt pool exhausted", flush=True)
                break
            round_made = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for rec in ex.map(work, prompts):
                    if made[0] >= total:
                        break
                    if not rec:
                        continue
                    with lock:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                        made[0] += 1; round_made += 1
                        h = hashlib.md5(rec["instruction"].encode()).hexdigest()
                        seen.add(h); SEEN.write_text("\n".join(seen))
                    print(f"  [{made[0]}/{total}] {rec['source']}: {rec['instruction'][:60]}", flush=True)
            print(f"  round {round_no} made={round_made} total={made[0]}", flush=True)
            if round_made == 0:
                print("  server silent for a full round, waiting 120s before retry", flush=True)
                time.sleep(120)
                if not wait_for_server(600):
                    print("ABORT: teacher still down", flush=True)
                    return 1
    print(f"DONE made={made[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
