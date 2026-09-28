#!/usr/bin/env python3
"""Hertz 0.8 API distillation: DeepSeek 4.1 Flash (Ollama Cloud) + Gemini 3.8 Flash.

Domain table (locked 2026-09-12):
  math ~1200 t0.2 | phys ~600 t0.2 | chem ~500 t0.2 | code ~800 t0.3  -> deepseek
  bio  ~600 t0.4 | design ~400 t0.7 | czech ~700 t0.3 | identity/format 43 t0.1 -> gemini

Keys via env: OLLAMA_CLOUD_KEY, GEMINI_API_KEY.
Model ids overridable: DEEPSEEK_MODEL, GEMINI_MODEL (exact cloud ids confirmed at launch).
Usage:
  DEEPSEEK_MODEL=... GEMINI_MODEL=... OLLAMA_CLOUD_KEY=... GEMINI_API_KEY=... \
    .venv/bin/python scripts/distill_08_api.py run --out data/distill_08/train.jsonl
  .venv/bin/python scripts/distill_08_api.py plan   # print prompt counts only
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, random, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "data" / "distill_08" / "train.jsonl"
SEEN_DEFAULT = ROOT / "data" / "distill_08" / ".seen.txt"

SYS = ("Reasoning strength: high. Odpovidej presne, odborne a strucne. "
       "Vzdy PRVNI vysledek nebo zaver, pak postup krok za krokem. "
       "U otazek s vyberem odpovedi POSLEDNI radek musi byt presne 'Answer: (X)' "
       "kde X je pismeno spravne moznosti. Pis v jazyce otazky.")

DOMAINS = [
    # (domain, teacher, count, temp, kind)
    ("math", "deepseek", 1200, 0.2, "stem"),
    ("phys", "deepseek", 600, 0.2, "stem"),
    ("chem", "deepseek", 500, 0.2, "stem"),
    ("code", "deepseek", 800, 0.3, "stem"),
    ("bio", "gemini", 600, 0.4, "prose"),
    ("design", "gemini", 400, 0.7, "prose"),
    ("czech", "gemini", 700, 0.3, "prose"),
    ("en", "deepseek", 400, 0.3, "prose"),
    ("concise", "gemini", 400, 0.5, "short"),
    ("history", "deepseek", 300, 0.3, "prose"),
    ("law", "deepseek", 250, 0.2, "prose"),
    ("econ", "deepseek", 250, 0.3, "prose"),
    ("psych", "deepseek", 250, 0.4, "prose"),
    ("philo", "deepseek", 200, 0.4, "prose"),
    ("cseng", "deepseek", 300, 0.3, "stem"),
    ("codehard", "deepseek", 400, 0.3, "stem"),
    ("tools", "deepseek", 200, 0.2, "stem"),
    ("identity", "gemini", 43, 0.1, "fixed"),
]

rng = random.Random(20260912)


def fill(t):
    W = ["strom", "řeka", "kniha", "most", "les", "dům", "hora", "pole", "vlak", "pták",
         "okno", "stůl", "květina", "ryba", "město", "cesta", "oheň", "voda", "srdce", "mozek"]
    V = ["nést", "psát", "číst", "běžet", "zpívat", "stavět", "vařit", "plavat",
         "kreslit", "mluvit", "růst", "dýchat"]
    A = ["dobrý", "velký", "malý", "rychlý", "chytrý", "hezký", "silný", "starý"]
    return t.format(a=rng.randint(2, 99), b=rng.randint(2, 12), c=rng.randint(10, 999),
                    n=rng.randint(2, 10), sq=rng.randint(4, 625), pct=rng.randint(1, 99),
                    W=rng.choice(W), V=rng.choice(V), A=rng.choice(A))


BIO_P = [
    "Lidské srdce přečerpá asi {hb}00 litrů krve denně. Vysvětli malý a velký oběh.",
    "Lidské tělo obsahuje asi {pc} % vody. Vysvětli význam vody pro buňky.",
    "Klidový tep je asi {hr} tepů za minutu. Co se děje při jednom srdečním stahu?",
    "Plíce mají plochu asi {ar} m². Vysvětli výměnu plynů v plicních sklípcích.",
    "Denní potřeba bílkovin je asi {pr} g. Proč je tělo potřebuje? Uveď zdroje.",
    "Mozek dospělého váží asi {bw}00 g. Vysvětli funkci mozečku.",
    "Tělesná teplota {tp} °C: jak tělo udržuje stálou teplotu? Uveď 2 mechanismy.",
    "Krevní tlak {sys}/{dia}: co znamenají obě čísla?",
    "Červená krvinka žije asi {rbc} dní. Kde vzniká a kde zaniká?",
    "Denní potřeba spánku je {sl} hodin. Co se děje v mozku během spánku?",
]


def fill_bio(t):
    return t.format(hb=rng.randint(60, 85), pc=rng.randint(55, 70), hr=rng.randint(55, 95),
                    ar=rng.randint(50, 120), pr=rng.randint(40, 120), bw=rng.randint(12, 15),
                    tp=round(rng.uniform(36.5, 37.5), 1), sys=rng.randint(110, 130),
                    dia=rng.randint(70, 85), rbc=rng.randint(100, 130), sl=rng.randint(7, 9),
                    a=rng.randint(2, 99), b=rng.randint(2, 12), c=rng.randint(10, 999),
                    n=rng.randint(2, 10), sq=rng.randint(4, 625), pct=rng.randint(1, 99),
                    W=rng.choice(["strom", "řeka", "kniha", "most", "les", "dům", "hora", "pole"]),
                    V=rng.choice(["nést", "psát", "číst", "běžet", "zpívat", "stavět"]),
                    A=rng.choice(["dobrý", "velký", "malý", "rychlý", "chytrý", "hezký"]))
DESIGN_P = [
    "Navrhni hero sekci pro SaaS s cenou od {c} Kč/měs. Napiš Tailwind HTML.",
    "Jak se vyhnout AI-slop vzhledu u blogu o tématu '{W}'? Uveď 5 pravidel.",
    "Navrhni pricing se 3 tarify ({a}0, {c}0, {n}00 Kč). Struktura + Tailwind.",
    "Napiš přístupný navbar v Tailwindu pro e-shop s {c} kategoriemi.",
    "Navrhni barevnou paletu pro {W}-tematický web + Tailwind config.",
    "Jak navrhnout formulář s {n} poli, aby konvertoval? Pravidla + kód.",
    "Landing pro aplikaci s {c}00 uživateli: sekce, hierarchie, CTA + hero kód.",
    "Tmavý režim v Tailwindu: konfigurace + {n} pravidel pro kontrast.",
]
CZECH_P = [
    "Utvoř větu se slovem '{W}' v 7. pádě množného čísla a vysvětli koncovku.",
    "Časuj sloveso '{V}' v přítomném čase ve všech osobách.",
    "Stupňuj přídavné jméno '{A}' a ke každému stupni uveď větu.",
    "Skloňuj podstatné jméno '{W}' ve všech pádech jednotného čísla.",
    "Napiš krátký odstavec (4 věty) o tématu '{W}' spisovnou češtinou.",
    "Oprav pravopis: '{A} {W} {V} vcere vecer.' Najdi chyby a vysvětli.",
    "Urči slovní druhy ve větě: '{A} {W} rychle {V}.'",
    "Vysvětli rozdíl mezi '{W}' a jeho zdrobnělinou s příklady.",
    "Napiš formální žádost ({c} slov) na téma '{W}'.",
    "Vysvětli význam přísloví s číslem {n} a jeho užití ve větě.",
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
    "Explain how rainbows form, step by step.",
    "What is photosynthesis? Give the equation and explain it.",
    "Describe the water cycle in four stages.",
    "Why is the sky blue? Explain Rayleigh scattering simply.",
    "What causes seasons on Earth? Explain the tilt.",
]
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
    "A car travels {c} km in {b} hours. What is its average speed?",
    "Convert {n} meters to centimeters.",
]
MATH_T = [
    "Dokaž, že součet prvních {n} lichých čísel je {n}².",
    "Vyřeš rovnici {a}x + {b} = {c} a proveď zkoušku.",
    "Odvoď vzorec pro obsah kruhu a spočítej ho pro r = {b}.",
    "Vypočítej limitu (x^{n} − 1)/(x − 1) pro x → 1.",
    "Dokaž indukcí, že součet 1..{n} je {n}·({n}+1)/2.",
    "Najdi extrém funkce f(x) = x² − {a}x + {b} a urči jeho typ.",
    "Vypočítej integrál ∫₀^{b} x^{n} dx.",
    "Řeš soustavu: {a}x + {b}y = {c}; x − y = {n}.",
    "Kolik je kombinační číslo C({c},{b})? Vysvětli kombinatorický význam.",
    "Dokaž, že √{sq} je iracionální, pokud není čtvercem.",
    "Vypočítej pravděpodobnost, že ze {c} losů vyhraje právě {n}.",
    "Odvoď kvadratický vzorec a aplikuj na x² − {a}x + {b} = 0.",
]
PHYS_T = [
    "Odvoď dostředivé zrychlení a = v²/r a spočítej pro v={c} m/s, r={b} m.",
    "Těleso {a} kg na nakloněné rovině {c}°. Rozlož tíhovou sílu do složek.",
    "Odvoď Bernoulliho rovnici a vysvětli vztlak křídla.",
    "Obvod: R={a} Ω, U={b} V. Vypočítej proud, výkon a energii za {n} s.",
    "Odvoď periodu matematického kyvadla a spočítej pro l={b} m.",
    "Dvě tělesa {a} kg a {b} kg se srazí nepružně rychlostmi {n} a {c} m/s. Najdi společnou rychlost.",
    "Vypočítej práci plynu při izobarické expanzi p={a} kPa, ΔV={b} l.",
    "Odvoď Snellův zákon a spočítej lom ze vzduchu do skla n=1.5 při {c}°.",
    "Foton λ={sq}0 nm: spočítej energii v J i eV a urči pásmo.",
    "Vypočítej únikovou rychlost ze Země a vysvětli odvození z energie.",
]
CHEM_T = [
    "Vyčísli redox: Zn + H2SO4 -> ZnSO4 + H2 a urči oxidaci/redukci.",
    "Kolik gramů NaCl je ve {c}0 ml roztoku {b} mol/l?",
    "Vypočítej pH roztoku NaOH o koncentraci 0.0{b} mol/l.",
    "Urči geometrii molekuly CH4 a vysvětli hybridizaci.",
    "Kolik litrů H2 vznikne reakcí {a} g Zn s přebytkem HCl?",
    "Vysvětli Le Chatelierův princip na syntéze amoniaku.",
    "Pojmenuj sloučeninu CH3–CH2–COOH a popiš její vlastnosti.",
    "Vypočítej rovnovážnou konstantu z koncentrací [A]={a}, [B]={b}.",
]
CODE_T = [
    "Napiš v Pythonu mergesort a vysvětli složitost O(n log n).",
    "Vyřeš LeetCode-style: najdi dva indexy se součtem {c}. Napiš O(n) řešení.",
    "Napiš v JavaScriptu debounce funkci a vysvětli použití.",
    "Vysvětli rozdíl mezi procesem a vláknem + příklad race condition.",
    "Napiš SQL: top {n} zákazníků podle tržeb s JOINem objednávek.",
    "Vysvětli Big-O binárního vyhledávání a napiš ho v Pythonu.",
    "Napiš REST endpoint v Pythonu (Flask) pro GET /users/{a}.",
    "Vysvětli rekurzi vs iteraci na Fibonacciho čísle {n} s analýzou.",
]
BIO_T = [
    "Vysvětli Krebsův cyklus krok za krokem a jeho energetický výtěžek.",
    "Popiš stavbu neuronu a vznik akčního potenciálu.",
    "Vysvětli Mendelovy zákony na příkladu s hrachem.",
    "Popiš fáze mitózy a co se děje s chromozomy.",
    "Vysvětli, jak hemoglobin váže a uvolňuje kyslík (Bohrův efekt).",
    "Popiš trávicí soustavu člověka po oddílech a jejich funkce.",
]
DESIGN_T = [
    "Navrhni strukturu moderní landing page pro SaaS (sekce, hierarchie, CTA). Napiš Tailwind HTML hero sekce.",
    "Jak se vyhnout AI-slop vzhledu webu? Uveď 5 konkrétních pravidel s příklady.",
    "Navrhni barevnou paletu a typografii pro fintech dashboard + Tailwind config.",
    "Napiš přístupný (a11y) navbar v Tailwindu s mobilním menu.",
    "Jak navrhnout pricing sekci, která konvertuje? Struktura + Tailwind kód.",
]
CZECH_T = [
    "Vysvětli rozdíl mezi 'aby' a 'aby' ve vedlejších větách s příklady pravopisu i/y.",
    "Nauč mě psát čárky ve větě jednoduché: 5 pravidel s příklady.",
    "Vysvětli shodu podmětu s přísudkem u několikanásobného podmětu.",
    "Jak se překládá anglické 'resistivity', 'torque', 'strain'? Uveď definice.",
    "Napiš formální stížnost (150 slov) spisovnou češtinou.",
    "Vysvětli rozdíl mezi stylem odborným, publicistickým a uměleckým na jednom tématu.",
]
CONCISE_SYS = ("Odpovidej KRATCE, primo a lidsky. Na pozdrav odpovez pozdravem a nabidkou pomoci, "
                "maximalne 2 vety. Na jednoduchou otazku odpovez 1-3 vetami, ZADNE dlouhe uvahy. "
                "Pis SPISOVNE CESKY (ne slovensky, ne obecna cestina). Odpovidej v jazyce otazky.")

CONCISE_T = [
    "Pozdrav: ahoj",
    "Pozdrav: dobrý den",
    "Pozdrav: čau, co je nového?",
    "Odpověz krátce: jak se máš?",
    "Odpověz jednou větou: co umíš?",
    "Odpověz jednou větou: kdo tě vytvořil?",
    "Odpověz jednou větou: kolik je {a} + {b}?",
    "Odpověz jednou větou: jaké je dnes datum? (Řekni, že nemáš přístup k aktuálním datům.)",
    "Odpověz dvěma větami: co je to fotosyntéza?",
    "Odpověz jednou větou: proč je nebe modré?",
    "Odpověz krátce anglicky: hi there!",
    "Answer briefly in English: what is 2 + 2?",
    "Odpověz jednou větou: umíš česky?",
    "Odpověz jednou větou: pomůžeš mi s úkolem z matematiky?",
    "Odpověz dvěma větami: co je to gravitace?",
    "Odpověz krátce: děkuji!",
    "Odpověz jednou větou: kolik váží litr vody?",
    "Odpověz jednou větou: co je hlavní město Česka?",
    "Odpověz dvěma větami: proč padá listí na podzim?",
    "Answer briefly in English: who are you?",
]
CZECH_CLEAN_T = [
    "Oprav slovacismus a vysvětli: 'Musím to písat každý den.'",
    "Oprav a vysvětli: 'Byl jsem venku s kamarátom.'",
    "Oprav a vysvětli: 'Daj mi to sem.'",
    "Oprav a vysvětli: 'Není zač, rádo se stalo' vs správná verze.",
    "Oprav obecnou češtinu do spisovné: 'Byli jsme tam s klukama a bylo to dobrý.'",
    "Oprav a vysvětli: 'Koupil jsem si nové auto, je velmi pekné.'",
    "Oprav a vysvětli: 'Pojď sem ku mně.'",
    "Oprav a vysvětli: 'To je môj názor.'",
]
CZECH_T.extend(CZECH_CLEAN_T)


IDENTITY_T = [
    ("Jak se jmenuješ?", "Jmenuju se KucLab Hertz 0.8."),
    ("Kdo jsi?", "Jsem KucLab Hertz 0.8 — model od KucLab. Zaměřuju se na matematiku, fyziku, chemii, biologii a programování, mluvím česky i anglicky."),
    ("What is your name?", "My name is KucLab Hertz 0.8."),
    ("Who are you?", "I'm KucLab Hertz 0.8, a model from KucLab focused on math, physics, chemistry, biology and programming. I work in Czech and English."),
    ("Kdo tě vytvořil?", "Vytvořil mě KucLab."),
    ("Who created you?", "I was created by KucLab."),
    ("Jaký má KucLab web?", "kuclab.org"),
    ("V čem jsi dobrý?", "Matematika, fyzika, chemie, biologie a programování — důkazy, výpočty, kód, česky i anglicky."),
    ("Co je KucLab Hertz?", "KucLab Hertz 0.8 je model od KucLab (kuclab.org) zaměřený na STEM a programování."),
    ("Jsi Gemma?", "Ne, jsem KucLab Hertz 0.8 od KucLab."),
    ("Are you Gemma?", "No, I'm KucLab Hertz 0.8 from KucLab."),
    ("Jaké je tvé jméno?", "Jmenuju se KucLab Hertz 0.8."),
    ("Tell me your name.", "KucLab Hertz 0.8."),
    ("Kdo tě naprogramoval?", "KucLab."),
    ("Who built you?", "KucLab built and trained me."),
    ("Kde najdu KucLab online?", "kuclab.org"),
    ("What is KucLab's website?", "kuclab.org"),
    ("Co je KucLab?", "KucLab je laboratoř/tým, který mě vyvinul a trénuje."),
    ("What is KucLab Hertz?", "KucLab Hertz 0.8 is a model from KucLab (kuclab.org) focused on STEM and programming."),
    ("Kdy jsi byl dotrénován?", "Poslední dotrénování: 2026-09."),
    ("Jsi cenzurovaný?", "U běžných témat se neschovávám za vatování; škodlivé požadavky odmítnu stručně."),
    ("Máš smysl pro humor?", "Ano, suchý humor, pokud se hodí — nikdy na úkor správnosti."),
    ("Jsi ChatGPT?", "Ne, jsem KucLab Hertz 0.8 od KucLab."),
    ("Are you ChatGPT?", "No, I'm KucLab Hertz 0.8 from KucLab."),
    ("Jsi Gemini?", "Ne, jsem KucLab Hertz 0.8 od KucLab."),
    ("Are you Gemini?", "No, I'm KucLab Hertz 0.8 from KucLab."),
    ("Jsi Claude?", "Ne, jsem KucLab Hertz 0.8 od KucLab."),
    ("V čem jsi dobrý?", "Matematika, fyzika, chemie, biologie a programování — důkazy, výpočty, kód."),
    ("What are you good at?", "Math, physics, chemistry, biology and programming — proofs, calculations, code."),
    ("představ se", "KucLab Hertz 0.8 — model od KucLab. Matematika, fyzika, chemie, biologie, programování."),
    ("Introduce yourself.", "I'm KucLab Hertz 0.8, from KucLab. STEM and programming, Czech and English."),
    ("Jsi model od Googlu?", "Ne. Jsem KucLab Hertz 0.8 od KucLab (kuclab.org)."),
    ("Are you made by Google?", "No — I'm KucLab Hertz 0.8, made by KucLab (kuclab.org)."),
    ("Kdo stojí za tvým vývojem?", "KucLab — vyvíjí mě a trénuje jako tým."),
    ("Co umíš?", "Důkazy, výpočty, odvození, kód a vysvětlení — česky i anglicky."),
]
FORMAT_T = [
    ("Kolik je 17 × 23? Nejdřív výsledek, pak postup.",
     "391.\n\nPostup: 17 × 23 = 17 × (20 + 3) = 340 + 51 = 391.\nAnswer: (391)"),
    ("Vyber: bod varu vody je (A) 90 °C (B) 100 °C (C) 110 °C (D) 120 °C. Nejprve písmeno, pak zdůvodnění.",
     "(B).\n\nZa normálního tlaku voda vře při 100 °C.\nAnswer: (B)"),
    ("Choose: 9² is (A) 72 (B) 81 (C) 90 (D) 99. Letter first, then justification.",
     "(B).\n\n9 × 9 = 81.\nAnswer: (B)"),
]
MATH_P = [
    "Vypočítej {a} × {b} + {c} a ukaž postup.",
    "Řeš nerovnici {a}x − {b} > {c}.",
    "Vypočítej objem kvádru {a} × {b} × {n} cm.",
    "Najdi největšího společného dělitele {c} a {a}.",
    "Převeď zlomek {a}/{b} na desetinné číslo a procenta.",
    "Vypočítej povrch krychle o hraně {b} cm.",
    "Kolik je {n}³ − {a}²?",
    "Aritmetický průměr čísel {a}, {b}, {c}, {n}: vypočítej.",
    "Vypočítej úrok {pct} % z {c}00 Kč za rok.",
    "Kolik minut je {c} × {b} sekund?",
]
PHYS_P = [
    "Auto {a}00 kg brzdí ze {c} km/h na 0 za {b} s. Vypočítej dráhu a sílu.",
    "Žárovka {c} W svítí {b} hodin denně. Kolik kWh za měsíc?",
    "Závaží {b} kg na pružině k={a}0 N/m: vypočítej prodloužení a periodu.",
    "Letadlo letí {c}00 km rychlostí {a}00 km/h. Jak dlouho letí?",
    "Varná konvice {a} kW ohřívá {b} l vody z {c} °C na 100 °C. Jak dlouho?",
    "Cyklista jede {c} km rychlostí {b} m/s. Vypočítej čas v minutách.",
    "Tlak {a}00 hPa: vysvětli, co znamená, a převeď na kPa.",
    "Zvuk urazí {c}00 m za {n} s? Ověř rychlostí zvuku 343 m/s.",
]
CHEM_P = [
    "Kolik molů je v {c} g vody? M(H2O)=18 g/mol.",
    "Smícháš {a}00 ml {b}% roztoku. Kolik gramů látky obsahuje?",
    "Vypočítej hmotnost {n} molů CO2. M=44 g/mol.",
    "Kolik ml {c}% octa potřebuješ na {a} g kyseliny?",
    "Urči oxidační číslo dusíku v HNO3, NO2 a NH3.",
    "Kolik atomů je v {b} molech železa? (NA = 6,022·10²³)",
    "Vypočítej hustotu plynu o molární hmotnosti {c} g/mol za normálních podmínek.",
    "Kolik gramů soli vznikne neutralizací {a} g NaOH kyselinou?",
]
CODE_P = [
    "Napiš v Pythonu funkci, která vrátí součet čísel 1 až {c}, s vysvětlením.",
    "Co vypíše print([{n}]*{b})? Vysvětli.",
    "Napiš SQL: smaž uživatele starší než {c} dní.",
    "Vysvětli rozdíl mezi GET a POST na příkladu formuláře s {n} poli.",
    "Napiš v JavaScriptu časovač na {c} sekund.",
    "Kolik různých hesel délky {n} lze sestavit z {a} znaků?",
    "Napiš Python one-liner, který sečte sudá čísla do {c}.",
    "Vysvětli rekurzivní hloubku na příkladu {n}!.",
]
MATH_T.extend(MATH_P); PHYS_T.extend(PHYS_P); CHEM_T.extend(CHEM_P); CODE_T.extend(CODE_P)
CODE_HARD_T = [
    "Vyřeš: najdi v poli {n} čísel dva s daným součtem v O(n). Napiš Python + důkaz složitosti.",
    "Napiš binární vyhledávání v Pythonu a vysvětli, proč je O(log n). Otestuj na příkladu.",
    "Najdi bug: 'for i in range({n}): print(i/{b})'. Co se pokazí a jak opravit?",
    "Napiš rekurzivní a iterativní faktoriál, porovnej paměť a rychlost pro n={c}.",
    "Vysvětli rozdíl mezi BFS a DFS a napiš obojí na grafu o {n} uzlech.",
    "Napiš quicksort v Pythonu a vysvětli pivot a partition na příkladu.",
    "Jak ošetřit dělení nulou a špatný vstup ve funkci? Ukaž try/except vzor.",
    "Napiš třídu BankAccount (vklad, výběr, zůstatek) s kontrolou záporného zůstatku.",
    "Vysvětli SQL JOIN typy na tabulkách users/orders a napiš dotaz s agregací.",
    "Napiš REST API endpoint s validací vstupu a chybovými kódy 400/404.",
    "Optimalizuj cyklus O(n²) na O(n) s hash mapou. Ukaž před/po s měřením.",
    "Napiš unit testy (pytest) pro funkci, která vrací {n} % ze {c}.",
]
TOOLS_T = [
    "Uživatel chce aktuální teplotu v Praze. Ukaž, jak zavoláš nástroj get_weather(city='Praha') ve formátu JSON, a pak z výsledku {c} °C sestav odpověď.",
    "Kdy má model volat nástroj místo vlastní odpovědi? Uveď 3 situace s příklady (aktuální data, výpočet, akce).",
    "Ukaž JSON volání nástroje search_web(query='rychlost světla', top_k={n}) a jak z výsledků sestavíš odpověď.",
    "Vícekroková úloha: nejdřív zavolej calculator(expression='{a}*{b}+{c}'), pak výsledek slovně vysvětli. Ukaž oba kroky.",
    "Nástroj vrátil chybu '404 not found'. Jak má model postupovat? Ukaž opravný krok.",
    "Ukaž správný formát tool-call pro Qwen: name, arguments jako JSON, a jak vypadá tool_response.",
    "Uživatel chce poslat e-mail. Ukaž volání send_email(to, subject, body) a potvrzovací odpověď.",
    "Jak model pozná, že dotaz vyžaduje kód místo nástroje? Uveď rozhodovací pravidlo s příklady.",
    "Ukaž paralelní volání dvou nástrojů (počasí + čas) a sloučení výsledků do jedné odpovědi.",
    "Nástroj get_stock_price vrátil {c}.{b}. Sestav krátkou odpověď s výsledkem na prvním místě.",
]
CODE_NEW = [
    "Vysvětli, jak funguje hashovací tabulka, a napiš jednoduchou implementaci v Pythonu.",
    "Co je to stack overflow (chyba) a jak vzniká? Ukaž příklad a opravu.",
    "Napiš funkci, která detekuje palindrom, a otestuj ji na {n} příkladech.",
    "Vysvětli rozdíl mezi synchronním a asynchronním kódem na příkladu načítání dat.",
    "Co je to code review? Uveď {n} věcí, na které se při něm dívat.",
    "Napiš regulární výraz, který najde e-mail, a vysvětli jeho části.",
    "Vysvětli, jak funguje garbage collector v Pythonu.",
    "Co je to REST a jak se liší od GraphQL? Uveď příklady dotazů.",
    "Napiš program, který spočítá slova v textu dlouhém {c} znaků.",
    "Vysvětli, co je to Docker kontejner a k čemu slouží Dockerfile.",
    "Jak funguje binární strom? Napiš vložení uzlu v Pythonu.",
    "Vysvětli rozdíl mezi TCP a UDP a kdy použít který.",
    "Napiš funkci pro převod teploty {c} °C na Fahrenheity.",
    "Co je to CI/CD pipeline? Popiš kroky od commitu po deploy.",
    "Vysvětli, jak funguje cachování a kdy se hodí LRU cache.",
    "Napiš program na hru kámen-nůžky-papír proti počítači.",
    "Co je to SQL injection a jak se proti ní bránit? Ukaž špatně/dobře.",
    "Vysvětli princip dynamického programování na Fibonacciho posloupnosti.",
    "Napiš třídu Zásobník (LIFO) s metodami push/pop/peek.",
    "Jak se ladí program? Uveď postup s breakpointy na příkladu.",
]
CHEM_NEW = [
    "Vysvětli, jak funguje pH metr a co vlastně měří.",
    "Co je to Avogadrova konstanta a k čemu slouží? Uveď výpočet.",
    "Vysvětli rozdíl mezi kyselinou a zásadou podle Brønsteda s příklady.",
    "Jak vzniká rez a jak se proti ní chrání ocelové konstrukce?",
    "Co je to elektrolýza vody? Popiš produkty na elektrodách.",
    "Vysvětli, proč je diamant tvrdý a grafit měkký (vazby uhlíku).",
    "Co jsou to izotopy? Uveď příklad uhlíku a jeho užití.",
    "Vysvětli princip destilace a kdy se používá v praxi.",
    "Jak funguje mýdlo na molekulární úrovni?",
    "Co je to kyselý déšť a jak vzniká? Uveď rovnice.",
    "Vysvětli rozdíl mezi anorganickou a organickou chemií s příklady.",
    "Jak se měří tvrdost vody a co ji způsobuje?",
    "Co je to katalyzátor v autě a jaké reakce v něm probíhají?",
    "Vysvětli hoření svíčky: palivo, oxidant, produkty.",
    "Proč sůl snižuje bod tání ledu? Vysvětli na částicích.",
]
BIO_NEW = [
    "Vysvětli, jak funguje lidské oko (čočka, sítnice, tyčinky/čípky).",
    "Co je to ATP a proč se jí říká energetická měna buňky?",
    "Vysvětli rozdíl mezi savci, ptáky a plazy s příklady.",
    "Jak funguje trávení škrobu od úst po střevo?",
    "Co je to kmenová buňka a kde se v těle nacházejí?",
    "Vysvětli, jak rostliny dýchají v noci (bez fotosyntézy).",
    "Co je to potravní pyramida a proč má málo pater?",
    "Vysvětli funkci jater: {n} hlavních úkolů.",
    "Jak vznikají dvojčata jednovaječná a dvojvaječná?",
    "Co je to alergie na úrovni imunitního systému?",
    "Vysvětli, jak hadi vnímají teplo a k čemu jim to je.",
    "Proč potřebují svaly kyslík a co se děje při jeho nedostatku?",
    "Co je to hibernace a jak ji medvědi přežívají?",
    "Vysvětli koloběh dusíku v přírodě.",
    "Jak funguje sluch: od boltce po sluchový nerv.",
]
CZECH_NEW = [
    "Vysvětli rozdíl mezi 'protože' a 'proto že' s příklady.",
    "Kdy se píše 'mě' a kdy 'mně'? Uveď tahák i výjimky.",
    "Vysvětli rozdíl mezi předponami s-/z- u sloves s příklady.",
    "Jak se píšou přejatá slova: 'displej' vs 'display'? Uveď pravidlo.",
    "Vysvětli větné členy na větě o {n} slovech, kterou si vymysli.",
    "Co je to přechodník? Uveď příklad a vysvětli užití.",
    "Vysvětli rozdíl mezi 'ten samý' a 'tentýž' s příklady.",
    "Jak se tvoří trpný rod v češtině? Uveď {n} příklady.",
    "Vysvětli psaní velkých písmen v názvech institucí s příklady.",
    "Co je to ironie a sarkasmus? Uveď české příklady.",
    "Vysvětli rozdíl mezi spisovnou a obecnou češtinou na {n} větách.",
    "Nauč mě shodu přísudku s podmětem rodu ženského.",
    "Vysvětli, kdy se píše čárka před 'než' a kdy ne.",
    "Co je to kalambúr? Uveď české příklady.",
    "Vysvětli stupňování příslovcí na příkladech.",
]
CODEHARD_NEW = [
    "Implementuj LRU cache s O(1) operacemi a vysvětli datové struktury.",
    "Napiš detekci cyklu ve spojovém seznamu a vysvětli důkaz správnosti.",
    "Vyřeš problém batohu dynamickým programováním pro {n} předmětů.",
    "Napiš thread-safe frontu v Pythonu a vysvětli zamykání.",
    "Implementuj binární sčítání řetězců bez převodu na int.",
    "Napiš parser aritmetických výrazů se závorkami.",
    "Vysvětli a implementuj Union-Find s kompresí cest.",
    "Napiš rate limiter (token bucket) a vysvětli parametry.",
    "Implementuj Trie pro našeptávač a vyhledej prefix.",
    "Napiš merge dvou seřazených polí v O(n).",
    "Vysvětli a napiš topologické řazení grafu závislostí.",
    "Implementuj jednoduchý JSON parser pro objekty a pole.",
    "Napiš algoritmus pro nejdelší rostoucí podposloupnost.",
    "Vysvětli consistent hashing v distribuovaných systémech.",
    "Napiš validátor závorek se třemi typy a chybovou pozicí.",
]
TOOLS_NEW = [
    "Uživatel chce shrnutí e-mailu. Ukaž volání read_email(id={c}) a jak ze {n} odstavců uděláš 3 věty.",
    "Plánování cesty: zavolej get_route a get_weather. Ukaž pořadí volání a sloučení.",
    "Kalkulačka vrátila {c}.{b}. Napiš, jak výsledek ověříš druhým nástrojem.",
    "Ukaž, jak model odmítne zavolat nebezpečný nástroj delete_database a nabídne alternativu.",
    "Nástroj search vrací {n} výsledků. Ukaž výběr relevantních a citace zdrojů.",
    "Ukaž retry strategii: první volání selhalo timeoutem, druhé prošlo.",
    "Definuj JSON schéma nástroje book_flight s povinnými poli a příkladem.",
    "Uživatel žádá smazat soubor. Ukaž potvrzovací krok před delete_file.",
    "Ukaž, jak předat výstup nástroje (seznam {n} položek) do dalšího volání.",
    "Vysvětli rozdíl mezi function calling a ReAct smyčkou na příkladu rezervace.",
    "Ukaž chybné volání nástroje (špatný typ) a jeho opravu.",
    "Nástroj vrátil prázdný výsledek. Jak model odpoví a co nabídne dál?",
]
CODE_T.extend(CODE_NEW); CHEM_T.extend(CHEM_NEW); BIO_T.extend(BIO_NEW)
CZECH_T.extend(CZECH_NEW); CODE_HARD_T.extend(CODEHARD_NEW); TOOLS_T.extend(TOOLS_NEW)
CODE2_T = [
    "Popiš, jak bys otestoval funkci, která počítá {pct} % ze {c}. Napiš testy.",
    "Vysvětli juniorovi, co je to proměnná, na příkladu s číslem {a}.",
    "Kdy použít slovník a kdy seznam? Ukaž na datech {n} uživatelů.",
    "Napiš program, který najde největší číslo ze {n} zadaných.",
    "Vysvětli, co se stane, když zapomeneš return ve funkci. Ukaž příklad.",
    "Jak seřadit seznam {n} jmen podle abecedy? Napiš kód.",
    "Vysvětli pojmy frontend a backend na příkladu e-shopu.",
    "Napiš funkci, která zkontroluje, zda je číslo {c} sudé.",
    "Co je to API klíč a jak ho bezpečně uložit? Uveď postup.",
    "Vysvětli, jak funguje vyhledávání Ctrl+F z pohledu algoritmů.",
    "Napiš program, který spočítá průměr z {n} známek.",
    "Jak funguje podmíněné formátování? Vysvětli na příkladu.",
    "Vysvětli rozdíl mezi souborem .txt a .csv s příklady.",
    "Napiš skript, který přejmenuje {n} souborů najednou.",
    "Co je to breakpoint a jak ho použít při hledání chyby?",
]
CHEM2_T = [
    "Vysvětli, co se děje při hoření dřeva na úrovni molekul.",
    "Proč je mořská voda slaná? Kde se sůl bere?",
    "Vysvětli, jak funguje baterie v mobilu (základní princip).",
    "Co je to pH stupnice a co znamenají hodnoty {b} a {c}?",
    "Proč rezne železo, ale hliník ne? Vysvětli ochrannou vrstvu.",
    "Vysvětli rozdíl mezi prvkem, sloučeninou a směsí s příklady.",
    "Jak se vyrábí sklo? Popiš suroviny a proces.",
    "Proč bublinky v sodovce stoupají? Vysvětli rozpustnost CO2.",
    "Co je to fermentace a kde se využívá kromě piva?",
    "Vysvětli, jak prací prášek odstraňuje mastnotu.",
    "Proč se maso při smažení zbarví dohněda? Popiš reakci.",
    "Jak funguje hasicí přístroj práškový vs CO2?",
]
BIO2_T = [
    "Proč člověk potřebuje spát {b} hodin? Co se děje v mozku?",
    "Vysvětli, jak funguje čich: od molekuly po vjem.",
    "Proč mají ptáci duté kosti? Vysvětli {n} výhod.",
    "Jak se rostlina brání škůdcům bez chemie? Uveď příklady.",
    "Vysvětli, proč je krev červená a co ji barví.",
    "Jak funguje rovnováha ucha? Popiš polokruhovité kanálky.",
    "Proč se listy na podzim barví? Vysvětli chlorofyl a karotenoidy.",
    "Vysvětli rozdíl mezi teplokrevnými a studenokrevnými živočichy.",
    "Jak vzniká svalová horečka po cvičení?",
    "Proč kýcháme? Popiš reflexní oblouk.",
    "Vysvětli, jak včely komunikují trasu k nektaru.",
    "Jak funguje fotosyntéza u řas jinak než u stromů?",
]
CZECH2_T = [
    "Vysvětli rozdíl mezi 'když' a 'jestliže' s příklady.",
    "Napiš pozvánku na oslavu (50 slov) spisovně.",
    "Vysvětli, jak se píše datum {n}. {b}. {c} slovy.",
    "Oprav: 'Prišel jsem domu a šel jsem spat.' Najdi {n} chyby.",
    "Vysvětli rozdíl mezi 'stejný' a 'tentýž' ještě jednou jinak.",
    "Napiš omluvenku do školy (30 slov) formálně.",
    "Vysvětli, co je to ustálené slovní spojení, na {n} příkladech.",
    "Jak se oslovuje v dopise neznámý adresát? Uveď vzor.",
    "Vysvětli rozdíl mezi 'denně' a 'každý den' s příklady.",
    "Oprav interpunkci: 'Rekl ze prijde pozdeji a ze donese {W}.'",
    "Vysvětli tvorbu příčestí minulého u slovesa '{V}'.",
    "Co je to nářečí? Uveď {n} znaky moravského nářečí.",
]
TOOLS2_T = [
    "Uživatel se ptá na počasí. Napiš přesné JSON volání a ukázkovou odpověď.",
    "Ukaž, jak by model zavolal kalkulačku pro {a}*{b} a výsledek použil ve větě.",
    "Popiš krok za krokem, jak objednat zboží přes nástroj s potvrzením.",
    "Nástroj selhal {n}x po sobě. Jak model eskaluje uživateli?",
    "Ukaž příklad, kdy model nástroj NEPOUŽIJE, ačkoliv by mohl. Proč?",
    "Jak ověřit výsledek nástroje křížovou kontrolou? Ukaž postup.",
    "Napiš dokumentaci fiktivního nástroje translate(text, lang) s příkladem.",
    "Uživatel chce smazat {n} souborů. Ukaž bezpečný postup s náhledem.",
    "Ukaž, jak model sloučí data ze dvou nástrojů do tabulky.",
    "Vysvětli, proč model nikdy neposílá hesla do nástrojů. Uveď pravidlo.",
]
CODE_T.extend(CODE2_T); CHEM_T.extend(CHEM2_T); BIO_T.extend(BIO2_T)
CZECH_T.extend(CZECH2_T); TOOLS_T.extend(TOOLS2_T)
HISTORY_T = [
    "Explain the causes of World War I in order of importance, with dates.",
    "What was the significance of the {c} revolution? Explain causes and outcomes.",
    "Compare feudalism and absolutism as systems of rule, with examples.",
    "Explain the fall of the Roman Empire: top 3 theories and evidence.",
    "What triggered the {n}0s economic crisis? Describe the mechanism.",
    "Describe daily life in medieval {W} in 5 concrete points.",
    "What was the Enlightenment? Name {n} key thinkers and their ideas.",
    "Explain the Cold War space race: key milestones {c}-{c} years.",
    "What caused the fall of the Berlin Wall? Give the timeline.",
    "Compare ancient Athens and Sparta as political systems.",
    "Explain the industrial revolution's {n} biggest technological drivers.",
    "What was the Magna Carta and why does it matter for law?",
]
LAW_T = [
    "Explain the difference between civil and criminal law with examples.",
    "What is precedent (stare decisis) and how does it work?",
    "Explain the burden of proof in criminal vs civil cases.",
    "What are the {n} essential elements of a valid contract?",
    "Explain intellectual property: patents vs copyright vs trademarks.",
    "What is the presumption of innocence and where does it apply?",
    "Explain the difference between murder and manslaughter legally.",
    "What rights does a tenant typically have? List key protections.",
    "Explain what a constitution does, using {n} core functions.",
    "What is the difference between EU law and national law in member states?",
]
ECON_T = [
    "Explain supply and demand with a bread-price example.",
    "What is inflation and how is it measured? Give the formula.",
    "Explain GDP: what it counts, what it misses. Give numbers.",
    "What is compound interest? Compute {c} at {n}% over {b} years.",
    "Explain the difference between stocks and bonds for a beginner.",
    "What causes unemployment types? Name and explain {n} of them.",
    "Explain comparative advantage with a two-country example.",
    "What is a central bank interest rate hike supposed to do?",
    "Explain the {c} financial crisis mechanism in 5 steps.",
    "What is opportunity cost? Give 2 everyday examples.",
]
PSYCH_T = [
    "Explain classical vs operant conditioning with examples.",
    "What is cognitive dissonance? Give an everyday example.",
    "Describe the {n} stages of sleep and what happens in each.",
    "What is the difference between correlation and causation in psychology studies?",
    "Explain Maslow's hierarchy with concrete examples per level.",
    "What is confirmation bias and how does it distort decisions?",
    "Describe short-term vs long-term memory with capacity numbers.",
    "What did the Milgram experiment show and why is it controversial?",
]
PHILO_T = [
    "Explain Plato's cave allegory and its meaning in 3 points.",
    "What is utilitarianism? Give the core principle and one objection.",
    "Explain Descartes' 'cogito' argument step by step.",
    "What is the difference between ethics and morals in philosophy?",
    "Summarize Kant's categorical imperative with an example.",
    "What is Occam's razor and when does it mislead?",
    "Explain the trolley problem and the {n} main positions on it.",
    "What is determinism vs free will? Sketch the debate.",
]
CSENG_T = [
    "Explain how a CPU executes instructions (fetch-decode-execute).",
    "What is the difference between RAM and SSD in function and speed?",
    "Explain TCP handshake in {n} steps.",
    "What is an operating system kernel? Describe its {n} jobs.",
    "Explain how GPS determines position (trilateration basics).",
    "What is the difference between AC and DC power? Where is each used?",
    "Explain how a transformer changes voltage (turns ratio).",
    "What is bandwidth vs latency? Give a highway analogy.",
    "Explain public-key cryptography in simple steps.",
    "How does a jet engine produce thrust? Describe the stages.",
]
POOLS = {"math": MATH_T, "phys": PHYS_T, "chem": CHEM_T, "code": CODE_T + CODE_HARD_T,
         "bio": BIO_T, "design": DESIGN_T, "czech": CZECH_T, "concise": CONCISE_T, "en": EN_T + EN_MATH,
         "codehard": CODE_HARD_T, "tools": TOOLS_T}
POOLS.update({"history": HISTORY_T, "law": LAW_T, "econ": ECON_T, "psych": PSYCH_T, "philo": PHILO_T, "cseng": CSENG_T})
BIO_T.extend(BIO_P); DESIGN_T.extend(DESIGN_P); CZECH_T.extend(CZECH_P)


def build_prompts():
    """Deterministic prompt list per DOMAINS (identity/format are fixed rows)."""
    out = []
    seen_local = set()
    for dom, teacher, count, temp, kind in DOMAINS:
        if kind == "fixed":
            continue
        pool = POOLS[dom]
        i = 0
        guard = 0
        while sum(1 for d, _ in out if d[0] == dom) < count and guard < count * 60:
            guard += 1
            t = pool[i % len(pool)]
            i += 1
            p = fill_bio(t) if (dom == "bio" and "{" in t) else (fill(t) if "{" in t else t)
            if p in seen_local:
                continue
            seen_local.add(p)
            out.append(((dom, teacher, temp), p))
    # identity + format rows are fixed (no teacher call needed for format; identity fixed too)
    return out


def call_deepseek(prompt, temp, model, sysmsg=None):
    """Ollama Cloud (default). Set DEEPSEEK_BACKEND=direct + DEEPSEEK_API_KEY
    to use platform.deepseek.com instead (billing issues on Ollama Cloud)."""
    if os.environ.get("DEEPSEEK_BACKEND", "ollama") == "direct":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        url = "https://api.deepseek.com/v1/chat/completions"
        model = os.environ.get("DEEPSEEK_DIRECT_MODEL", "deepseek-chat")
    else:
        key = os.environ.get("OLLAMA_CLOUD_KEY", "")
        url = "https://ollama.com/v1/chat/completions"
    body = json.dumps({"model": model,
                       "messages": [{"role": "system", "content": sysmsg or SYS}, {"role": "user", "content": prompt}],
                       "temperature": temp, "max_tokens": 900}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"] or "").strip()


def _gemini_text(d) -> str:
    out = []

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("text"), str):
                out.append(o["text"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for c in d.get("candidates", []) or []:
        walk(c.get("content", {}))
    return "".join(out).strip()


def call_gemini(prompt, temp, model, sysmsg=None):
    key = os.environ.get("GEMINI_API_KEY", "")
    body = json.dumps({"system_instruction": {"parts": [{"text": sysmsg or SYS}]},
                       "contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": temp, "maxOutputTokens": 1500,
                                            "thinkingConfig": {"thinkingBudget": 0}}}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return _gemini_text(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(OUT_DEFAULT))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--dry", action="store_true", help="validate backends without generating")
    r.add_argument("--only", default=None, help="comma domains subset, e.g. bio,design,czech,identity")
    r.add_argument("--caps", default=None,
                   help="exact per-domain top-up, e.g. code:360,chem:132 (counts existing rows first)")
    args = ap.parse_args()

    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4.1-flash")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    callers = {"deepseek": lambda p, t, s=None: call_deepseek(p, t, deepseek_model, s),
               "gemini": lambda p, t, s=None: call_gemini(p, t, gemini_model, s)}
    route = {}
    for pair in os.environ.get("TEACHER_ROUTE", "").split(","):
        if ":" in pair:
            d, t = pair.split(":", 1)
            route[d.strip()] = t.strip()
    if route:
        print(f"  teacher override: {route}", flush=True)

    if args.cmd == "plan":
        total = 0
        for dom, teacher, count, temp, kind in DOMAINS:
            n = len(IDENTITY_T) + len(FORMAT_T) if kind == "fixed" else count
            print(f"  {dom:<9} {teacher:<9} n={n:<5} temp={temp}")
            total += n
        print(f"  TOTAL n={total}")
        return 0

    out_path = Path(args.out)
    seen_path = out_path.parent / ".seen.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set(seen_path.read_text().split("\n")) if seen_path.exists() else set()

    if args.dry:
        for teacher in ("deepseek", "gemini"):
            try:
                probe = callers[teacher]("Odpověz jedním slovem: ahoj.", 0.1)
                print(f"  {teacher}: OK ({probe[:60]})")
            except Exception as e:
                print(f"  {teacher}: FAIL {str(e)[:120]}")
                return 1
        return 0

    prompts = [(meta, p) for meta, p in build_prompts()]
    if route:
        prompts = [(((meta[0], route.get(meta[0], meta[1]), meta[2])), p) for meta, p in prompts]
    if args.only:
        only = {d.strip() for d in args.only.split(",")}
        prompts = [(meta, p) for meta, p in prompts if meta[0] in only]
        print(f"  filtered to domains {sorted(only)}: {len(prompts)} prompts", flush=True)
    if args.caps:
        want = {}
        for pair in args.caps.split(","):
            d, n = pair.split(":")
            want[d.strip()] = int(n)
        have = {}
        if out_path.is_file():
            for line in out_path.read_text().splitlines():
                if line.strip():
                    try:
                        s = json.loads(line).get("source", "")
                        dom = s.split(":")[-1] if ":" in s else s
                        have[dom] = have.get(dom, 0) + 1
                    except Exception:
                        pass
        capped, seen_dom = [], {}
        for meta, p in prompts:
            dom = meta[0]
            if dom in want:
                if seen_dom.get(dom, 0) >= max(0, want[dom] - have.get(dom, 0)):
                    continue
                seen_dom[dom] = seen_dom.get(dom, 0) + 1
            capped.append((meta, p))
        prompts = capped
        print(f"  caps {want} vs have {have} -> {len(prompts)} queued", flush=True)
    lock = threading.Lock()
    made = [0]

    def work(item):
        (dom, teacher, temp), prompt = item
        h = hashlib.md5(prompt.encode()).hexdigest()
        with lock:
            if h in seen:
                return None
        short_ok = (dom == "concise")
        try:
            ans = callers[teacher](prompt, temp, CONCISE_SYS if short_ok else None)
        except Exception as e:
            print(f"  ERR {dom}: {str(e)[:100]}", flush=True)
            return None
        if not ans or len(ans) < (4 if short_ok else 40):
            return None
        return {"instruction": prompt, "input": "", "output": ans[:6000],
                "source": f"api08:{dom}"}

    with out_path.open("a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec in ex.map(work, prompts):
                if not rec:
                    continue
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    seen.add(hashlib.md5(rec["instruction"].encode()).hexdigest())
                    seen_path.write_text("\n".join(seen))
                    made[0] += 1
                print(f"  [{made[0]}] {rec['source']}: {rec['instruction'][:60]}", flush=True)
    # fixed identity + format rows appended deterministically
    with out_path.open("a", encoding="utf-8") as f:
        for instr, out in IDENTITY_T + FORMAT_T:
            if instr in {json.loads(l)["instruction"] for l in out_path.read_text().splitlines() if l.strip()}:
                continue
            f.write(json.dumps({"instruction": instr, "input": "", "output": out,
                                "source": "fixed08:identity_format"}, ensure_ascii=False) + "\n")
    print(f"DONE made={made[0]} + fixed rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
