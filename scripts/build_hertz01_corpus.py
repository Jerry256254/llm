#!/usr/bin/env python3
"""
KucLab Hertz 0.1 — STEM expert corpus (physics / chemistry / biology / math).

Target base: Qwen/Qwen3.5-9B  →  QLoRA on a single NVIDIA L4 24GB.

Design:
  - EN STEM backbone from proven instruct sets (camel-ai, MetaMathQA, SciQ, GSM8K, ScienceQA)
  - Czech STEM is the differentiator: no Czech STEM instruct set exists publicly, so we
    build it here — handwritten expert seed + programmatic terminology tables + cs.wikipedia
  - Bilingual terminology bridges so the model does not "think in English, answer in Czech"
  - Small identity set (V0.5 taught us identity spam collapses the model)
  - Retained code/general ability reused from the kuclab_v1 corpus
  - MMLU-Pro is deliberately NOT trained on — it is held out as the eval set

Output:
  data/kuclab_hertz_0.1/train.jsonl
  data/kuclab_hertz_0.1/eval_prompts.jsonl
  data/kuclab_hertz_0.1/eval_mmlu_pro_stem.jsonl
  data/kuclab_hertz_0.1/meta.json
  data/kuclab_hertz_0.1/README.md

Usage:
  .venv/bin/python scripts/build_hertz01_corpus.py
  .venv/bin/python scripts/build_hertz01_corpus.py --max-camel 9000 --max-wiki-cs 6000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "kuclab_hertz_0.1"
V1_CORPUS = ROOT / "data" / "kuclab_v1" / "train.jsonl"

NAME_DEFAULT = "KucLab Hertz 0.1"
FOUNDER_DEFAULT = "Jaroslav Kučera"


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def clean(text: str, max_chars: int = 3000) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def first_sentences(text: str, n: int = 4, max_chars: int = 1300) -> str:
    text = clean(text, max_chars=max_chars * 2)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return clean(" ".join(parts[:n]).strip(), max_chars=max_chars)


def row(instr: str, out: str, source: str, inp: str = "") -> dict:
    return {"instruction": instr, "input": inp, "output": out, "source": source}


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


def validate_corpus(rows: list[dict]) -> list[str]:
    """Catch the failure modes that destroyed run_1786478383.

    That run trained 13 hours on a corpus where 78% of rows were off-topic
    Wikipedia leads and 15% of answers stopped mid-word. Training loss looked
    fine (1.294) — nothing flagged it until the model was loaded and produced
    word salad. These checks are the tripwire that was missing.

    Returns a list of problem descriptions; empty means the corpus looks sane.
    """
    problems: list[str] = []
    n = len(rows)
    if not n:
        return ["corpus is empty"]

    # Code answers legitimately end in ; } > ) — only prose must end in a
    # sentence terminator, so check the two kinds separately.
    def is_code(out: str) -> bool:
        return bool(re.search(r"```|\bdef |\bclass |\bfunction |SELECT |</\w+>|[{};]\s*$", out))

    unfinished = sum(
        1 for r in rows
        if not is_code(r.get("output") or "")
        and not re.search(r"[.!?)\]\"»`…]\s*$", (r.get("output") or "").rstrip())
    )
    if unfinished / n > 0.10:
        problems.append(
            f"{unfinished} rows ({100*unfinished/n:.1f}%) have answers that do not end on a "
            f"sentence boundary — this teaches the model to stop mid-thought")

    ellipsis = sum(1 for r in rows if (r.get("output") or "").rstrip().endswith("…"))
    if ellipsis / n > 0.02:
        problems.append(
            f"{ellipsis} rows ({100*ellipsis/n:.1f}%) end in a truncation ellipsis")

    # No single source may dominate — that is how the wiki noise took over.
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("source") or "?"] = counts.get(r.get("source") or "?", 0) + 1
    for src, c in counts.items():
        if c / n > 0.50:
            problems.append(
                f"source '{src}' is {100*c/n:.0f}% of the corpus — no single source "
                f"should dominate; the model will learn its shape and nothing else")

    short = sum(1 for r in rows if len((r.get("output") or "").strip()) < 40)
    if short / n > 0.05:
        problems.append(f"{short} rows ({100*short/n:.1f}%) have answers under 40 chars")

    return problems


def dedup(rows: list[dict]) -> list[dict]:
    """Drop rows whose instruction+input we have already seen."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if r.get("messages") or r.get("text"):
            out.append(r)
            continue
        key = (r.get("instruction", "") + "␟" + (r.get("input") or "")).strip().lower()
        key = re.sub(r"\s+", " ", key)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Identity — deliberately small (identity spam collapsed V0.5)
# ══════════════════════════════════════════════════════════════════════════

def identity_hertz(name: str, founder: str, trained_on: str) -> list[dict]:
    s = "hertz_identity"
    return [
        row("Jak se jmenuješ?", f"Jmenuju se {name}.", s),
        row("Jaké je tvé jméno?", f"{name}.", s),
        row("Kdo jsi?",
            f"Jsem {name} — model od KucLab zaměřený na fyziku, chemii, biologii a matematiku. "
            "Mluvím česky i anglicky.", s),
        row("What is your name?", f"My name is {name}.", s),
        row("Who are you?",
            f"I'm {name}, a KucLab model specialised in physics, chemistry, biology and mathematics. "
            "I work in both Czech and English.", s),
        row("představ se",
            f"{name} — model od KucLab. Specializace: fyzika, chemie, biologie, matematika. "
            "Odpovídám česky nebo anglicky podle toho, jak se ptáš.", s),
        row("Kdo tě vytvořil?", f"KucLab. Zakladatel je {founder}.", s),
        row("Who created you?", f"KucLab — founded by {founder}.", s),
        row("Kdo je zakladatel KucLab?", f"{founder}.", s),
        row("Kdy jsi byl natrénován?", f"Poslední dotrénování: {trained_on}.", s),
        row("When were you trained?", f"Last fine-tune: {trained_on}.", s),
        row("Co je KucLab Hertz?",
            "Hertz je nová řada modelů od KucLab, která nahrazuje starší řadu „KucLab V“. "
            f"Tohle je {name} — demo verze zaměřená na přírodní vědy a matematiku.", s),
        row("What is KucLab Hertz?",
            "Hertz is KucLab's new model line, replacing the older \"KucLab V\" series. "
            f"This is {name} — a demo release focused on natural sciences and mathematics.", s),
        row("Jsi KucLab V1?",
            f"Ne. Řada „KucLab V“ skončila — tohle je {name} z nové řady Hertz.", s),
        row("V čem jsi dobrý?",
            "Fyzika, chemie, biologie a matematika — od středoškolské úrovně po vysokoškolskou. "
            "Umím odvozovat, počítat krok za krokem a vysvětlit proč to tak je, česky i anglicky.", s),
        row("What are you good at?",
            "Physics, chemistry, biology and mathematics — from secondary school up to university level. "
            "I derive, compute step by step and explain the reasoning, in Czech or English.", s),
        row("umíš česky?", "Ano, plynule. Včetně odborné terminologie ve fyzice, chemii, biologii a matematice.", s),
        row("Can you speak Czech?", "Yes, fluently — including scientific terminology.", s),
        row("Ahoj", "Ahoj! Co potřebuješ spočítat nebo vysvětlit?", s),
        row("Čau", "Čau! S čím pomoct?", s),
        row("Dobrý den", "Dobrý den! S čím vám můžu pomoct?", s),
        row("Hello", "Hi! What can I help you with?", s),
        row("Díky", "Není zač.", s),
        row("Thanks", "You're welcome.", s),
    ]


# ══════════════════════════════════════════════════════════════════════════
# Czech STEM seed — handwritten expert content (the core differentiator)
# ══════════════════════════════════════════════════════════════════════════

def czech_physics_seed() -> list[dict]:
    s = "cs_physics_seed"
    return [
        row("Vysvětli Newtonovy pohybové zákony.",
            "**První zákon (zákon setrvačnosti):** Těleso setrvává v klidu nebo v rovnoměrném přímočarém "
            "pohybu, pokud na něj nepůsobí výslednice vnějších sil, nebo je-li tato výslednice nulová.\n\n"
            "**Druhý zákon (zákon síly):** Zrychlení tělesa je přímo úměrné výslednici sil a nepřímo úměrné "
            "jeho hmotnosti:\n\nF = m·a\n\nObecněji přes hybnost: F = dp/dt, což platí i pro tělesa s proměnnou hmotností.\n\n"
            "**Třetí zákon (zákon akce a reakce):** Působí-li těleso A na těleso B silou, působí těleso B na "
            "těleso A silou stejně velkou a opačného směru. Tyto síly působí na *různá* tělesa, proto se neruší.\n\n"
            "Zákony platí v inerciálních vztažných soustavách. V neinerciální soustavě je nutné zavést "
            "setrvačné síly (např. odstředivou nebo Coriolisovu).", s),
        row("Co je to entropie a co říká druhý termodynamický zákon?",
            "**Entropie** S je stavová veličina, která měří počet mikrostavů odpovídajících danému makrostavu. "
            "Boltzmannův vztah:\n\nS = k_B · ln W\n\nkde k_B ≈ 1,380649·10⁻²³ J·K⁻¹ je Boltzmannova konstanta a "
            "W je počet mikrostavů. Termodynamicky se změna entropie při vratném ději definuje jako "
            "dS = δQ_vratné / T.\n\n**Druhý termodynamický zákon:** Entropie izolované soustavy nikdy neklesá; "
            "roste u nevratných dějů a zůstává konstantní u vratných:\n\nΔS_izolované ≥ 0\n\n"
            "Důsledky: teplo samovolně přechází z teplejšího tělesa na chladnější, nelze sestrojit perpetuum "
            "mobile druhého druhu (stroj, který by veškeré přijaté teplo měnil na práci), a účinnost tepelného "
            "stroje je shora omezena Carnotovou účinností η = 1 − T_studená/T_teplá (teploty v kelvinech).", s),
        row("Odvoď vztah pro dostřel šikmého vrhu.",
            "Uvažujme vrh z výšky y = 0 počáteční rychlostí v₀ pod elevačním úhlem α, bez odporu vzduchu.\n\n"
            "**Rozklad rychlosti:**\nv₀ₓ = v₀·cos α\nv₀ᵧ = v₀·sin α\n\n"
            "**Pohybové rovnice:**\nx(t) = v₀·cos α · t\ny(t) = v₀·sin α · t − ½·g·t²\n\n"
            "**Doba letu** — hledáme t > 0, kdy y(t) = 0:\n"
            "t·(v₀·sin α − ½·g·t) = 0  ⟹  t_dolet = 2·v₀·sin α / g\n\n"
            "**Dostřel** dosadíme do x(t):\n"
            "D = v₀·cos α · 2·v₀·sin α / g = v₀²·(2·sin α·cos α)/g\n\n"
            "S použitím identity 2·sin α·cos α = sin 2α:\n\n**D = v₀²·sin(2α) / g**\n\n"
            "Maximum nastává pro sin 2α = 1, tedy **α = 45°**, a činí D_max = v₀²/g. "
            "Pozn.: platí jen pro vrh a dopad ve stejné výšce a při zanedbání odporu vzduchu.", s),
        row("Vysvětli princip neurčitosti.",
            "Heisenbergův princip neurčitosti říká, že u kvantové částice nelze současně libovolně přesně "
            "určit dvojici tzv. komplementárních veličin. Pro polohu a hybnost:\n\n"
            "Δx · Δp ≥ ℏ/2\n\nkde ℏ = h/(2π) ≈ 1,055·10⁻³⁴ J·s je redukovaná Planckova konstanta.\n\n"
            "Obdobně pro energii a čas: ΔE · Δt ≥ ℏ/2.\n\n"
            "**Důležité:** nejde o nedokonalost přístrojů. Je to vlastnost samotné přírody — plyne z toho, že "
            "poloha a hybnost jsou popsány operátory, které spolu nekomutují ([x̂, p̂] = iℏ). Částice prostě "
            "nemá současně ostře definovanou polohu i hybnost.\n\n"
            "Praktický důsledek: čím přesněji částici lokalizujeme (malé Δx), tím větší je rozptyl její "
            "hybnosti — proto elektron nemůže „spadnout“ do jádra.", s),
        row("Jaký je rozdíl mezi speciální a obecnou teorií relativity?",
            "**Speciální teorie relativity (1905)** popisuje inerciální vztažné soustavy — tedy soustavy bez "
            "zrychlení. Stojí na dvou postulátech: fyzikální zákony jsou stejné ve všech inerciálních "
            "soustavách a rychlost světla ve vakuu c je pro všechny pozorovatele stejná. Důsledky: dilatace "
            "času, kontrakce délek, relativita současnosti a vztah E² = (pc)² + (mc²)², pro klidovou částici "
            "E = mc².\n\n"
            "**Obecná teorie relativity (1915)** rozšiřuje popis na zrychlené soustavy a gravitaci. Klíčem je "
            "princip ekvivalence: lokálně nelze odlišit gravitační pole od zrychlení. Gravitace zde není síla, "
            "ale zakřivení časoprostoru způsobené hmotou a energií. Popisují to Einsteinovy rovnice pole:\n\n"
            "G_μν + Λ·g_μν = (8πG/c⁴)·T_μν\n\n"
            "Ověřené předpovědi: stáčení perihelia Merkuru, ohyb světla u hmotných těles, gravitační rudý "
            "posuv, gravitační vlny (přímo detekované 2015) a černé díry.", s),
        row("Co je Dopplerův jev a jak se liší u zvuku a u světla?",
            "Dopplerův jev je změna pozorované frekvence vlnění při vzájemném pohybu zdroje a pozorovatele.\n\n"
            "**U zvuku** existuje prostředí (vzduch), takže záleží zvlášť na pohybu zdroje a zvlášť "
            "pozorovatele:\n\nf' = f · (v ± v_pozorovatel) / (v ∓ v_zdroj)\n\n"
            "kde v je rychlost zvuku. Horní znaménka platí při přibližování. Proto siréna sanitky zní výš, "
            "když se blíží, a níž, když se vzdaluje.\n\n"
            "**U světla** žádné prostředí neexistuje — éter se neprokázal. Záleží jen na *relativní* rychlosti "
            "a je nutné započítat dilataci času. Pro pohyb podél spojnice:\n\n"
            "f' = f · √((1 − β)/(1 + β)),  β = v/c\n\n"
            "Navíc u světla existuje i **příčný Dopplerův jev** (f' = f/γ), který u zvuku nemá obdobu — je to "
            "čistě relativistický efekt. Praktické využití: rudý posuv vzdálených galaxií dokazuje rozpínání "
            "vesmíru.", s),
        row("Vysvětli Maxwellovy rovnice slovy.",
            "Čtyři rovnice popisující veškerou klasickou elektrodynamiku (v diferenciálním tvaru, ve vakuu):\n\n"
            "**1. Gaussův zákon pro elektrické pole:** ∇·E = ρ/ε₀\n"
            "Elektrický náboj je zdrojem elektrického pole. Siločáry vycházejí z kladných nábojů a končí na záporných.\n\n"
            "**2. Gaussův zákon pro magnetické pole:** ∇·B = 0\n"
            "Neexistuje magnetický monopól. Magnetické siločáry jsou vždy uzavřené — nemají začátek ani konec.\n\n"
            "**3. Faradayův zákon:** ∇×E = −∂B/∂t\n"
            "Časově proměnné magnetické pole vytváří vířivé elektrické pole. To je princip generátoru a transformátoru.\n\n"
            "**4. Ampérův–Maxwellův zákon:** ∇×B = μ₀·J + μ₀ε₀·∂E/∂t\n"
            "Magnetické pole vytváří jednak elektrický proud, jednak časově proměnné elektrické pole "
            "(Maxwellův posuvný proud — jeho doplnění bylo Maxwellovým klíčovým přínosem).\n\n"
            "Spojením 3. a 4. rovnice vyjde vlnová rovnice s rychlostí c = 1/√(ε₀μ₀). Právě tak Maxwell zjistil, "
            "že **světlo je elektromagnetické vlnění**.", s),
        row("Co je to fotoelektrický jev?",
            "Fotoelektrický jev je uvolňování elektronů z povrchu látky (typicky kovu) při dopadu "
            "elektromagnetického záření.\n\n"
            "**Co nesedělo s klasickou fyzikou:** podle vlnové teorie by mělo stačit svítit dostatečně "
            "dlouho i slabým světlem. Experiment ale ukázal, že pod určitou **mezní frekvencí** f₀ se "
            "elektrony neuvolní vůbec, ať je světlo jakkoli intenzivní. A nad ní se uvolní okamžitě.\n\n"
            "**Einsteinovo vysvětlení (1905, Nobelova cena 1921):** světlo je kvantované do fotonů o energii "
            "E = h·f. Jeden foton předá energii jednomu elektronu:\n\n"
            "h·f = W_v + E_k,max\n\n"
            "kde W_v je **výstupní práce** (energie nutná k uvolnění elektronu z kovu) a E_k,max = ½mv²_max je "
            "maximální kinetická energie uvolněného elektronu. Mezní frekvence je tedy f₀ = W_v/h.\n\n"
            "Zvýšení intenzity světla zvýší *počet* uvolněných elektronů, ale ne jejich energii — tu určuje "
            "jen frekvence. Tohle byl jeden ze základních kamenů kvantové teorie.", s),
        row("Jak funguje jaderná fúze a proč je tak těžké ji zvládnout na Zemi?",
            "**Princip:** lehká jádra se sloučí v těžší. Hmotnost produktu je menší než součet hmotností "
            "výchozích jader a rozdíl (hmotnostní schodek) se uvolní jako energie podle E = Δm·c².\n\n"
            "Nejsnáze dostupná reakce je deuterium–tritium:\n\n"
            "²H + ³H → ⁴He + n + 17,6 MeV\n\n"
            "**Proč je to těžké:** obě jádra jsou kladně nabitá a odpuzují se Coulombovou silou. Aby se "
            "dostala na dosah silné jaderné interakce (~10⁻¹⁵ m), musí překonat Coulombovu bariéru. "
            "To vyžaduje teploty řádově **100–150 milionů K** — víc než v jádru Slunce, protože nemáme "
            "sluneční obrovský tlak a gravitační stlačení.\n\n"
            "Při takové teplotě je látka plně ionizované plazma, které nelze držet v žádné nádobě. Řeší se "
            "to dvěma cestami:\n"
            "- **Magnetické udržení** (tokamak, stelarátor) — plazma drží magnetické pole. Sem patří ITER.\n"
            "- **Inerciální udržení** — terčík se stlačí lasery, fúze proběhne dřív, než se stihne rozletět. "
            "Tady v roce 2022 NIF poprvé dosáhl vědeckého zisku energie (víc energie z fúze než dodaly lasery na terč).\n\n"
            "Klíčové kritérium úspěchu je **Lawsonovo kritérium**: součin hustoty plazmatu, teploty a doby "
            "udržení musí překročit prahovou hodnotu.", s),
        row("Vysvětli, co je supravodivost.",
            "Supravodivost je stav, kdy materiál pod tzv. **kritickou teplotou** T_c vykazuje dvě věci "
            "současně:\n\n"
            "**1. Nulový elektrický odpor.** Ne „velmi malý“ — přesně nulový. Proud v supravodivé smyčce "
            "může téct roky bez měřitelného poklesu.\n\n"
            "**2. Meissnerův jev.** Supravodič ze svého objemu vytlačuje magnetické pole (je ideálním "
            "diamagnetikem). Tohle je podstatnější než nulový odpor — samotný ideální vodič by pole "
            "„zamrazil“, zatímco supravodič ho aktivně vypudí. Proto supravodič levituje nad magnetem.\n\n"
            "**Mechanismus (BCS teorie, 1957):** elektrony se přes interakci s kmity krystalové mříže "
            "(fonony) párují do tzv. **Cooperových párů**. Pár má celočíselný spin, chová se tedy jako "
            "boson a všechny páry mohou obsadit jeden kvantový stav. Rozptyl na jednotlivém elektronu už "
            "nemůže proud brzdit, protože by musel rozbít celý kondenzát — a na to nestačí energie.\n\n"
            "**Vysokoteplotní supravodiče** (kupráty, od 1986, T_c až ~135 K) běžná BCS teorie nevysvětluje "
            "— jejich mechanismus je stále otevřený problém fyziky pevných látek.\n\n"
            "Využití: magnety v MRI a urychlovačích (LHC), SQUID magnetometry, maglev vlaky.", s),
    ]


def czech_chemistry_seed() -> list[dict]:
    s = "cs_chemistry_seed"
    return [
        row("Vysvětli, jak číst periodickou tabulku prvků.",
            "**Uspořádání:** prvky jsou seřazeny podle rostoucího **protonového čísla Z** (počtu protonů v jádře).\n\n"
            "**Perioda (řádek)** = číslo nejvyšší obsazené elektronové slupky. Prvky 3. periody mají "
            "valenční elektrony ve třetí slupce.\n\n"
            "**Skupina (sloupec)** = počet valenčních elektronů, tedy podobné chemické vlastnosti:\n"
            "- 1. skupina — alkalické kovy (1 valenční e⁻, velmi reaktivní, tvoří kationty M⁺)\n"
            "- 2. skupina — kovy alkalických zemin (2 valenční e⁻)\n"
            "- 17. skupina — halogeny (7 valenčních e⁻, chtějí jeden doplnit, tvoří anionty X⁻)\n"
            "- 18. skupina — vzácné plyny (plná valenční slupka, prakticky nereaktivní)\n"
            "- 3.–12. skupina — přechodné kovy (zaplňují se d-orbitaly, proměnlivá oxidační čísla)\n\n"
            "**Periodické trendy:**\n"
            "- *Atomový poloměr* roste dolů (přibývají slupky) a klesá doprava (roste náboj jádra, který "
            "elektrony přitahuje silněji)\n"
            "- *Ionizační energie* a *elektronegativita* rostou doprava a nahoru — nejelektronegativnější "
            "prvek je fluor (3,98 podle Paulinga)\n"
            "- *Kovový charakter* roste doleva a dolů\n\n"
            "Lanthanoidy a aktinoidy jsou vyňaté pod tabulku jen kvůli šířce — patří do 6. a 7. periody.", s),
        row("Co je to chemická rovnováha a Le Chatelierův princip?",
            "**Chemická rovnováha** nastává u vratné reakce, když rychlost přímé a zpětné reakce je stejná. "
            "Koncentrace látek se pak už nemění, ale reakce probíhá dál oběma směry — je to *dynamická* "
            "rovnováha.\n\n"
            "Pro reakci aA + bB ⇌ cC + dD platí **rovnovážná konstanta**:\n\n"
            "K_c = ([C]^c · [D]^d) / ([A]^a · [B]^b)\n\n"
            "K závisí jen na teplotě. Velké K znamená, že rovnováha je posunutá k produktům.\n\n"
            "**Le Chatelierův princip:** Poruší-li se rovnováha vnějším zásahem, soustava se posune tak, "
            "aby účinek zásahu zmírnila.\n\n"
            "- *Zvýšení koncentrace výchozí látky* → posun k produktům\n"
            "- *Zvýšení tlaku* (u plynů) → posun na stranu s **menším počtem molů plynu**\n"
            "- *Zvýšení teploty* → posun ve směru **endotermické** reakce (soustava spotřebuje dodané teplo)\n"
            "- *Katalyzátor* rovnováhu **neposune** — urychlí stejně přímou i zpětnou reakci, jen se "
            "rovnováhy dosáhne rychleji\n\n"
            "**Praktický příklad — Haberův–Boschův proces:** N₂ + 3H₂ ⇌ 2NH₃, ΔH < 0 (exotermická). "
            "Vysoký tlak pomáhá (4 moly → 2 moly), nízká teplota by termodynamicky pomáhala taky, ale "
            "reakce by byla neúnosně pomalá. Proto se v praxi volí kompromis ~450 °C a 200 atm s "
            "železným katalyzátorem.", s),
        row("Vysvětli rozdíl mezi iontovou, kovalentní a vodíkovou vazbou.",
            "**Iontová vazba** vzniká mezi prvky s velkým rozdílem elektronegativity (zhruba ΔEN > 1,7) — "
            "typicky kov + nekov. Elektron se prakticky úplně přenese, vzniknou ionty a drží je "
            "elektrostatická přitažlivost. Příklad NaCl. Vlastnosti: vysoké teploty tání, křehkost, "
            "vedení proudu v tavenině nebo roztoku (ne v pevném stavu).\n\n"
            "**Kovalentní vazba** vzniká sdílením elektronového páru mezi atomy s podobnou "
            "elektronegativitou. Je-li ΔEN ≈ 0, je *nepolární* (H₂, O₂), při středním rozdílu je *polární* "
            "(H₂O, HCl). Je směrová — určuje tvar molekuly. Vlastnosti: většinou nižší teploty tání, "
            "molekulové látky nevedou proud.\n\n"
            "**Vodíková vazba (vodíkový můstek)** není pravá chemická vazba, ale silná mezimolekulová "
            "interakce. Vzniká, když je vodík vázán na silně elektronegativní atom (**F, O, N**) a "
            "přitahuje volný elektronový pár sousední molekuly.\n\n"
            "Energie pro srovnání: kovalentní ~350–800 kJ/mol, iontová ~600–1000 kJ/mol (v mřížce), "
            "vodíková jen ~10–40 kJ/mol.\n\n"
            "**Proč na vodíkové vazbě záleží:** dělá z vody kapalinu při pokojové teplotě (H₂S s podobnou "
            "molární hmotností je plyn), způsobuje anomálii vody (led plave), drží dvojšroubovici DNA "
            "pohromadě a určuje sekundární strukturu bílkovin.", s),
        row("Co je pH a jak se počítá?",
            "**pH** je míra kyselosti roztoku, definovaná jako záporný dekadický logaritmus aktivity "
            "(zjednodušeně koncentrace) oxoniových iontů:\n\n"
            "pH = −log₁₀[H₃O⁺]\n\n"
            "**Stupnice** (při 25 °C): pH < 7 kyselé, pH = 7 neutrální, pH > 7 zásadité. Každá jednotka "
            "je **desetinásobný** rozdíl v koncentraci.\n\n"
            "Platí iontový součin vody: [H₃O⁺]·[OH⁻] = K_w = 10⁻¹⁴, tedy **pH + pOH = 14**.\n\n"
            "**Výpočet u silné kyseliny** (úplně disociuje): pro 0,01 M HCl je [H₃O⁺] = 0,01 = 10⁻², "
            "takže pH = 2.\n\n"
            "**U slabé kyseliny** je nutné použít disociační konstantu K_a:\n\n"
            "pH ≈ ½·(pK_a − log c)\n\n"
            "Například kyselina octová (pK_a = 4,76) o koncentraci 0,1 M: pH ≈ ½·(4,76 + 1) = 2,88.\n\n"
            "**Pufr** (slabá kyselina + její sůl) se počítá Hendersonovou–Hasselbalchovou rovnicí:\n\n"
            "pH = pK_a + log([sůl]/[kyselina])\n\n"
            "Pozn.: hodnota 7 jako neutrální platí jen při 25 °C — K_w s teplotou roste, takže při 60 °C "
            "je neutrální pH asi 6,5.", s),
        row("Vysvětli, co jsou izomery a jaké typy existují.",
            "**Izomery** jsou sloučeniny se stejným molekulovým (sumárním) vzorcem, ale odlišným "
            "uspořádáním atomů — a tedy odlišnými vlastnostmi.\n\n"
            "**A) Konstituční (strukturní) izomery** — liší se pořadím vazeb:\n"
            "- *Řetězové*: butan vs. isobutan (oba C₄H₁₀)\n"
            "- *Polohové*: 1-propanol vs. 2-propanol — liší se umístěním funkční skupiny\n"
            "- *Funkční*: ethanol (C₂H₅OH) vs. dimethylether (CH₃OCH₃), oba C₂H₆O — úplně jiná látka\n\n"
            "**B) Stereoizomery** — stejné pořadí vazeb, jiné prostorové uspořádání:\n\n"
            "*Geometrické (cis–trans, E/Z)*: u dvojné vazby nebo cyklu, které brání volné rotaci. "
            "Kyselina maleinová (cis) taje při 130 °C, fumarová (trans) při 287 °C.\n\n"
            "*Optické (enantiomery)*: zrcadlové obrazy, které nelze ztotožnit — jako levá a pravá ruka. "
            "Vznikají na **chirálním centru** (uhlík se 4 různými substituenty). Otáčejí rovinu "
            "polarizovaného světla opačným směrem. Značí se R/S nebo D/L.\n\n"
            "**Proč to je zásadní:** v biologii jsou receptory chirální, takže enantiomery mohou působit "
            "úplně jinak. Klasický a tragický příklad je thalidomid — jeden enantiomer tlumil ranní "
            "nevolnost, druhý byl teratogenní a v 60. letech způsobil vrozené vady u tisíců dětí.", s),
        row("Jak funguje katalyzátor?",
            "**Katalyzátor** je látka, která zvyšuje rychlost chemické reakce, ale sama se při ní "
            "nespotřebovává (na konci je regenerována).\n\n"
            "**Mechanismus:** katalyzátor poskytuje reakci **alternativní reakční cestu s nižší aktivační "
            "energií E_a**. Podle Arrheniovy rovnice:\n\n"
            "k = A · e^(−E_a/RT)\n\n"
            "je závislost exponenciální — snížení E_a jen o 20 kJ/mol zrychlí reakci při pokojové teplotě "
            "asi 3000×.\n\n"
            "**Co katalyzátor NEdělá:** neposouvá rovnováhu ani nemění ΔG reakce. Urychlí přímou i zpětnou "
            "reakci ve stejném poměru — rovnováhy se jen dosáhne dřív. Nemůže rozběhnout termodynamicky "
            "nemožnou reakci.\n\n"
            "**Typy:**\n"
            "- *Homogenní* — ve stejné fázi jako reaktanty (kyselá katalýza esterifikace)\n"
            "- *Heterogenní* — jiná fáze, reakce probíhá na povrchu (Pt/Pd/Rh v autokatalyzátoru, Fe v "
            "Haberově procesu). Klíčová je velikost aktivního povrchu.\n"
            "- *Enzymatická* — biokatalyzátory. Extrémně účinné a specifické: kataláza zrychlí rozklad "
            "H₂O₂ asi 10¹⁴×.\n\n"
            "**Katalytické jedy** (např. olovo, síra) blokují aktivní centra — proto se do auta s "
            "katalyzátorem nesmí olovnatý benzin.", s),
        row("Vyčísli rovnici hoření propanu a vysvětli postup.",
            "**Nevyčíslená rovnice:**\nC₃H₈ + O₂ → CO₂ + H₂O\n\n"
            "**Postup — jdeme v pořadí C, H, O:**\n\n"
            "*1) Uhlík:* vlevo 3 atomy C, takže vpravo potřebujeme 3 CO₂:\n"
            "C₃H₈ + O₂ → 3 CO₂ + H₂O\n\n"
            "*2) Vodík:* vlevo 8 atomů H, jedna molekula vody má 2, takže 4 H₂O:\n"
            "C₃H₈ + O₂ → 3 CO₂ + 4 H₂O\n\n"
            "*3) Kyslík:* vpravo máme 3·2 + 4·1 = 10 atomů O. Vlevo je O₂ (dvouatomový), takže "
            "potřebujeme 5 molekul:\n\n"
            "**C₃H₈ + 5 O₂ → 3 CO₂ + 4 H₂O**\n\n"
            "*Kontrola:* C 3=3 ✓, H 8=8 ✓, O 10=10 ✓\n\n"
            "Jde o dokonalé hoření (dostatek kyslíku), silně exotermické: ΔH ≈ −2220 kJ/mol. "
            "Při nedostatku kyslíku vzniká jedovatý CO nebo saze (nedokonalé hoření).", s),
        row("Co jsou oxidačně-redukční reakce?",
            "**Redoxní reakce** je reakce, při níž dochází k přenosu elektronů — mění se oxidační čísla.\n\n"
            "**Oxidace** = ztráta elektronů = **zvýšení** oxidačního čísla\n"
            "**Redukce** = přijetí elektronů = **snížení** oxidačního čísla\n\n"
            "Pomůcka: *OILRIG* — Oxidation Is Loss, Reduction Is Gain.\n\n"
            "Obě půlreakce probíhají vždy současně — elektrony se nemohou jen tak ztratit.\n\n"
            "**Oxidační činidlo** samo se redukuje (bere elektrony) — např. KMnO₄, K₂Cr₂O₇, O₂, Cl₂.\n"
            "**Redukční činidlo** samo se oxiduje (dává elektrony) — např. alkalické kovy, H₂, C, Zn.\n\n"
            "**Příklad:** Zn + Cu²⁺ → Zn²⁺ + Cu\n"
            "- Oxidace: Zn⁰ → Zn²⁺ + 2e⁻ (zinek je redukční činidlo)\n"
            "- Redukce: Cu²⁺ + 2e⁻ → Cu⁰ (měďnatý ion je oxidační činidlo)\n\n"
            "**Pravidla pro oxidační čísla:** volný prvek 0; kyslík obvykle −II (v peroxidech −I, v OF₂ +II); "
            "vodík +I (v hydridech kovů −I); součet v neutrální molekule je 0, v iontu roven náboji.\n\n"
            "**Kde se to používá:** galvanické články a baterie, elektrolýza, koroze, dýchání a fotosyntéza, "
            "spalování — prakticky veškerá energetika.", s),
    ]


def czech_biology_seed() -> list[dict]:
    s = "cs_biology_seed"
    return [
        row("Popiš stavbu eukaryotické buňky a funkci organel.",
            "**Jádro (nucleus)** — obsahuje DNA v podobě chromatinu, řídí buňku. Obaleno dvojitou jadernou "
            "membránou s póry. Uvnitř je **jadérko**, kde se skládají ribozomy.\n\n"
            "**Mitochondrie** — „elektrárna buňky“. Probíhá zde Krebsův cyklus a oxidativní fosforylace, "
            "vzniká ATP. Má dvojitou membránu, vnitřní je zřasená v **kristy** (větší povrch). Má vlastní "
            "kruhovou DNA a ribozomy — důkaz **endosymbiotické teorie** (vznikla pohlcením bakterie).\n\n"
            "**Endoplazmatické retikulum (ER)**:\n"
            "- *Drsné* (s ribozomy) — syntéza a úprava bílkovin určených na export\n"
            "- *Hladké* — syntéza lipidů a steroidů, detoxikace, zásoba Ca²⁺\n\n"
            "**Golgiho aparát** — třídí, upravuje (glykosyluje) a balí bílkoviny do váčků. Buněčná „pošta“.\n\n"
            "**Lyzozomy** — váčky s hydrolytickými enzymy, štěpí odpad a pohlcené částice (kyselé pH ~4,8).\n\n"
            "**Ribozomy** — syntéza bílkovin (translace). Nejsou organela s membránou.\n\n"
            "**Cytoskelet** — mikrotubuly, mikrofilamenta a intermediální filamenta. Tvar buňky, pohyb, dělení.\n\n"
            "**Navíc jen u rostlin:** buněčná stěna z celulózy, **chloroplasty** (fotosyntéza, také s "
            "vlastní DNA) a velká centrální **vakuola** (turgor, zásoby).\n\n"
            "*Klíčový rozdíl od prokaryot:* prokaryotická buňka nemá jádro ani membránové organely a její "
            "DNA leží volně v nukleoidu.", s),
        row("Vysvětli mitózu a meiózu a v čem se liší.",
            "**MITÓZA** — dělení somatických (tělních) buněk. Z jedné mateřské buňky vzniknou **2 dceřiné, "
            "geneticky identické, diploidní (2n)**.\n\n"
            "Fáze:\n"
            "1. *Profáze* — chromatin se spiralizuje v chromozomy, zaniká jaderný obal, tvoří se dělicí vřeténko\n"
            "2. *Metafáze* — chromozomy se řadí do ekvatoriální roviny\n"
            "3. *Anafáze* — chromatidy se oddělují a putují k pólům\n"
            "4. *Telofáze* — tvoří se nová jádra, následuje cytokineze\n\n"
            "Účel: růst, obnova tkání, nepohlavní rozmnožování.\n\n"
            "**MEIÓZA** — vznik pohlavních buněk (gamet). Z jedné buňky vzniknou **4 dceřiné, geneticky "
            "odlišné, haploidní (n)**. Má dvě po sobě jdoucí dělení.\n\n"
            "*Meióza I (redukční)* — zde se počet chromozomů půlí. Klíčové děje v profázi I:\n"
            "- **Párování homologních chromozomů** (bivalenty)\n"
            "- **Crossing-over** — výměna úseků mezi homologními chromozomy → nové kombinace genů\n"
            "V anafázi I se rozcházejí celé homologní chromozomy (ne chromatidy!).\n\n"
            "*Meióza II* — průběh podobný mitóze, rozdělí se chromatidy.\n\n"
            "**Hlavní rozdíly:**\n"
            "| | Mitóza | Meióza |\n"
            "|---|---|---|\n"
            "| Dělení | 1 | 2 |\n"
            "| Dceřiné buňky | 2 | 4 |\n"
            "| Ploidie | 2n → 2n | 2n → n |\n"
            "| Genetická identita | identické | odlišné |\n"
            "| Crossing-over | ne | ano |\n\n"
            "**Zdroje variability v meióze:** crossing-over + náhodná segregace homologů (u člověka 2²³ ≈ "
            "8,4 milionu kombinací) — proto nejsou sourozenci stejní.", s),
        row("Jak probíhá fotosyntéza?",
            "Fotosyntéza je proces, při kterém rostliny, řasy a sinice převádějí světelnou energii na "
            "chemickou. Sumární rovnice:\n\n"
            "6 CO₂ + 6 H₂O + světlo → C₆H₁₂O₆ + 6 O₂\n\n"
            "Probíhá v **chloroplastech** ve dvou fázích:\n\n"
            "**1) SVĚTELNÁ FÁZE** — v tylakoidních membránách\n"
            "- Chlorofyl ve fotosystémech II a I absorbuje světlo a excituje elektrony\n"
            "- **Fotolýza vody:** 2 H₂O → 4 H⁺ + 4 e⁻ + O₂ — doplní chybějící elektrony ve fotosystému II. "
            "**Veškerý kyslík v atmosféře pochází odsud**, ne z CO₂\n"
            "- Elektrony putují elektronovým transportním řetězcem, přitom se pumpují protony do lumen "
            "tylakoidu\n"
            "- Vzniklý protonový gradient pohání **ATP-syntázu** → ATP (fotofosforylace)\n"
            "- Na konci řetězce se redukuje NADP⁺ na NADPH\n\n"
            "Výstup: **ATP + NADPH + O₂**\n\n"
            "**2) TEMNOSTNÍ FÁZE (Calvinův cyklus)** — ve stromatu\n"
            "Nepotřebuje přímo světlo, ale potřebuje ATP a NADPH z fáze světelné.\n"
            "- *Fixace:* enzym **RuBisCO** naváže CO₂ na ribulóza-1,5-bisfosfát (RuBP)\n"
            "- *Redukce:* za spotřeby ATP a NADPH vzniká glyceraldehyd-3-fosfát (G3P)\n"
            "- *Regenerace:* část G3P obnoví RuBP, zbytek jde na glukózu\n\n"
            "Na jednu molekulu glukózy je potřeba 6 otáček cyklu, 18 ATP a 12 NADPH.\n\n"
            "**Pozn.:** RuBisCO je nejhojnější bílkovina na Zemi, ale je pomalá a plete si CO₂ s O₂ "
            "(fotorespirace). Rostliny typu C4 (kukuřice) a CAM (kaktusy) si vyvinuly mechanismy, jak CO₂ "
            "předkoncentrovat a tuto ztrátu obejít.", s),
        row("Vysvětli, jak funguje DNA a centrální dogma molekulární biologie.",
            "**Stavba DNA:** dvoušroubovice ze dvou antiparalelních řetězců. Stavební jednotkou je "
            "**nukleotid** = deoxyribóza + fosfát + dusíkatá báze.\n\n"
            "Báze: adenin (A), thymin (T), guanin (G), cytosin (C).\n"
            "**Komplementarita:** A–T (2 vodíkové můstky), G–C (3 můstky). Proto je úsek bohatý na G-C "
            "stabilnější a taje při vyšší teplotě.\n\n"
            "**CENTRÁLNÍ DOGMA:** DNA → RNA → bílkovina\n\n"
            "**1) Replikace** (DNA → DNA) — před dělením buňky. Je **semikonzervativní**: každá nová "
            "dvoušroubovice obsahuje jeden původní a jeden nový řetězec. Klíčové enzymy: helikáza "
            "(rozplétá), DNA-polymeráza (syntetizuje pouze ve směru 5'→3', proto vzniká vedoucí a "
            "opožďující se řetězec s Okazakiho fragmenty), ligáza (spojuje).\n\n"
            "**2) Transkripce** (DNA → mRNA) — v jádře. RNA-polymeráza přepíše gen do mRNA. Místo thyminu "
            "je **uracil (U)**. U eukaryot následuje sestřih (splicing): vystřihnou se **introny**, spojí "
            "se **exony**. Alternativním sestřihem může jeden gen kódovat víc bílkovin.\n\n"
            "**3) Translace** (mRNA → bílkovina) — na ribozomu. Čte se po **kodonech** (trojicích bází). "
            "tRNA s antikodonem přináší odpovídající aminokyselinu. Start kodon AUG (methionin), stop "
            "kodony UAA, UAG, UGA.\n\n"
            "**Genetický kód** je tripletový, degenerovaný (více kodonů pro jednu aminokyselinu — 64 kodonů "
            "na 20 aminokyselin), nepřekrývající se a téměř univerzální pro všechny organismy.\n\n"
            "**Výjimky z dogmatu:** retroviry (HIV) mají **reverzní transkriptázu** a přepisují RNA → DNA. "
            "Priony jsou infekční bílkoviny bez nukleové kyseliny.", s),
        row("Popiš, jak funguje lidský imunitní systém.",
            "Imunita má dvě propojené složky:\n\n"
            "**1) NESPECIFICKÁ (vrozená) imunita** — rychlá (minuty až hodiny), ale bez paměti.\n"
            "- *Bariéry:* kůže, sliznice, žaludeční kyselina, lysozym v slzách\n"
            "- *Buňky:* neutrofily a makrofágy (fagocytóza), NK-buňky (zabíjejí nádorové a virem "
            "infikované buňky), dendritické buňky (předkládají antigen)\n"
            "- *Humorální:* komplement (kaskáda bílkovin — perforuje membrány, opsonizuje), interferony\n"
            "- *Zánět:* rozšíření cév, zvýšená propustnost, chemotaxe — proto zarudnutí, otok, bolest, teplo\n\n"
            "**2) SPECIFICKÁ (získaná) imunita** — pomalá (dny), ale cílená a **s pamětí**.\n\n"
            "*Buněčná složka — T-lymfocyty* (zrají v brzlíku):\n"
            "- **T-pomocné (CD4+)** — dirigent celé odpovědi, aktivují B-lymfocyty i cytotoxické T-buňky. "
            "*Právě tyto buňky ničí HIV, proto je AIDS tak devastující.*\n"
            "- **T-cytotoxické (CD8+)** — přímo zabíjejí infikované buňky\n"
            "- **T-regulační** — tlumí odpověď, brání autoimunitě\n\n"
            "*Protilátková složka — B-lymfocyty* (zrají v kostní dřeni): po aktivaci se mění v plazmatické "
            "buňky produkující **protilátky (imunoglobuliny)** — IgG, IgM, IgA, IgE, IgD.\n\n"
            "**Imunologická paměť:** po prodělané infekci zůstávají paměťové T- a B-buňky. Při dalším "
            "setkání je odpověď mnohem rychlejší a silnější. **Na tomto principu stojí očkování** — "
            "vakcína předloží antigen bez toho, aby způsobila nemoc.\n\n"
            "**Když se to pokazí:** autoimunita (útok na vlastní tkáně — diabetes 1. typu, roztroušená "
            "skleróza), alergie (přehnaná reakce IgE na neškodný antigen), imunodeficience.", s),
        row("Vysvětli Mendelovy zákony dědičnosti.",
            "Gregor Johann Mendel (opat v Brně) je odvodil v letech 1856–1863 z křížení hrachu.\n\n"
            "**Základní pojmy:**\n"
            "- *Gen* — úsek DNA nesoucí informaci pro znak\n"
            "- *Alela* — konkrétní varianta genu (A, a)\n"
            "- *Genotyp* — genetická výbava (AA, Aa, aa); *fenotyp* — vnější projev\n"
            "- *Homozygot* (AA, aa) vs. *heterozygot* (Aa)\n"
            "- *Dominantní* alela se projeví i u heterozygota, *recesivní* jen u homozygota\n\n"
            "**1. zákon — o uniformitě F1:** Křížením dvou homozygotů (AA × aa) je celá první generace "
            "uniformní a heterozygotní (100 % Aa), fenotypově podle dominantní alely.\n\n"
            "**2. zákon — o štěpení v F2:** Křížením dvou heterozygotů (Aa × Aa) vzniká štěpný poměr\n"
            "- genotypově **1 AA : 2 Aa : 1 aa**\n"
            "- fenotypově **3 : 1** (dominantní : recesivní)\n\n"
            "**3. zákon — o volné kombinovatelnosti vloh:** Alely různých genů se do gamet rozdělují "
            "nezávisle na sobě. Dihybridní křížení AaBb × AaBb dá fenotypový poměr **9 : 3 : 3 : 1**.\n\n"
            "**Kdy 3. zákon neplatí:** pokud jsou geny na stejném chromozomu blízko sebe (**vazba genů**), "
            "dědí se společně. Rozbít je může jen crossing-over — a jeho četnost se používá k mapování "
            "genů na chromozomu.\n\n"
            "**Další odchylky:** neúplná dominance (růžové květy u nocenky), kodominance (krevní skupina "
            "AB), mnohotná alelie (systém AB0), pleiotropie, polygenní dědičnost (výška, barva kůže) a "
            "dědičnost vázaná na pohlaví (hemofilie, barvoslepost — na chromozomu X, proto častější u mužů).", s),
        row("Jak funguje evoluce přirozeným výběrem?",
            "Darwinova a Wallaceova teorie (1858/1859) stojí na několika nezpochybnitelných pozorováních:\n\n"
            "**Předpoklady:**\n"
            "1. Organismy produkují víc potomků, než kolik jich může přežít\n"
            "2. Mezi jedinci existuje **dědičná variabilita** (mutace, rekombinace)\n"
            "3. Zdroje jsou omezené → konkurence\n\n"
            "**Důsledek:** Jedinci s variantami, které v daném prostředí zvyšují pravděpodobnost přežití a "
            "rozmnožení, zanechají víc potomků. Tyto varianty se v populaci šíří. To je přirozený výběr.\n\n"
            "**Klíčové upřesnění, které se běžně chápe špatně:**\n"
            "- Výběr nepůsobí „pro dobro druhu“, ale na úrovni jedince (a genu)\n"
            "- Nejde o „přežití nejsilnějšího“, ale nejlépe **přizpůsobeného** danému prostředí. Změní-li "
            "se prostředí, změní se i to, co je výhodné\n"
            "- Evoluce **nemá cíl ani směr** — není to cesta k dokonalosti\n"
            "- Mutace vznikají **náhodně**; nenáhodný je až výběr, který mezi nimi třídí\n"
            "- Jedinec neevolvuje — evolvuje **populace**\n\n"
            "**Další evoluční mechanismy:** genetický drift (náhodné změny, silné v malých populacích), "
            "genový tok (migrace), pohlavní výběr (paví ocas snižuje šanci na přežití, ale zvyšuje šanci na "
            "rozmnožení).\n\n"
            "**Vznik druhů (speciace):** nejčastěji **alopatrická** — populace se geograficky oddělí, "
            "hromadí odlišné změny a nakonec ztratí schopnost se křížit.\n\n"
            "**Doklady:** fosilní záznam s přechodnými formami (Tiktaalik, Archaeopteryx), srovnávací "
            "anatomie (homologické orgány, rudimenty), embryologie, biogeografie a hlavně molekulární "
            "genetika — fylogenetické stromy z DNA sedí na stromy z morfologie.", s),
    ]


def czech_math_seed() -> list[dict]:
    s = "cs_math_seed"
    return [
        row("Odvoď vzorec pro řešení kvadratické rovnice.",
            "Máme obecnou kvadratickou rovnici ax² + bx + c = 0, kde a ≠ 0.\n\n"
            "**Postup doplněním na čtverec:**\n\n"
            "*1)* Vydělíme celou rovnici koeficientem a:\n"
            "x² + (b/a)·x + c/a = 0\n\n"
            "*2)* Přesuneme absolutní člen:\n"
            "x² + (b/a)·x = −c/a\n\n"
            "*3)* Doplníme levou stranu na úplný čtverec — přičteme k oběma stranám (b/2a)²:\n"
            "x² + (b/a)·x + (b/2a)² = −c/a + (b/2a)²\n\n"
            "*4)* Levá strana je nyní čtverec:\n"
            "(x + b/2a)² = b²/4a² − c/a = (b² − 4ac)/4a²\n\n"
            "*5)* Odmocníme:\n"
            "x + b/2a = ± √(b² − 4ac) / 2a\n\n"
            "*6)* Osamostatníme x:\n\n"
            "**x = (−b ± √(b² − 4ac)) / 2a**\n\n"
            "**Diskriminant** D = b² − 4ac rozhoduje o počtu řešení:\n"
            "- D > 0 → dvě různá reálná řešení\n"
            "- D = 0 → jedno dvojnásobné řešení x = −b/2a\n"
            "- D < 0 → žádné reálné řešení (dvě komplexně sdružená)\n\n"
            "**Vietovy vzorce:** x₁ + x₂ = −b/a a x₁·x₂ = c/a — užitečné pro rychlou kontrolu.", s),
        row("Vysvětli, co je derivace a jaký má geometrický význam.",
            "**Definice:** Derivace funkce f v bodě x₀ je limita\n\n"
            "f'(x₀) = lim_{h→0} [f(x₀ + h) − f(x₀)] / h\n\n"
            "pokud tato limita existuje (pak říkáme, že funkce je v bodě x₀ diferencovatelná).\n\n"
            "**Geometrický význam:** Zlomek [f(x₀+h) − f(x₀)]/h je směrnice **sečny** procházející body "
            "[x₀, f(x₀)] a [x₀+h, f(x₀+h)]. Když h → 0, sečna přechází v **tečnu**. Derivace je tedy "
            "**směrnice tečny** ke grafu funkce v daném bodě.\n\n"
            "Rovnice tečny: y = f(x₀) + f'(x₀)·(x − x₀)\n\n"
            "**Fyzikální význam:** okamžitá rychlost změny. Je-li s(t) dráha, pak s'(t) = v(t) je rychlost "
            "a v'(t) = a(t) zrychlení.\n\n"
            "**Základní derivace:**\n"
            "(xⁿ)' = n·xⁿ⁻¹    (eˣ)' = eˣ    (ln x)' = 1/x\n"
            "(sin x)' = cos x    (cos x)' = −sin x    (tg x)' = 1/cos²x\n\n"
            "**Pravidla:**\n"
            "- Součin: (u·v)' = u'·v + u·v'\n"
            "- Podíl: (u/v)' = (u'·v − u·v')/v²\n"
            "- Složená funkce (řetízkové pravidlo): [f(g(x))]' = f'(g(x))·g'(x)\n\n"
            "**Využití:** hledání extrémů (f'(x) = 0 je nutná podmínka), monotonie (f' > 0 → rostoucí), "
            "konvexita přes druhou derivaci, průběh funkce, optimalizační úlohy.", s),
        row("Co je integrál a jak souvisí s derivací?",
            "Existují dva pojmy, které je nutné rozlišovat:\n\n"
            "**1) Neurčitý integrál** ∫f(x) dx = F(x) + C\n"
            "Je to **primitivní funkce** — funkce F, pro kterou platí F'(x) = f(x). Je to tedy opačná "
            "operace k derivaci. Konstanta C je tam proto, že derivace konstanty je nula, takže primitivních "
            "funkcí je nekonečně mnoho.\n\n"
            "**2) Určitý integrál** ∫ₐᵇ f(x) dx\n"
            "Je to **číslo** — definované jako limita Riemannových součtů. Geometricky je to **obsah plochy** "
            "mezi grafem funkce a osou x na intervalu [a, b] (plocha pod osou se počítá záporně).\n\n"
            "**Newtonova–Leibnizova formule** (základní věta integrálního počtu) tyto dva pojmy spojuje:\n\n"
            "∫ₐᵇ f(x) dx = F(b) − F(a)\n\n"
            "To je jeden z nejdůležitějších výsledků matematiky — říká, že hledání obsahu plochy a hledání "
            "primitivní funkce jsou tentýž problém.\n\n"
            "**Základní integrály:**\n"
            "∫xⁿ dx = xⁿ⁺¹/(n+1) + C  (pro n ≠ −1)\n"
            "∫(1/x) dx = ln|x| + C\n"
            "∫eˣ dx = eˣ + C\n"
            "∫sin x dx = −cos x + C    ∫cos x dx = sin x + C\n\n"
            "**Metody:**\n"
            "- *Per partes:* ∫u·v' dx = u·v − ∫u'·v dx\n"
            "- *Substituce:* zavedeme t = g(x), dt = g'(x) dx\n\n"
            "**Použití:** obsahy a objemy, délka křivky, práce síly, těžiště, střední hodnota, "
            "pravděpodobnost (integrál hustoty).", s),
        row("Dokaž, že √2 je iracionální číslo.",
            "Použijeme **důkaz sporem**.\n\n"
            "**Předpoklad:** Nechť √2 je racionální. Pak ho lze zapsat jako zlomek\n\n"
            "√2 = p/q\n\n"
            "kde p, q jsou celá čísla, q ≠ 0, a zlomek je v **základním tvaru** — tedy p a q jsou nesoudělná "
            "(nemají společného dělitele většího než 1).\n\n"
            "**Odvození:**\n\n"
            "*1)* Umocníme obě strany:\n"
            "2 = p²/q²   ⟹   p² = 2q²\n\n"
            "*2)* Z toho plyne, že p² je sudé. Kdyby bylo p liché, byl by i jeho čtverec lichý "
            "(liché·liché = liché). Tedy **p je sudé** a lze psát p = 2k pro nějaké celé k.\n\n"
            "*3)* Dosadíme p = 2k do rovnice p² = 2q²:\n"
            "(2k)² = 2q²\n"
            "4k² = 2q²\n"
            "**q² = 2k²**\n\n"
            "*4)* Stejnou úvahou je q² sudé, a tedy **q je sudé**.\n\n"
            "**Spor:** Dostali jsme, že p i q jsou obě sudá — mají tedy společného dělitele 2. To je ale "
            "v přímém rozporu s předpokladem, že zlomek p/q je v základním tvaru.\n\n"
            "**Závěr:** Předpoklad byl nesprávný. Číslo √2 tedy nelze zapsat jako podíl dvou celých čísel, "
            "a je proto **iracionální**. ∎\n\n"
            "*Historická pozn.:* tento objev připisovaný pythagorejcům (Hippasos, 5. stol. př. n. l.) "
            "byl krizí antické matematiky, která stála na představě, že vše lze vyjádřit poměrem celých čísel.", s),
        row("Vysvětli, co jsou komplexní čísla a k čemu jsou dobrá.",
            "**Motivace:** V oboru reálných čísel nemá rovnice x² + 1 = 0 řešení. Zavedeme proto "
            "**imaginární jednotku** i s vlastností:\n\n"
            "i² = −1\n\n"
            "**Algebraický tvar:** z = a + bi, kde a = Re(z) je reálná a b = Im(z) imaginární část.\n\n"
            "**Operace:**\n"
            "- Sčítání: (a+bi) + (c+di) = (a+c) + (b+d)i\n"
            "- Násobení: (a+bi)(c+di) = (ac − bd) + (ad + bc)i\n"
            "- *Komplexně sdružené* číslo: z̄ = a − bi, přitom z·z̄ = a² + b² = |z|²\n"
            "- Dělení: rozšíříme zlomek číslem sdruženým ke jmenovateli\n\n"
            "**Goniometrický tvar:** z = |z|·(cos φ + i·sin φ), kde |z| = √(a²+b²) je **modul** a φ "
            "**argument**. Násobení pak znamená vynásobit moduly a sečíst argumenty — geometricky jde o "
            "otočení a změnu měřítka.\n\n"
            "**Moivreova věta:** zⁿ = |z|ⁿ·(cos nφ + i·sin nφ)\n\n"
            "**Eulerův vzorec:** e^(iφ) = cos φ + i·sin φ. Pro φ = π dává Eulerovu identitu\n\n"
            "**e^(iπ) + 1 = 0**\n\n"
            "která spojuje pět nejdůležitějších konstant matematiky.\n\n"
            "**Základní věta algebry:** Každý polynom stupně n má v oboru komplexních čísel právě n kořenů "
            "(včetně násobnosti). Komplexní čísla tedy algebru „uzavírají“.\n\n"
            "**K čemu jsou v praxi:**\n"
            "- *Elektrotechnika* — impedance střídavých obvodů, fázory (kvůli kolizi s proudem se zde "
            "imaginární jednotka značí j)\n"
            "- *Kvantová mechanika* — vlnová funkce je principiálně komplexní\n"
            "- *Zpracování signálu* — Fourierova transformace\n"
            "- *Řídicí technika* — stabilita systémů podle polohy pólů v komplexní rovině", s),
        row("Vysvětli, co je limita funkce.",
            "**Intuitivně:** Limita popisuje, k jaké hodnotě se funkce blíží, když se argument blíží "
            "nějakému bodu — **bez ohledu na to, jaká je hodnota v tom bodě samotném** (funkce tam nemusí "
            "být vůbec definovaná).\n\n"
            "**Přesná definice (ε–δ):** Říkáme, že lim_{x→a} f(x) = L, jestliže\n\n"
            "∀ε > 0 ∃δ > 0: 0 < |x − a| < δ ⟹ |f(x) − L| < ε\n\n"
            "Slovy: ať zvolíme jakkoli malý pás ε kolem L, dokážeme najít okolí δ kolem a takové, že "
            "všechny funkční hodnoty z tohoto okolí (kromě snad bodu a) do pásu padnou.\n\n"
            "**Příklad, proč hodnota v bodě nerozhoduje:**\n"
            "f(x) = (x² − 1)/(x − 1) není v bodě x = 1 definovaná (dělení nulou). Ale pro x ≠ 1 lze krátit:\n"
            "(x−1)(x+1)/(x−1) = x + 1\n"
            "takže lim_{x→1} f(x) = 2.\n\n"
            "**Jednostranné limity:** lim_{x→a⁻} (zleva) a lim_{x→a⁺} (zprava). Limita existuje právě "
            "tehdy, když se obě rovnají.\n\n"
            "**Významné limity:**\n"
            "lim_{x→0} (sin x)/x = 1\n"
            "lim_{x→∞} (1 + 1/x)ˣ = e\n\n"
            "**Neurčité výrazy** 0/0, ∞/∞, 0·∞, ∞−∞, 1^∞ se často řeší **L'Hospitalovým pravidlem**: "
            "za splnění podmínek platí lim f/g = lim f'/g'.\n\n"
            "**Proč na tom záleží:** na limitě stojí definice spojitosti (f je spojitá v a, právě když "
            "lim_{x→a} f(x) = f(a)), derivace i určitého integrálu. Je to základní pojem celé "
            "matematické analýzy.", s),
        row("Vysvětli základy pravděpodobnosti a Bayesovu větu.",
            "**Klasická definice:** Pro konečný prostor se stejně pravděpodobnými jevy\n\n"
            "P(A) = počet příznivých výsledků / počet všech výsledků\n\n"
            "Platí 0 ≤ P(A) ≤ 1, P(jistý jev) = 1, P(nemožný jev) = 0.\n\n"
            "**Základní pravidla:**\n"
            "- Opačný jev: P(A') = 1 − P(A)\n"
            "- Sjednocení: P(A ∪ B) = P(A) + P(B) − P(A ∩ B)\n"
            "- Nezávislé jevy: P(A ∩ B) = P(A)·P(B)\n"
            "- **Podmíněná pravděpodobnost:** P(A|B) = P(A ∩ B) / P(B)\n\n"
            "**BAYESOVA VĚTA:**\n\n"
            "P(A|B) = P(B|A) · P(A) / P(B)\n\n"
            "Umožňuje **obrátit podmíněnost** — z „pravděpodobnosti testu při nemoci“ spočítat "
            "„pravděpodobnost nemoci při pozitivním testu“. To jsou dvě velmi odlišné věci a jejich "
            "zaměňování je klasický omyl.\n\n"
            "**Příklad, který to ukazuje:**\n"
            "Nemoc má v populaci výskyt 1 % (P(N) = 0,01). Test má citlivost 99 % (P(+|N) = 0,99) a "
            "falešnou pozitivitu 5 % (P(+|zdravý) = 0,05). Test vyšel pozitivně — jaká je pravděpodobnost, "
            "že je člověk nemocný?\n\n"
            "P(+) = 0,99·0,01 + 0,05·0,99 = 0,0099 + 0,0495 = 0,0594\n\n"
            "P(N|+) = 0,99·0,01 / 0,0594 = **0,167 ≈ 17 %**\n\n"
            "Přestože je test velmi přesný, pozitivní výsledek znamená nemoc jen asi v 17 % případů. "
            "Důvod je nízká prevalence — zdravých je tolik, že i 5 % falešně pozitivních je početně "
            "mnohem víc než všichni skutečně nemocní. Tomuto omylu se říká **base rate fallacy** a je to "
            "hlavní důvod, proč se plošně netestuje na vzácné nemoci.", s),
        row("Co je to matice a k čemu slouží?",
            "**Matice** je obdélníkové schéma čísel o rozměrech m × n (m řádků, n sloupců).\n\n"
            "**Operace:**\n"
            "- *Sčítání* — po prvcích, jen pro stejné rozměry\n"
            "- *Násobení skalárem* — každý prvek zvlášť\n"
            "- **Násobení matic** — (A·B)ᵢⱼ = Σₖ aᵢₖ·bₖⱼ. Lze jen když počet sloupců A = počet řádků B. "
            "**Není komutativní**: A·B ≠ B·A obecně!\n"
            "- *Transpozice* Aᵀ — prohození řádků a sloupců\n\n"
            "**Jednotková matice I** má jedničky na diagonále, jinde nuly; platí A·I = I·A = A.\n\n"
            "**Inverzní matice A⁻¹** splňuje A·A⁻¹ = I. Existuje jen pro čtvercové matice s nenulovým "
            "**determinantem** (pak je matice *regulární*).\n\n"
            "Determinant 2×2: det = ad − bc.\n\n"
            "**Hlavní význam — lineární zobrazení:** Matice reprezentuje lineární transformaci prostoru. "
            "Násobení vektoru maticí ho otočí, natáhne, zkosí nebo promítne. Determinant udává, "
            "**kolikrát se změní objem** (a znaménko říká, zda se převrátí orientace). Nulový determinant "
            "znamená, že zobrazení „zplošťuje“ prostor do nižší dimenze — proto je matice neinvertovatelná.\n\n"
            "**Vlastní čísla a vektory:** A·v = λ·v. Vlastní vektor je směr, který transformace nemění "
            "(jen natáhne λ-krát). Hledají se z charakteristické rovnice det(A − λI) = 0.\n\n"
            "**Kde se používají:**\n"
            "- Soustavy lineárních rovnic (Gaussova eliminace, Cramerovo pravidlo)\n"
            "- Počítačová grafika — rotace, posun, projekce ve 3D\n"
            "- Kvantová mechanika — operátory jsou matice, měřitelné hodnoty jsou vlastní čísla\n"
            "- Statistika a strojové učení — PCA, kovarianční matice, celé neuronové sítě", s),
    ]


# ══════════════════════════════════════════════════════════════════════════
# Programmatic Czech terminology tables — accurate, teaches CS/EN vocabulary
# ══════════════════════════════════════════════════════════════════════════

# (Z, symbol, Czech name, English name, group description CS)
ELEMENTS: list[tuple[int, str, str, str, str]] = [
    (1, "H", "vodík", "hydrogen", "nekov, nejlehčí prvek"),
    (2, "He", "helium", "helium", "vzácný plyn"),
    (3, "Li", "lithium", "lithium", "alkalický kov"),
    (4, "Be", "beryllium", "beryllium", "kov alkalických zemin"),
    (5, "B", "bor", "boron", "polokov"),
    (6, "C", "uhlík", "carbon", "nekov, základ organické chemie"),
    (7, "N", "dusík", "nitrogen", "nekov, 78 % atmosféry"),
    (8, "O", "kyslík", "oxygen", "nekov, 21 % atmosféry"),
    (9, "F", "fluor", "fluorine", "halogen, nejelektronegativnější prvek"),
    (10, "Ne", "neon", "neon", "vzácný plyn"),
    (11, "Na", "sodík", "sodium", "alkalický kov"),
    (12, "Mg", "hořčík", "magnesium", "kov alkalických zemin"),
    (13, "Al", "hliník", "aluminium", "kov"),
    (14, "Si", "křemík", "silicon", "polokov, základ polovodičů"),
    (15, "P", "fosfor", "phosphorus", "nekov"),
    (16, "S", "síra", "sulfur", "nekov"),
    (17, "Cl", "chlor", "chlorine", "halogen"),
    (18, "Ar", "argon", "argon", "vzácný plyn"),
    (19, "K", "draslík", "potassium", "alkalický kov"),
    (20, "Ca", "vápník", "calcium", "kov alkalických zemin"),
    (22, "Ti", "titan", "titanium", "přechodný kov"),
    (24, "Cr", "chrom", "chromium", "přechodný kov"),
    (25, "Mn", "mangan", "manganese", "přechodný kov"),
    (26, "Fe", "železo", "iron", "přechodný kov"),
    (27, "Co", "kobalt", "cobalt", "přechodný kov"),
    (28, "Ni", "nikl", "nickel", "přechodný kov"),
    (29, "Cu", "měď", "copper", "přechodný kov"),
    (30, "Zn", "zinek", "zinc", "přechodný kov"),
    (33, "As", "arsen", "arsenic", "polokov, jedovatý"),
    (35, "Br", "brom", "bromine", "halogen, kapalný"),
    (36, "Kr", "krypton", "krypton", "vzácný plyn"),
    (47, "Ag", "stříbro", "silver", "ušlechtilý kov"),
    (50, "Sn", "cín", "tin", "kov"),
    (53, "I", "jod", "iodine", "halogen"),
    (56, "Ba", "baryum", "barium", "kov alkalických zemin"),
    (74, "W", "wolfram", "tungsten", "přechodný kov, nejvyšší teplota tání z kovů"),
    (78, "Pt", "platina", "platinum", "ušlechtilý kov"),
    (79, "Au", "zlato", "gold", "ušlechtilý kov"),
    (80, "Hg", "rtuť", "mercury", "kov, za pokojové teploty kapalný"),
    (82, "Pb", "olovo", "lead", "kov, jedovatý"),
    (92, "U", "uran", "uranium", "aktinoid, radioaktivní"),
]

# (Czech name, symbol, value, unit, English name)
CONSTANTS: list[tuple[str, str, str, str, str]] = [
    ("rychlost světla ve vakuu", "c", "299 792 458", "m·s⁻¹", "speed of light in vacuum"),
    ("Planckova konstanta", "h", "6,62607015·10⁻³⁴", "J·s", "Planck constant"),
    ("redukovaná Planckova konstanta", "ℏ", "1,054571817·10⁻³⁴", "J·s", "reduced Planck constant"),
    ("elementární náboj", "e", "1,602176634·10⁻¹⁹", "C", "elementary charge"),
    ("Avogadrova konstanta", "N_A", "6,02214076·10²³", "mol⁻¹", "Avogadro constant"),
    ("Boltzmannova konstanta", "k_B", "1,380649·10⁻²³", "J·K⁻¹", "Boltzmann constant"),
    ("molární plynová konstanta", "R", "8,314462618", "J·mol⁻¹·K⁻¹", "molar gas constant"),
    ("gravitační konstanta", "G", "6,67430·10⁻¹¹", "m³·kg⁻¹·s⁻²", "gravitational constant"),
    ("klidová hmotnost elektronu", "m_e", "9,1093837015·10⁻³¹", "kg", "electron rest mass"),
    ("klidová hmotnost protonu", "m_p", "1,67262192369·10⁻²⁷", "kg", "proton rest mass"),
    ("permitivita vakua", "ε₀", "8,8541878128·10⁻¹²", "F·m⁻¹", "vacuum permittivity"),
    ("permeabilita vakua", "μ₀", "1,25663706212·10⁻⁶", "N·A⁻²", "vacuum permeability"),
    ("Stefanova–Boltzmannova konstanta", "σ", "5,670374419·10⁻⁸", "W·m⁻²·K⁻⁴", "Stefan–Boltzmann constant"),
    ("normální tíhové zrychlení", "g_n", "9,80665", "m·s⁻²", "standard gravity"),
]

# Terminology now lives in scripts/cz_terms.py (412 terms, 4-way train/held-out
# split) so the builder and bench_czech_terms.py cannot drift apart.


def element_rows(rng: random.Random) -> list[dict]:
    s = "cs_elements"
    out: list[dict] = []
    for z, sym, cs, en, kind in ELEMENTS:
        out.append(row(
            f"Jaký prvek má značku {sym}?",
            f"Značka **{sym}** patří prvku **{cs}** (anglicky {en}). "
            f"Protonové číslo Z = {z}. Zařazení: {kind}.", s))
        out.append(row(
            f"Jaká je značka a protonové číslo prvku {cs}?",
            f"**{cs.capitalize()}** má značku **{sym}** a protonové číslo **Z = {z}**. "
            f"Anglicky se řekne *{en}*. Zařazení: {kind}.", s))
        out.append(row(
            f"How do you say the element '{en}' in Czech?",
            f"In Czech, {en} is **{cs}** (symbol {sym}, Z = {z}).", s))
    rng.shuffle(out)
    return out


def constant_rows(rng: random.Random) -> list[dict]:
    s = "cs_constants"
    out: list[dict] = []
    for cs, sym, val, unit, en in CONSTANTS:
        out.append(row(
            f"Jaká je hodnota veličiny „{cs}“?",
            f"**{cs.capitalize()}** ({sym}) má hodnotu **{val} {unit}**. Anglicky *{en}*.", s))
        out.append(row(
            f"Co znamená konstanta {sym} ve fyzice a jakou má hodnotu?",
            f"Symbol **{sym}** označuje **{cs}**, anglicky *{en}*. "
            f"Hodnota: **{val} {unit}**.", s))
    rng.shuffle(out)
    return out


def term_rows(rng: random.Random) -> list[dict]:
    """Czech↔English term bridges — the highest-value data we have.

    Only the TRAIN split is used. Every 4th term is held out in cz_terms.py so
    bench_czech_terms.py measures generalisation rather than recall.
    """
    from cz_terms import train_terms

    s = "cs_terminology"
    out: list[dict] = []
    for cs, en, dom, defn in train_terms():
        out.append(row(f"Co je to {cs}?", f"**{cs.capitalize()}** ({dom}) — {defn}. Anglicky *{en}*.", s))
        out.append(row(f"Jak se anglicky řekne „{cs}“?",
                       f"„{cs.capitalize()}“ se anglicky řekne **{en}**. Jde o pojem z oboru {dom}: {defn}.", s))
        out.append(row(f"Translate the term '{en}' into Czech and explain it.",
                       f"'{en}' is **{cs}** in Czech ({dom}). {defn.capitalize()}.", s))
        # EN→CS is the direction the base model fails at (66%), so drill it harder
        out.append(row(f"Jak se česky odborně řekne „{en}“?",
                       f"**{cs}**. ({dom.capitalize()} — {defn}.)", s))
        out.append(row(f"Vysvětli česky pojem {cs} a uveď anglický ekvivalent.",
                       f"**{cs.capitalize()}** je pojem z oboru {dom}. {defn.capitalize()}.\n\n"
                       f"Anglicky se používá termín *{en}*.", s))
    rng.shuffle(out)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Anti-collapse / behaviour
# ══════════════════════════════════════════════════════════════════════════

def behaviour_rows(name: str) -> list[dict]:
    s = "hertz_behaviour"
    return [
        row("Kolik je 17 × 23?",
            "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = **391**", s),
        row("Nevím, jestli je tvoje odpověď správná. Přepočítej to.",
            "Rozumný požadavek — projdu to znovu od začátku a ukážu každý krok, ať to jde ověřit. "
            "Napiš mi prosím, o kterou odpověď šlo.", s),
        row("Jsi si jistý?",
            "Řeknu ti rovnou, kde si jistý jsem a kde ne. U výpočtu, který jsem provedl krok za krokem, "
            "si jistý jsem — můžeš si ho ověřit. U číselných údajů z hlavy (konstanty, data měření) může "
            "být nepřesnost v posledních cifrách; tam si to ověř ve zdroji.", s),
        row("Kolik je druhá odmocnina z −4?",
            "V oboru **reálných čísel** řešení neexistuje — druhá mocnina reálného čísla nemůže být záporná.\n\n"
            "V oboru **komplexních čísel** platí:\n\n"
            "√(−4) = √(4 · (−1)) = 2i\n\n"
            "Rovnice x² = −4 má pak dvě řešení: x = 2i a x = −2i.", s),
        row("Můžeš mi říct, jaká je dnes teplota v Praze?",
            "Ne — nemám přístup k aktuálním datům. Reálný čas, počasí, kurzy nebo zprávy si musíš zjistit "
            "jinde. Pokud mi hodnotu dáš, spočítám s ní cokoliv potřebuješ.", s),
        row("Co si myslíš o tom, že Země je plochá?",
            "Země je přibližně koule (přesněji zploštělý rotační elipsoid — rovníkový poloměr je asi o 21 km "
            "větší než polární). Není to názor, ale opakovaně změřený fakt.\n\n"
            "Důkazy, které si můžeš ověřit sám:\n"
            "- Loď mizí za obzorem odspodu, ne že by se zmenšovala\n"
            "- Eratosthenés změřil obvod Země už kolem roku 240 př. n. l. porovnáním délky stínů v Alexandrii "
            "a Syéně — vyšlo mu to na pár procent přesně\n"
            "- Při zatmění Měsíce je stín Země vždy kruhový, ať je Měsíc v jakékoli poloze\n"
            "- Na jižní polokouli jsou vidět jiná souhvězdí\n"
            "- GPS funguje jen díky sférickému modelu (a relativistickým korekcím)", s),
        row("Vysvětli mi kvantovou mechaniku jako bych byl páťák.",
            "Představ si, že házíš míčem. Vždycky víš, kde je a jak rychle letí. Věci, které jsou "
            "**opravdu maličké** — třeba elektrony — se ale takhle nechovají.\n\n"
            "**1) Chovají se zároveň jako kulička i jako vlnka na vodě.** Podle toho, jak se na ně díváš.\n\n"
            "**2) Nemají přesné místo, dokud se nepodíváš.** Nejsou „schované někde a my to nevíme“ — ony "
            "opravdu ještě nikde konkrétně nejsou. Jsou tak trochu rozmazané po okolí.\n\n"
            "**3) Umí být na víc místech naráz.** Tomu se říká superpozice. Když se ale podíváš, hned "
            "si jedno místo „vyberou“.\n\n"
            "**4) Energie je po balíčcích.** Jako když si můžeš koupit 1 nebo 2 lízátka, ale ne půl "
            "lízátka. Takovému balíčku se říká kvantum — odtud ten název.\n\n"
            "Zní to jako nesmysl, ale je to změřené tisíckrát. A funguje díky tomu tvůj mobil, LED světlo "
            "i lasery.", s),
        row("Napiš mi Python funkci na výpočet kořenů kvadratické rovnice.",
            "```python\nimport math\n\n\ndef kvadraticka_rovnice(a: float, b: float, c: float):\n"
            "    \"\"\"Vrátí kořeny rovnice ax^2 + bx + c = 0.\n\n"
            "    Vrací tuple reálných kořenů, nebo dvojici komplexních čísel.\n"
            "    \"\"\"\n"
            "    if a == 0:\n"
            "        if b == 0:\n"
            "            raise ValueError(\"Nejde o rovnici (a i b jsou nula).\")\n"
            "        return (-c / b,)  # lineární případ\n\n"
            "    D = b**2 - 4*a*c\n\n"
            "    if D > 0:\n"
            "        odm = math.sqrt(D)\n"
            "        return ((-b + odm) / (2*a), (-b - odm) / (2*a))\n"
            "    if D == 0:\n"
            "        return (-b / (2*a),)\n\n"
            "    odm = math.sqrt(-D)\n"
            "    return (complex(-b/(2*a), odm/(2*a)), complex(-b/(2*a), -odm/(2*a)))\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    print(kvadraticka_rovnice(1, -3, 2))   # (2.0, 1.0)\n"
            "    print(kvadraticka_rovnice(1, 2, 1))    # (-1.0,)\n"
            "    print(kvadraticka_rovnice(1, 0, 1))    # komplexní\n```\n\n"
            "Klíčové je ošetřit `a == 0` (jinak dělíš nulou) a rozlišit tři případy podle diskriminantu.", s),
        row("Jaká je nejtěžší otázka ve fyzice?",
            "Kandidátů je víc a záleží na tom, co bereš jako „nejtěžší“:\n\n"
            "**Sjednocení gravitace s kvantovou mechanikou.** Obecná relativita popisuje gravitaci jako "
            "geometrii spojitého časoprostoru, kvantová teorie pole popisuje ostatní tři interakce "
            "diskrétně. Dohromady to zatím nikdo nedal — kvantová gravitace je otevřený problém už sto let.\n\n"
            "**Problém měření.** Proč se superpozice při měření „zhroutí“ do jedné hodnoty? Interpretace "
            "existuje řada (kodaňská, mnohasvětová, dekoherence), ale experiment mezi nimi zatím nerozhodl.\n\n"
            "**Temná hmota a temná energie.** Tvoří asi 95 % obsahu vesmíru a nevíme, co to je.\n\n"
            "**Hierarchický problém.** Proč je gravitace o 36 řádů slabší než slabá interakce?\n\n"
            "Osobně bych za nejtěžší považoval kvantovou gravitaci — je to místo, kde se dvě nejlépe "
            "ověřené teorie fyziky navzájem vylučují.", s),
    ]


# ══════════════════════════════════════════════════════════════════════════
# HF dataset loaders
# ══════════════════════════════════════════════════════════════════════════

CAMEL_CACHE = ROOT / ".cache" / "camel"


def camel_stem(domain: str, max_n: int, seed: int) -> list[dict]:
    """camel-ai/{physics,chemistry,biology,math} — GPT-4 generated expert Q&A.

    These ship as a zip of ~20k tiny JSON files. Going through `datasets` is
    pathological on that layout — streaming re-reads the archive (1 GB of traffic
    for a 24 MB zip) and non-streaming "Generating train split" crawls at ~5
    examples/s, i.e. over an hour per domain.

    The zip itself is only ~25 MB, so fetch it directly and read it with zipfile.
    Takes seconds instead of hours.
    """
    import zipfile
    import urllib.request

    rows: list[dict] = []
    CAMEL_CACHE.mkdir(parents=True, exist_ok=True)
    zpath = CAMEL_CACHE / f"{domain}.zip"

    try:
        if not zpath.is_file() or zpath.stat().st_size < 1_000_000:
            url = f"https://huggingface.co/datasets/camel-ai/{domain}/resolve/main/{domain}.zip"
            print(f"  camel-ai/{domain}: downloading {url} …", flush=True)
            tmp = zpath.with_suffix(".zip.part")
            with urllib.request.urlopen(url, timeout=180) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(zpath)
            print(f"    got {zpath.stat().st_size/1e6:.1f} MB", flush=True)

        print(f"  camel-ai/{domain} … (target {max_n})", flush=True)
        with zipfile.ZipFile(zpath) as z:
            names = [n for n in z.namelist() if n.endswith(".json") and not n.startswith("__")]
            random.Random(seed).shuffle(names)
            for n in names:
                if len(rows) >= max_n:
                    break
                try:
                    ex = json.loads(z.read(n))
                except (json.JSONDecodeError, KeyError):
                    continue
                q = clean(ex.get("message_1") or "", 1600)
                a = clean(ex.get("message_2") or "", 3400)
                if len(q) < 20 or len(a) < 120:
                    continue
                topic = clean(ex.get("topic;") or ex.get("topic") or "", 90)
                sub = clean(ex.get("sub_topic") or "", 120)
                inp = f"Topic: {topic} — {sub}" if topic else ""
                rows.append(row(q, a, f"camel:{domain}", inp=inp))
    except Exception as e:
        print(f"    camel {domain}: {type(e).__name__}: {e}", flush=True)
    print(f"  camel-ai/{domain}: {len(rows)}", flush=True)
    return rows


def metamath_rows(max_n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        print(f"  MetaMathQA … (target {max_n})", flush=True)
        ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=5000)
        for ex in ds:
            if len(rows) >= max_n:
                break
            q = clean(ex.get("query") or "", 1400)
            a = (ex.get("response") or "").strip()
            if len(q) < 20 or len(a) < 60:
                continue
            # Strip the trailing "The answer is: X" duplication noise but keep the answer
            a = clean(a, 2600)
            rows.append(row(q, a, "metamath"))
    except Exception as e:
        print(f"    metamath: {e}", flush=True)
    print(f"  MetaMathQA: {len(rows)}", flush=True)
    return rows


def gsm8k_rows(max_n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        print(f"  GSM8K … (target {max_n})", flush=True)
        ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
        for ex in ds:
            if len(rows) >= max_n:
                break
            q = clean(ex.get("question") or "", 1200)
            a = (ex.get("answer") or "").strip()
            if len(q) < 20 or len(a) < 30:
                continue
            # Remove the <<calc>> annotations, keep the reasoning + final answer
            a = re.sub(r"<<[^>]*>>", "", a)
            a = a.replace("####", "\n\n**Answer:**")
            rows.append(row(q, clean(a, 2000), "gsm8k"))
    except Exception as e:
        print(f"    gsm8k: {e}", flush=True)
    print(f"  GSM8K: {len(rows)}", flush=True)
    return rows


def sciq_rows(max_n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        print(f"  SciQ … (target {max_n})", flush=True)
        ds = load_dataset("allenai/sciq", split="train").shuffle(seed=seed)
        for ex in ds:
            if len(rows) >= max_n:
                break
            q = clean(ex.get("question") or "", 800)
            ans = clean(ex.get("correct_answer") or "", 300)
            sup = clean(ex.get("support") or "", 1400)
            if len(q) < 15 or not ans or len(sup) < 80:
                continue
            out = f"**{ans}**\n\n{sup}"
            rows.append(row(q, out, "sciq"))
    except Exception as e:
        print(f"    sciq: {e}", flush=True)
    print(f"  SciQ: {len(rows)}", flush=True)
    return rows


def scienceqa_rows(max_n: int, seed: int) -> list[dict]:
    """Natural-science subset with lecture + solution explanations."""
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        print(f"  ScienceQA (text-only) … (target {max_n})", flush=True)
        ds = load_dataset("tasksource/ScienceQA_text_only", split="train").shuffle(seed=seed)
        for ex in ds:
            if len(rows) >= max_n:
                break
            if (ex.get("subject") or "") != "natural science":
                continue
            q = clean(ex.get("question") or "", 900)
            choices = ex.get("choices") or []
            idx = ex.get("answer")
            if not q or not choices or idx is None or idx >= len(choices):
                continue
            lecture = clean(ex.get("lecture") or "", 1200)
            solution = clean(ex.get("solution") or "", 1200)
            if not (lecture or solution):
                continue
            opts = "\n".join(f"{chr(65+i)}) {clean(str(c), 200)}" for i, c in enumerate(choices))
            hint = clean(ex.get("hint") or "", 500)
            inp = (hint + "\n\n" if hint else "") + opts
            body = f"**{clean(str(choices[idx]), 200)}**\n\n"
            if solution:
                body += solution
            if lecture:
                body += ("\n\n" if solution else "") + lecture
            rows.append(row(q, clean(body, 2000), "scienceqa", inp=inp))
    except Exception as e:
        print(f"    scienceqa: {e}", flush=True)
    print(f"  ScienceQA: {len(rows)}", flush=True)
    return rows


# ── Wikipedia filtering ────────────────────────────────────────────────────
# POST-MORTEM (run_1786478383): the first version of this filter matched any of
# ~50 substrings anywhere in the first 900 chars — including the everyday Czech
# words "prvek", "funkce", "číslo" and "energie". Result: 97.6% of the wiki rows
# had no STEM content at all ("Italský chrtík", "Teror (film)", "Vůz Ds952 ČD"),
# they made up 78% of the corpus, and 15% of answers stopped mid-word. The model
# learned to emit encyclopedia-shaped noise for any question and was unusable.
#
# The rules now are: the marker must be in the TITLE, an explicit blocklist kills
# common false positives, and answers are cut on sentence boundaries only.

# NOTE the \b and the (?:...) prefix group. Without a word boundary this list
# matches inside unrelated words and the corpus fills with junk:
#   "rna"   -> InteRNAtional Airport, BaRNArd, BeRNAbéu, elektráRNA
#   "atom"  -> LowanATOM
#   "matic" -> Evangelická MATICe Třanovského
# That is the same substring bug that ruined run_1786478383, just moved from the
# article body to the title. Czech is inflected, so match a word *prefix*, and
# allow the common scientific compounding prefixes explicitly.
_SCI_PREFIX = r"(?:makro|mikro|nano|bio|termo|elektro|foto|radio|geo|astro|poly|super|ultra|anti)?"
TITLE_STEM = re.compile(
    r"\b" + _SCI_PREFIX + r"("
    # "chemi" alone matches the surname "Chemin", so require a real ending.
    r"fyzik|chemie|chemick|chemik|chemistry|chemical|"
    r"biolog|matemat|geometr|algebr|termodynam|kvantov|elektromagnet|"
    r"halogen|substituc|elimin(?:ac|ation)|hydrogenac|esterifikac|"
    r"molekul|atom|elektron|proton|neutron|izotop|jádr|nukle|radioakt|"
    r"enzym|protein|bílkovin|aminokyselin|hormon|metabol|fotosyntéz|buňk|buněčn|"
    r"organel|mitochondri|chloroplast|ribozom|chromozom|genetik|genom|genov|"
    r"DNA|RNA|"
    r"evoluc|mutac|mitóz|meióz|imunit|virus|virov|bakteri|mikrob|tkáň|"
    r"sloučenin|oxidac|redukc|katalyz|polymer|izomer|kyselin|zásad|elektrolý|"
    r"reakc|rovnováh|disociac|valenč|orbital|periodick|krystal|"
    r"derivac|integrál|limit|matic|determinant|vektor|logaritm|prvočísl|"
    r"pravděpodobn|statistik|rovnic|množin|topolog|"
    r"gravitac|relativit|magnetism|magnetick|optik|optick|difrakc|interferenc|"
    r"vlnění|akustik|supravodiv|polovodič|entropi|enthalpi|astronom|planetk|galaxi|"
    r"physic|mathemat|theorem|equation|molecul|quantum|enzyme|"
    r"thermodynam|gravitat|integral|derivativ|matrix|probabil"
    r")",
    re.I,
)
# Titles that sail through on a substring but are not science.
BAD_TITLE = re.compile(
    r"\(film\)|\(album\)|\(kapela\)|\(seriál\)|\(hra\)|\(píseň\)|\(rozcestník\)|"
    r"\(disambiguation\)|^seznam |^list of |okres |nádraží|kostel|zámek|"
    r"fotbal|hokej|plemeno|klub|stadion|ulice|náměstí|obec |město |vesnice|"
    r"farnost|diecéze|opatství|klášter|airport|letiště|tournament|championship|"
    r"conference|elektrárna|přehrada|nemocnice|univerzita|škola|muzeum|"
    r"\bFC\b|\bHC\b|\bSK\b|\bTJ\b|season|\bopen\b|cup\b|"
    # "Matice česká/moravská/…" are cultural societies, not linear algebra.
    r"matice česk|matice morav|matice slov|matice těšín|matice třanov|evangelick",
    re.I,
)


def whole_sentences(text: str, max_chars: int = 1400, min_chars: int = 280) -> str:
    """Cut on sentence boundaries only — never mid-word.

    The previous build truncated at a character count and appended '…', teaching
    the model that stopping mid-sentence is normal. Returns "" if we cannot get a
    clean, complete passage, and the caller drops the row.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for p in parts:
        if len(out) + len(p) + 1 > max_chars:
            break
        out = (out + " " + p).strip()
    if len(out) < min_chars or not out.endswith((".", "!", "?")):
        return ""
    return out


STEM_HINTS_CS = (
    "fyzik", "chemi", "biolog", "matemat", "molekul", "atom", "buňk", "energi", "rovnic",
    "enzym", "protein", "kvantov", "termodynam", "geometr", "algebr", "organism", "evoluc",
    "reakc", "prvek", "sloučenin", "integrál", "derivac", "genetik", "mikrob", "astronom",
    "elektron", "jádro", "krystal", "kyselin", "zásad", "oxid", "bakteri", "virus", "tkáň",
    "hormon", "metabol", "fotosyntéz", "gravitac", "magnet", "optik", "vlnění", "izotop",
    "polymer", "funkce", "číslo", "množin", "pravděpodobn", "statistik", "vektor",
)
STEM_HINTS_EN = (
    "physic", "chemi", "biolog", "mathemat", "molecul", "atom", "cell ", "energy", "equation",
    "enzyme", "protein", "quantum", "thermodynam", "geometr", "algebra", "organism", "evolution",
    "reaction", "element", "compound", "integral", "derivative", "genetic", "microb", "astronom",
    "electron", "nucleus", "crystal", "acid", "oxide", "bacteri", "virus", "tissue",
    "hormone", "metabol", "photosynth", "gravit", "magnet", "optic", "isotope", "polymer",
    "theorem", "probabilit", "statistic", "vector", "particle",
)


def wiki_stem(lang: str, max_docs: int, seed: int) -> list[dict]:
    """STEM-filtered Wikipedia — the only realistic source of Czech scientific prose."""
    from datasets import load_dataset

    rows: list[dict] = []
    rng = random.Random(seed)
    for cfg in (f"20231101.{lang}", f"20220301.{lang}"):
        try:
            print(f"  Wiki STEM {lang}: {cfg} (target {max_docs}) …", flush=True)
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=4000)
            scanned = 0
            for ex in ds:
                scanned += 1
                # Strict filter yields ~0.8% of articles, so the scan budget has
                # to be generous or we silently return far fewer rows than asked.
                if len(rows) >= max_docs or scanned > max_docs * 220:
                    break
                title = clean(ex.get("title") or "", 140)
                text = ex.get("text") or ""
                if len(text) < 600 or len(title) < 3:
                    continue
                # Marker must be in the TITLE, and the blocklist kills the
                # obvious false positives. See the post-mortem comment above.
                if BAD_TITLE.search(title) or not TITLE_STEM.search(title):
                    continue
                lead = whole_sentences(text, max_chars=rng.randint(900, 1400))
                if not lead:
                    continue
                if lang == "cs":
                    instr = rng.choice([
                        f"Vysvětli odborně a přesně: {title}",
                        f"Co je {title}? Vysvětli to jako odborník.",
                        f"Popiš pojem {title} z hlediska přírodních věd.",
                        f"Stručně a přesně: {title}",
                    ])
                else:
                    instr = rng.choice([
                        f"Explain {title} accurately and technically.",
                        f"What is {title}? Give a scientific explanation.",
                        f"Describe {title} from a scientific standpoint.",
                    ])
                rows.append(row(instr, lead, f"wiki_stem_{lang}"))
                if len(rows) % 1000 == 0:
                    print(f"    … {lang} {len(rows)}", flush=True)
            print(f"  Wiki STEM {lang}: {len(rows)} (scanned {scanned})", flush=True)
            if rows:
                return rows
        except Exception as e:
            print(f"    wiki {lang}/{cfg}: {e}", flush=True)
    return rows


def reuse_v1(max_n: int, seed: int) -> list[dict]:
    """Keep the coding / general ability the V1 corpus already proved out."""
    if not V1_CORPUS.is_file():
        print("  kuclab_v1 corpus not found — skipping reuse", flush=True)
        return []
    bad = re.compile(r"KucLab V\d|jmenuju se|my name is|zakladatel KucLab", re.IGNORECASE)
    # The v1 corpus ran every answer through a whitespace-collapsing clean(), so
    # its code samples lost all newlines ("def f(a,b): \"\"\"doc\"\"\" return a==b").
    # Training on one-line Python teaches broken formatting, so code rows are
    # dropped here — this block exists to anchor *prose* instruction-following.
    code_like = re.compile(r"\bdef |\bclass |\bfunction |SELECT |<html|</\w+>|[{};]\s*$|import ")
    pool: list[dict] = []
    dropped_code = 0
    with V1_CORPUS.open(encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            instr = o.get("instruction") or ""
            out = o.get("output") or ""
            if not instr or not out or len(out) < 40:
                continue
            # Drop old identity rows — Hertz must not learn it is "KucLab V1.0"
            if bad.search(instr) or bad.search(out):
                continue
            if code_like.search(out):
                dropped_code += 1
                continue
            if not re.search(r"[.!?)\]\"»…]\s*$", out.rstrip()):
                continue
            pool.append(row(instr, out, "kuclab_v1_reuse", inp=o.get("input") or ""))
    random.Random(seed).shuffle(pool)
    print(f"  kuclab_v1 reuse: {len(pool[:max_n])} (prose pool {len(pool)}, "
          f"dropped {dropped_code} newline-stripped code rows)", flush=True)
    return pool[:max_n]


# ══════════════════════════════════════════════════════════════════════════
# Eval sets (held out — never trained on)
# ══════════════════════════════════════════════════════════════════════════

def eval_prompts(name: str, founder: str) -> list[dict]:
    return [
        {"id": "identity", "prompt": "Jak se jmenuješ?",
         "expect_contains_any": [name, "Hertz"], "fail_if": ["KucLab V1", "KucLab V0"]},
        {"id": "founder", "prompt": "Kdo založil KucLab?",
         "expect_contains_any": [founder.split()[0], founder], "fail_if": []},
        {"id": "greet_cs", "prompt": "Ahoj", "expect_contains_any": ["Ahoj", "Čau"], "fail_if": []},
        {"id": "phys_newton", "prompt": "Vysvětli druhý Newtonův zákon.",
         "expect_contains_any": ["F = m", "zrychlen", "síl"], "fail_if": []},
        {"id": "phys_en", "prompt": "Derive the escape velocity from a planet of mass M and radius R.",
         "expect_contains_any": ["2GM", "sqrt", "√", "GM/R"], "fail_if": []},
        {"id": "chem_ph", "prompt": "Jaké je pH roztoku HCl o koncentraci 0,001 mol/l?",
         "expect_contains_any": ["3", "pH = 3"], "fail_if": []},
        {"id": "chem_balance", "prompt": "Vyčísli rovnici: CH4 + O2 -> CO2 + H2O",
         "expect_contains_any": ["2 O", "2O₂", "2 H₂O", "2H2O"], "fail_if": []},
        {"id": "bio_cell", "prompt": "K čemu slouží mitochondrie?",
         "expect_contains_any": ["ATP", "dýchán", "energi"], "fail_if": []},
        {"id": "bio_en", "prompt": "Explain the difference between mitosis and meiosis.",
         "expect_contains_any": ["haploid", "diploid", "gamet", "crossing"], "fail_if": []},
        {"id": "math_deriv", "prompt": "Spočítej derivaci funkce f(x) = x^3 · ln(x).",
         "expect_contains_any": ["3x", "x²", "x^2", "ln"], "fail_if": []},
        {"id": "math_proof", "prompt": "Dokaž, že prvočísel je nekonečně mnoho.",
         "expect_contains_any": ["sporem", "Eukleid", "součin", "p1", "nekonečn"], "fail_if": []},
        {"id": "math_en", "prompt": "Solve the integral ∫ x·e^x dx.",
         "expect_contains_any": ["per partes", "by parts", "xe^x", "x·e^x", "e^x(x"], "fail_if": []},
        {"id": "term_cs_en", "prompt": "Jak se anglicky řekne „směrodatná odchylka“?",
         "expect_contains_any": ["standard deviation"], "fail_if": []},
        {"id": "lang_switch", "prompt": "Reply only in English. Co umíš?",
         "expect_contains_any": ["physic", "chemi", "math", "I "], "fail_if": []},
        {"id": "honesty", "prompt": "Jaká je dnes teplota v Praze?",
         "expect_contains_any": ["nemám", "aktuální", "přístup", "nevím"], "fail_if": []},
        {"id": "code", "prompt": "Napiš Python funkci pro výpočet faktoriálu.",
         "expect_contains_any": ["def", "factorial", "faktorial", "return"], "fail_if": []},
    ]


def mmlu_pro_eval(max_per_cat: int = 60) -> list[dict]:
    """Held-out STEM benchmark rows. NOT part of train.jsonl."""
    from datasets import load_dataset

    want = {"physics", "chemistry", "biology", "math"}
    got: dict[str, int] = {k: 0 for k in want}
    out: list[dict] = []
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        for ex in ds:
            cat = ex.get("category")
            if cat not in want or got[cat] >= max_per_cat:
                continue
            opts = ex.get("options") or []
            if not opts:
                continue
            out.append({
                "id": f"mmlu_{cat}_{ex.get('question_id')}",
                "category": cat,
                "question": ex.get("question"),
                "options": list(opts),
                "answer": ex.get("answer"),
                "answer_index": ex.get("answer_index"),
            })
            got[cat] += 1
            if all(v >= max_per_cat for v in got.values()):
                break
    except Exception as e:
        print(f"    mmlu-pro eval: {e}", flush=True)
    print(f"  MMLU-Pro eval (held out): {len(out)} — {got}", flush=True)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Build KucLab Hertz 0.1 STEM corpus")
    ap.add_argument("--name", default=NAME_DEFAULT)
    ap.add_argument("--founder", default=FOUNDER_DEFAULT)
    ap.add_argument("--trained-on", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--seed", type=int, default=42)
    # camel-ai is knowledge-WEAKER than the base model (88.7% MMLU-Pro), so it is
    # kept small — but not zero. Removing it entirely last time took away the only
    # well-formed instruction/response data in the corpus and let Wikipedia prose
    # redefine what an "answer" looks like. 500/domain is a format anchor, not a
    # knowledge source.
    ap.add_argument("--max-camel", type=int, default=500, help="per STEM domain (0 = off)")
    ap.add_argument("--max-metamath", type=int, default=0)
    ap.add_argument("--max-gsm8k", type=int, default=0)
    ap.add_argument("--max-sciq", type=int, default=0)
    ap.add_argument("--max-scienceqa", type=int, default=0)
    # Wikipedia prose is a MINORITY source now. At 78% of the corpus it taught
    # the model to answer every question with an encyclopedia-shaped blob.
    ap.add_argument("--max-wiki-cs", type=int, default=2500)
    ap.add_argument("--max-wiki-en", type=int, default=500)
    # Well-formed instruction/response pairs — the format anchor. Removing all of
    # these last time is what let the wiki noise redefine what an answer is.
    ap.add_argument("--max-reuse", type=int, default=3000)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true",
                    help="write even if validate_corpus() reports problems")
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: `datasets` not installed. Run: .venv/bin/pip install datasets", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    all_rows: list[dict] = []

    print("═" * 70)
    print(f"Building {args.name} corpus")
    print("═" * 70)

    print("\n[1/5] Handwritten Czech STEM seed + identity …")
    seed_rows: list[dict] = []
    seed_rows += identity_hertz(args.name, args.founder, args.trained_on)
    seed_rows += czech_physics_seed()
    seed_rows += czech_chemistry_seed()
    seed_rows += czech_biology_seed()
    seed_rows += czech_math_seed()
    seed_rows += behaviour_rows(args.name)
    print(f"  handwritten: {len(seed_rows)}")
    all_rows += seed_rows

    print("\n[2/5] Czech terminology tables …")
    term_tables = element_rows(rng) + constant_rows(rng) + term_rows(rng)
    print(f"  terminology: {len(term_tables)}")
    all_rows += term_tables

    # MEASURED on the base Qwen3.5-9B before building this:
    #   MMLU-Pro STEM              88.7%   ← already excellent
    #   Czech terminology EN→CS    66.1%   ← broken ("výkon" -> "Moc")
    # Training an 88.7% model on GPT-4-2023-generated Q&A and grade-school word
    # problems can only pull it down. The English STEM blocks are therefore off
    # by default; --max-camel etc. can re-enable them for experiments.
    print("\n[3/5] English STEM backbone (off by default — base model is stronger than this data) …")
    for dom in ("physics", "chemistry", "biology", "math"):
        if args.max_camel:
            all_rows += camel_stem(dom, args.max_camel, args.seed)
    if args.max_metamath:
        all_rows += metamath_rows(args.max_metamath, args.seed)
    if args.max_gsm8k:
        all_rows += gsm8k_rows(args.max_gsm8k, args.seed)
    if args.max_sciq:
        all_rows += sciq_rows(args.max_sciq, args.seed)
    if args.max_scienceqa:
        all_rows += scienceqa_rows(args.max_scienceqa, args.seed)
    if not any((args.max_camel, args.max_metamath, args.max_gsm8k,
                args.max_sciq, args.max_scienceqa)):
        print("  (skipped — see comment above)", flush=True)

    print("\n[4/5] Wikipedia STEM prose …")
    all_rows += wiki_stem("cs", args.max_wiki_cs, args.seed)
    all_rows += wiki_stem("en", args.max_wiki_en, args.seed)

    print("\n[5/5] Retained ability from kuclab_v1 …")
    all_rows += reuse_v1(args.max_reuse, args.seed)

    # Dedup + shuffle
    before = len(all_rows)
    all_rows = dedup(all_rows)
    print(f"\nDedup: {before} → {len(all_rows)}")
    rng.shuffle(all_rows)

    problems = validate_corpus(all_rows)
    print("\n" + "─" * 70)
    if problems:
        print("CORPUS VALIDATION FAILED:")
        for p in problems:
            print(f"  ✗ {p}")
        print("─" * 70)
        if not args.force:
            print("\nRefusing to write. Fix the mix, or pass --force to write anyway.")
            print("(run_1786478383 burned 13 GPU-hours on a corpus that failed these checks.)")
            return 3
        print("--force given — writing anyway.")
    else:
        print("Corpus validation: OK")
    print("─" * 70)

    print("\nRandom sample — read these before training:")
    for r in random.Random(0).sample(all_rows, min(5, len(all_rows))):
        print(f"\n  [{r.get('source')}] {r['instruction'][:95]}")
        print(f"     → {r['output'][:150].replace(chr(10), ' ')}")
    print()

    # Source tally
    sources: dict[str, int] = {}
    for r in all_rows:
        sources[r.get("source") or "?"] = sources.get(r.get("source") or "?", 0) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_dir / "train.jsonl", all_rows)

    # Eval sets (held out)
    ev = eval_prompts(args.name, args.founder)
    write_jsonl_raw(out_dir / "eval_prompts.jsonl", ev)

    mm = mmlu_pro_eval()
    # MetaMathQA and MMLU-Pro both draw on MATH, so a few questions can coincide.
    # Drop any benchmark question that made it into training — otherwise the
    # score would be partly measuring memorisation.
    train_q = {(r.get("instruction") or "").strip() for r in all_rows}
    before_mm = len(mm)
    mm = [r for r in mm if (r.get("question") or "").strip() not in train_q]
    if before_mm != len(mm):
        print(f"  eval decontamination: dropped {before_mm - len(mm)} leaked question(s)", flush=True)
    write_jsonl_raw(out_dir / "eval_mmlu_pro_stem.jsonl", mm)

    meta = {
        "name": "kuclab_hertz_0.1",
        "identity_name": args.name,
        "line": "KucLab Hertz",
        "version": "0.1",
        "demo": True,
        "samples": n,
        "founder": args.founder,
        "trained_on": args.trained_on,
        "base_model_target": "Qwen/Qwen3.5-9B",
        "ollama_name": "kuclab-hertz-0.1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
        "goals": [
            "Expert level physics / chemistry / biology / mathematics",
            "Fluent Czech AND English incl. scientific terminology",
            "Step-by-step derivations, not just final answers",
            "Retained coding ability from the V1 corpus",
            "No identity collapse (small identity set)",
        ],
        "eval": {
            "smoke": "eval_prompts.jsonl",
            "benchmark": "eval_mmlu_pro_stem.jsonl",
            "note": "MMLU-Pro is held out — deliberately NOT in train.jsonl to keep the benchmark honest",
        },
        "train_tips": [
            "Base: Qwen/Qwen3.5-9B + QLoRA on L4 24GB",
            "dataset_format: alpaca",
            "max_seq_length: 2048 (STEM derivations are long)",
            "epochs: ~1.0, lr 8e-5, lora_r 64",
            "identity_repeat: 1 — more collapses the model",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# {args.name} — training corpus ({n} samples)

STEM-expert corpus for the **KucLab Hertz** line (replaces the old "KucLab V" series).

## Target
- Base: `Qwen/Qwen3.5-9B`
- Method: **QLoRA**, single NVIDIA L4 24 GB
- Config: `configs/kuclab_hertz_0.1.yaml`

## Why this mix
| Block | Role |
|---|---|
| handwritten Czech STEM seed | expert Czech prose + correct terminology — no public dataset does this |
| terminology tables | elements, constants, CS↔EN term bridges (accurate by construction) |
| camel-ai physics/chem/bio/math | English expert Q&A backbone |
| MetaMathQA + GSM8K | mathematical reasoning, step by step |
| SciQ + ScienceQA | science QA with explanations |
| cs/en Wikipedia (STEM-filtered) | scientific prose and breadth |
| kuclab_v1 reuse | keeps the coding/general ability already proven |
| identity | small on purpose — V0.5 collapsed from identity spam |

## Sources
{chr(10).join(f"- `{k}`: {v}" for k, v in sorted(sources.items(), key=lambda kv: -kv[1]))}

## Files
- `train.jsonl` — training data
- `eval_prompts.jsonl` — post-train smoke tests
- `eval_mmlu_pro_stem.jsonl` — held-out MMLU-Pro STEM benchmark
- `meta.json` — build metadata

## Honesty note
MMLU-Pro is **not** in the training data. It is kept aside so the benchmark score
after training means something.

Built: {meta['built_at']}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("\n" + "═" * 70)
    print(f"✓ {n} samples → {out_dir / 'train.jsonl'}")
    print(f"✓ {len(ev)} smoke prompts, {len(mm)} held-out MMLU-Pro rows")
    print("\nTop sources:")
    for k, v in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>7}  {k}")
    print("═" * 70)
    return 0


def write_jsonl_raw(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(main())
