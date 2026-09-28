#!/usr/bin/env python3
"""Hertz 0.9 bulk generation via Meta Model API (Muse Spark 1.3).

14 MMLU-Pro subjects x ~3000 rows. Answer-first + 'Answer: (X)' discipline.
Key: read from ~/.config/muse/auth.json api_key (never logged/printed).
Usage:
  .venv/bin/python scripts/gen_09_muse.py plan
  .venv/bin/python scripts/gen_09_muse.py run --only math,physics --workers 8
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, random, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "data" / "distill_09" / "train.jsonl"
SEEN_DEFAULT = ROOT / "data" / "distill_09" / ".seen.txt"
API = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3"

SYS = ("Reasoning strength: high. Odpovidej presne, odborne a strucne. "
       "Vzdy PRVNI vysledek nebo zaver, pak postup krok za krokem. "
       "U otazek s vyberem odpovedi POSLEDNI radek musi byt presne 'Answer: (X)' "
       "kde X je pismeno spravne moznosti. Pis v jazyce otazky.")

rng = random.Random(20260913)


def _key() -> str:
    d = json.load(open(os.path.expanduser("~/.config/muse/auth.json")))
    return d["providers"]["meta"]["api_key"]


DOMAINS = [
    ("biology", 3000, 0.4), ("business", 3000, 0.4), ("chemistry", 3000, 0.3),
    ("computer_science", 3000, 0.3), ("economics", 3000, 0.3), ("engineering", 3000, 0.3),
    ("health", 3000, 0.4), ("history", 3000, 0.4), ("law", 3000, 0.2),
    ("math", 3000, 0.2), ("philosophy", 3000, 0.5), ("physics", 3000, 0.2),
    ("psychology", 3000, 0.4), ("other", 3000, 0.4),
]


def fill(t):
    W = ["strom", "řeka", "kniha", "most", "les", "dům", "hora", "pole", "vlak", "pták"]
    return t.format(a=rng.randint(2, 99), b=rng.randint(2, 25), c=rng.randint(10, 999),
                    n=rng.randint(2, 12), sq=rng.randint(4, 625), pct=rng.randint(1, 99),
                    yr=rng.randint(1700, 2025), W=rng.choice(W))


BIOLOGY_T = [
    "Explain the stages of mitosis and what happens to chromosomes in each.",
    "Describe the human heart cycle (systole/diastole) with pressures.",
    "What is the function of hemoglobin? Explain oxygen binding cooperativity.",
    "Compare DNA replication in prokaryotes vs eukaryotes.",
    "Explain natural selection using antibiotic resistance as the example.",
    "Describepondělí: Vysvětli rozdíl mezi dýcháním a buněčným dýcháním.",
    "How do vaccines train adaptive immunity? Cover B and T cells.",
    "Explain the sliding filament theory of muscle contraction.",
    "What limits population growth? Explain {n} density-dependent factors.",
    "Describe the nephron's role in urine formation step by step.",
    "Explain Mendelian inheritance with a dihybrid cross of {W} traits.",
    "What is an action potential? Describe depolarization and repolarization.",
    "Compare C3, C4 and CAM photosynthesis pathways and their trade-offs.",
    "Explain the endosymbiotic theory evidence in {n} points.",
    "How does insulin regulate blood glucose? Include feedback loop.",
]
BUSINESS_T = [
    "Explain the difference between revenue, profit and cash flow.",
    "What is a balance sheet? Describe its {n} main sections.",
    "Define break-even point and compute it for fixed costs {c}00.",
    "Explain Porter's five forces on a coffee-shop example.",
    "What is the time value of money? Discount {c}00 due in {n} years.",
    "Compare debt vs equity financing for a startup.",
    "What is market segmentation? Give {n} bases with examples.",
    "Explain supply chain disruption using a {W} shortage scenario.",
    "What do liquidity ratios measure? Interpret current ratio {b}.{n}.",
    "Define price elasticity with a gasoline example.",
    "Explain the principal-agent problem in corporate governance.",
    "What is an IPO and why do companies go public? List {n} reasons.",
    "Compare FIFO vs LIFO inventory methods and their tax effect.",
    "Explain network effects with a platform example.",
    "What is working capital? Compute from current assets {c}0 and liabilities {a}0.",
]
CHEMISTRY_T = [
    "Balance: C3H8 + O2 -> CO2 + H2O. Show atom counts.",
    "Compute the pH of 0.0{b} M HCl and 0.0{b} M NaOH.",
    "How many grams of NaCl are in {c}0 mL of {b} M solution?",
    "Explain Le Chatelier with the Haber process (N2+3H2).",
    "Determine oxidation states in K2Cr2O7.",
    "What is the ideal gas law? Compute V for {n} mol at STP.",
    "Explain SN1 vs SN2 mechanisms with an example each.",
    "Calculate the molarity of {a} g NaOH in {b}00 mL.",
    "What is an electrochemical cell? Anode, cathode, salt bridge.",
    "Explain hydrogen bonding and why ice floats.",
    "Name CH3-CH2-COOH and describe its acidity.",
    "What is the rate law? Determine orders from given data table.",
    "Explain VSEPR for CH4, NH3 and H2O geometries.",
    "How many moles are in {c} g of CaCO3?",
    "What is a buffer? Show Henderson-Hasselbalch for pH {n}.",
]
CS_T = [
    "Explain quicksort partitioning on an example array of {n} elements.",
    "What is Big-O of binary search? Prove it.",
    "Write a Python function detecting cycles in a linked list (Floyd).",
    "Explain the difference between TCP and UDP with use cases.",
    "What is a deadlock? Give the {n} Coffman conditions.",
    "Describe how a hash table resolves collisions (chaining vs probing).",
    "Explain public-key cryptography (RSA idea) in steps.",
    "What is normalization to 3NF? Normalize a sample table.",
    "Write SQL: top {n} customers by revenue with a JOIN.",
    "Explain the CAP theorem with a database example.",
    "What is memoization? Show Fibonacci with and without it.",
    "Describe fetch-decode-execute for instruction ADD R1, R2.",
    "Explain gradient descent in {n} steps with a tiny example.",
    "What is a race condition? Show a Python threading fix.",
    "Compare BFS vs DFS: when is each better?",
]
ECONOMICS_T = [
    "Derive equilibrium price when Qd={c}-2P and Qs=3P-{a}.",
    "What is GDP deflator? Compute from nominal {c}0 and real {a}0.",
    "Explain inflation targeting by central banks (2% goal).",
    "Compute compound growth: {c} at {n}% for {b} years.",
    "What is a recession technically? Give the common rule.",
    "Explain fiscal vs monetary stimulus with lags.",
    "What is purchasing power parity? Big Mac example.",
    "Define unemployment rate and compute from {c}0 labor force, {a}0 unemployed.",
    "Explain the Phillips curve and its breakdown in the 1970s.",
    "What are externalities? Positive and negative example each.",
    "Explain comparative advantage numerically (2 goods, 2 countries).",
    "What does quantitative easing do to bond yields?",
    "Define Gini coefficient and interpret 0.{b}{n}.",
    "What is stagflation and why is it hard to fix?",
    "Explain how exchange rates affect imports priced at {c}.",
]
ENGINEERING_T = [
    "Compute stress in a steel rod ({a} mm²) under {c} kN load.",
    "Explain the Rankine cycle stages of a steam plant.",
    "What is resonance in structures? Give the bridge example.",
    "Ohm's law: R={a} Ω, U={b} V. Find current, power, energy in {n} s.",
    "Explain how a transformer steps {c} V to {b} V (turns ratio).",
    "What is factor of safety? Apply to a {c} kN design load.",
    "Describe PID control with a thermostat example.",
    "Explain beam bending: where is max stress and why?",
    "What is thermal efficiency? Compute for W={a} kJ, Q={c} kJ.",
    "How does a four-stroke engine work? Name the strokes.",
    "Explain signal aliasing and the Nyquist rate with numbers.",
    "What is fatigue failure? S-N curve in plain words.",
    "Compute Reynolds number for water at {b} m/s in a {n} cm pipe.",
    "Explain how GPS trilateration needs {n} satellites minimum.",
    "What is a heat exchanger? Counter-flow vs parallel.",
]
HEALTH_T = [
    "Explain how blood pressure {c}0/{a}0 is interpreted clinically.",
    "What does BMI {b}{n} mean? Categories and limits of BMI.",
    "Describe type 1 vs type 2 diabetes mechanisms.",
    "How do antibiotics work? Explain {n} main classes.",
    "What is an ECG wave (PQRST)? What does each deflection mean?",
    "Explain vaccine types: mRNA, inactivated, live-attenuated.",
    "What is hypertension management? Lifestyle + {n} drug classes.",
    "Describe the stages of wound healing.",
    "What is anemia? Iron-deficiency mechanism and labs.",
    "Explain how X-rays image bone but not soft tissue well.",
    "What is cholesterol (LDL vs HDL) and target ranges?",
    "Describe symptoms and first aid for stroke (FAST).",
    "What is antibiotic resistance and how does stewardship help?",
    "Explain the difference between epidemic and pandemic.",
    "How does insulin dosing relate to carbohydrate counting?",
]
HISTORY_T = [
    "List {n} causes of WWI in order of importance with dates.",
    "Explain the significance of the printing press (c. 1440).",
    "Compare Athenian democracy with the Roman Republic.",
    "What caused the 1929 crash? Describe the mechanism.",
    "Explain the fall of Rome: top {n} theories with evidence.",
    "Describe the Silk Road: goods, routes, cultural exchange.",
    "What was the Reformation? Causes in {n} points.",
    "Explain decolonization of Africa in the 1960s.",
    "What triggered the French Revolution? Estates, debt, ideas.",
    "Describe daily life in ancient Egypt across classes.",
    "What was the Marshall Plan and did it work?",
    "Explain the Cuban Missile Crisis day by day (13 days).",
    "What ended the Cold War? Give {n} converging causes.",
    "Describe the Ottoman Empire at its {c}00s peak.",
    "What was the Green Revolution in agriculture?",
]
LAW_T = [
    "Distinguish civil vs criminal liability with examples.",
    "What are the elements of negligence? Apply to a car crash.",
    "Explain hearsay and its main exceptions.",
    "What makes a contract enforceable? List {n} elements.",
    "Compare common law vs civil law traditions.",
    "What is double jeopardy and what are its limits?",
    "Explain copyright fair use via the {n}-factor test.",
    "What is due process? Procedural vs substantive.",
    "Define mens rea levels from negligence to purpose.",
    "What is an injunction? When do courts grant one?",
    "Explain the burden of proof: preponderance vs beyond reasonable doubt.",
    "What is GDPR's lawful-basis requirement in one paragraph?",
    "Distinguish theft, robbery and burglary legally.",
    "What is arbitration vs litigation? Pros and cons.",
    "Explain strict liability with product-defect examples.",
]
MATH_T = [
    "Prove the quadratic formula by completing the square.",
    "Compute ∫₀^{b} x^{n} dx and explain each step.",
    "Solve {a}x + {b} = {c} and verify by substitution.",
    "Prove √{sq} is irrational unless a perfect square.",
    "Find the derivative of sin(x²) using the chain rule.",
    "What is the central limit theorem? State it precisely.",
    "Compute C({c},{b}) and explain its combinatorial meaning.",
    "Prove by induction: sum 1..{n} = {n}({n}+1)/2.",
    "Find eigenvalues of [[{a},{b}],[{n},{c}]].",
    "Explain Bayes' theorem with a medical-test example.",
    "What is a p-value? Correct interpretation in 3 sentences.",
    "Solve the system: {a}x+{b}y={c}; x−y={n}.",
    "Define a limit (epsilon-delta) and apply to 2x at x={n}.",
    "What is the difference between correlation r=0.{b} and causation?",
    "Compute the area between y=x² and y={a}x.",
]
PHILOSOPHY_T = [
    "Explain the trolley problem and {n} major responses.",
    "What is the veil of ignorance? Apply to tax policy.",
    "Summarize Kant's categorical imperative with an example.",
    "Explain the hard problem of consciousness.",
    "What is the Ship of Theseus paradox? {n} resolutions.",
    "Define utilitarianism and state its strongest objection.",
    "Explain Hume's problem of induction.",
    "What is existentialism? Sartre's 'existence precedes essence'.",
    "Distinguish deontology from consequentialism clearly.",
    "What is the Gettier problem? Give the classic case.",
    "Explain Occam's razor and when it fails.",
    "What is virtue ethics? Aristotle's golden mean example.",
    "Summarize the mind-body debate positions.",
    "What is Rawls vs Nozick on justice? Contrast in 5 lines.",
    "Explain logical fallacy: strawman, with a debate example.",
]
PHYSICS_T = [
    "Derive v² = v0² + 2aΔx from definitions.",
    "A {a} kg object at {b} m/s: momentum and kinetic energy?",
    "Explain Bernoulli with an airplane-wing calculation.",
    "Circuit: R={a} Ω, U={b} V. Current, power, energy in {n} s.",
    "Derive the period of a {b} m pendulum.",
    "Two masses collide inelastically: find final velocity.",
    "Compute photon energy for λ={sq}0 nm in J and eV.",
    "Explain half-life with a {c} g sample, T½={n} days.",
    "What is entropy? Second law plus one everyday proof.",
    "Derive escape velocity for Earth numerically.",
    "Explain interference with the double-slit setup.",
    "How does a transformer work? Turns ratio example.",
    "Compute de Broglie wavelength for an electron at {c} eV.",
    "What is the photoelectric effect? Threshold frequency idea.",
    "Explain Kepler's third law with Earth/Mars numbers.",
]
PSYCHOLOGY_T = [
    "Explain Pavlovian vs operant conditioning with examples.",
    "What is the Big Five? Describe each trait briefly.",
    "Describe the Milgram experiment and its ethical issues.",
    "What is cognitive dissonance? Festinger's classic study.",
    "Explain attachment styles (secure, avoidant, anxious).",
    "What is the Stroop effect and what does it reveal?",
    "Describe Piaget's {n} stages of cognitive development.",
    "What is confirmation bias? Debiasing technique included.",
    "Explain the placebo effect mechanisms (not 'just psychology').",
    "What is the working-memory limit 7±2? Modern update.",
    "Describe fight-or-flight physiology step by step.",
    "What is learned helplessness? Seligman's dogs experiment.",
    "Explain growth vs fixed mindset (Dweck) with classroom tips.",
    "What is the bystander effect? Darley & Latané findings.",
    "Describe REM sleep and why we dream (leading theories).",
]
OTHER_T = [
    "Explain how the internet routes a packet across continents.",
    "What is climate change attribution science in 5 lines?",
    "Describe how vaccines are tested (phases 1-{n}).",
    "Explain blockchain consensus (proof-of-work) simply.",
    "What is GDP vs happiness (Easterlin paradox)?",
    "How does a microwave oven heat food? Physics in brief.",
    "Explain the greenhouse effect gas by gas.",
    "What is open source licensing (MIT vs GPL)?",
    "Describe how LLMs predict the next token (high level).",
    "What is inflation hedging for ordinary savers? {n} options.",
    "Explain air-traffic control basics.",
    "How do solar panels convert light? Efficiency limits.",
    "What is the Electoral College? Arguments for/against.",
    "Explain antibiotic stewardship in hospitals.",
    "What is urban heat island effect? Mitigations.",
]
POOLS = {"biology": BIOLOGY_T, "business": BUSINESS_T, "chemistry": CHEMISTRY_T,
         "computer_science": CS_T, "economics": ECONOMICS_T, "engineering": ENGINEERING_T,
         "health": HEALTH_T, "history": HISTORY_T, "law": LAW_T, "math": MATH_T,
         "philosophy": PHILOSOPHY_T, "physics": PHYSICS_T, "psychology": PSYCHOLOGY_T,
         "other": OTHER_T}


def build_prompts():
    out, seen = [], set()
    for dom, count, temp in DOMAINS:
        pool = POOLS[dom]
        i = guard = 0
        made = 0
        while made < count and guard < count * 80:
            guard += 1
            t = pool[i % len(pool)]
            i += 1
            p = fill(t) if "{" in t else t
            if p in seen:
                continue
            seen.add(p)
            out.append(((dom, temp), p))
    return out


def call_muse(prompt: str, temp: float) -> str:
    import json as _j, os as _o, urllib.request as _u
    d = _j.load(open(_o.path.expanduser("~/.config/muse/auth.json")))
    key = d["providers"]["meta"]["api_key"]
    body = _j.dumps({"model": MODEL,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": prompt}],
                       "temperature": temp, "max_tokens": 2000}).encode()
    req = _u.Request(API, data=body,
                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with _u.urlopen(req, timeout=300) as r:
        dd = _j.load(r)
    return (dd["choices"][0]["message"]["content"] or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(OUT_DEFAULT))
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--only", default=None)
    r.add_argument("--caps", default=None)
    r.add_argument("--delay", type=float, default=0.0, help="seconds between calls (rate-limit safety)")
    args = ap.parse_args()

    if args.cmd == "plan":
        print(f"  14 subjects x 3000 = {sum(c for _, c, _ in DOMAINS)} prompts")
        return 0

    out_path = Path(args.out)
    seen_path = out_path.parent / ".seen.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set(seen_path.read_text().split("\n")) if seen_path.exists() else set()
    prompts = [(meta, p) for meta, p in build_prompts()]
    if args.only:
        only = {d.strip() for d in args.only.split(",")}
        prompts = [(meta, p) for meta, p in prompts if meta[0] in only]
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
        print(f"  queued {len(prompts)}", flush=True)
    lock = threading.Lock()
    made = [0]

    def work(item):
        (dom, temp), prompt = item
        h = hashlib.md5(prompt.encode()).hexdigest()
        with lock:
            if h in seen:
                return None
        if args.delay > 0:
            time.sleep(args.delay)
        for attempt in range(3):
            try:
                ans = call_muse(prompt, temp)
                break
            except Exception as e:
                print(f"  ERR {dom}: {str(e)[:100]}", flush=True)
                time.sleep(10)
        else:
            return None
        if not ans or len(ans) < 40:
            return None
        return {"instruction": prompt, "input": "", "output": ans[:6000],
                "source": f"muse09:{dom}"}

    with out_path.open("a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec in ex.map(work, prompts):
                if not rec:
                    continue
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    made[0] += 1
                    seen.add(hashlib.md5(rec["instruction"].encode()).hexdigest())
                    seen_path.write_text("\n".join(seen))
                if made[0] % 50 == 0:
                    print(f"  [{made[0]}]", flush=True)
    print(f"DONE made={made[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
