# KucLab Hertz 0.1 — STEM demo

První model nové řady **KucLab Hertz** (nahrazuje starší řadu „KucLab V").
Zaměření: **fyzika, chemie, biologie, matematika** — česky i anglicky.

---

## Zadání vs. realita hardwaru

Zadání bylo: model 7–10 B, expert na přírodní vědy a matematiku, perfektní
čeština i angličtina, dobrá efficiency; případně natrénovat ~15 B a zkomprimovat.

**Trénink 15 B a následná komprese na tomto stroji nejde.** K dispozici je
jedna **NVIDIA L4 (23 GB VRAM)** a **15 GB systémové RAM**. Skutečná komprese
(distilace) vyžaduje učitele i studenta rezidentní v paměti současně — samotný
15B učitel má v fp16 asi 30 GB. To je úloha pro multi-GPU.

**Zvolené řešení:** QLoRA na 9B base a expertízu nacpat do **dat**.
Kvantizace na `q4_k_m` při exportu je zároveň ta „komprese" — výsledný GGUF
má kolem 5–6 GB.

---

## Base model

**`Qwen/Qwen3.5-9B`** — 9 B, tedy uvnitř požadovaného rozsahu 7–10 B.

Proč právě tenhle:

| Kritérium | Proč Qwen3.5-9B |
|---|---|
| STEM | Nejsilnější matematika a přírodní vědy mezi 9B modely |
| Efficiency | Hybridní **linear attention** (3 linear : 1 full) — výrazně rychlejší a paměťově úspornější než dense 9B |
| Kontext | 262 144 tokenů nativně |
| Dostupnost | Ungated, bez HF tokenu |
| Pipeline | Repo už má fix pro Qwen3.5 MTP vrstvu (`lib/convert_gguf.py`) a `_lora_target_modules()` umí `in_proj_qkv`/`out_proj` z linear-attention bloků |

Slabší stránka je čeština (Gemma je na evropské jazyky lepší) — to dohání
korpus, viz níže.

---

## Naměřený baseline — proč korpus vypadá, jak vypadá

Před stavbou dat byl base model změřen (q4 přes Ollamu, stejná kvantizace jako
finální model):

| Test | Base Qwen3.5-9B |
|---|---|
| MMLU-Pro STEM (239 držených otázek) | **88,7 %** |
| Česká terminologie CS→EN (103 držených) | 79,6 % |
| **Česká terminologie EN→CS (103 držených)** | **61,2 %** |

Ukázky selhání v češtině:

| Správně | Model řekl |
|---|---|
| výkon | „Moc" |
| stejnosměrný proud | **„střídavý proud"** |
| endotermická reakce | **„exotermická reakce"** |
| oxidační číslo | „Oxidacionní číslo" |
| poločas rozpadu | „polčas rozpadu" |
| rezistivita | „odporivost" |

**Závěr, který přepsal původní plán:** model vědu umí a data, kterými jsem ho
chtěl učit (camel-ai generované GPT-4 v roce 2023, GSM8K = úlohy pro základku),
jsou slabší než on sám. Trénink na nich by ho stáhl dolů. Anglická STEM páteř
(~50 000 řádků) byla proto z korpusu **odstraněna**.

Co model neumí, je česká odborná terminologie. To je jediné místo, kde jde
reálně přidat hodnotu — a shodou okolností to nikdo jiný neřeší, protože
český STEM korpus veřejně neexistuje.

**Cíl běhu:** udržet MMLU-Pro na 88,7 % a vytáhnout EN→CS z 61,2 % nahoru.

Přísnost hodnocení: „časová dilatace" místo „dilatace času" počítám jako chybu,
takže absolutní čísla jsou podhodnocená. Pro měření *rozdílu* před/po to nevadí —
stejná přísnost platí pro oba.

## Data — `data/kuclab_hertz_0.1/`

Staví se skriptem `scripts/build_hertz01_corpus.py`.

**21 686 vzorků, ~76 % česky.**

| Zdroj | Počet |
|---|---|
| `wiki_stem_cs` — česká Wikipedie, STEM filtr | 15 000 |
| `kuclab_v1_reuse` — drží programování a obecné schopnosti | 3 924 |
| `cs_terminology` — 309 trénovacích termínů × 5 formulací | 1 545 |
| `wiki_stem_en` — malá anglická kotva | 1 000 |
| `cs_elements` — 41 prvků | 123 |
| `cs_constants` — 14 fyzikálních konstant | 28 |
| identita + chování + ručně psaný seed | 66 |

Anglické STEM zdroje (camel-ai, MetaMathQA, GSM8K, SciQ, ScienceQA) jsou
v builderu ponechané, ale **vypnuté** (`--max-camel 0` atd.) — dají se zapnout
pro experiment, výchozí stav je off z důvodů popsaných výše.

### Česká část — hlavní přidaná hodnota
Veřejný český STEM instruct dataset **neexistuje**. Proto se staví zde:

1. **Ručně psaný odborný seed** — fyzika, chemie, biologie, matematika.
   Newtonovy zákony, entropie a 2. termodynamický zákon, princip neurčitosti,
   Maxwellovy rovnice, fotoelektrický jev, supravodivost, periodická tabulka,
   chemická rovnováha a Le Chatelier, pH, izomerie, redoxní reakce, stavba
   buňky, mitóza/meióza, fotosyntéza, centrální dogma, imunita, Mendelovy
   zákony, evoluce, kvadratická rovnice, derivace, integrál, důkaz
   iracionality √2, komplexní čísla, limity, Bayesova věta, matice.

2. **Terminologické tabulky** (`scripts/cz_terms.py`) — přesné konstrukcí:
   - **412 odborných termínů** (107 fyzika, 99 chemie, 99 biologie, 107 matematika)
     s českou definicí a anglickým ekvivalentem
   - 41 prvků (značka, Z, český i anglický název, zařazení)
   - 14 fyzikálních konstant s hodnotami a jednotkami

   **Každý 4. termín je držený stranou** (309 trénink / 103 eval), takže
   `bench_czech_terms.py` měří zobecnění na neviděnou terminologii, ne
   memorování seznamu. Ověřeno: z 103 držených definic se v tréninku objevila
   1 (náhodou z Wikipedie).

3. **Česká Wikipedie** filtrovaná na STEM (~13 % článků projde filtrem)

### Zachování schopností
Filtrovaný výběr z korpusu `kuclab_v1` — drží programování a obecné znalosti.
Identita staré řady je odfiltrovaná, aby se model neučil, že je „KucLab V1.0".

### Identita
Záměrně **malá** (~24 řádků). V0.5 se rozpadla právě přemírou identity dat.

---

## Eval — poctivě oddělený

**MMLU-Pro se netrénuje.** Je odložený jako benchmark:

- `eval_prompts.jsonl` — smoke testy (identita, přepínání jazyka, čestnost, základní STEM)
- `eval_mmlu_pro_stem.jsonl` — držený MMLU-Pro (physics / chemistry / biology / math)

```bash
# po tréninku — obojí proti stejným drženým sadám jako baseline
.venv/bin/python scripts/bench_mmlu_pro.py    --model kuclab-hertz-0.1
.venv/bin/python scripts/bench_czech_terms.py --model kuclab-hertz-0.1

# tabulka před/po
.venv/bin/python scripts/bench_mmlu_pro.py    --compare qwen3.5:9b kuclab-hertz-0.1
.venv/bin/python scripts/bench_czech_terms.py --compare qwen3.5:9b kuclab-hertz-0.1

# smoke testy (identita, přepínání jazyka, čestnost)
.venv/bin/python scripts/eval_hertz01.py --model kuclab-hertz-0.1
```

Kdyby byl MMLU-Pro v trénovacích datech, výsledné skóre by neznamenalo nic.

**Kritéria úspěchu:**
- MMLU-Pro STEM ≥ 88,7 % (nesmí klesnout)
- Česká terminologie EN→CS výrazně nad 61,2 %

Pokud MMLU-Pro klesne, je běh neúspěšný bez ohledu na češtinu — sniž `lora_r`
nebo `learning_rate` a opakuj.

### Poznámka k rychlosti benchmarku
Ollama běží s `-np 1`, takže paralelní requesty se serializují; plný MMLU-Pro
sweep trvá ~45 min. `--num-predict` nesmí být nízký: při 640 tokenech se
odpovědi usekávaly před řádkem „Answer: (X)" a skript pak bral náhodné písmeno
z textu úvahy — tichá záměna „došly tokeny" za „špatná odpověď".

---

## Trénink

Config: `configs/kuclab_hertz_0.1.yaml`

| Parametr | Hodnota |
|---|---|
| metoda | QLoRA, 4-bit |
| lora_r / alpha | **32 / 64** |
| max_seq_length | 2048 |
| batch_size × grad_accum | 1 × 16 |
| epochs | 1.0 (1 356 kroků) |
| learning_rate | 8e-5 |
| kvantizace exportu | q4_k_m |
| Ollama tag | `kuclab-hertz-0.1` |
| VRAM | 15,8 GiB z 23 GB |
| odhad | **~18,7 h** |

`lora_r: 32` je zvolený záměrně nízko. Hlavní riziko tohoto běhu není, že se
model češtinu nenaučí — je to, že si rozbije těch 88,7 %. Rank je přímá páka na
to, kolik toho adaptér přepíše, a terminologie není high-rank dovednost.

`max_seq_length: 2048` je **minimum** — odvození ve fyzice a matematice jsou
dlouhá a kratší kontext by je usekával uprostřed výpočtu.

### Odhad doby
Předchozí běh (Gemma 2 9B, seq 512) jel 7,65 s/krok. Při seq 2048 je krok
podstatně dražší; Qwen3.5 to částečně vrací linear attention. Reálný odhad
si nech spočítat tlačítkem **Odhad** ve web UI — počítá z aktuálního počtu
vzorků a naměřené propustnosti.

---

## Spuštění

Web UI běží v tmuxu `train`, aby přežilo odpojení:

```bash
tmux attach -t train      # připojit se
# Ctrl+B pak D            # odpojit, běží dál
```

Ve web UI je **vše předvyplněné** — stačí zmáčknout **Začít učení**.
