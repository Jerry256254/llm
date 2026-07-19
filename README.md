# LLM Fine-Tune Pipeline (Google Cloud + NVIDIA GPU → Ollama)

Univerzální **Docker-first** Python pipeline pro fine-tuning LLM modelů na NVIDIA GPU (typicky Google Cloud VM), s automatickým exportem do **GGUF** a importem do **Ollama**.

## Jediný příkaz

```bash
python run.py
```

To je vše. Skript:

1. doinstaluje host závislosti (pokud chybí),
2. na pozadí připraví Docker / NVIDIA prostředí,
3. spustí **moderní web UI** na `0.0.0.0:8080` (veřejná IP na GCE),
4. vypíše **access token** do terminálu.

Otevřete v prohlížeči `http://<VAŠE_PUBLIC_IP>:8080`, zadejte token, nastavte model/dataset a spusťte pipeline.

### Web UI (výchozí)

| Akce | Popis |
|------|--------|
| Analyzovat | Odhad VRAM, času, nákladů |
| Spustit pipeline | Setup → train → GGUF → Ollama |
| Live log | Streaming logů + progress |
| Zrušit | Požadavek na zastavení jobu |

```bash
python run.py                     # web na :8080 + auto token
python run.py --port 3000
python run.py --token tajneheslo
python run.py --no-token          # bez auth (jen privátní síť!)
python run.py --skip-setup        # bez auto instalace balíčků
```

**Google Cloud firewall** (jednou):

```bash
gcloud compute firewall-rules create llm-ui \
  --allow=tcp:8080 \
  --target-tags=llm-train \
  --source-ranges=0.0.0.0/0
```

(případně omezte `source-ranges` na vaši IP).

### CLI (volitelné)

```bash
python run.py --cli
python run.py --cli -y --model unsloth/llama-3.2-1b-instruct \
  --dataset ./data/sample_alpaca.jsonl --dry-run
# nebo přímo:
python train_pipeline.py -y --dataset ./data/sample_alpaca.jsonl
```

## Co pipeline dělá

1. **Automatizace prostředí** — detekce Debian/Fedora, Docker + NVIDIA Container Toolkit, image s CUDA/Unsloth/llama.cpp.
2. **Web UI nebo CLI** — model, formát dat, dataset, hyperparametry, limity.
3. **Analýza** — parametry (LoRA/QLoRA), VRAM, čas, USD.
4. **Trénink v Dockeru** — QLoRA/LoRA/full na GPU.
5. **GGUF + Ollama** — merge, llama.cpp, Modelfile, `ollama create`.
6. **Safety** — watchdog max. čas / náklady.

## Požadavky (Google Cloud)

| Položka | Doporučení |
|--------|------------|
| VM | Ubuntu 22.04 LTS nebo Debian 12 / Fedora (GPU image) |
| GPU | T4 16GB (malé modely), L4 24GB, A100 40/80GB |
| Disk | ≥ 100 GB SSD (modely + Docker layer cache) |
| Driver | NVIDIA driver + `nvidia-smi` funkční na hostu |
| Síť | HF + Docker Hub; TCP 8080 pro web UI |

Vytvoření GPU VM (příklad):

```bash
gcloud compute instances create llm-train \
  --zone=europe-west4-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --tags=llm-train
```

## Formáty dat

Podporováno: **Alpaca** (JSON/JSONL), **ShareGPT**, **OpenAI chat messages**, plain **text**, nebo **HF dataset ID**.

Ukázka Alpaca JSONL: `data/sample_alpaca.jsonl`.

## Struktura projektu

```
llm/
├── run.py                     # ★ jediný vstup (web UI výchozí)
├── train_pipeline.py          # CLI pipeline
├── requirements.txt
├── web/                       # control panel (HTML/CSS/JS)
├── configs/default_train.yaml
├── data/sample_alpaca.jsonl
├── docker/
│   ├── Dockerfile.unsloth
│   └── Dockerfile.axolotl
├── scripts/train_inside_container.py
└── lib/
    ├── job_manager.py         # job + logy pro web/CLI
    ├── web_app.py             # FastAPI
    ├── env_setup.py
    ├── interactive.py
    ├── analysis.py
    ├── safety.py
    ├── training.py
    ├── convert_gguf.py
    └── ollama_export.py
```

## Výstupy běhu

```
outputs/run_<timestamp>/
├── train_config.json
├── estimate.json
├── train.log
├── adapter/          # LoRA váhy
├── merged/           # sloučený HF model
├── gguf/*.gguf
├── Modelfile
└── TRAINING_OK
```

Spuštění modelu:

```bash
ollama run demo-1b-ft
```

Obnovení jen konverze z hotového runu:

```bash
python train_pipeline.py --run-dir outputs/run_XXXX --ollama-name my-model
```

## Safety limity

- `--max-hours` — hard stop watchdogu (SIGTERM → SIGKILL).
- `--max-cost` + `--gpu-hourly-usd` — odhad nákladů = čas × cena × počet GPU.
- Před startem **preflight**: pokud odhad překročí limity, trénink se nespustí (override: `--allow-over-limit`).

## Poznámky k produkci

- Odhad VRAM/času je **heuristický** — ověřte na malém běhu (`--epochs 0.1` / `max_steps`).
- Privátní HF modely: `export HF_TOKEN=...` (proměnná se propsala do kontejneru přes mount cache; pro token přidejte `-e HF_TOKEN` v `lib/training.py` dle potřeby).
- První `docker build` trvá dlouho (PyTorch, Unsloth, llama.cpp).
- Na GCE vypněte VM po tréninku, ať neplatíte GPU idle.

## Licence

MIT — použijte na vlastní riziko; sledujte licence base modelů a datasetů.
