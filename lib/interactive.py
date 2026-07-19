"""Interactive CLI prompts for the fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

console = Console()

# Common HF model IDs with approximate parameter counts (billions)
KNOWN_MODELS: dict[str, float] = {
    # base (less chat-aligned / better for "uncensored" / full training)
    "unsloth/llama-3.2-1b": 1.0,
    "unsloth/llama-3.2-3b": 3.0,
    "unsloth/meta-llama-3.1-8b": 8.0,
    "unsloth/qwen2.5-7b": 7.0,
    "unsloth/qwen2.5-3b": 3.0,
    "unsloth/qwen2.5-1.5b": 1.5,
    "unsloth/mistral-7b-v0.3": 7.0,
    # Newer open bases (HF) — full-weight training on your data ("od nuly" v praxi)
    "google/gemma-2-2b": 2.0,
    "google/gemma-2-9b": 9.0,
    "google/gemma-3-1b-pt": 1.0,
    "google/gemma-3-4b-pt": 4.0,
    "google/gemma-3-12b-pt": 12.0,
    "Qwen/Qwen2.5-1.5B": 1.5,
    "Qwen/Qwen2.5-3B": 3.0,
    "Qwen/Qwen2.5-7B": 7.0,
    "meta-llama/Llama-3.2-1B": 1.0,
    "meta-llama/Llama-3.2-3B": 3.0,
    # instruct / chat
    "unsloth/llama-3.2-1b-instruct": 1.0,
    "unsloth/llama-3.2-3b-instruct": 3.0,
    "unsloth/meta-llama-3.1-8b-instruct": 8.0,
    "unsloth/mistral-7b-instruct-v0.3": 7.0,
    "unsloth/qwen2.5-7b-instruct": 7.0,
    "unsloth/qwen2.5-3b-instruct": 3.0,
    "unsloth/qwen2.5-1.5b-instruct": 1.5,
    "unsloth/gemma-2-2b-it": 2.0,
    "unsloth/gemma-2-9b-it": 9.0,
    "meta-llama/Llama-3.2-1B": 1.0,
    "meta-llama/Llama-3.2-1B-Instruct": 1.0,
    "meta-llama/Llama-3.2-3B-Instruct": 3.0,
    "meta-llama/Meta-Llama-3.1-8B": 8.0,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 8.0,
    "mistralai/Mistral-7B-v0.3": 7.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
    "Qwen/Qwen2.5-7B": 7.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "google/gemma-2-2b-it": 2.0,
    "microsoft/Phi-3.5-mini-instruct": 3.8,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.1,
}

DATASET_FORMAT_HELP = """
JAK MAJÍ VYPADAT UČEBNÍ DATA (jednoduše)

────────────────────────────────────────
1) OTÁZKA + ODPOVĚĎ  (nejjednodušší)
   Soubor .jsonl — každý řádek je jeden příklad:

{"instruction": "Vysvětli fotosyntézu", "input": "", "output": "Fotosyntéza je proces..."}
{"instruction": "Přelož do angličtiny", "input": "Ahoj světe", "output": "Hello world"}

   • instruction = co má model udělat / otázka
   • input = doplňující text (může být prázdný "")
   • output = správná odpověď, kterou se má naučit

────────────────────────────────────────
2) PROSTÝ TEXT / KNIHA  (učení „od nuly“ na textech)

{"text": "Dlouhý odstavec nebo kapitola, kterou má model číst a učit se z ní..."}

   Nebo soubor .txt s odstavci oddělenými prázdným řádkem.

────────────────────────────────────────
3) CHAT (user / assistant)

{"messages": [
  {"role": "user", "content": "Ahoj!"},
  {"role": "assistant", "content": "Ahoj, jak pomohu?"}
]}

────────────────────────────────────────
4) DIALOG ShareGPT

{"conversations": [
  {"from": "human", "value": "Ahoj!"},
  {"from": "gpt", "value": "Ahoj!"}
]}

────────────────────────────────────────
TIPY
• Čím víc kvalitních příkladů, tím líp (stovky až desetitisíce).
• Pište v jednom jazyce a stejném stylu, jaký chcete od modelu.
• Soubor musí být UTF-8.
• Ukázka v projektu: ./data/sample_alpaca.jsonl
"""


@dataclass
class PipelineConfig:
    model_id: str
    model_params_b: float
    dataset_path: str
    dataset_format: str  # alpaca | sharegpt | chat | text | hf
    output_dir: Path
    framework: str = "unsloth"  # unsloth | axolotl
    method: str = "qlora"  # lora | qlora | full
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    batch_size: int = 2
    grad_accum: int = 4
    epochs: float = 1.0
    learning_rate: float = 2e-4
    max_steps: int = -1  # -1 = by epochs
    load_in_4bit: bool = True
    gguf_quant: str = "q4_k_m"
    ollama_name: str = ""
    max_train_hours: float = 4.0
    max_cost_usd: float = 20.0
    gpu_hourly_usd: float = 0.35  # default ~T4 on-demand-ish estimate
    seed: int = 42
    extra: dict = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        safe = self.model_id.replace("/", "_").replace(" ", "-")
        return f"{safe}-ft"


def show_dataset_format_help() -> None:
    console.print(Panel(Markdown(DATASET_FORMAT_HELP), title="Formát datasetu", border_style="cyan"))


def _parse_params_from_name(model_id: str) -> Optional[float]:
    import re

    # e.g. 7b, 1.5B, 70B, 3b-instruct
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)[bB][-_]", model_id)
    if m:
        return float(m.group(1))
    return None


def resolve_model_params(model_id: str) -> float:
    if model_id in KNOWN_MODELS:
        return KNOWN_MODELS[model_id]
    # case-insensitive lookup
    lower = {k.lower(): v for k, v in KNOWN_MODELS.items()}
    if model_id.lower() in lower:
        return lower[model_id.lower()]
    parsed = _parse_params_from_name(model_id)
    if parsed is not None:
        return parsed
    return FloatPrompt.ask(
        "Nepodařilo se odhadnout počet parametrů. Zadejte velikost modelu v miliardách (např. 7)",
        default=7.0,
    )


def detect_dataset_format(path: Path) -> str:
    if not path.exists():
        # treat as HF hub id
        return "hf"
    if path.is_dir():
        samples = list(path.glob("**/*.{json,jsonl,parquet,csv,arrow}"))[:5]
        if not samples:
            return "text"
        path = samples[0]

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return "text"
    if suffix in {".json", ".jsonl"}:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:8000]
            if '"conversations"' in text or '"from":' in text:
                return "sharegpt"
            if '"messages"' in text and '"role"' in text:
                return "chat"
            if '"instruction"' in text and '"output"' in text:
                return "alpaca"
            if '"text"' in text:
                return "text"
        except OSError:
            pass
        return "alpaca"
    return "alpaca"


def list_known_models() -> None:
    table = Table(title="Příklady modelů (Hugging Face / Unsloth)")
    table.add_column("Model ID", style="cyan")
    table.add_column("Parametry (B)", justify="right")
    for mid, pb in sorted(KNOWN_MODELS.items(), key=lambda x: (x[1], x[0]))[:16]:
        table.add_row(mid, f"{pb:g}")
    console.print(table)


def collect_config_interactive(
    *,
    defaults: Optional[PipelineConfig] = None,
    non_interactive: bool = False,
    overrides: Optional[dict] = None,
) -> PipelineConfig:
    """Prompt user for all pipeline settings (or apply CLI overrides)."""
    overrides = overrides or {}

    if not non_interactive:
        console.print(
            Panel.fit(
                "[bold]LLM Fine-Tune Pipeline[/]\n"
                "Google Cloud · NVIDIA GPU · Docker · Unsloth/Axolotl → GGUF → Ollama",
                border_style="green",
            )
        )
        list_known_models()
        show_dataset_format_help()

    model_id = overrides.get("model_id") or (
        "unsloth/llama-3.2-1b-instruct"
        if non_interactive
        else Prompt.ask("Název / HF ID modelu", default="unsloth/llama-3.2-1b-instruct")
    )
    model_params_b = overrides.get("model_params_b") or resolve_model_params(model_id)

    dataset_path = overrides.get("dataset_path") or (
        "./data/train.jsonl"
        if non_interactive
        else Prompt.ask(
            "Cesta k datasetu (soubor/složka) nebo HF dataset ID",
            default="./data/train.jsonl",
        )
    )

    ds_path = Path(dataset_path)
    if ds_path.exists() or "/" not in dataset_path.replace("\\", "/"):
        # local path or simple name — try detect; HF ids usually org/name
        pass
    dataset_format = overrides.get("dataset_format") or detect_dataset_format(ds_path)
    if not non_interactive and "dataset_format" not in overrides:
        dataset_format = Prompt.ask(
            "Formát datasetu",
            choices=["alpaca", "sharegpt", "chat", "text", "hf"],
            default=dataset_format if dataset_format in {"alpaca", "sharegpt", "chat", "text", "hf"} else "alpaca",
        )

    framework = overrides.get("framework") or (
        "unsloth"
        if non_interactive
        else Prompt.ask("Framework", choices=["unsloth", "axolotl"], default="unsloth")
    )
    method = overrides.get("method") or (
        "qlora"
        if non_interactive
        else Prompt.ask("Metoda tréninku", choices=["qlora", "lora", "full"], default="qlora")
    )

    output_dir = Path(
        overrides.get("output_dir")
        or (
            "./outputs"
            if non_interactive
            else Prompt.ask("Výstupní adresář", default="./outputs")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if non_interactive:
        lora_r = int(overrides.get("lora_r", 16))
        max_seq = int(overrides.get("max_seq_length", 2048))
        batch = int(overrides.get("batch_size", 2))
        epochs = float(overrides.get("epochs", 1.0))
        gguf_quant = overrides.get("gguf_quant", "q4_k_m")
        max_hours = float(overrides.get("max_train_hours", 4.0))
        max_cost = float(overrides.get("max_cost_usd", 20.0))
        gpu_rate = float(overrides.get("gpu_hourly_usd", 0.35))
        ollama_name = overrides.get("ollama_name") or f"{model_id.split('/')[-1].lower()}-ft"
    else:
        advanced = Confirm.ask("Upravit pokročilé hyperparametry?", default=False)
        if advanced:
            lora_r = IntPrompt.ask("LoRA rank (r)", default=16)
            max_seq = IntPrompt.ask("Max sequence length", default=2048)
            batch = IntPrompt.ask("Per-device batch size", default=2)
            epochs = FloatPrompt.ask("Epochs", default=1.0)
            gguf_quant = Prompt.ask(
                "GGUF quantizace",
                choices=["q4_k_m", "q5_k_m", "q8_0", "f16", "q3_k_m"],
                default="q4_k_m",
            )
            max_hours = FloatPrompt.ask("Max. doba tréninku (hodiny)", default=4.0)
            max_cost = FloatPrompt.ask("Max. odhadované náklady (USD)", default=20.0)
            gpu_rate = FloatPrompt.ask("Hodinová cena GPU (USD)", default=0.35)
        else:
            lora_r, max_seq, batch, epochs = 16, 2048, 2, 1.0
            gguf_quant, max_hours, max_cost, gpu_rate = "q4_k_m", 4.0, 20.0, 0.35
        ollama_name = Prompt.ask(
            "Název modelu v Ollama",
            default=f"{model_id.split('/')[-1].lower()}-ft",
        )

    load_in_4bit = method == "qlora"
    if method == "full":
        load_in_4bit = False

    cfg = PipelineConfig(
        model_id=model_id,
        model_params_b=float(model_params_b),
        dataset_path=str(dataset_path),
        dataset_format=dataset_format,
        output_dir=output_dir.resolve(),
        framework=framework,
        method=method,
        lora_r=lora_r,
        lora_alpha=int(overrides.get("lora_alpha", lora_r * 2)),
        max_seq_length=max_seq,
        batch_size=batch,
        grad_accum=int(overrides.get("grad_accum", 4)),
        epochs=epochs,
        learning_rate=float(overrides.get("learning_rate", 2e-4)),
        load_in_4bit=load_in_4bit,
        gguf_quant=gguf_quant,
        ollama_name=ollama_name,
        max_train_hours=max_hours,
        max_cost_usd=max_cost,
        gpu_hourly_usd=gpu_rate,
        seed=int(overrides.get("seed", 42)),
    )
    return cfg
