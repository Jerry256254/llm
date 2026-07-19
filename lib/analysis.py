"""VRAM, parameter, and training-time estimation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .interactive import PipelineConfig
from .env_setup import GpuInfo

console = Console()

# Bytes per parameter for common dtypes
DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "nf4": 0.5,  # 4-bit approx
    "fp4": 0.5,
}


@dataclass
class TrainEstimate:
    model_params_total: int
    trainable_params: int
    trainable_pct: float
    base_weights_gib: float
    optimizer_gib: float
    activations_gib: float
    overhead_gib: float
    total_vram_gib: float
    recommended_vram_gib: float
    fits_gpus: bool
    num_samples: int
    steps_per_epoch: int
    total_steps: int
    est_seconds_per_step: float
    est_train_seconds: float
    est_train_hours: float
    est_cost_usd: float
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_model_params(params_b: float) -> int:
    return int(params_b * 1e9)


def estimate_lora_trainable(
    params_total: int,
    lora_r: int,
    *,
    # Rough: LoRA on q,k,v,o,gate,up,down ≈ 7 matrices per layer
    # Fraction of params that are attention/MLP linear ≈ 0.55–0.7 of model
    target_modules_fraction: float = 0.6,
    # For rank r, trainable ≈ 2 * r * (d_in + d_out) per matrix;
    # empirical: trainable ≈ params * (2 * r / hidden) * n_modules_factor
    # Simpler heuristic used in practice:
    hidden_proxy: Optional[int] = None,
) -> int:
    """
    Heuristic LoRA trainable parameter count.
    For 7B r=16, typically ~20–40M trainable params (~0.3–0.6%).
    Rule of thumb: trainable ≈ 2 * r * sqrt(params) * k  with k~4–8
    Better: fraction ≈ (2 * r) / d_model for each adapted weight.
    """
    # Approximate d_model from param count (Transformer rule of thumb)
    # params ≈ 12 * n_layers * d^2 → d ≈ sqrt(params / (12 * n_layer))
    # Use: trainable_ratio ≈ 2 * r / 4096 for 7B-class (d≈4096)
    if params_total >= 60e9:
        d_model = 8192
    elif params_total >= 30e9:
        d_model = 6656
    elif params_total >= 10e9:
        d_model = 5120
    elif params_total >= 6e9:
        d_model = 4096
    elif params_total >= 2e9:
        d_model = 2560
    elif params_total >= 1e9:
        d_model = 2048
    else:
        d_model = 1024

    if hidden_proxy:
        d_model = hidden_proxy

    # LoRA on ~target_modules_fraction of linear weights, each with 2*r*d effective
    # trainable ≈ n_params_adapted * (2 * r / d_model)
    trainable = int(params_total * target_modules_fraction * (2.0 * lora_r / d_model))
    return max(trainable, lora_r * 1024)


def count_dataset_samples(dataset_path: str, dataset_format: str) -> int:
    """Best-effort sample count for local files / HF ids."""
    path = Path(dataset_path)
    if not path.exists():
        # HF hub — unknown without download; use placeholder
        return 1000

    if path.is_file():
        return _count_file_samples(path)

    total = 0
    for p in path.rglob("*"):
        if p.suffix.lower() in {".json", ".jsonl", ".jsonlines", ".txt", ".csv"}:
            total += _count_file_samples(p)
    return max(total, 1)


def _count_file_samples(path: Path) -> int:
    try:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".jsonlines"}:
            n = 0
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        n += 1
            return n
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                for key in ("data", "examples", "train"):
                    if key in data and isinstance(data[key], list):
                        return len(data[key])
                return 1
            return 1
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            # treat non-empty paragraphs as samples if many, else 1 document
            paras = [p for p in text.split("\n\n") if p.strip()]
            return max(len(paras), 1)
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return max(sum(1 for _ in f) - 1, 1)
    except Exception:
        return 100
    return 100


def estimate_vram_gib(
    params_total: int,
    trainable: int,
    *,
    method: str,
    max_seq_length: int,
    batch_size: int,
    grad_accum: int,  # does not multiply activation peak much if micro-batch is batch_size
) -> tuple[float, float, float, float, float]:
    """
    Return (base_weights, optimizer, activations, overhead, total) in GiB.
    """
    if method == "qlora":
        # 4-bit base + small fp16 LoRA adapters
        base = params_total * DTYPE_BYTES["nf4"] / (1024**3)
        # adapters stored fp16
        adapters = trainable * DTYPE_BYTES["fp16"] / (1024**3)
        base_weights = base + adapters
        # AdamW states for trainable only (m,v) in fp32 ≈ 8 bytes/param + fp32 master ≈ 4
        optimizer = trainable * 12 / (1024**3)
    elif method == "lora":
        base_weights = params_total * DTYPE_BYTES["bf16"] / (1024**3)
        adapters = trainable * DTYPE_BYTES["fp16"] / (1024**3)
        base_weights += adapters
        optimizer = trainable * 12 / (1024**3)
    else:  # full
        base_weights = params_total * DTYPE_BYTES["bf16"] / (1024**3)
        optimizer = params_total * 12 / (1024**3)  # adamw fp32 states heuristic

    # Activation memory (very rough): scales with batch * seq * hidden * layers
    # Use params as proxy: activations ≈ k * batch * seq * sqrt(params)
    hidden = max(1024, int(math.sqrt(params_total / 12)))  # rough
    n_layers = max(8, int(params_total / (12 * hidden * hidden)))
    # bytes: batch * seq * hidden * n_layers * 2 (fp16) * factor for residuals/attention
    act_factor = 8.0
    activations = (
        batch_size * max_seq_length * hidden * n_layers * 2 * act_factor / (1024**3)
    )
    # Cap insane estimates for tiny models
    activations = max(0.3, min(activations, base_weights * 4 + 8))

    overhead = 1.5 + 0.15 * base_weights  # CUDA context, fragmentation, etc.
    total = base_weights + optimizer + activations + overhead
    return base_weights, optimizer, activations, overhead, total


def estimate_step_time_seconds(
    params_b: float,
    method: str,
    max_seq_length: int,
    batch_size: int,
    gpu_memory_mib: int,
) -> float:
    """
    Empirical-ish step time. Calibrated roughly for consumer/datacenter GPUs.
    """
    # Baseline: 7B QLoRA, seq 2048, bs 2 on ~24GB ≈ 1.5–3s/step
    base = 2.0
    size_scale = params_b / 7.0
    seq_scale = max_seq_length / 2048.0
    batch_scale = batch_size / 2.0
    method_scale = {"qlora": 1.0, "lora": 1.25, "full": 2.5}.get(method, 1.0)

    # Faster GPUs with more memory often have higher FLOPS
    if gpu_memory_mib >= 80000:  # A100 80GB
        gpu_scale = 0.35
    elif gpu_memory_mib >= 40000:  # A100 40 / A10 / L40
        gpu_scale = 0.5
    elif gpu_memory_mib >= 24000:  # 4090 / L4 24
        gpu_scale = 0.7
    elif gpu_memory_mib >= 16000:  # T4 16 / 4080
        gpu_scale = 1.2
    else:
        gpu_scale = 1.8

    return max(0.2, base * size_scale * seq_scale * batch_scale * method_scale * gpu_scale)


def analyze(
    cfg: PipelineConfig,
    gpus: list[GpuInfo],
) -> TrainEstimate:
    params_total = estimate_model_params(cfg.model_params_b)
    notes: list[str] = []

    if cfg.method == "full":
        trainable = params_total
    else:
        trainable = estimate_lora_trainable(params_total, cfg.lora_r)

    base_w, opt, act, oh, total_vram = estimate_vram_gib(
        params_total,
        trainable,
        method=cfg.method,
        max_seq_length=cfg.max_seq_length,
        batch_size=cfg.batch_size,
        grad_accum=cfg.grad_accum,
    )
    recommended = total_vram * 1.15  # safety margin

    total_gpu_mib = sum(g.memory_mib for g in gpus) if gpus else 0
    total_gpu_gib = total_gpu_mib / 1024.0
    fits = total_gpu_gib >= recommended if gpus else False
    if not gpus:
        notes.append("Žádná GPU detekována — odhad VRAM je teoretický.")
    elif not fits:
        notes.append(
            f"Odhad {recommended:.1f} GiB přesahuje dostupných {total_gpu_gib:.1f} GiB. "
            "Snižte batch/seq length, použijte QLoRA, nebo větší GPU."
        )
    if cfg.method == "full" and cfg.model_params_b >= 7:
        notes.append("Full fine-tune u 7B+ obvykle vyžaduje multi-GPU nebo velmi velkou VRAM.")
    if cfg.load_in_4bit and cfg.method == "lora":
        notes.append("Pro 4-bit základ je vhodnější method=qlora.")

    n_samples = count_dataset_samples(cfg.dataset_path, cfg.dataset_format)
    effective_batch = cfg.batch_size * cfg.grad_accum * max(len(gpus), 1)
    steps_per_epoch = max(1, math.ceil(n_samples / effective_batch))
    if cfg.max_steps and cfg.max_steps > 0:
        total_steps = cfg.max_steps
    else:
        total_steps = max(1, int(steps_per_epoch * cfg.epochs))

    primary_mem = gpus[0].memory_mib if gpus else 16000
    sec_per_step = estimate_step_time_seconds(
        cfg.model_params_b,
        cfg.method,
        cfg.max_seq_length,
        cfg.batch_size,
        primary_mem,
    )
    # multi-GPU data parallel: steps don't reduce much wall time if same steps/gpu
    # but throughput increases → fewer steps for same epochs if global batch grows
    # (already accounted via effective_batch)
    train_sec = total_steps * sec_per_step
    train_hours = train_sec / 3600.0
    cost = train_hours * cfg.gpu_hourly_usd * max(len(gpus), 1)

    est = TrainEstimate(
        model_params_total=params_total,
        trainable_params=trainable,
        trainable_pct=100.0 * trainable / params_total if params_total else 0.0,
        base_weights_gib=base_w,
        optimizer_gib=opt,
        activations_gib=act,
        overhead_gib=oh,
        total_vram_gib=total_vram,
        recommended_vram_gib=recommended,
        fits_gpus=fits,
        num_samples=n_samples,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        est_seconds_per_step=sec_per_step,
        est_train_seconds=train_sec,
        est_train_hours=train_hours,
        est_cost_usd=cost,
        notes=notes,
    )
    return est


def print_analysis(cfg: PipelineConfig, est: TrainEstimate, gpus: list[GpuInfo]) -> None:
    table = Table(title="Analýza tréninku", show_header=True, header_style="bold magenta")
    table.add_column("Položka")
    table.add_column("Hodnota", justify="right")

    table.add_row("Model", cfg.model_id)
    table.add_row("Parametry celkem", f"{est.model_params_total / 1e9:.2f} B")
    table.add_row(
        "Trénovatelné (LoRA/full)",
        f"{est.trainable_params / 1e6:.2f} M ({est.trainable_pct:.3f}%)",
    )
    table.add_row("Metoda", cfg.method.upper())
    table.add_row("LoRA r / alpha", f"{cfg.lora_r} / {cfg.lora_alpha}")
    table.add_row("Max seq length", str(cfg.max_seq_length))
    table.add_row("Micro-batch × accum", f"{cfg.batch_size} × {cfg.grad_accum}")
    table.add_row("Epochs", str(cfg.epochs))
    table.add_row("Dataset samples (odhad)", str(est.num_samples))
    table.add_row("Steps / epoch", str(est.steps_per_epoch))
    table.add_row("Celkem steps", str(est.total_steps))
    table.add_row("—", "—")
    table.add_row("VRAM váhy", f"{est.base_weights_gib:.2f} GiB")
    table.add_row("VRAM optimizer", f"{est.optimizer_gib:.2f} GiB")
    table.add_row("VRAM aktivace", f"{est.activations_gib:.2f} GiB")
    table.add_row("VRAM overhead", f"{est.overhead_gib:.2f} GiB")
    table.add_row("VRAM celkem (odhad)", f"{est.total_vram_gib:.2f} GiB")
    table.add_row("VRAM doporučeno (+15%)", f"{est.recommended_vram_gib:.2f} GiB")
    if gpus:
        table.add_row(
            "Dostupná VRAM",
            f"{sum(g.memory_mib for g in gpus) / 1024:.2f} GiB ({len(gpus)} GPU)",
        )
        table.add_row("Vejde se?", "[green]ANO[/]" if est.fits_gpus else "[red]NE / těsně[/]")
    table.add_row("—", "—")
    table.add_row("Čas / step", f"{est.est_seconds_per_step:.2f} s")
    table.add_row("Odhadovaný čas", f"{est.est_train_hours:.2f} h ({est.est_train_seconds/60:.1f} min)")
    table.add_row("Odhadované náklady", f"${est.est_cost_usd:.2f}")
    table.add_row("Limit času", f"{cfg.max_train_hours:.2f} h")
    table.add_row("Limit nákladů", f"${cfg.max_cost_usd:.2f}")

    console.print(table)
    if est.notes:
        console.print(Panel("\n".join(f"• {n}" for n in est.notes), title="Poznámky", border_style="yellow"))


def save_analysis(est: TrainEstimate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(est.to_dict(), indent=2), encoding="utf-8")
