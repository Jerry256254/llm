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
    # Context window (tokens)
    native_context: int = 0          # architecture max (e.g. 262144 for Qwen3.5)
    train_context: int = 0           # max_seq_length used while training
    ollama_num_ctx: int = 0          # recommended Ollama PARAMETER num_ctx
    context_expandable_by_train: bool = False  # training can improve use of long ctx, not invent new architecture max

    def to_dict(self) -> dict:
        return asdict(self)


# Known native context lengths (tokens) when config.json is not on disk yet
KNOWN_NATIVE_CONTEXT: dict[str, int] = {
    "google/gemma-4-E2B": 131072,
    "google/gemma-4-E2B-it": 131072,
    "google/gemma-4-E4B": 131072,
    "google/gemma-4-E4B-it": 131072,
    "Qwen/Qwen3.5-0.8B-Base": 262144,
    "Qwen/Qwen3.5-2B-Base": 262144,
    "Qwen/Qwen3.5-4B-Base": 262144,
    "Qwen/Qwen3.5-9B-Base": 262144,
    "Qwen/Qwen3.5-0.8B": 262144,
    "Qwen/Qwen3.5-2B": 262144,
    "Qwen/Qwen3.5-4B": 262144,
    "Qwen/Qwen3.5-9B": 262144,
    "Qwen/Qwen2.5-1.5B": 32768,
    "Qwen/Qwen2.5-3B": 32768,
    "Qwen/Qwen2.5-7B": 131072,
    "unsloth/llama-3.2-1b": 131072,
    "unsloth/llama-3.2-3b": 131072,
    "meta-llama/Llama-3.2-1B": 131072,
    "meta-llama/Llama-3.2-3B": 131072,
    "google/gemma-2-2b": 8192,
    "google/gemma-2-9b": 8192,
}


def resolve_native_context(model_id: str) -> int:
    """Best-effort native max position embeddings for a HF model id or local path."""
    mid = (model_id or "").strip()
    if mid in KNOWN_NATIVE_CONTEXT:
        return KNOWN_NATIVE_CONTEXT[mid]
    # Local path / downloaded models/
    candidates: list[Path] = []
    p = Path(mid).expanduser()
    if p.is_dir():
        candidates.append(p / "config.json")
    try:
        from .model_source import local_model_dir

        candidates.append(local_model_dir(mid) / "config.json")
    except Exception:
        pass
    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            # Qwen3.5 nests under text_config
            for block in (data, data.get("text_config") or {}):
                if not isinstance(block, dict):
                    continue
                for key in (
                    "max_position_embeddings",
                    "model_max_length",
                    "max_sequence_length",
                    "n_positions",
                ):
                    if key in block and block[key]:
                        return int(block[key])
        except Exception:
            continue
    # Heuristic from name
    low = mid.lower()
    if "qwen3.5" in low or "qwen3_5" in low:
        return 262144
    if "llama-3" in low or "llama3" in low:
        return 131072
    if "qwen2.5" in low:
        return 32768
    return 8192


def recommend_ollama_num_ctx(
    *,
    train_seq: int,
    native_ctx: int,
    explicit: Optional[int] = None,
) -> int:
    """
    Ollama inference context.

    - Training with max_seq_length teaches the model *within* that window well.
    - Architecture may support far more (Qwen3.5 ~256k); runtime num_ctx can be
      higher than train_seq, but quality on ultra-long prompts is better if you
      also train with longer sequences (VRAM allowing).
    - Training does NOT raise architecture max beyond native_ctx.
    """
    if explicit and explicit > 0:
        return min(int(explicit), max(native_ctx, int(explicit)))
    # Default chat: comfortable window without huge KV cache
    # Prefer at least 4k, up to max(train_seq * 2, 8192), capped by native and 32k for small GPUs
    target = max(4096, int(train_seq) * 2, 8192)
    target = min(target, int(native_ctx) if native_ctx > 0 else target)
    # Cap default export at 32k so Ollama stays snappy on L4; user can raise
    target = min(target, 32768)
    # Round to power-of-two-ish common sizes
    for size in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
        if target <= size:
            return min(size, native_ctx) if native_ctx else size
    return min(target, native_ctx) if native_ctx else target


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
    else:  # full — pipeline uses adamw_8bit + frozen vision on L4
        base_weights = params_total * DTYPE_BYTES["bf16"] / (1024**3)
        # 8-bit Adam: ~2 bytes/param states (vs ~12 for fp32 AdamW) — critical for 2B on 24GB
        optimizer = params_total * 2.5 / (1024**3)

    # Activation memory (very rough): scales with batch * seq * hidden * layers
    # Use params as proxy: activations ≈ k * batch * seq * sqrt(params)
    hidden = max(1024, int(math.sqrt(params_total / 12)))  # rough
    n_layers = max(8, int(params_total / (12 * hidden * hidden)))
    # bytes: batch * seq * hidden * n_layers * 2 (fp16) * factor for residuals/attention
    act_factor = 8.0
    activations = (
        batch_size * max_seq_length * hidden * n_layers * 2 * act_factor / (1024**3)
    )
    # Gradient checkpointing + frozen vision lowers activation peak a lot
    if method == "full":
        activations *= 0.55
    activations = max(0.3, min(activations, base_weights * 3 + 6))

    overhead = 1.2 + 0.12 * base_weights  # CUDA context, fragmentation, etc.
    total = base_weights + optimizer + activations + overhead
    return base_weights, optimizer, activations, overhead, total


def estimate_step_time_seconds(
    params_b: float,
    method: str,
    max_seq_length: int,
    batch_size: int,
    gpu_memory_mib: int,
    *,
    packing: bool = True,
    grad_accum: int = 1,
) -> float:
    """
    Empirical optimizer-step time for L4-class GPUs.

    Calibration (measured):
      Qwen3.5-0.8B full, seq=2048, micro-bs=1, ga=8, no packing → ~31 s/step
      (~17.5 h for 2026 steps).

    Optimized path assumptions (packing + seq 1024 + tf32 + frozen vision):
      ~0.45× from shorter seq, ~0.7× from packing efficiency, ~0.9× other opts
      → full 0.8B ≈ 8–10 s/step; full 2B ≈ 18–28 s/step (order of magnitude).
    """
    # Re-base on measured full-FT step at seq 2048, 0.8B
    measured_full_0_8b_seq2048 = 31.0
    size_scale = max(params_b, 0.3) / 0.8
    seq_scale = (max(max_seq_length, 256) / 2048.0) ** 1.35
    # micro-batch >1 increases step time sub-linearly with checkpointing
    batch_scale = (max(batch_size, 1) / 1.0) ** 0.85

    if method == "full":
        method_scale = 1.0
    elif method == "lora":
        method_scale = 0.18
    else:  # qlora
        method_scale = 0.12

    pack_scale = 0.72 if packing and max_seq_length >= 512 else 1.0
    opt_scale = 0.88  # tf32 + frozen vision + 8bit adam + sdpa (expected)

    if gpu_memory_mib >= 80000:
        gpu_scale = 0.45
    elif gpu_memory_mib >= 40000:
        gpu_scale = 0.6
    elif gpu_memory_mib >= 24000:
        gpu_scale = 1.0
    elif gpu_memory_mib >= 16000:
        gpu_scale = 1.35
    else:
        gpu_scale = 1.9

    t = (
        measured_full_0_8b_seq2048
        * size_scale
        * seq_scale
        * batch_scale
        * method_scale
        * pack_scale
        * opt_scale
        * gpu_scale
    )
    # grad_accum is already "inside" one optimizer step wall time in HF logs
    return max(0.4, t)


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

    native_ctx = resolve_native_context(cfg.model_id)
    train_ctx = int(cfg.max_seq_length)
    explicit_ctx = None
    if cfg.extra:
        raw = cfg.extra.get("num_ctx") or cfg.extra.get("ollama_num_ctx")
        if raw is not None and str(raw).strip() != "":
            try:
                explicit_ctx = int(raw)
            except (TypeError, ValueError):
                explicit_ctx = None
    ollama_ctx = recommend_ollama_num_ctx(
        train_seq=train_ctx, native_ctx=native_ctx, explicit=explicit_ctx
    )
    if train_ctx > native_ctx:
        notes.append(
            f"Train max_seq_length ({train_ctx}) > nativní context modelu ({native_ctx}). "
            f"Snižte na ≤ {native_ctx} — trénink nemůže architektonicky „natáhnout“ window výš."
        )
        train_ctx = native_ctx
    elif train_ctx < 2048 and native_ctx >= 8192:
        notes.append(
            f"Trénujete na {train_ctx} tokenech; model umí až ~{native_ctx}. "
            "Delší max_seq_length (2048–8192) zlepší dlouhé chaty, ale žere VRAM a čas."
        )
    if cfg.method == "qlora":
        notes.append(
            "QLoRA/LoRA = pořád VÁŠ model (vaše jméno, vaše data, váš GGUF v Ollama). "
            "Mění se adaptér/váhy na vašich datech; základ dává jen startovní znalosti jako u full FT."
        )

    n_samples = count_dataset_samples(cfg.dataset_path, cfg.dataset_format)
    effective_batch = cfg.batch_size * cfg.grad_accum * max(len(gpus), 1)
    steps_per_epoch = max(1, math.ceil(n_samples / effective_batch))
    if cfg.max_steps and cfg.max_steps > 0:
        total_steps = cfg.max_steps
    else:
        total_steps = max(1, int(steps_per_epoch * cfg.epochs))

    primary_mem = gpus[0].memory_mib if gpus else 16000
    packing = True
    if cfg.extra and cfg.extra.get("packing") is not None:
        packing = bool(cfg.extra.get("packing"))
    sec_per_step = estimate_step_time_seconds(
        cfg.model_params_b,
        cfg.method,
        cfg.max_seq_length,
        cfg.batch_size,
        primary_mem,
        packing=packing,
        grad_accum=cfg.grad_accum,
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
        native_context=native_ctx,
        train_context=train_ctx,
        ollama_num_ctx=ollama_ctx,
        context_expandable_by_train=train_ctx < native_ctx,
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
    table.add_row("Max seq length (train)", str(cfg.max_seq_length))
    table.add_row("Nativní context (model)", str(est.native_context or "?"))
    table.add_row("Ollama num_ctx (doporučeno)", str(est.ollama_num_ctx or "?"))
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
