"""Orchestrate fine-tuning inside Docker (Unsloth / Axolotl)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from .analysis import TrainEstimate, save_analysis
from .env_setup import (
    PROJECT_ROOT,
    build_or_pull_image,
    docker_run_base_args,
    recommend_cuda_tag,
    GpuInfo,
)
from .interactive import PipelineConfig
from .safety import SafetyLimits, TrainingWatchdog, preflight_cost_check

console = Console()


def write_run_config(cfg: PipelineConfig, estimate: TrainEstimate, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": cfg.model_id,
        "model_params_b": cfg.model_params_b,
        "dataset_path": cfg.dataset_path,
        "dataset_format": cfg.dataset_format,
        "output_dir": str(cfg.output_dir),
        "framework": cfg.framework,
        "method": cfg.method,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "max_seq_length": cfg.max_seq_length,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "max_steps": cfg.max_steps,
        "load_in_4bit": cfg.load_in_4bit,
        "seed": cfg.seed,
        "gguf_quant": cfg.gguf_quant,
        "ollama_name": cfg.ollama_name,
        "identity_name": (cfg.extra or {}).get("identity_name") or cfg.ollama_name,
        "teach_identity": (cfg.extra or {}).get("teach_identity", True),
        "identity_repeat": (cfg.extra or {}).get("identity_repeat", 3),
        "system_prompt": (cfg.extra or {}).get("system_prompt"),
    }
    path = run_dir / "train_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_analysis(estimate, run_dir / "estimate.json")
    return path


def _resolve_dataset_mount(cfg: PipelineConfig, work_root: Path) -> tuple[str, list[str]]:
    """
    Returns (container_dataset_path, extra_docker_-v_args).
    Copies HF ids as-is; binds local paths under /data.
    """
    p = Path(cfg.dataset_path)
    if p.exists():
        host = p.resolve()
        if host.is_file():
            cont = f"/data/{host.name}"
            return cont, ["-v", f"{host}:/data/{host.name}:ro"]
        cont = "/data/dataset"
        return cont, ["-v", f"{host}:/data/dataset:ro"]
    # Hugging Face id
    return cfg.dataset_path, []


def run_training(
    cfg: PipelineConfig,
    estimate: TrainEstimate,
    gpus: list[GpuInfo],
    *,
    skip_if_over_limit: bool = True,
    dry_run: bool = False,
) -> Path:
    """
    Launch training container. Returns path to adapter/model output directory.
    """
    run_dir = cfg.output_dir / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = run_dir / "adapter"
    merged_dir = run_dir / "merged"
    adapter_dir.mkdir(exist_ok=True)
    merged_dir.mkdir(exist_ok=True)

    write_run_config(cfg, estimate, run_dir)

    limits = SafetyLimits.from_config(
        cfg.max_train_hours,
        cfg.max_cost_usd,
        cfg.gpu_hourly_usd,
        num_gpus=max(len(gpus), 1),
    )
    if not preflight_cost_check(
        estimate.est_train_hours,
        estimate.est_cost_usd,
        limits,
        abort_if_over=skip_if_over_limit,
    ):
        raise RuntimeError("Training aborted by preflight safety limits")

    if dry_run:
        console.print("[yellow]Dry-run: trénink se nespouští.[/]")
        return run_dir

    if not gpus:
        raise RuntimeError("Nelze spustit trénink bez NVIDIA GPU")

    cuda_tag = recommend_cuda_tag(gpus)
    image = build_or_pull_image(framework=cfg.framework, cuda_tag=cuda_tag)

    # Materialize config for container (dataset path rewritten)
    cont_dataset, extra_vols = _resolve_dataset_mount(cfg, run_dir)
    cont_cfg = json.loads((run_dir / "train_config.json").read_text())
    cont_cfg["dataset_path"] = cont_dataset
    cont_cfg["output_dir"] = "/workspace/adapter"
    cont_cfg["merged_dir"] = "/workspace/merged"
    cont_cfg_path = run_dir / "train_config.container.json"
    cont_cfg_path.write_text(json.dumps(cont_cfg, indent=2), encoding="utf-8")

    # Host scripts mounted
    train_script = PROJECT_ROOT / "scripts" / "train_inside_container.py"
    if not train_script.exists():
        raise FileNotFoundError(train_script)

    extra_env = {
        "NVIDIA_VISIBLE_DEVICES": "all",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(key):
            extra_env[key] = os.environ[key]
            break

    args = docker_run_base_args(
        image,
        run_dir,
        gpus="all",
        extra_env=extra_env,
    )
    # Insert volume mounts before image name
    # docker_run_base_args ends with image; rebuild carefully
    image_idx = args.index(image)
    prefix, suffix = args[:image_idx], args[image_idx:]
    # mount scripts + config
    prefix.extend(
        [
            "-v",
            f"{train_script}:/opt/pipeline/train_inside_container.py:ro",
            "-v",
            f"{cont_cfg_path}:/workspace/train_config.json:ro",
        ]
    )
    prefix.extend(extra_vols)
    cmd = prefix + suffix + ["python", "/opt/pipeline/train_inside_container.py", "/workspace/train_config.json"]

    console.print(f"[bold green]Spouštím trénink v Dockeru…[/]")
    console.print(f"[dim]{' '.join(cmd[:12])} …[/]")

    log_path = run_dir / "train.log"
    watchdog: Optional[TrainingWatchdog] = None

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # own process group for watchdog kill
        )
        watchdog = TrainingWatchdog(limits)
        watchdog.start(proc.pid)

        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                # full line-by-line to host terminal + train.log + (via LogTee) web
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C — ukončuji trénink…[/]")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                proc.terminate()
            proc.wait(timeout=60)
            raise
        finally:
            if watchdog:
                watchdog.stop()

        rc = proc.wait()
        if watchdog and watchdog.triggered_reason:
            raise RuntimeError(f"Training stopped by safety watchdog: {watchdog.triggered_reason}")
        if rc != 0:
            raise RuntimeError(f"Training container exited with code {rc}. See {log_path}")

    # Expect adapter or merged weights
    if any(adapter_dir.iterdir()) or any(merged_dir.iterdir()):
        console.print(f"[green]Trénink dokončen.[/] Výstup: {run_dir}")
    else:
        console.print(f"[yellow]Trénink skončil, ale výstupní složky vypadají prázdné: {run_dir}[/]")

    # Marker for downstream steps
    (run_dir / "TRAINING_OK").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return run_dir
