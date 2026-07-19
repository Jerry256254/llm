"""Shared job state, log buffer, and pipeline execution for CLI + Web UI."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .analysis import analyze
from .convert_gguf import convert_to_gguf
from .env_setup import Distro, EnvReport, GpuInfo, detect_gpus, prepare_environment
from .interactive import (
    KNOWN_MODELS,
    PipelineConfig,
    detect_dataset_format,
)
from .ollama_export import export_and_import
from .training import run_training

ROOT = Path(__file__).resolve().parent.parent


class JobPhase(str, Enum):
    IDLE = "idle"
    SETUP = "setup"
    ANALYZE = "analyze"
    TRAIN = "train"
    GGUF = "gguf"
    OLLAMA = "ollama"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class JobStatus:
    phase: str = JobPhase.IDLE.value
    message: str = "Připraveno"
    progress: float = 0.0  # 0–100
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    run_dir: Optional[str] = None
    estimate: Optional[dict] = None
    error: Optional[str] = None
    ollama_name: Optional[str] = None
    gguf_path: Optional[str] = None
    config: Optional[dict] = None
    env: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class LogTee:
    """Tee writes to original stream + callback (for Web UI log streaming)."""

    def __init__(self, original, callback: Callable[[str], None]):
        self._original = original
        self._callback = callback
        self._buf = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            self._original.write(data)
        except Exception:
            pass
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._callback(line)
        return len(data)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        if self._buf:
            self._callback(self._buf)
            self._buf = ""

    def isatty(self) -> bool:
        return False


class JobManager:
    """Singleton-style manager: one active pipeline job at a time."""

    def __init__(self, max_log_lines: int = 4000):
        self._lock = threading.RLock()
        self._status = JobStatus()
        self._logs: deque[str] = deque(maxlen=max_log_lines)
        self._log_seq = 0
        self._subscribers: list[threading.Event] = []
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._env: Optional[EnvReport] = None
        self._prepared = False

    # ── logging ──────────────────────────────────────────────

    def log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        text = line if line.startswith("[") else f"[{ts}] {line}"
        with self._lock:
            self._logs.append(text)
            self._log_seq += 1
            subs = list(self._subscribers)
        for ev in subs:
            ev.set()

    def get_logs(self, after: int = 0) -> tuple[int, list[str]]:
        with self._lock:
            # after is absolute seq; we only keep last N lines
            total = self._log_seq
            lines = list(self._logs)
            # approximate: if after < total - len(lines), send all
            kept = len(lines)
            start_seq = total - kept
            if after <= start_seq:
                return total, lines
            offset = after - start_seq
            return total, lines[offset:]

    def subscribe(self) -> threading.Event:
        ev = threading.Event()
        with self._lock:
            self._subscribers.append(ev)
        return ev

    def unsubscribe(self, ev: threading.Event) -> None:
        with self._lock:
            if ev in self._subscribers:
                self._subscribers.remove(ev)

    # ── status ───────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            d = self._status.to_dict()
            d["running"] = self._thread is not None and self._thread.is_alive()
            d["log_seq"] = self._log_seq
            return d

    def _set(
        self,
        *,
        phase: Optional[str] = None,
        message: Optional[str] = None,
        progress: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if phase is not None:
                self._status.phase = phase
            if message is not None:
                self._status.message = message
            if progress is not None:
                self._status.progress = progress
            for k, v in kwargs.items():
                if hasattr(self._status, k):
                    setattr(self._status, k, v)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── environment ──────────────────────────────────────────

    def prepare_env(self, *, install_packages: bool = True, framework: str = "unsloth") -> dict:
        self._set(phase=JobPhase.SETUP.value, message="Připravuji prostředí…", progress=5)
        self.log("Příprava host prostředí (Docker, NVIDIA runtime)…")
        try:
            self._env = prepare_environment(
                install_packages=install_packages,
                framework=framework,
            )
            self._prepared = True
            env_info = {
                "distro": self._env.distro_pretty,
                "docker_ok": self._env.docker_ok,
                "nvidia_runtime_ok": self._env.nvidia_runtime_ok,
                "cuda_host": self._env.cuda_host,
                "gpus": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "memory_mib": g.memory_mib,
                        "driver": g.driver_version,
                        "cuda": g.cuda_version,
                    }
                    for g in self._env.gpus
                ],
            }
            self._set(env=env_info, message="Prostředí připraveno", progress=15)
            self.log(f"OS: {self._env.distro_pretty}")
            if self._env.gpus:
                for g in self._env.gpus:
                    self.log(f"GPU {g.index}: {g.name} ({g.memory_mib} MiB)")
            else:
                self.log("Varování: žádná NVIDIA GPU detekována.")
            return env_info
        except Exception as e:
            self.log(f"CHYBA setup: {e}")
            self._set(phase=JobPhase.ERROR.value, message=str(e), error=str(e))
            raise

    def get_env_snapshot(self) -> dict:
        gpus = detect_gpus()
        return {
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_mib": g.memory_mib,
                    "driver": g.driver_version,
                    "cuda": g.cuda_version,
                }
                for g in gpus
            ],
            "prepared": self._prepared,
        }

    # ── config helpers ───────────────────────────────────────

    def build_config(self, data: dict) -> PipelineConfig:
        model_id = (data.get("model_id") or "unsloth/llama-3.2-1b").strip()
        default_data = str(ROOT / "data" / "test_multilang_code" / "train.jsonl")
        dataset_path = (data.get("dataset_path") or default_data).strip()
        # map friendly train_mode → method (UI usually sends both)
        train_mode = (data.get("train_mode") or "from_scratch").strip()
        method = (data.get("method") or "").strip().lower()
        if method not in ("qlora", "lora", "full"):
            if train_mode in ("finetune", "fine_tune", "teach_answers", "light_adapt"):
                method = "qlora"
            else:
                method = "full"
        out = Path(data.get("output_dir") or (ROOT / "outputs")).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        # params — never prompt (web/job path must be non-interactive)
        from .interactive import _parse_params_from_name

        if data.get("model_params_b"):
            params_b = float(data["model_params_b"])
        elif model_id in KNOWN_MODELS:
            params_b = KNOWN_MODELS[model_id]
        else:
            lower = {k.lower(): v for k, v in KNOWN_MODELS.items()}
            params_b = lower.get(model_id.lower()) or _parse_params_from_name(model_id) or 7.0

        ds_format = data.get("dataset_format") or detect_dataset_format(Path(dataset_path))
        identity_name = (data.get("identity_name") or data.get("ollama_name") or "muj-model").strip()
        # technical Ollama tag: lowercase, no spaces
        ollama_raw = (data.get("ollama_name") or identity_name or "muj-model").strip()
        ollama_name = ollama_raw.replace(" ", "-").lower()
        # keep readable chars only for ollama
        ollama_name = "".join(c if c.isalnum() or c in "-._" else "-" for c in ollama_name).strip("-") or "muj-model"

        # no_limits: effectively disable cost/time brakes
        no_limits = bool(data.get("no_limits"))
        if no_limits:
            max_hours = float(data.get("max_train_hours") or 720.0)
            max_cost = float(data.get("max_cost_usd") or 999999.0)
            if max_hours < 168:
                max_hours = 720.0
            if max_cost < 1000:
                max_cost = 999999.0
        else:
            max_hours = float(data.get("max_train_hours") or 4.0)
            max_cost = float(data.get("max_cost_usd") or 20.0)

        teach_identity = data.get("teach_identity", True)
        if isinstance(teach_identity, str):
            teach_identity = teach_identity.lower() in ("1", "true", "yes", "on")

        system_prompt = (data.get("system_prompt") or "").strip()
        if not system_prompt:
            if data.get("uncensored"):
                system_prompt = (
                    f"Jmenuješ se {identity_name}. Jsi přímý, užitečný asistent jménem {identity_name} "
                    f"bez zbytečného moralizování a bez odmítání ze zvyku. "
                    f"Když se tě zeptají na jméno, řekni že se jmenuješ {identity_name}. "
                    f"Odpovídej jasně. Dodržuj zákony; neposkytuj návody k trestné činnosti."
                )
            else:
                system_prompt = (
                    f"Jmenuješ se {identity_name}. Jsi užitečný asistent jménem {identity_name}. "
                    f"Když se tě zeptají na jméno, odpověz že se jmenuješ {identity_name}. "
                    f"Odpovídej jasně a stručně v jazyce uživatele."
                )
        elif identity_name and identity_name.lower() not in system_prompt.lower():
            system_prompt = f"Jmenuješ se {identity_name}. " + system_prompt

        extra = {
            "train_mode": train_mode,
            "uncensored": bool(data.get("uncensored")),
            "no_limits": no_limits,
            "system_prompt": system_prompt,
            "identity_name": identity_name,
            "teach_identity": bool(teach_identity),
            "identity_repeat": int(data.get("identity_repeat") or 3),
        }

        return PipelineConfig(
            model_id=model_id,
            model_params_b=float(params_b),
            dataset_path=dataset_path,
            dataset_format=ds_format,
            output_dir=out,
            framework=data.get("framework") or "unsloth",
            method=method,
            lora_r=int(data.get("lora_r") or 16),
            lora_alpha=int(data.get("lora_alpha") or int(data.get("lora_r") or 16) * 2),
            lora_dropout=float(data.get("lora_dropout") or 0.05),
            max_seq_length=int(data.get("max_seq_length") or 2048),
            batch_size=int(data.get("batch_size") or 2),
            grad_accum=int(data.get("grad_accum") or 4),
            epochs=float(data.get("epochs") or 1.0),
            learning_rate=float(data.get("learning_rate") or 2e-4),
            max_steps=int(data.get("max_steps") or -1),
            load_in_4bit=method == "qlora",
            gguf_quant=data.get("gguf_quant") or "q4_k_m",
            ollama_name=ollama_name,
            max_train_hours=max_hours,
            max_cost_usd=max_cost,
            gpu_hourly_usd=float(data.get("gpu_hourly_usd") or 0.35),
            seed=int(data.get("seed") or 42),
            extra=extra,
        )

    def analyze_only(self, data: dict) -> dict:
        cfg = self.build_config(data)
        gpus = self._env.gpus if self._env else detect_gpus()
        est = analyze(cfg, gpus)
        result = {
            "config": {
                "model_id": cfg.model_id,
                "model_params_b": cfg.model_params_b,
                "dataset_path": cfg.dataset_path,
                "dataset_format": cfg.dataset_format,
                "method": cfg.method,
                "lora_r": cfg.lora_r,
                "max_seq_length": cfg.max_seq_length,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "gguf_quant": cfg.gguf_quant,
                "ollama_name": cfg.ollama_name,
            },
            "estimate": est.to_dict(),
        }
        self._set(estimate=est.to_dict(), config=result["config"])
        return result

    # ── job lifecycle ────────────────────────────────────────

    def start_job(self, data: dict) -> dict:
        if self.is_running():
            raise RuntimeError("Job už běží")
        self._cancel.clear()
        with self._lock:
            self._status = JobStatus(
                phase=JobPhase.SETUP.value,
                message="Startuji…",
                progress=0,
                started_at=time.time(),
                config=data,
            )
            self._logs.clear()
            self._log_seq = 0

        self._thread = threading.Thread(
            target=self._run_job,
            args=(data,),
            name="pipeline-job",
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def request_cancel(self) -> None:
        self._cancel.set()
        self.log("Požadavek na zrušení…")
        self._set(message="Ruším…")

    def _run_job(self, data: dict) -> None:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = LogTee(old_out, self.log)
        sys.stderr = LogTee(old_err, self.log)
        try:
            self._execute(data)
        except Exception as e:
            self.log(f"FATAL: {e}")
            self.log(traceback.format_exc())
            self._set(
                phase=JobPhase.ERROR.value,
                message=str(e),
                error=str(e),
                finished_at=time.time(),
                progress=100,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise RuntimeError("Zrušeno uživatelem")

    def _execute(self, data: dict) -> None:
        dry_run = bool(data.get("dry_run"))
        skip_gguf = bool(data.get("skip_gguf"))
        skip_ollama = bool(data.get("skip_ollama"))
        skip_train = bool(data.get("skip_train"))
        # no_limits always allows over estimate; allow_over_limit also
        allow_over = bool(data.get("allow_over_limit")) or bool(data.get("no_limits"))
        install = not bool(data.get("skip_setup"))

        # Setup
        self._set(phase=JobPhase.SETUP.value, message="Setup prostředí…", progress=5)
        if not self._prepared or install:
            try:
                self.prepare_env(
                    install_packages=install,
                    framework=data.get("framework") or "unsloth",
                )
            except Exception:
                # continue with partial env for analyze/dry-run
                if not dry_run and not skip_train:
                    raise
                self._env = self._env or EnvReport(
                    distro=Distro.UNKNOWN,
                    distro_pretty="unknown",
                    docker_ok=False,
                    nvidia_runtime_ok=False,
                    gpus=detect_gpus(),
                    cuda_host=None,
                )

        self._check_cancel()
        cfg = self.build_config(data)
        gpus: list[GpuInfo] = self._env.gpus if self._env else detect_gpus()

        # Analyze
        self._set(phase=JobPhase.ANALYZE.value, message="Počítám odhad paměti a času…", progress=20)
        mode = (cfg.extra or {}).get("train_mode") or "?"
        identity = (cfg.extra or {}).get("identity_name") or cfg.ollama_name
        self.log(
            f"Režim: {mode} | jméno AI: „{identity}“ | model: {cfg.model_id} | "
            f"data: {cfg.dataset_path} | metoda: {cfg.method} | "
            f"bez cenzury: {(cfg.extra or {}).get('uncensored')} | "
            f"bez limitů: {(cfg.extra or {}).get('no_limits')} | "
            f"učit jméno: {(cfg.extra or {}).get('teach_identity', True)}"
        )
        est = analyze(cfg, gpus)
        self._set(estimate=est.to_dict(), progress=30)
        self.log(
            f"Odhad: grafika ~{est.recommended_vram_gib:.1f} GB, "
            f"čas ~{est.est_train_hours:.2f} h, ~${est.est_cost_usd:.2f}, "
            f"učí se {est.trainable_params/1e6:.1f}M parametrů"
        )

        if dry_run or skip_train:
            self._set(
                phase=JobPhase.DONE.value,
                message="Odhad hotov (model se neučil)",
                progress=100,
                finished_at=time.time(),
            )
            self.log("Hotovo — jen odhad, bez učení.")
            return

        self._check_cancel()

        # Train
        self._set(phase=JobPhase.TRAIN.value, message="Učení modelu běží (Docker + GPU)…", progress=35)
        self.log("Spouštím učení…")
        run_dir = run_training(
            cfg,
            est,
            gpus,
            skip_if_over_limit=not allow_over,
            dry_run=False,
        )
        self._set(run_dir=str(run_dir), progress=70)
        self.log(f"Trénink hotov: {run_dir}")

        meta = {
            "ollama_name": cfg.ollama_name,
            "gguf_quant": cfg.gguf_quant,
            "model_id": cfg.model_id,
            "train_mode": (cfg.extra or {}).get("train_mode"),
            "uncensored": (cfg.extra or {}).get("uncensored"),
            "system_prompt": (cfg.extra or {}).get("system_prompt"),
            "identity_name": (cfg.extra or {}).get("identity_name"),
            "teach_identity": (cfg.extra or {}).get("teach_identity", True),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self._check_cancel()
        gguf_path = None

        if not skip_gguf:
            self._set(phase=JobPhase.GGUF.value, message="Balím model (GGUF)…", progress=78)
            gguf_path = convert_to_gguf(run_dir, quant=cfg.gguf_quant, gpus=gpus)
            self._set(gguf_path=str(gguf_path), progress=88)
            self.log(f"GGUF: {gguf_path}")
        else:
            found = list((run_dir / "gguf").glob("*.gguf")) if (run_dir / "gguf").exists() else []
            gguf_path = found[0] if found else None

        self._check_cancel()

        if not skip_ollama and gguf_path is not None:
            self._set(phase=JobPhase.OLLAMA.value, message="Instaluji do Ollama…", progress=92)
            system_prompt = (cfg.extra or {}).get("system_prompt")
            export_and_import(
                run_dir,
                Path(gguf_path),
                cfg.ollama_name,
                system_prompt=system_prompt,
            )
            self._set(ollama_name=cfg.ollama_name, progress=98)
            self.log(f"Hotovo. Spusťte: ollama run {cfg.ollama_name}")

        self._set(
            phase=JobPhase.DONE.value,
            message=f"Hotovo — ollama run {cfg.ollama_name}",
            progress=100,
            finished_at=time.time(),
            ollama_name=cfg.ollama_name,
            gguf_path=str(gguf_path) if gguf_path else None,
        )
        self.log("=== UČENÍ DOKONČENO ===")


# Global instance used by web + run.py
manager = JobManager()
