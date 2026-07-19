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
from .env_setup import (
    Distro,
    EnvReport,
    GpuInfo,
    detect_gpus,
    env_ready_for_train,
    prepare_environment,
)
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
    progress: float = 0.0  # 0–100 overall pipeline
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    run_dir: Optional[str] = None
    estimate: Optional[dict] = None
    error: Optional[str] = None
    ollama_name: Optional[str] = None
    gguf_path: Optional[str] = None
    config: Optional[dict] = None
    env: Optional[dict] = None
    # Live training sub-progress (parsed from trainer logs)
    train_progress: Optional[dict] = None
    progress_detail: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Overall pipeline progress bands (proportional UX)
# setup 0–8 | analyze 8–15 | docker 15–28 | train 28–88 | gguf 88–95 | ollama 95–100
_BAND = {
    "setup": (0.0, 8.0),
    "analyze": (8.0, 15.0),
    "docker": (15.0, 28.0),
    "train": (28.0, 88.0),
    "gguf": (88.0, 95.0),
    "ollama": (95.0, 100.0),
    "done": (100.0, 100.0),
    "error": (100.0, 100.0),
}


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
        # strip ANSI so web deník is readable
        import re
        plain = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", data)
        self._buf += plain
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

    def __init__(self, max_log_lines: int = 100_000):
        self._lock = threading.RLock()
        self._status = JobStatus()
        # Large ring buffer — full training + docker logs
        self._logs: deque[str] = deque(maxlen=max_log_lines)
        self._log_seq = 0
        self._subscribers: list[threading.Event] = []
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._env: Optional[EnvReport] = None
        self._prepared = False
        self._log_file = ROOT / "outputs" / "live_terminal.log"
        self._est_total_steps: Optional[int] = None
        self._est_epochs: float = 1.0
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # ── logging ──────────────────────────────────────────────

    @staticmethod
    def _strip_ansi(s: str) -> str:
        import re
        s = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)
        s = re.sub(r"\x1b\].*?\x07", "", s)
        s = re.sub(r"\r", "", s)
        return s

    def _map_band(self, band: str, frac: float, message: Optional[str] = None) -> None:
        """Map 0..1 fraction inside a pipeline band to overall 0..100 progress."""
        lo, hi = _BAND.get(band, (0.0, 100.0))
        frac = max(0.0, min(1.0, float(frac)))
        overall = lo + (hi - lo) * frac
        # Never go backwards during a running job (except error/done)
        with self._lock:
            if band not in ("error", "done") and overall < self._status.progress:
                overall = self._status.progress
            self._status.progress = overall
            if message:
                self._status.message = message
            if band in ("setup", "analyze", "train", "gguf", "ollama"):
                # keep phase as-is if already train/gguf; docker maps to train phase for UI
                if band == "docker":
                    self._status.phase = JobPhase.TRAIN.value
                elif band == "train":
                    self._status.phase = JobPhase.TRAIN.value
                elif band == "setup":
                    self._status.phase = JobPhase.SETUP.value
                elif band == "analyze":
                    self._status.phase = JobPhase.ANALYZE.value
                elif band == "gguf":
                    self._status.phase = JobPhase.GGUF.value
                elif band == "ollama":
                    self._status.phase = JobPhase.OLLAMA.value

    def _parse_train_progress(self, line: str) -> None:
        """Extract step/epoch/% from HF Trainer / tqdm / Unsloth logs → overall progress."""
        import re

        step = total = None
        epoch = None
        pct = None
        detail = None

        # Docker build: [ 45%] Building
        m = re.search(r"\[\s*(\d+)%\s*\]", line)
        if m and ("Building" in line or "ggml" in line.lower() or "%" in line):
            try:
                if any(x in line for x in ("Building", "ggml", "cmake", "Linking", "%]")):
                    bp = float(m.group(1)) / 100.0
                    # only treat as docker if not clearly a trainer line
                    if "loss" not in line.lower() and "epoch" not in line.lower():
                        self._map_band("docker", min(0.99, bp), f"Docker/build ~{m.group(1)}%")
                        return
            except Exception:
                pass

        if "Building image" in line or "Čekám na zámek Docker" in line:
            self._map_band("docker", 0.05, "Stavím Docker image…")
            return
        if "Image hotový" in line or "Docker image ready" in line:
            self._map_band("docker", 1.0, "Docker image ready")
            return
        if "Spouštím trénink" in line or "Starting training" in line:
            self._map_band("train", 0.0, "Start tréninku…")
            return
        if "Loading model" in line or "from_pretrained" in line:
            self._map_band("train", 0.02, "Načítám model…")
            return
        if "ollama create" in line.lower() or "Import do Ollama" in line:
            self._map_band("ollama", 0.4, "Import do Ollama…")
            return

        # tqdm: 45%|
        m = re.search(r"(\d+(?:\.\d+)?)%\s*\|", line)
        if m:
            pct = float(m.group(1))

        m = re.search(r"\b(\d+)\s*/\s*(\d+)\s*\[", line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))

        m = re.search(r"['\"]epoch['\"]\s*:\s*([\d.]+)", line)
        if m:
            epoch = float(m.group(1))

        m = re.search(r"\bstep\s*[:=]?\s*(\d+)\s*/\s*(\d+)", line, re.I)
        if m:
            step, total = int(m.group(1)), int(m.group(2))

        # HF: {'loss': 0.9, 'learning_rate': ..., 'epoch': 0.5}
        if "loss" in line.lower() and epoch is None:
            m = re.search(r"epoch['\"]?\s*[:=]\s*([\d.]+)", line, re.I)
            if m:
                epoch = float(m.group(1))

        frac = None
        if step is not None and total and total > 0:
            frac = step / total
            detail = f"step {step}/{total}"
            if pct is None:
                pct = frac * 100.0
        elif epoch is not None and self._est_epochs > 0:
            frac = min(1.0, epoch / max(self._est_epochs, 1e-6))
            detail = f"epoch {epoch:.3f}/{self._est_epochs:g}"
            if pct is None:
                pct = frac * 100.0
        elif pct is not None:
            frac = pct / 100.0
            detail = f"{pct:.1f}%"
        elif step is not None and self._est_total_steps:
            frac = min(1.0, step / max(self._est_total_steps, 1))
            detail = f"step {step}/{self._est_total_steps}"
            pct = frac * 100.0

        if frac is not None:
            self._map_band(
                "train",
                frac,
                f"Učení {pct:.1f}%" if pct is not None else "Učení…",
            )
            with self._lock:
                self._status.train_progress = {
                    "percent": float(pct) if pct is not None else frac * 100.0,
                    "step": step,
                    "total_steps": total or self._est_total_steps,
                    "epoch": epoch,
                }
                self._status.progress_detail = detail
                if detail:
                    self._status.message = f"Učení — {detail}"

    def log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        # docker / multiprogress may send multi-line chunks
        for raw in self._strip_ansi(line).splitlines():
            clean = raw.rstrip()
            if not clean:
                continue
            text = clean if clean.startswith("[") else f"[{ts}] {clean}"
            with self._lock:
                self._logs.append(text)
                self._log_seq += 1
                subs = list(self._subscribers)
            try:
                with self._log_file.open("a", encoding="utf-8") as f:
                    f.write(text + "\n")
            except OSError:
                pass
            # Update live progress from this log line
            try:
                self._parse_train_progress(clean)
            except Exception:
                pass
            for ev in subs:
                ev.set()

    def get_logs(self, after: int = 0) -> tuple[int, list[str]]:
        with self._lock:
            total = self._log_seq
            lines = list(self._logs)
            kept = len(lines)
            start_seq = total - kept
            if after <= start_seq:
                return total, lines
            offset = after - start_seq
            return total, lines[offset:]

    def get_logs_text(self) -> str:
        """Full log bundle for copy/download (memory + file fallback)."""
        with self._lock:
            mem = "\n".join(self._logs)
            seq = self._log_seq
        header = (
            f"=== LLM pipeline terminal log ===\n"
            f"exported_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"lines_in_memory={seq}\n"
            f"{'=' * 40}\n"
        )
        # Prefer full file if it has more history
        try:
            if self._log_file.exists():
                file_text = self._log_file.read_text(encoding="utf-8", errors="replace")
                if file_text.count("\n") >= mem.count("\n"):
                    return header + file_text
        except OSError:
            pass
        return header + mem + ("\n" if mem and not mem.endswith("\n") else "")

    def clear_logs_ui_only(self) -> None:
        """Does not wipe file history — only used if we add UI clear of view."""
        pass

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

    def prepare_env(
        self,
        *,
        install_packages: bool = True,
        framework: str = "unsloth",
        build_image: bool = True,
    ) -> dict:
        self._set(phase=JobPhase.SETUP.value, message="Připravuji prostředí…", progress=5)
        self.log("Příprava host prostředí (Docker, NVIDIA runtime)…")
        try:
            self._env = prepare_environment(
                install_packages=install_packages,
                framework=framework,
                build_image=build_image,
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
            # Keep previous session in file; reset ring for UI stream from this job
            self._logs.clear()
            self._log_seq = 0
        try:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(
                    f"\n\n########## NEW JOB {time.strftime('%Y-%m-%d %H:%M:%S')} ##########\n"
                    f"model={data.get('model_id')} mode={data.get('train_mode')} "
                    f"method={data.get('method')}\n"
                )
        except OSError:
            pass
        self.log("=== NOVÝ JOB — logy jdou na web terminál i do outputs/live_terminal.log ===")

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
        user_skip = bool(data.get("skip_setup"))
        ready, ready_why = env_ready_for_train()
        # Skip apt if user asked OR host already has docker+GPU (prevents hang on sudo/apt)
        install = (not user_skip) and (not ready)
        if user_skip:
            self.log("Setup balíčků přeskočen (skip_setup).")
        elif ready:
            self.log(f"Setup balíčků přeskočen — {ready_why}")
        else:
            self.log(f"Host není kompletní ({ready_why}) — zkusím setup (max. jednotky minut)…")

        # Setup (host packages only — Docker image se staví až při tréninku, 1× se zámkem)
        self._map_band("setup", 0.2, "Kontrola prostředí…")
        self.log("Kontrola Docker + GPU…")
        try:
            self.prepare_env(
                install_packages=install,
                framework=data.get("framework") or "unsloth",
                build_image=False,
            )
            self._map_band("setup", 1.0, "Prostředí OK")
            self.log("Kontrola prostředí hotová.")
        except Exception as e:
            self.log(f"Setup varování: {e}")
            if not dry_run and not skip_train and not ready:
                # still try — maybe docker works without fresh apt
                self.log("Pokračuji i po chybě setupu (možná už máte docker+GPU)…")
            self._env = self._env or EnvReport(
                distro=Distro.UNKNOWN,
                distro_pretty="unknown",
                docker_ok=ready,
                nvidia_runtime_ok=ready,
                gpus=detect_gpus(),
                cuda_host=None,
            )

        self._check_cancel()
        cfg = self.build_config(data)
        gpus: list[GpuInfo] = (self._env.gpus if self._env and self._env.gpus else detect_gpus())
        if not gpus:
            self.log("VAROVÁNÍ: žádná GPU v detekci — trénink může selhat.")
        else:
            self.log(f"GPU: {gpus[0].name} ({gpus[0].memory_mib} MiB)")

        # Analyze FIRST (live estimate in UI before long docker build)
        self._map_band("analyze", 0.2, "Počítám odhad paměti a času…")
        mode = (cfg.extra or {}).get("train_mode") or "?"
        identity = (cfg.extra or {}).get("identity_name") or cfg.ollama_name
        self.log("=" * 48)
        self.log(f"MODEL (přesně to, co se učí): {cfg.model_id}")
        self.log(f"Režim: {mode} | metoda: {cfg.method} | jméno AI: „{identity}“")
        self.log(f"Data: {cfg.dataset_path} | formát: {cfg.dataset_format}")
        self.log(
            f"bez cenzury: {(cfg.extra or {}).get('uncensored')} | "
            f"bez limitů: {(cfg.extra or {}).get('no_limits')} | "
            f"učit jméno: {(cfg.extra or {}).get('teach_identity', True)}"
        )
        if "qwen" in cfg.model_id.lower() and mode in ("from_scratch", "from_scratch_full"):
            self.log("POZNÁMKA: zvolili jste Qwen model záměrně (ID výše).")
        est = analyze(cfg, gpus)
        self._est_total_steps = est.total_steps
        self._est_epochs = float(cfg.epochs) if cfg.epochs else 1.0
        self._set(estimate=est.to_dict(), config={
            "model_id": cfg.model_id,
            "method": cfg.method,
            "train_mode": mode,
            "identity_name": identity,
            "dataset_path": cfg.dataset_path,
            "ollama_name": cfg.ollama_name,
        })
        self._map_band("analyze", 1.0, "Odhad hotov")
        self.log(
            f"ODHAD (živě): VRAM ~{est.recommended_vram_gib:.1f} GB | "
            f"čas ~{est.est_train_hours:.2f} h | ~${est.est_cost_usd:.2f} | "
            f"samples {est.num_samples} | steps {est.total_steps} | "
            f"trainable {est.trainable_params/1e6:.1f}M "
            f"({est.trainable_pct:.1f}%) | vejde se na GPU: {est.fits_gpus}"
        )
        self.log("=" * 48)

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

        # Train: overall 28–88% follows trainer logs via _parse_train_progress
        self._map_band("docker", 0.0, "Docker / start tréninku…")
        self.log("Spouštím učení…")
        run_dir = run_training(
            cfg,
            est,
            gpus,
            skip_if_over_limit=not allow_over,
            dry_run=False,
        )
        self._map_band("train", 1.0, "Trénink dokončen")
        self._set(run_dir=str(run_dir))
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
            self._map_band("gguf", 0.1, "Balím model (GGUF)…")
            gguf_path = convert_to_gguf(run_dir, quant=cfg.gguf_quant, gpus=gpus)
            self._set(gguf_path=str(gguf_path))
            self._map_band("gguf", 1.0, "GGUF hotovo")
            self.log(f"GGUF: {gguf_path}")
        else:
            found = list((run_dir / "gguf").glob("*.gguf")) if (run_dir / "gguf").exists() else []
            gguf_path = found[0] if found else None
            self._map_band("gguf", 1.0, "GGUF přeskočen")

        self._check_cancel()

        if not skip_ollama and gguf_path is not None:
            self._map_band("ollama", 0.2, "Import do Ollama…")
            system_prompt = (cfg.extra or {}).get("system_prompt")
            export_and_import(
                run_dir,
                Path(gguf_path),
                cfg.ollama_name,
                system_prompt=system_prompt,
            )
            self._set(ollama_name=cfg.ollama_name)
            self._map_band("ollama", 1.0, f"Ollama: {cfg.ollama_name}")
            self.log(f"Hotovo. Spusťte: ollama run {cfg.ollama_name}")
        elif skip_ollama:
            self.log("Ollama import přeskočen.")
        elif gguf_path is None:
            self.log("VAROVÁNÍ: chybí GGUF — Ollama import nelze.")

        self._set(
            phase=JobPhase.DONE.value,
            message=f"Hotovo — ollama run {cfg.ollama_name}",
            progress=100,
            finished_at=time.time(),
            ollama_name=cfg.ollama_name,
            gguf_path=str(gguf_path) if gguf_path else None,
            progress_detail="100%",
        )
        self.log("=== UČENÍ DOKONČENO ===")
        self.log(f"=== ollama run {cfg.ollama_name} ===")


# Global instance used by web + run.py
manager = JobManager()
