"""Training time / cost limits and safe process termination."""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from rich.console import Console

console = Console()


@dataclass
class SafetyLimits:
    max_train_seconds: float
    max_cost_usd: float
    gpu_hourly_usd: float
    num_gpus: int = 1

    @classmethod
    def from_config(cls, max_hours: float, max_cost_usd: float, gpu_hourly_usd: float, num_gpus: int = 1):
        return cls(
            max_train_seconds=max(60.0, max_hours * 3600.0),
            max_cost_usd=max(0.01, max_cost_usd),
            gpu_hourly_usd=max(0.0, gpu_hourly_usd),
            num_gpus=max(1, num_gpus),
        )

    def estimated_cost(self, elapsed_seconds: float) -> float:
        hours = elapsed_seconds / 3600.0
        return hours * self.gpu_hourly_usd * self.num_gpus


class TrainingWatchdog:
    """
    Background watchdog that kills a process group when time or cost limits hit.
    Also exposes a soft-check for cooperative cancellation.
    """

    def __init__(
        self,
        limits: SafetyLimits,
        *,
        on_limit: Optional[Callable[[str], None]] = None,
        poll_interval: float = 15.0,
    ):
        self.limits = limits
        self.on_limit = on_limit
        self.poll_interval = poll_interval
        self._start: Optional[float] = None
        self._pid: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.triggered_reason: Optional[str] = None
        self._lock = threading.Lock()

    def start(self, pid: int) -> None:
        self._pid = pid
        self._start = time.monotonic()
        self._stop.clear()
        self.triggered_reason = None
        self._thread = threading.Thread(target=self._loop, name="train-watchdog", daemon=True)
        self._thread.start()
        console.print(
            f"[cyan]Safety watchdog aktivní:[/] max {self.limits.max_train_seconds/3600:.2f} h, "
            f"max ${self.limits.max_cost_usd:.2f} "
            f"(~${self.limits.gpu_hourly_usd}/GPU·h × {self.limits.num_gpus} GPU)"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    def check_soft(self) -> Optional[str]:
        """Return reason string if limits exceeded (without killing)."""
        if self._start is None:
            return None
        elapsed = self.elapsed
        if elapsed >= self.limits.max_train_seconds:
            return f"Překročen časový limit ({elapsed/3600:.2f} h ≥ {self.limits.max_train_seconds/3600:.2f} h)"
        cost = self.limits.estimated_cost(elapsed)
        if cost >= self.limits.max_cost_usd:
            return f"Překročen nákladový limit (${cost:.2f} ≥ ${self.limits.max_cost_usd:.2f})"
        return None

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            reason = self.check_soft()
            if reason:
                with self._lock:
                    self.triggered_reason = reason
                console.print(f"[red bold]WATCHDOG:[/] {reason}")
                if self.on_limit:
                    try:
                        self.on_limit(reason)
                    except Exception as e:
                        console.print(f"[red]on_limit callback error: {e}[/]")
                self._terminate_process()
                break

    def _terminate_process(self) -> None:
        pid = self._pid
        if not pid:
            return
        console.print(f"[yellow]Ukončuji trénink (PID {pid})…[/]")
        try:
            # Try process group first (docker/child shells)
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Grace period then SIGKILL
        time.sleep(10)
        try:
            os.kill(pid, 0)  # still alive?
            console.print("[red]SIGKILL…[/]")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        except OSError:
            pass  # already dead


def preflight_cost_check(
    est_hours: float,
    est_cost: float,
    limits: SafetyLimits,
    *,
    abort_if_over: bool = True,
) -> bool:
    """
    Return True if safe to proceed.
    If estimate exceeds limits, warn and optionally abort.
    """
    over_time = est_hours * 3600 > limits.max_train_seconds
    over_cost = est_cost > limits.max_cost_usd
    if not over_time and not over_cost:
        return True

    console.print("[yellow bold]Předběžný odhad překračuje limity:[/]")
    if over_time:
        console.print(
            f"  Čas: {est_hours:.2f} h > limit {limits.max_train_seconds/3600:.2f} h"
        )
    if over_cost:
        console.print(f"  Náklady: ${est_cost:.2f} > limit ${limits.max_cost_usd:.2f}")

    if abort_if_over:
        console.print(
            "[red]Trénink nebude spuštěn. Zvyšte limity, snižte epochs/seq/model, "
            "nebo potvrďte přepsání limitu v konfiguraci.[/]"
        )
        return False
    return True
