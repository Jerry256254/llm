#!/usr/bin/env python3
"""
Jediný vstupní bod — spustí vše.

Výchozí režim: web UI na 0.0.0.0 (vhodné pro Google Cloud public IP).
CLI režim:     python run.py --cli

Příklady:
  python run.py
  python run.py --port 8080
  python run.py --no-token          # bez auth (jen v důvěryhodné síti!)
  python run.py --cli -y --model unsloth/llama-3.2-1b-instruct --dataset ./data/sample_alpaca.jsonl
  python run.py --cli --dry-run
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
MARKER = VENV_DIR / ".deps_ok"

# Import-name -> pip package (for diagnostics only)
REQUIRED_IMPORTS = (
    "rich",
    "yaml",  # PyYAML
    "fastapi",
    "uvicorn",
    "pydantic",
    "psutil",
    "huggingface_hub",
)


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _in_project_venv() -> bool:
    """True if current interpreter is this project's .venv."""
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


def _missing_imports() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[run.py] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check)


def _create_venv() -> None:
    print(f"[run.py] Vytvářím virtuální prostředí: {VENV_DIR}", flush=True)
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False)
    builder.create(str(VENV_DIR))


def _pip_install(python: Path) -> None:
    if not REQ_FILE.exists():
        raise FileNotFoundError(f"Chybí {REQ_FILE}")
    # upgrade pip first (helps on older venv bootstrap)
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=False)
    print(f"[run.py] Instaluji závislosti z {REQ_FILE.name} …", flush=True)
    _run([str(python), "-m", "pip", "install", "-r", str(REQ_FILE)])
    MARKER.write_text("ok\n", encoding="utf-8")


def _reexec_in_venv(argv: list[str]) -> None:
    """Replace current process with project venv Python running this script."""
    py = _venv_python()
    if not py.exists():
        raise RuntimeError(f"Venv python neexistuje: {py}")
    print(f"[run.py] Restartuji v .venv: {py}", flush=True)
    # os.execv replaces process — no return
    os.execv(str(py), [str(py), str(ROOT / "run.py"), *argv])


def ensure_runtime(argv: list[str]) -> None:
    """
    Guarantee we run inside project .venv with all host deps installed.
    May os.execv() and never return.
    """
    # Allow skip for advanced users
    if os.environ.get("LLM_SKIP_BOOTSTRAP") == "1":
        return

    py = _venv_python()
    need_venv = not _in_project_venv()

    if need_venv:
        if not py.exists():
            try:
                _create_venv()
            except Exception as e:
                print(
                    f"[run.py] CHYBA: nelze vytvořit .venv ({e}).\n"
                    f"  Manuálně: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(1) from e
        # Install deps with venv pip if marker missing or force
        if not MARKER.exists() or os.environ.get("LLM_FORCE_REINSTALL") == "1":
            try:
                _pip_install(py)
            except subprocess.CalledProcessError as e:
                print(
                    f"[run.py] CHYBA: pip install selhal (exit {e.returncode}).\n"
                    f"  Zkuste: {py} -m pip install -r requirements.txt",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(1) from e
        _reexec_in_venv(argv)
        return  # unreachable

    # Already in project venv — ensure imports work
    missing = _missing_imports()
    if missing or not MARKER.exists() or os.environ.get("LLM_FORCE_REINSTALL") == "1":
        if missing:
            print(f"[run.py] V .venv chybí: {', '.join(missing)} — doinstaluji…", flush=True)
        try:
            _pip_install(Path(sys.executable))
        except subprocess.CalledProcessError as e:
            print(
                f"[run.py] CHYBA: pip install selhal (exit {e.returncode}).\n"
                f"  Zkuste: {sys.executable} -m pip install -r requirements.txt",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1) from e
        # re-check
        missing = _missing_imports()
        if missing:
            print(
                f"[run.py] CHYBA: po instalaci stále chybí: {', '.join(missing)}\n"
                f"  Interpreter: {sys.executable}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)


def _local_ips() -> list[str]:
    ips = ["127.0.0.1"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips


def run_web(args: argparse.Namespace) -> int:
    # path for local imports (lib/)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from rich.console import Console
    from rich.panel import Panel

    from lib.web_app import create_app, generate_token
    import uvicorn

    console = Console()

    if args.no_token:
        token = ""
        os.environ.pop("LLM_UI_TOKEN", None)
    elif args.token:
        token = args.token
        os.environ["LLM_UI_TOKEN"] = token
    else:
        token = os.environ.get("LLM_UI_TOKEN") or generate_token()
        os.environ["LLM_UI_TOKEN"] = token

    host = args.host
    port = args.port
    app = create_app(access_token=token)

    if not args.skip_setup:
        def _bg_setup():
            try:
                from lib.job_manager import manager
                manager.log("Automatický setup prostředí na pozadí…")
                manager.prepare_env(install_packages=True, framework="unsloth")
            except Exception as e:
                from lib.job_manager import manager
                manager.log(f"Setup varování (můžete spustit z UI): {e}")

        import threading
        threading.Thread(target=_bg_setup, name="bg-setup", daemon=True).start()

    urls = [f"http://{ip}:{port}" for ip in _local_ips()]
    lines = [
        "[bold]Web Control Panel běží[/]",
        "",
        f"Lokálně:   {urls[0]}",
    ]
    for u in urls[1:]:
        lines.append(f"Síť:       {u}")
    lines.append(f"Bind:      http://{host}:{port}")
    lines.append(f"Python:    {sys.executable}")
    lines.append("")
    if token:
        lines.append(f"[bold yellow]Access token:[/] {token}")
        lines.append("Zadejte token ve web UI (nebo hlavička X-Token).")
        lines.append("Vypnout auth: python run.py --no-token")
    else:
        lines.append("[red]Auth vypnutá[/] — nevystavujte na veřejný internet bez firewallu.")
    lines.append("")
    lines.append("Na Google Cloud otevřete firewall pro TCP port " + str(port) + ":")
    lines.append(
        f"  gcloud compute firewall-rules create llm-ui --allow=tcp:{port} --source-ranges=0.0.0.0/0"
    )
    lines.append("")
    lines.append("Ctrl+C ukončí server.")

    console.print(Panel("\n".join(lines), title="Učení AI modelu", border_style="cyan"))
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


def run_cli(argv: list[str]) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from train_pipeline import main as pipeline_main
    return pipeline_main(argv)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="LLM fine-tune: jeden příkaz pro vše (web UI výchozí)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--cli", action="store_true", help="CLI pipeline místo web UI")
    p.add_argument("--host", default="0.0.0.0", help="Web bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=8080, help="Web port (default 8080)")
    p.add_argument("--token", default=None, help="Access token pro web UI")
    p.add_argument("--no-token", action="store_true", help="Web bez autentizace")
    p.add_argument("--skip-setup", action="store_true", help="Nespouštět auto setup host balíčků")
    p.add_argument("--log-level", default="info", help="uvicorn log level")
    args, rest = p.parse_known_args(argv)
    return args, rest


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    # Bootstrap before parsing? Keep parse after bootstrap so --help works in venv too.
    # For --help on outer python without deps, argparse still works (stdlib only).
    if any(a in ("-h", "--help") for a in argv) and not _in_project_venv():
        # show help without installing
        parse_args(argv)
        return 0

    ensure_runtime(argv)  # may re-exec into .venv

    args, rest = parse_args(argv)
    if args.cli:
        return run_cli(rest)
    return run_web(args)


if __name__ == "__main__":
    raise SystemExit(main())
