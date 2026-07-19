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
import shutil
import socket
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
MARKER = VENV_DIR / ".deps_ok"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

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
    # Debian sometimes only creates python3
    for name in ("python", "python3"):
        p = VENV_DIR / "bin" / name
        if p.exists():
            return p
    return VENV_DIR / "bin" / "python3"


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


def _rm_broken_venv() -> None:
    if VENV_DIR.exists():
        print(f"[run.py] Mažu neúplné .venv: {VENV_DIR}", flush=True)
        shutil.rmtree(VENV_DIR, ignore_errors=True)


def _try_apt_install_venv() -> bool:
    """Best-effort: install python3-venv via sudo (GCE / Ubuntu)."""
    if os.geteuid() == 0:
        sudo: list[str] = []
    elif shutil.which("sudo"):
        sudo = ["sudo"]
    else:
        return False
    # Detect versioned package
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    packages = [f"python{ver}-venv", "python3-venv", "python3-pip"]
    print("[run.py] Zkouším doinstalovat python3-venv přes apt (může chtít heslo sudo)…", flush=True)
    r = subprocess.run(
        sudo + ["apt-get", "update", "-y"],
        check=False,
    )
    r2 = subprocess.run(
        sudo + ["apt-get", "install", "-y"] + packages,
        check=False,
    )
    return r2.returncode == 0


def _bootstrap_pip(python: Path) -> None:
    """Install pip into a venv that was created with --without-pip."""
    # Already has pip?
    r = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True, text=True)
    if r.returncode == 0:
        return
    print("[run.py] Do venv instaluji pip (get-pip.py)…", flush=True)
    get_pip = ROOT / ".get-pip.py"
    try:
        urllib.request.urlretrieve(GET_PIP_URL, get_pip)
        _run([str(python), str(get_pip)], check=True)
    finally:
        if get_pip.exists():
            get_pip.unlink(missing_ok=True)  # type: ignore[arg-type]


def _create_venv() -> Path:
    """
    Create project .venv. Handles Debian/Ubuntu without ensurepip:
    1) normal venv + pip
    2) venv --without-pip + get-pip.py
    3) sudo apt install python3-venv and retry
    """
    print(f"[run.py] Vytvářím virtuální prostředí: {VENV_DIR}", flush=True)
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)

    def try_with_pip() -> None:
        builder = venv.EnvBuilder(with_pip=True, clear=True, upgrade_deps=False)
        builder.create(str(VENV_DIR))

    def try_without_pip() -> None:
        # clear broken partial env first
        _rm_broken_venv()
        print("[run.py] ensurepip chybí — zkouším venv bez pip + get-pip.py …", flush=True)
        builder = venv.EnvBuilder(with_pip=False, clear=True, upgrade_deps=False)
        builder.create(str(VENV_DIR))
        py = _venv_python()
        if not py.exists():
            raise RuntimeError(f"Po vytvoření venv chybí interpreter: {py}")
        _bootstrap_pip(py)

    errors: list[str] = []

    try:
        try_with_pip()
    except Exception as e1:
        errors.append(f"venv+pip: {e1}")
        try:
            try_without_pip()
        except Exception as e2:
            errors.append(f"venv--without-pip: {e2}")
            # apt + retry
            if _try_apt_install_venv():
                try:
                    _rm_broken_venv()
                    try_with_pip()
                except Exception as e3:
                    errors.append(f"po apt: {e3}")
                    try:
                        try_without_pip()
                    except Exception as e4:
                        errors.append(f"po apt without-pip: {e4}")
                        _print_venv_help(errors)
                        raise SystemExit(1) from e4
            else:
                _print_venv_help(errors)
                raise SystemExit(1) from e2

    py = _venv_python()
    if not py.exists():
        _print_venv_help(errors + ["python v .venv/bin nenalezen"])
        raise SystemExit(1)
    # ensure pip exists even if with_pip left a broken state
    try:
        _bootstrap_pip(py)
    except Exception as e:
        print(f"[run.py] Varování: bootstrap pip: {e}", flush=True)
    return py


def _print_venv_help(errors: list[str]) -> None:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(
        "\n[run.py] CHYBA: nejde vytvořit .venv (na GCE/Debian často chybí python3-venv).\n"
        "\n"
        "Spusťte jednou s právy root/sudo:\n"
        f"  sudo apt-get update\n"
        f"  sudo apt-get install -y python{ver}-venv python3-pip python3-venv\n"
        f"  rm -rf {VENV_DIR}\n"
        f"  python3 run.py\n"
        "\n"
        "Nebo ručně bez ensurepip:\n"
        f"  rm -rf {VENV_DIR}\n"
        f"  python3 -m venv --without-pip {VENV_DIR}\n"
        f"  curl -sS {GET_PIP_URL} -o /tmp/get-pip.py\n"
        f"  {VENV_DIR}/bin/python3 /tmp/get-pip.py\n"
        f"  {VENV_DIR}/bin/pip install -r requirements.txt\n"
        f"  {VENV_DIR}/bin/python run.py\n"
        "\n"
        f"Detaily: {'; | '.join(errors)}\n",
        file=sys.stderr,
        flush=True,
    )


def _pip_install(python: Path) -> None:
    if not REQ_FILE.exists():
        raise FileNotFoundError(f"Chybí {REQ_FILE}")
    # ensure pip module works
    _bootstrap_pip(python)
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
    os.execv(str(py), [str(py), str(ROOT / "run.py"), *argv])


def ensure_runtime(argv: list[str]) -> None:
    """
    Guarantee we run inside project .venv with all host deps installed.
    May os.execv() and never return.
    """
    if os.environ.get("LLM_SKIP_BOOTSTRAP") == "1":
        return

    py = _venv_python()
    need_venv = not _in_project_venv()

    if need_venv:
        # Broken partial venv from previous failed ensurepip attempt
        if VENV_DIR.exists() and not py.exists():
            _rm_broken_venv()
            py = _venv_python()

        if not py.exists():
            py = _create_venv()

        # pip missing inside existing venv?
        probe = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True)
        if probe.returncode != 0:
            try:
                _bootstrap_pip(py)
            except Exception as e:
                _print_venv_help([str(e)])
                raise SystemExit(1) from e

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


def _banner_text(host: str, port: int, token: str) -> str:
    urls = [f"http://{ip}:{port}" for ip in _local_ips()]
    lines = [
        "Web UI běží — otevřete v prohlížeči:",
        "",
        f"  Lokálně:  {urls[0]}",
    ]
    for u in urls[1:]:
        lines.append(f"  Síť:      {u}")
    lines.append(f"  Bind:     http://{host}:{port}")
    lines.append("")
    if token:
        lines.append(f"  Access token:  {token}")
        lines.append("  (vložte dole ve webu; nebo: python3 run.py --no-token)")
    else:
        lines.append("  Auth: VYPNUTÁ (--no-token)")
    lines.append("")
    lines.append("  Firewall GCE: gcloud compute firewall-rules create llm-ui \\")
    lines.append(f"      --allow=tcp:{port} --source-ranges=0.0.0.0/0")
    lines.append("")
    lines.append("  Ctrl+C  =  bezpečně ukončit server")
    return "\n".join(lines)


def run_web(args: argparse.Namespace) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import signal
    import threading

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

    # Background: only packages + GPU detect — NOT docker image (avoids dual builds)
    if not args.skip_setup:
        def _bg_setup():
            try:
                from lib.job_manager import manager
                manager.log("Automatický setup (balíčky/GPU, bez Docker buildu)…")
                manager.prepare_env(
                    install_packages=True,
                    framework="unsloth",
                    build_image=False,
                )
                manager.log("Host setup hotov. Docker image se postaví až při Startu (1×).")
            except Exception as e:
                from lib.job_manager import manager
                manager.log(f"Setup varování: {e}")

        threading.Thread(target=_bg_setup, name="bg-setup", daemon=True).start()

    banner = _banner_text(host, port, token)
    # Save for easy re-print
    (ROOT / ".ui_banner.txt").write_text(
        banner.replace("[bold]", "").replace("[/]", "").replace("[bold yellow]", "")
        .replace("[/bold yellow]", "").replace("[red]", "").replace("[/red]", ""),
        encoding="utf-8",
    )

    def print_footer(title: str = "Učení AI modelu — URL + TOKEN (vždy dole)") -> None:
        console.print()
        console.print(Panel(banner, title=title, border_style="bright_cyan", expand=True))
        console.print("[dim]Logy serveru jsou NÁD tímto panelem. Ctrl+C ukončí vše.[/]\n")

    print_footer()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=args.log_level,
        access_log=False,  # méně šumu; job logy jdou do webu
    )
    server = uvicorn.Server(config)

    def _on_signal(signum, frame):
        console.print("\n[yellow]Ctrl+C — ukončuji server…[/]")
        server.should_exit = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Re-print banner after startup so it sits under uvicorn "Started" lines
    def _reprint():
        import time
        time.sleep(1.2)
        if not server.should_exit:
            print_footer("Připomenutí — URL + TOKEN")

    threading.Thread(target=_reprint, name="banner-footer", daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        console.print("[green]Server ukončen.[/]")
        print_footer("Ukončeno — token pro příští start")
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
