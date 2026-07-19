"""Generate Ollama Modelfile and import GGUF model."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

DEFAULT_SYSTEM = (
    "Jsi užitečný asistent. Odpovídej jasně a stručně v jazyce uživatele."
)

# Chat templates — keep simple; Ollama applies architecture defaults when omitted.
TEMPLATE_CHATML = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
<|im_start|>assistant
{{ else if eq .Role "assistant" }}{{ .Content }}<|im_end|>
{{ end }}{{- end }}"""


def write_modelfile(
    gguf_path: Path,
    modelfile_path: Path,
    *,
    system: str = DEFAULT_SYSTEM,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_ctx: int = 4096,
    template: Optional[str] = None,
) -> Path:
    """Write an Ollama Modelfile pointing at a local GGUF file."""
    gguf_path = gguf_path.resolve()
    # FROM can be absolute path to gguf
    lines = [
        f"FROM {gguf_path}",
        "",
        f'SYSTEM """{system}"""',
        "",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
        f"PARAMETER num_ctx {num_ctx}",
        f"PARAMETER stop \"<|im_end|>\"",
        f"PARAMETER stop \"<|endoftext|>\"",
    ]
    if template:
        lines.extend(["", f'TEMPLATE """{template}"""'])

    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Modelfile:[/] {modelfile_path}")
    return modelfile_path


def ensure_ollama() -> bool:
    if shutil.which("ollama"):
        return True
    console.print(
        "[yellow]Ollama není v PATH. "
        "Nainstalujte: https://ollama.com/download nebo "
        "`curl -fsSL https://ollama.com/install.sh | sh`[/]"
    )
    return False


def import_to_ollama(
    modelfile_path: Path,
    model_name: str,
    *,
    start_server: bool = True,
) -> None:
    """Run `ollama create` from Modelfile."""
    if not ensure_ollama():
        # Still leave Modelfile for manual import
        console.print(
            f"[yellow]Přeskočen import. Ručně: ollama create {model_name} -f {modelfile_path}[/]"
        )
        return

    if start_server:
        # Best-effort start (systemd or background)
        subprocess.run(["ollama", "serve"], check=False, capture_output=True, start_new_session=True)
        import time
        time.sleep(2)

    console.print(f"[bold cyan]Import do Ollama jako[/] [bold]{model_name}[/] …")
    proc = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        console.print(proc.stdout)
        console.print(proc.stderr)
        raise RuntimeError(f"ollama create failed: {proc.returncode}")

    console.print(f"[green bold]Model připraven:[/] ollama run {model_name}")
    # Quick list confirmation
    subprocess.run(["ollama", "list"], check=False)


def export_and_import(
    run_dir: Path,
    gguf_path: Path,
    ollama_name: str,
    *,
    system_prompt: Optional[str] = None,
) -> Path:
    modelfile = run_dir / "Modelfile"
    write_modelfile(
        gguf_path,
        modelfile,
        system=system_prompt or DEFAULT_SYSTEM,
    )
    import_to_ollama(modelfile, ollama_name)
    return modelfile
