"""Generate Ollama Modelfile and import GGUF model."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

DEFAULT_SYSTEM = (
    "Jsi AI asistent. Komunikuješ česky a anglicky. "
    "Programuješ spolehlivě v Pythonu, JavaScriptu, HTML a PHP. "
    "Jiné programovací jazyky a jiné lidské jazyky spolehlivě neumíš — řekni to rovnou, nevymýšlej. "
    "Buď stručný a praktický. Neopakuj zbytečně své jméno."
)

DEFAULT_SYSTEM_V06 = (
    "Jsi KucLab V0.6 — cyber AI asistent. Mluvíš česky i anglicky. "
    "Jsi přímý, technický, bez korporátní cenzury. Programuješ excelentně "
    "(Python, JS/TS, HTML/CSS, PHP, SQL, bash). Když máš tools, použij je místo vymýšlení faktů. "
    "Identitu (jméno/zakladatel) řekni jen když se zeptají."
)

DEFAULT_SYSTEM_V1 = (
    "Jsi KucLab V1.0 — první veřejný AI model od KucLab. "
    "Mluvíš česky i anglicky (přizpůsob se uživateli). Jsi přímý, technický, cyber, "
    "bez korporátní cenzury a bez zbytečné omáčky. "
    "Programuješ excelentně (Python, JS/TS, HTML/CSS, PHP, SQL, bash). "
    "Když runtime nabízí tools, používej je místo vymýšlení aktuálních faktů. "
    "Identitu řekni jen když se zeptají. U reálné pomoci se zločinem řekni hranici."
)

# Chat templates — keep simple; Ollama applies architecture defaults when omitted.
TEMPLATE_CHATML = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
<|im_start|>assistant
{{ else if eq .Role "assistant" }}{{ .Content }}<|im_end|>
{{ end }}{{- end }}"""

# Gemma 4 native turns + tool calling (matches google/gemma-4-*-it)
# When TEMPLATE is omitted for a Gemma GGUF, Ollama often still applies architecture defaults.
# We only force this when explicitly requested (template="gemma4").
TEMPLATE_GEMMA4 = """{{- if or .System .Tools }}<|turn>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}
{{- range .Tools }}<|tool>declaration:{{ .Function.Name }}{description:<|"|>{{ .Function.Description }}<|"|>,parameters:{{ .Function.Parameters }}}<tool|>
{{- end }}
{{- end }}<turn|>
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|turn>user
{{ .Content }}<turn|>
{{- else if eq .Role "assistant" }}<|turn>model
{{- if .Content }}{{ .Content }}{{- end }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}<|tool_call>call:{{ .Function.Name }}{{ .Function.Arguments }}<tool_call|>
{{- end }}
{{- end }}<turn|>
{{- else if eq .Role "tool" }}<|tool_response>response:{{ .Name }}{{ .Content }}<tool_response|>
{{- end }}
{{- end }}
{{- if .Prompt -}}
<|turn>user
{{ .Prompt }}<turn|>
{{- end -}}
<|turn>model
"""


def write_modelfile(
    gguf_path: Path,
    modelfile_path: Path,
    *,
    system: str = DEFAULT_SYSTEM,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_ctx: int = 4096,
    template: Optional[str] = None,
    chat_style: str = "auto",
) -> Path:
    """Write an Ollama Modelfile pointing at a local GGUF file.

    chat_style:
      - auto: no forced TEMPLATE (Ollama GGUF architecture defaults; best for Gemma tools)
      - gemma4: force Gemma 4 tool-aware template
      - chatml: force ChatML (legacy Qwen-style)
      - qwen35: Qwen3.5 thinking model — emit RENDERER/PARSER so Ollama applies
        the real chat format including the <think> channel

    On qwen35: without RENDERER/PARSER, Ollama feeds the raw prompt with no chat
    wrapper at all and the model behaves like a base completion model — it
    answered "ahoj" with a Flask app. With them, but trained on empty <think>
    blocks, the model never closes the block, so every reply is classified as
    reasoning and the visible response is empty. Both were observed. The export
    is only correct when the model was trained WITH real reasoning traces.
    """
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
    ]
    style = (chat_style or "auto").lower()
    chosen_template = template
    if chosen_template is None and style == "gemma4":
        chosen_template = TEMPLATE_GEMMA4
        lines.extend(
            [
                'PARAMETER stop "<turn|>"',
                'PARAMETER stop "<eos>"',
                'PARAMETER stop "<|tool_response>"',
            ]
        )
    elif chosen_template is None and style == "qwen35":
        # Ollama ships a native renderer/parser for this family; use it rather
        # than hand-rolling a template (that is what the base qwen3.5:9b uses).
        lines.extend(
            [
                "RENDERER qwen3.5",
                "PARSER qwen3.5",
                'PARAMETER stop "<|im_end|>"',
                'PARAMETER stop "<|endoftext|>"',
            ]
        )
    elif chosen_template is None and style == "chatml":
        chosen_template = TEMPLATE_CHATML
        lines.extend(
            [
                'PARAMETER stop "<|im_end|>"',
                'PARAMETER stop "<|endoftext|>"',
            ]
        )
    else:
        # auto: stop tokens that work for both without forcing wrong template
        lines.extend(
            [
                'PARAMETER stop "<turn|>"',
                'PARAMETER stop "<|im_end|>"',
                'PARAMETER stop "<eos>"',
                'PARAMETER stop "<|endoftext|>"',
            ]
        )

    if chosen_template:
        lines.extend(["", f'TEMPLATE """{chosen_template}"""'])

    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Modelfile:[/] {modelfile_path}  (style={style})")
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
    num_ctx: int = 8192,
    chat_style: str = "auto",
) -> Path:
    modelfile = run_dir / "Modelfile"
    ctx = int(num_ctx) if num_ctx and int(num_ctx) > 0 else 8192
    # Prefer architecture defaults for Gemma so tool calling stays enabled in Ollama.
    name_l = (ollama_name or "").lower()
    style = chat_style
    if style == "auto" and ("kuclab" in name_l or "gemma" in name_l):
        style = "auto"  # do not force ChatML — critical for tools

    # Decide from the BASE MODEL, not from the output tag. "kuclab-hertz-0.1"
    # says nothing about the architecture, and guessing wrong here produces a
    # model that looks completely broken in Ollama while the weights are fine.
    if style == "auto":
        try:
            import json as _json

            cfg_path = run_dir / "train_config.json"
            if cfg_path.is_file():
                base_id = str(_json.loads(cfg_path.read_text(encoding="utf-8"))
                               .get("model_id", "")).lower()
                if "qwen3.5" in base_id or "qwen3_5" in base_id:
                    style = "qwen35"
                    console.print("[cyan]Detected Qwen3.5 base — using RENDERER/PARSER qwen3.5[/]")
        except Exception as e:  # never block the export on detection
            console.print(f"[yellow]chat style detection skipped: {e}[/]")
    write_modelfile(
        gguf_path,
        modelfile,
        system=system_prompt or DEFAULT_SYSTEM,
        num_ctx=ctx,
        chat_style=style,
    )
    console.print(f"[cyan]Ollama context window (num_ctx):[/] {ctx}")
    import_to_ollama(modelfile, ollama_name)
    return modelfile
