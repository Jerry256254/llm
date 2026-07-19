#!/usr/bin/env python3
"""
Univerzální pipeline: prostředí → interaktivní config → analýza VRAM/času
→ Docker fine-tune (Unsloth/Axolotl) → GGUF → Ollama.

Příklad:
  python train_pipeline.py
  python train_pipeline.py --non-interactive \\
      --model unsloth/llama-3.2-1b-instruct \\
      --dataset ./data/sample_alpaca.jsonl \\
      --max-hours 2 --max-cost 10
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from lib.analysis import analyze, print_analysis
from lib.convert_gguf import convert_to_gguf
from lib.env_setup import prepare_environment
from lib.interactive import collect_config_interactive, show_dataset_format_help
from lib.ollama_export import export_and_import
from lib.training import run_training

console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated LLM fine-tune → GGUF → Ollama (Docker + NVIDIA GPU)",
    )
    p.add_argument("--non-interactive", "-y", action="store_true", help="Bez interaktivních promptů")
    p.add_argument("--model", dest="model_id", default=None, help="HF / Unsloth model ID")
    p.add_argument("--dataset", dest="dataset_path", default=None, help="Cesta k datasetu nebo HF ID")
    p.add_argument("--dataset-format", default=None, choices=["alpaca", "sharegpt", "chat", "text", "hf"])
    p.add_argument("--output", dest="output_dir", default=None, help="Výstupní adresář")
    p.add_argument("--framework", choices=["unsloth", "axolotl"], default=None)
    p.add_argument("--method", choices=["qlora", "lora", "full"], default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--gguf-quant", default=None, help="q4_k_m, q5_k_m, q8_0, f16, …")
    p.add_argument("--ollama-name", default=None)
    p.add_argument("--max-hours", type=float, default=None, help="Safety: max trénink (h)")
    p.add_argument("--max-cost", type=float, default=None, help="Safety: max odhad USD")
    p.add_argument("--gpu-hourly-usd", type=float, default=None)
    p.add_argument("--skip-setup", action="store_true", help="Přeskočit instalaci host balíčků")
    p.add_argument("--skip-train", action="store_true", help="Jen setup + analýza")
    p.add_argument("--skip-gguf", action="store_true", help="Bez konverze GGUF")
    p.add_argument("--skip-ollama", action="store_true", help="Bez importu do Ollama")
    p.add_argument("--force-rebuild-image", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Analýza bez tréninku")
    p.add_argument("--allow-over-limit", action="store_true", help="Povolit běh i při překročení odhadu limitů")
    p.add_argument("--run-dir", default=None, help="Pokračovat od existujícího run adresáře (GGUF/Ollama)")
    p.add_argument("--show-formats", action="store_true", help="Zobrazit formáty datasetu a skončit")
    return p.parse_args(argv)


def overrides_from_args(args: argparse.Namespace) -> dict:
    mapping = {
        "model_id": args.model_id,
        "dataset_path": args.dataset_path,
        "dataset_format": args.dataset_format,
        "output_dir": args.output_dir,
        "framework": args.framework,
        "method": args.method,
        "lora_r": args.lora_r,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "gguf_quant": args.gguf_quant,
        "ollama_name": args.ollama_name,
        "max_train_hours": args.max_hours,
        "max_cost_usd": args.max_cost,
        "gpu_hourly_usd": args.gpu_hourly_usd,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.show_formats:
        show_dataset_format_help()
        return 0

    console.print(
        Panel.fit(
            "[bold]LLM Fine-Tune Pipeline v1[/]\n"
            "Docker · CUDA · Unsloth/Axolotl · GGUF · Ollama",
            border_style="bright_green",
        )
    )

    # --- 1) Environment ---
    try:
        env = prepare_environment(
            install_packages=not args.skip_setup,
            framework=args.framework or "unsloth",
            force_rebuild_image=args.force_rebuild_image,
        )
    except Exception as e:
        console.print(f"[red]Chyba přípravy prostředí:[/] {e}")
        traceback.print_exc()
        return 1

    if not env.docker_ok and not args.dry_run and not args.skip_train:
        console.print("[red]Docker není dostupný — trénink nelze spustit v kontejneru.[/]")
        if not args.non_interactive and not Confirm.ask("Pokračovat jen s analýzou?", default=True):
            return 1
        args.skip_train = True

    # Resume path: only GGUF + Ollama
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            console.print(f"[red]Run dir neexistuje:[/] {run_dir}")
            return 1
        return _post_train(run_dir, args, env.gpus)

    # --- 2) Interactive config ---
    try:
        cfg = collect_config_interactive(
            non_interactive=args.non_interactive,
            overrides=overrides_from_args(args),
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Zrušeno uživatelem.[/]")
        return 130

    # --- 3) Analysis ---
    estimate = analyze(cfg, env.gpus)
    print_analysis(cfg, estimate, env.gpus)

    summary_path = cfg.output_dir / "last_config.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_id": cfg.model_id,
                "dataset_path": cfg.dataset_path,
                "method": cfg.method,
                "estimate": estimate.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]Konfigurace uložena: {summary_path}[/]")

    if not estimate.fits_gpus and env.gpus:
        console.print("[yellow bold]Varování: model se nemusí vejít do VRAM.[/]")
        if not args.non_interactive and not args.dry_run:
            if not Confirm.ask("Přesto pokračovat?", default=False):
                return 1

    if args.skip_train or args.dry_run:
        if args.dry_run:
            try:
                run_training(
                    cfg,
                    estimate,
                    env.gpus,
                    skip_if_over_limit=not args.allow_over_limit,
                    dry_run=True,
                )
            except RuntimeError as e:
                console.print(f"[red]{e}[/]")
                return 1
        console.print("[cyan]Hotovo (bez tréninku).[/]")
        return 0

    if not args.non_interactive:
        if not Confirm.ask("Spustit trénink?", default=True):
            console.print("[yellow]Trénink přeskočen.[/]")
            return 0

    # --- 4) Train ---
    try:
        run_dir = run_training(
            cfg,
            estimate,
            env.gpus,
            skip_if_over_limit=not args.allow_over_limit,
            dry_run=False,
        )
    except Exception as e:
        console.print(f"[red bold]Trénink selhal:[/] {e}")
        traceback.print_exc()
        return 1

    # Persist ollama name for resume
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "ollama_name": cfg.ollama_name,
                "gguf_quant": cfg.gguf_quant,
                "model_id": cfg.model_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- 5–6) GGUF + Ollama ---
    return _post_train(run_dir, args, env.gpus, cfg_ollama=cfg.ollama_name, quant=cfg.gguf_quant)


def _post_train(run_dir, args, gpus, cfg_ollama: str | None = None, quant: str | None = None) -> int:
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    ollama_name = cfg_ollama or args.ollama_name or meta.get("ollama_name") or "finetuned-model"
    quant = quant or args.gguf_quant or meta.get("gguf_quant") or "q4_k_m"

    gguf_path = None
    if not args.skip_gguf:
        try:
            gguf_path = convert_to_gguf(run_dir, quant=quant, gpus=gpus)
        except Exception as e:
            console.print(f"[red]GGUF konverze selhala:[/] {e}")
            traceback.print_exc()
            return 1
    else:
        found = list((run_dir / "gguf").glob("*.gguf")) if (run_dir / "gguf").exists() else []
        gguf_path = found[0] if found else None

    if not args.skip_ollama:
        if gguf_path is None:
            console.print("[red]Chybí GGUF pro import do Ollama.[/]")
            return 1
        try:
            export_and_import(run_dir, gguf_path, ollama_name)
        except Exception as e:
            console.print(f"[red]Ollama import selhal:[/] {e}")
            console.print(
                f"[yellow]Modelfile může být k dispozici v {run_dir}/Modelfile — "
                f"zkuste: ollama create {ollama_name} -f {run_dir}/Modelfile[/]"
            )
            return 1

    console.print(
        Panel.fit(
            f"[bold green]Pipeline dokončena[/]\n"
            f"Run: {run_dir}\n"
            f"GGUF: {gguf_path or '—'}\n"
            f"Ollama: ollama run {ollama_name}",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
