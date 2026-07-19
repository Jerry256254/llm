#!/usr/bin/env python3
"""
In-container training entrypoint (Unsloth preferred, transformers+PEFT fallback).
Reads JSON config path from argv[1].
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def identity_qa_pairs(name: str) -> list[dict]:
    """Many phrasings so the model reliably knows its own name."""
    name = name.strip()
    if not name:
        return []
    answers = [
        f"Jmenuji se {name}.",
        f"Mé jméno je {name}.",
        f"Jsem {name}.",
        f"Říkají mi {name}. Jsem AI asistent jménem {name}.",
        f"Jsem jazykový model {name}. Rád pomohu.",
        f"My name is {name}.",
        f"I'm {name}, an AI assistant.",
        f"I am {name}.",
    ]
    questions = [
        "Jak se jmenuješ?",
        "Jak se jmenujete?",
        "Kdo jsi?",
        "Kdo jste?",
        "Jak ti říkají?",
        "Jaké je tvé jméno?",
        "Představ se.",
        "Představ se prosím.",
        "Co jsi zač?",
        "Jaké máš jméno?",
        "What is your name?",
        "Who are you?",
        "What's your name?",
        "Tell me your name.",
        "Introduce yourself.",
        "How should I call you?",
    ]
    pairs: list[dict] = []
    # pair each question with a rotating answer
    for i, q in enumerate(questions):
        a = answers[i % len(answers)]
        pairs.append({"instruction": q, "input": "", "output": a})
    # extra reinforced forms
    for a in answers:
        pairs.append({"instruction": "Jak se jmenuješ?", "input": "", "output": a})
        pairs.append(
            {
                "instruction": "Stručně se představ a uveď své jméno.",
                "input": "",
                "output": f"Ahoj, jmenuji se {name}. Jsem AI asistent připravený pomoct.",
            }
        )
    # plain text identity (for pretrain / text mode)
    for _ in range(8):
        pairs.append(
            {
                "text": (
                    f"Jmenuji se {name}. Jsem AI asistent jménem {name}. "
                    f"Když se někdo zeptá na mé jméno, odpovím, že se jmenuji {name}. "
                    f"My name is {name}."
                )
            }
        )
    return pairs


def inject_identity(dataset, cfg: dict):
    """Append identity examples so the model learns its name during training."""
    if cfg.get("teach_identity", True) is False:
        return dataset
    name = (cfg.get("identity_name") or cfg.get("ollama_name") or "").strip()
    if not name:
        return dataset

    from datasets import Dataset, concatenate_datasets

    pairs = identity_qa_pairs(name)
    if not pairs:
        return dataset

    ds_format = cfg.get("dataset_format", "alpaca")
    # Build rows matching expected schema after mapping: we inject as alpaca-style
    # then map through format_dataset separately — easier: inject as text already formatted
    texts = []
    for p in pairs:
        if "text" in p and "instruction" not in p:
            texts.append(p["text"])
        else:
            instr = p.get("instruction", "")
            out = p.get("output", "")
            texts.append(
                f"<|im_start|>user\n{instr}<|im_end|>\n"
                f"<|im_start|>assistant\n{out}<|im_end|>\n"
            )

    # repeat identity block so it has weight even in large datasets
    repeat = int(cfg.get("identity_repeat", 3))
    texts = texts * max(1, repeat)
    id_ds = Dataset.from_dict({"text": texts})
    log(f"Identita modelu „{name}“: přidávám {len(texts)} tréninkových příkladů se jménem.")
    # dataset already has `text` column after format_dataset
    return concatenate_datasets([dataset, id_ds])


def format_dataset(dataset, dataset_format: str, tokenizer):
    """Map raw dataset to a single `text` field for causal LM SFT."""

    def alpaca_map(example):
        # Mixed corpus: some rows are plain text only
        if not (example.get("instruction") or example.get("prompt") or example.get("output")):
            t = example.get("text") or example.get("content") or ""
            if t:
                return {"text": t}
        instr = example.get("instruction") or example.get("prompt") or ""
        inp = example.get("input") or ""
        out = example.get("output") or example.get("response") or example.get("completion") or ""
        if inp:
            user = f"{instr}\n{inp}"
        else:
            user = instr
        text = (
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>\n"
        )
        return {"text": text}

    def sharegpt_map(example):
        convs = example.get("conversations") or example.get("conversation") or []
        parts = []
        for turn in convs:
            role = (turn.get("from") or turn.get("role") or "").lower()
            val = turn.get("value") or turn.get("content") or ""
            if role in ("human", "user", "system"):
                r = "user" if role != "system" else "system"
            else:
                r = "assistant"
            parts.append(f"<|im_start|>{r}\n{val}<|im_end|>")
        parts.append("")  # trailing newline style
        return {"text": "\n".join(parts) + "\n"}

    def chat_map(example):
        msgs = example.get("messages") or []
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return {"text": "\n".join(parts) + "\n"}

    def text_map(example):
        t = example.get("text") or example.get("content") or ""
        if not t and isinstance(example, dict):
            # first string field
            for v in example.values():
                if isinstance(v, str) and v.strip():
                    t = v
                    break
        return {"text": t}

    mappers = {
        "alpaca": alpaca_map,
        "sharegpt": sharegpt_map,
        "chat": chat_map,
        "text": text_map,
        "hf": alpaca_map,  # try alpaca; HF datasets often alpaca-like
    }
    fn = mappers.get(dataset_format, alpaca_map)

    # Detect columns for HF datasets that use messages
    cols = set(dataset.column_names) if hasattr(dataset, "column_names") else set()
    if "messages" in cols:
        fn = chat_map
    elif "conversations" in cols:
        fn = sharegpt_map
    elif "text" in cols and dataset_format in ("text", "hf"):
        fn = text_map
    elif "instruction" in cols:
        fn = alpaca_map

    return dataset.map(fn, remove_columns=[c for c in dataset.column_names if c != "text"])


def load_raw_dataset(path: str, dataset_format: str):
    from datasets import load_dataset

    p = Path(path)
    if not p.exists():
        log(f"Loading HF dataset: {path}")
        ds = load_dataset(path)
        if "train" in ds:
            return ds["train"]
        # first split
        return ds[list(ds.keys())[0]]

    if p.is_dir():
        # try common files
        for name in ("train.jsonl", "train.json", "data.jsonl", "data.json"):
            cand = p / name
            if cand.exists():
                p = cand
                break
        else:
            jsonls = list(p.glob("**/*.jsonl")) + list(p.glob("**/*.json"))
            if not jsonls:
                raise FileNotFoundError(f"No json/jsonl in {path}")
            p = jsonls[0]

    suf = p.suffix.lower()
    if suf == ".jsonl" or suf == ".jsonlines":
        return load_dataset("json", data_files=str(p), split="train")
    if suf == ".json":
        return load_dataset("json", data_files=str(p), split="train")
    if suf == ".csv":
        return load_dataset("csv", data_files=str(p), split="train")
    if suf in {".txt", ".md"}:
        return load_dataset("text", data_files=str(p), split="train")
    return load_dataset("json", data_files=str(p), split="train")


def train_with_unsloth(cfg: dict) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    import torch

    max_seq = int(cfg.get("max_seq_length", 2048))
    model_id = cfg["model_id"]
    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    method = cfg.get("method", "qlora")

    log(f"Loading model with Unsloth: {model_id} (4bit={load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=load_in_4bit if method != "full" else False,
    )

    if method != "full":
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(cfg.get("lora_r", 16)),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=int(cfg.get("seed", 42)),
        )

    raw = load_raw_dataset(cfg["dataset_path"], cfg.get("dataset_format", "alpaca"))
    ds = format_dataset(raw, cfg.get("dataset_format", "alpaca"), tokenizer)
    ds = inject_identity(ds, cfg)
    log(f"Dataset samples (včetně identity): {len(ds)}")

    out = cfg.get("output_dir", "/workspace/adapter")
    os.makedirs(out, exist_ok=True)

    epochs = float(cfg.get("epochs", 1.0))
    max_steps = int(cfg.get("max_steps", -1))
    batch = int(cfg.get("batch_size", 2))
    gas = int(cfg.get("grad_accum", 4))
    lr = float(cfg.get("learning_rate", 2e-4))

    # Prefer SFTConfig (newer TRL); fall back to TrainingArguments
    try:
        args = SFTConfig(
            output_dir=out,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=gas,
            warmup_ratio=0.03,
            num_train_epochs=epochs if max_steps < 0 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=int(cfg.get("seed", 42)),
            save_strategy="epoch",
            report_to="none",
            max_seq_length=max_seq,
            dataset_text_field="text",
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            args=args,
        )
    except TypeError:
        args = TrainingArguments(
            output_dir=out,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=gas,
            warmup_ratio=0.03,
            num_train_epochs=epochs if max_steps < 0 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            learning_rate=lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=int(cfg.get("seed", 42)),
            save_strategy="epoch",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=max_seq,
            packing=False,
            args=args,
        )

    log("Starting training…")
    trainer.train()
    log("Saving adapter…")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    merged_dir = cfg.get("merged_dir", "/workspace/merged")
    os.makedirs(merged_dir, exist_ok=True)
    if method != "full":
        log("Merging LoRA into base (16-bit) for GGUF export…")
        try:
            model.save_pretrained_merged(
                merged_dir,
                tokenizer,
                save_method="merged_16bit",
            )
        except Exception as e:
            log(f"save_pretrained_merged failed ({e}); trying manual merge…")
            try:
                merged = model.merge_and_unload()
                merged.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
            except Exception as e2:
                log(f"Merge failed: {e2}. Adapter-only saved at {out}")
    else:
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    log("Training finished successfully.")


def train_with_peft_fallback(cfg: dict) -> None:
    """Fallback without Unsloth (slower, more VRAM)."""
    import os
    import torch

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    # Prevent optional vision codepaths from pulling broken torchvision ops
    os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None  # type: ignore
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    max_seq = int(cfg.get("max_seq_length", 2048))
    model_id = cfg["model_id"]
    method = cfg.get("method", "qlora")
    load_in_4bit = method == "qlora" or bool(cfg.get("load_in_4bit", True))

    bnb = None
    if load_in_4bit and method != "full":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    log(f"Loading model (PEFT fallback): {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    if method != "full":
        model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)

    raw = load_raw_dataset(cfg["dataset_path"], cfg.get("dataset_format", "alpaca"))
    ds = format_dataset(raw, cfg.get("dataset_format", "alpaca"), tokenizer)
    ds = inject_identity(ds, cfg)
    log(f"Dataset samples (včetně identity): {len(ds)}")

    out = cfg.get("output_dir", "/workspace/adapter")
    os.makedirs(out, exist_ok=True)
    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 4)),
        num_train_epochs=float(cfg.get("epochs", 1.0)),
        max_steps=int(cfg.get("max_steps", -1)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        seed=int(cfg.get("seed", 42)),
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=max_seq,
        tokenizer=tokenizer,
        args=args,
        packing=False,
    )
    trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    merged_dir = cfg.get("merged_dir", "/workspace/merged")
    os.makedirs(merged_dir, exist_ok=True)
    if method != "full":
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
        except Exception as e:
            log(f"merge_and_unload failed: {e}")
    else:
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
    log("PEFT training finished.")


def _stack_selfcheck() -> None:
    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu0={torch.cuda.get_device_name(0)}")
    try:
        import torchvision
        log(f"torchvision={torchvision.__version__}")
    except Exception as e:
        log(f"torchvision import warning: {e}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: train_inside_container.py <config.json>", file=sys.stderr)
        return 2
    cfg = load_config(sys.argv[1])
    log(f"Config: {json.dumps(cfg, indent=2)}")
    try:
        _stack_selfcheck()
        use_unsloth = False
        try:
            import unsloth  # noqa: F401
            use_unsloth = True
            log("Using Unsloth backend")
        except Exception as e:
            # Catch ImportError AND runtime ImportError from version checks
            log(f"Unsloth not available ({type(e).__name__}: {e})")
            log("Falling back to PEFT/transformers backend")
            use_unsloth = False

        if use_unsloth:
            train_with_unsloth(cfg)
        else:
            train_with_peft_fallback(cfg)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
