"""FastAPI web control panel for the fine-tune pipeline."""

from __future__ import annotations

import asyncio
import os
import secrets
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .interactive import DATASET_FORMAT_HELP, KNOWN_MODELS
from .job_manager import manager

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"


class TrainRequest(BaseModel):
    model_id: str = "unsloth/llama-3.2-1b"
    model_params_b: Optional[float] = None
    dataset_path: str = "./data/test_multilang_code/train.jsonl"
    dataset_format: str = "alpaca"
    output_dir: str = "./outputs"
    framework: str = "unsloth"
    method: str = "full"
    lora_r: int = 64
    lora_alpha: int = 128
    max_seq_length: int = 2048
    batch_size: int = 1
    grad_accum: int = 8
    epochs: float = 3.0
    learning_rate: float = 5e-5
    max_steps: int = -1
    gguf_quant: str = "q4_k_m"
    ollama_name: str = "muj-model"
    identity_name: str = "Můj Model"
    teach_identity: bool = True
    identity_repeat: int = 3
    max_train_hours: float = 720.0
    max_cost_usd: float = 999999.0
    gpu_hourly_usd: float = 0.35
    dry_run: bool = False
    skip_setup: bool = False
    skip_train: bool = False
    skip_gguf: bool = False
    skip_ollama: bool = False
    allow_over_limit: bool = True
    # friendly UI fields
    train_mode: str = "from_scratch"
    uncensored: bool = True
    no_limits: bool = True
    system_prompt: Optional[str] = None


def create_app(access_token: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Učení AI modelu", version="1.1.0")
    token = access_token if access_token is not None else (os.environ.get("LLM_UI_TOKEN") or "")

    async def verify(
        authorization: Optional[str] = Header(default=None),
        x_token: Optional[str] = Header(default=None),
    ) -> None:
        if not token:
            return
        provided = None
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        elif x_token:
            provided = x_token.strip()
        if provided != token:
            raise HTTPException(status_code=401, detail="Neplatný access token")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>Missing web/index.html</h1>", status_code=500)
        html = index_path.read_text(encoding="utf-8")
        html = html.replace("{{AUTH_REQUIRED}}", "true" if token else "false")
        return HTMLResponse(html)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "auth_required": bool(token)}

    @app.get("/api/models")
    async def models(_auth: None = Depends(verify)) -> list:
        return [
            {"id": k, "params_b": v}
            for k, v in sorted(KNOWN_MODELS.items(), key=lambda x: (x[1], x[0]))
        ]

    @app.get("/api/formats")
    async def formats(_auth: None = Depends(verify)) -> dict:
        return {"markdown": DATASET_FORMAT_HELP}

    @app.get("/api/status")
    async def status(_auth: None = Depends(verify)) -> dict:
        return manager.status()

    @app.get("/api/env")
    async def env_info(_auth: None = Depends(verify)) -> dict:
        return manager.get_env_snapshot()

    @app.get("/api/logs")
    async def logs(after: int = 0, _auth: None = Depends(verify)) -> dict:
        seq, lines = manager.get_logs(after)
        return {"seq": seq, "lines": lines, "total": seq}

    @app.get("/api/logs/full")
    async def logs_full(_auth: None = Depends(verify)) -> dict:
        """All logs as one string (for copy button)."""
        text = manager.get_logs_text()
        return {
            "text": text,
            "bytes": len(text.encode("utf-8")),
            "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
        }

    @app.get("/api/logs/download")
    async def logs_download(_auth: None = Depends(verify)) -> PlainTextResponse:
        text = manager.get_logs_text()
        return PlainTextResponse(
            text,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="llm-training-logs.txt"',
            },
        )

    @app.get("/api/logs/stream")
    async def logs_stream(request: Request, _auth: None = Depends(verify)) -> StreamingResponse:
        async def gen():
            last = 0
            ev = manager.subscribe()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    seq, lines = manager.get_logs(last)
                    if lines:
                        for line in lines:
                            safe = line.replace("\r", "").replace("\n", "\\n")
                            yield f"data: {safe}\n\n"
                        last = seq
                    yield f"event: ping\ndata: {seq}\n\n"
                    await asyncio.get_event_loop().run_in_executor(None, ev.wait, 2.0)
                    ev.clear()
            finally:
                manager.unsubscribe(ev)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/analyze")
    async def analyze(payload: TrainRequest, _auth: None = Depends(verify)) -> dict:
        try:
            return manager.analyze_only(payload.model_dump())
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/start")
    async def start(payload: TrainRequest, _auth: None = Depends(verify)) -> dict:
        try:
            return manager.start_job(payload.model_dump())
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/cancel")
    async def cancel(_auth: None = Depends(verify)) -> dict:
        manager.request_cancel()
        return {"ok": True, "status": manager.status()}

    @app.post("/api/setup")
    async def setup(_auth: None = Depends(verify)) -> dict:
        def _run():
            return manager.prepare_env(install_packages=True)

        try:
            loop = asyncio.get_event_loop()
            env = await loop.run_in_executor(None, _run)
            return env
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def generate_token() -> str:
    return secrets.token_urlsafe(24)
