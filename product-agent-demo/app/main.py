from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .mcp_client import StreamableHttpMcpProvider
from .model_client import BailianVisionClient
from .orchestrator import ConfiguredEvidenceProvider, ProductAgentOrchestrator
from .product_kb import ProductKbProvider
from .config import ModelConfig
from .lab import router as lab_router

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="L'Oréal Product Trust Agent", version="0.1.0")
app.include_router(lab_router)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "lab.html")


def _validate_image(upload: UploadFile) -> None:
    allowed = {"image/jpeg", "image/png", "image/webp", "image/heic"}
    if upload.content_type not in allowed:
        raise HTTPException(status_code=415, detail="请上传 JPG、PNG、WEBP 或 HEIC 图片")


async def _events(upload: UploadFile, model_config_json: str = "{}", prompt: str = "", skill: str = "", mcp_config_json: str = "{}"):
    _validate_image(upload)
    suffix = Path(upload.filename or "product.jpg").suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await upload.read())
        image_path = handle.name
    try:
        try:
            model_config = ModelConfig.from_dict(json.loads(model_config_json or "{}"))
            mcp_config = json.loads(mcp_config_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'配置 JSON 无效: {exc}'}, ensure_ascii=False)}\n\n"
            return
        vision = BailianVisionClient(
            api_key=model_config.api_key,
            model=model_config.model,
            base_url=model_config.base_url,
            prompt=prompt or None,
            skill=skill or None,
            timeout_seconds=model_config.timeout_seconds,
            temperature=model_config.temperature,
        )
        providers = [ProductKbProvider()]
        if mcp_config.get("url"):
            providers.append(StreamableHttpMcpProvider(
                url=mcp_config.get("url"), tool_name=mcp_config.get("tool_name"),
                api_key=mcp_config.get("api_key") or model_config.api_key,
                timeout_seconds=model_config.timeout_seconds,
            ))
        orchestrator = ProductAgentOrchestrator(vision, ConfiguredEvidenceProvider(providers))
        async for event in orchestrator.stream(image_path):
            yield f"data: {json.dumps(event.model_dump(), ensure_ascii=False)}\n\n"
    finally:
        Path(image_path).unlink(missing_ok=True)


@app.post("/api/analyze/stream")
async def analyze_stream(
    upload: UploadFile = File(...),
    model_settings: str = Form("{}"),
    prompt: str = Form(""),
    skill: str = Form(""),
    mcp_config: str = Form("{}"),
):
    return StreamingResponse(_events(upload, model_settings, prompt, skill, mcp_config), media_type="text/event-stream")


class ModelTestRequest(BaseModel):
    config_payload: dict = Field(default_factory=dict, alias="model_config")
    prompt: str
    system: str = ""
    model_config = {"populate_by_name": True}


@app.post("/api/model/test")
def model_test(request: ModelTestRequest):
    config = ModelConfig.from_dict(request.config_payload)
    try:
        client = BailianVisionClient(
            api_key=config.api_key, model=config.model, base_url=config.base_url,
            timeout_seconds=config.timeout_seconds, temperature=config.temperature,
        )
        result = client.complete_text(request.prompt, request.system)
        return {"status": "ok", "model": config.model, "request_id": client.last_request_id, "text": result}
    except Exception as exc:
        return {"status": "error", "model": config.model, "error": str(exc)}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "bailian_key_configured": bool(os.getenv("DASHSCOPE_API_KEY")),
        "mcp_configured": bool(os.getenv("BAILIAN_MCP_URL")),
    }
