"""
Step 5 — FastAPI inference endpoint with Prometheus metrics.

Endpoints:
  POST /summarise       — single article summarisation
  POST /summarise/batch — batch summarisation
  GET  /health          — liveness probe (used by Kubernetes)
  GET  /ready           — readiness probe (used by Kubernetes)
  GET  /metrics         — Prometheus metrics scrape endpoint
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import EncoderDecoderModel, RobertaTokenizerFast

# Prometheus metrics
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus metrics definitions ───────────────────────────────────────────
# Counter: monotonically increasing — total requests, errors
REQUEST_COUNT = Counter(
    "summarise_requests_total",
    "Total number of summarisation requests",
    ["endpoint", "status"],   # labels allow slicing in Grafana
)

# Histogram: captures distribution — used for p50, p90, p95, p99 latency
REQUEST_LATENCY = Histogram(
    "summarise_request_latency_seconds",
    "Latency of summarisation requests",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Gauge: value that goes up and down — model memory, batch size
MODEL_MEMORY_MB = Gauge(
    "model_memory_mb",
    "Approximate model memory usage in MB",
)

TOKENS_PROCESSED = Counter(
    "tokens_processed_total",
    "Total number of tokens processed",
    ["direction"],   # 'input' or 'output'
)

# ── Global state — loaded once at startup ────────────────────────────────────
_model: Optional[EncoderDecoderModel] = None
_tokenizer: Optional[RobertaTokenizerFast] = None
_device: str = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    global _model, _tokenizer, _device

    model_path = os.getenv("MODEL_PATH", "/app/models/lora_merged")
    _device    = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {model_path} on {_device}…")
    t0 = time.perf_counter()
    _tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    _model     = EncoderDecoderModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    ).to(_device)
    _model.eval()
    elapsed = time.perf_counter() - t0

    # Record memory usage
    mem_mb = sum(p.nelement() * p.element_size() for p in _model.parameters()) / 1024**2
    MODEL_MEMORY_MB.set(mem_mb)
    logger.info(f"Model loaded in {elapsed:.2f}s | {mem_mb:.0f} MB")

    yield   # ← server runs here

    logger.info("Shutting down — freeing model…")
    del _model
    del _tokenizer
    if _device == "cuda":
        torch.cuda.empty_cache()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Summarisation API",
    description="LoRA fine-tuned roberta2roberta summarisation service",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────
class SummariseRequest(BaseModel):
    article: str = Field(..., min_length=10, max_length=10_000,
                         description="News article text to summarise")
    max_new_tokens: int = Field(128, ge=10, le=512)
    num_beams: int = Field(4, ge=1, le=8)


class SummariseResponse(BaseModel):
    summary: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class BatchSummariseRequest(BaseModel):
    articles: list[str] = Field(..., min_length=1, max_length=16)
    max_new_tokens: int = Field(128, ge=10, le=512)


class BatchSummariseResponse(BaseModel):
    summaries: list[str]
    latency_ms: float


# ── Core inference function ───────────────────────────────────────────────────
def _run_inference(articles: list[str], max_new_tokens: int, num_beams: int) -> tuple:
    """Tokenise, run model, decode. Returns (summaries, input_len, output_len)."""
    inputs = _tokenizer(
        articles,
        return_tensors="pt",
        max_length=512,
        padding=True,
        truncation=True,
    ).to(_device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    output_len = out_ids.shape[1]
    summaries  = _tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    return summaries, input_len, output_len


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/summarise", response_model=SummariseResponse)
async def summarise(req: SummariseRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        summaries, in_len, out_len = _run_inference(
            [req.article], req.max_new_tokens, req.num_beams
        )
        latency = (time.perf_counter() - t0) * 1000

        # Update Prometheus metrics
        REQUEST_COUNT.labels(endpoint="summarise", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="summarise").observe(latency / 1000)
        TOKENS_PROCESSED.labels(direction="input").inc(in_len)
        TOKENS_PROCESSED.labels(direction="output").inc(out_len)

        return SummariseResponse(
            summary=summaries[0],
            latency_ms=latency,
            input_tokens=in_len,
            output_tokens=out_len,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="summarise", status="error").inc()
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarise/batch", response_model=BatchSummariseResponse)
async def summarise_batch(req: BatchSummariseRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        summaries, in_len, out_len = _run_inference(
            req.articles, req.max_new_tokens, num_beams=4
        )
        latency = (time.perf_counter() - t0) * 1000

        REQUEST_COUNT.labels(endpoint="summarise_batch", status="success").inc(len(req.articles))
        REQUEST_LATENCY.labels(endpoint="summarise_batch").observe(latency / 1000)
        TOKENS_PROCESSED.labels(direction="input").inc(in_len * len(req.articles))
        TOKENS_PROCESSED.labels(direction="output").inc(out_len * len(req.articles))

        return BatchSummariseResponse(summaries=summaries, latency_ms=latency)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="summarise_batch", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Kubernetes liveness probe — am I running at all?"""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Kubernetes readiness probe — am I ready to serve traffic?"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "device": _device}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint — returns all registered metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
async def root():
    return {
        "service": "summarisation-api",
        "endpoints": ["/summarise", "/summarise/batch", "/health", "/ready", "/metrics"],
        "docs": "/docs",
    }
