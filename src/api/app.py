"""FastAPI app for top-N recommendations from a packaged two-tower model."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.models.two_tower import load_two_tower_checkpoint


REQUESTS = Counter("recommendation_requests_total", "Recommendation API requests", ["status"])
LATENCY = Histogram("recommendation_request_seconds", "Recommendation request latency")
CACHE_HITS = Counter("recommendation_cache_hits_total", "Recommendation cache hits", ["backend"])
CACHE_ERRORS = Counter("recommendation_cache_errors_total", "Recommendation cache errors")


def _safe_extract(bundle: Path, destination: Path) -> None:
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            if Path(member.filename).name != member.filename:
                raise ValueError("model bundle may contain only top-level files")
        archive.extractall(destination)


def resolve_model_dir() -> Path:
    local = os.getenv("MODEL_DIR")
    if local:
        return Path(local)
    bucket, key = os.getenv("MODEL_S3_BUCKET"), os.getenv("MODEL_S3_KEY")
    if bucket and key:
        import boto3

        destination = Path(tempfile.gettempdir()) / "recommender-model"
        destination.mkdir(parents=True, exist_ok=True)
        bundle = destination / "model_bundle.zip"
        boto3.client("s3").download_file(bucket, key, str(bundle))
        extracted = destination / "bundle"
        extracted.mkdir(exist_ok=True)
        _safe_extract(bundle, extracted)
        return extracted
    raise RuntimeError("set MODEL_DIR or both MODEL_S3_BUCKET and MODEL_S3_KEY")


def load_runtime(model_dir: Path) -> dict:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        actual = hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"serving artifact checksum mismatch: {filename}")
    model, metadata = load_two_tower_checkpoint(model_dir / "best_model.pt", device="cpu")
    model.eval()
    item_ids = json.loads((model_dir / "item_ids.json").read_text(encoding="utf-8"))
    user_ids = json.loads((model_dir / "user_ids.json").read_text(encoding="utf-8"))
    seen = json.loads((model_dir / "seen_items.json").read_text(encoding="utf-8"))
    vectors = np.load(model_dir / "item_vectors.npy", mmap_mode="r")
    if vectors.shape != (model.num_items, model.embedding_dim) or len(item_ids) != model.num_items:
        raise ValueError("serving item vectors/IDs do not match model dimensions")
    if len(user_ids) != model.num_users:
        raise ValueError("serving user IDs do not match model dimensions")
    return {
        "model": model,
        "metadata": metadata,
        "item_ids": item_ids,
        "user_to_local": {int(source): local for local, source in enumerate(user_ids)},
        "seen": {int(local): set(items) for local, items in seen.items()},
        "item_vectors": torch.from_numpy(np.asarray(vectors).copy()),
        "cache": {},
        "redis": None,
    }


def create_app(model_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        path = model_dir or resolve_model_dir()
        application.state.runtime = load_runtime(path)
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            import redis

            application.state.runtime["redis"] = redis.Redis.from_url(redis_url, socket_timeout=0.2)
        try:
            yield
        finally:
            if application.state.runtime and application.state.runtime.get("redis") is not None:
                application.state.runtime["redis"].close()

    app = FastAPI(title="Amazon Reviews Recommender", version="0.1.0", lifespan=lifespan)
    app.state.runtime = None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if app.state.runtime is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return {"status": "ready"}

    @app.get("/model-info")
    def model_info() -> dict:
        runtime = app.state.runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return {
            "model_type": runtime["metadata"].get("model_type", "two_tower_id_embeddings"),
            "users": len(runtime["user_to_local"]),
            "items": len(runtime["item_ids"]),
            "embedding_dim": runtime["model"].embedding_dim,
        }

    @app.get("/recommendations/{user_idx}")
    def recommend(user_idx: int, limit: int = Query(default=10, ge=1, le=100)) -> dict:
        with LATENCY.time():
            runtime = app.state.runtime
            if runtime is None:
                REQUESTS.labels(status="not_ready").inc()
                raise HTTPException(status_code=503, detail="model is not loaded")
            local = runtime["user_to_local"].get(user_idx)
            if local is None:
                REQUESTS.labels(status="unknown_user").inc()
                raise HTTPException(status_code=404, detail="user is not in the model vocabulary")
            cache_key = (local, limit)
            redis_client = runtime.get("redis")
            redis_key = f"recommendations:{local}:{limit}"
            if redis_client is not None:
                try:
                    cached = redis_client.get(redis_key)
                    if cached:
                        CACHE_HITS.labels(backend="redis").inc()
                        REQUESTS.labels(status="ok").inc()
                        return json.loads(cached)
                except Exception:
                    # Cache is an optimization; a Redis outage must not take down serving.
                    CACHE_ERRORS.inc()
            if cache_key in runtime["cache"]:
                CACHE_HITS.labels(backend="memory").inc()
                REQUESTS.labels(status="ok").inc()
                return runtime["cache"][cache_key]
            model = runtime["model"]
            with torch.inference_mode():
                user = torch.tensor([local], dtype=torch.long)
                scores = model.encode_users(user) @ runtime["item_vectors"].T
                seen = runtime["seen"].get(local, set())
                valid_count = min(limit, model.num_items - len(seen))
                if valid_count < 1:
                    result_items: list[str] = []
                else:
                    if seen:
                        scores[0, list(seen)] = -torch.inf
                    top = torch.topk(scores[0], k=valid_count).indices.tolist()
                    result_items = [runtime["item_ids"][index] for index in top]
            result = {"user_idx": user_idx, "items": result_items, "model": "two_tower"}
            runtime["cache"][cache_key] = result
            if redis_client is not None:
                try:
                    redis_client.setex(redis_key, 300, json.dumps(result))
                except Exception:
                    CACHE_ERRORS.inc()
            REQUESTS.labels(status="ok").inc()
            return result

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if trace.get_tracer_provider().__class__.__name__ == "ProxyTracerProvider":
            provider = TracerProvider(resource=Resource.create({"service.name": "recommender-api"}))
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
            trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass
    return app


app = create_app()
