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
    required_files = {
        "best_model.pt", "item_vectors.npy", "item_ids.json", "user_ids.json",
        "seen_items.json", "model_metadata.json",
    }
    checksums = manifest.get("files")
    if not isinstance(checksums, dict) or not required_files.issubset(checksums):
        raise ValueError("model manifest is missing required serving artifact checksums")
    for filename, expected in checksums.items():
        if Path(filename).name != filename:
            raise ValueError("model manifest may reference only top-level files")
        actual = hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"serving artifact checksum mismatch: {filename}")
    model, metadata = load_two_tower_checkpoint(model_dir / "best_model.pt", device="cpu")
    model.eval()
    metadata_path = model_dir / "model_metadata.json"
    serving_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    item_ids = json.loads((model_dir / "item_ids.json").read_text(encoding="utf-8"))
    user_ids = json.loads((model_dir / "user_ids.json").read_text(encoding="utf-8"))
    seen = json.loads((model_dir / "seen_items.json").read_text(encoding="utf-8"))
    if not isinstance(item_ids, list) or not isinstance(user_ids, list):
        raise ValueError("serving item and user IDs must be JSON arrays")
    vectors = np.load(model_dir / "item_vectors.npy", mmap_mode="r")
    if vectors.shape != (model.num_items, model.embedding_dim) or len(item_ids) != model.num_items:
        raise ValueError("serving item vectors/IDs do not match model dimensions")
    if not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all():
        raise ValueError("serving item vectors must contain only finite floating-point values")
    expected_vectors = model.item_embedding.weight.detach().cpu().numpy()
    if not np.allclose(vectors, expected_vectors, rtol=1e-6, atol=1e-7):
        raise ValueError("serving item vectors do not match the model checkpoint")
    if len(user_ids) != model.num_users:
        raise ValueError("serving user IDs do not match model dimensions")
    if any(isinstance(source, bool) or not isinstance(source, int) for source in user_ids):
        raise ValueError("serving user IDs must be integers")
    if len(user_ids) != len(set(user_ids)):
        raise ValueError("serving user IDs must be unique")
    if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
        raise ValueError("serving item IDs must be non-empty strings")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("serving item IDs must be unique")
    if not isinstance(seen, dict):
        raise ValueError("seen items must be a mapping from local users to item indices")
    validated_seen: dict[int, set[int]] = {}
    for raw_user, raw_items in seen.items():
        try:
            local_user = int(raw_user)
        except (TypeError, ValueError) as error:
            raise ValueError("seen-item user keys must be valid local user indices") from error
        if str(local_user) != str(raw_user) or not 0 <= local_user < model.num_users:
            raise ValueError("seen-item user index is outside the model user table")
        if not isinstance(raw_items, list):
            raise ValueError("seen items for each user must be a list of catalog indices")
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            or item < 0 or item >= model.num_items
            for item in raw_items
        ):
            raise ValueError("seen item index is outside the model catalog")
        if len(raw_items) != len(set(raw_items)):
            raise ValueError("seen item indices for each user must be unique")
        validated_seen[local_user] = set(raw_items)
    popularity_path = model_dir / "popularity_item_indices.json"
    popularity = (
        json.loads(popularity_path.read_text(encoding="utf-8"))
        if popularity_path.is_file()
        else []
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= model.num_items
        for item in popularity
    ) or len(popularity) != len(set(popularity)):
        raise ValueError("popularity fallback list contains invalid item indices")
    return {
        "model": model,
        "metadata": metadata,
        "serving_metadata": serving_metadata,
        "serving_policy": serving_metadata.get("serving_policy", "two_tower"),
        "item_ids": item_ids,
        "cache_version": str(
            serving_metadata.get("model_version")
            or manifest.get("model_version")
            or checksums["best_model.pt"]
        ),
        "user_to_local": {int(source): local for local, source in enumerate(user_ids)},
        "seen": validated_seen,
        "item_vectors": torch.from_numpy(np.asarray(vectors).copy()),
        "popularity_item_indices": popularity,
        "cache": {},
        "redis": None,
    }


def _decode_cached_recommendation(
    cached: bytes | str,
    *,
    user_idx: int,
    limit: int,
    runtime: dict,
    local_user_idx: int,
) -> dict | None:
    """Return a cache entry only when it satisfies the current serving contract."""
    try:
        result = json.loads(cached)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if not isinstance(result, dict) or set(result) != {"user_idx", "items", "model"}:
        return None
    if (
        isinstance(result["user_idx"], bool)
        or not isinstance(result["user_idx"], int)
        or result["user_idx"] != user_idx
        or result["model"] != "two_tower"
        or not isinstance(result["items"], list)
        or len(result["items"]) > limit
    ):
        return None
    items = result["items"]
    valid_ids = set(runtime["item_ids"])
    seen_indices = runtime["seen"].get(local_user_idx, set())
    seen_ids = {runtime["item_ids"][idx] for idx in seen_indices}
    if (
        any(not isinstance(item, str) or item not in valid_ids for item in items)
        or len(items) != len(set(items))
        or not seen_ids.isdisjoint(items)
    ):
        return None
    return result


def create_app(model_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        path = model_dir or resolve_model_dir()
        application.state.runtime = load_runtime(path)
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            import redis

            application.state.runtime["redis"] = redis.Redis.from_url(
                redis_url,
                socket_timeout=0.2,
                socket_connect_timeout=0.2,
                retry_on_timeout=False,
            )
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
            "model_version": runtime["serving_metadata"].get("model_version"),
            "pipeline_run_id": runtime["serving_metadata"].get("pipeline_run_id"),
            "training_mlflow_run_id": runtime["serving_metadata"].get("training_mlflow_run_id"),
            "evaluation_mlflow_run_id": runtime["serving_metadata"].get("evaluation_mlflow_run_id"),
            "trained_at_utc": runtime["serving_metadata"].get("trained_at_utc"),
            "seed": runtime["serving_metadata"].get("seed"),
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
            fallback = runtime["popularity_item_indices"]
            if local is None or runtime["serving_policy"] == "popularity":
                if not fallback:
                    if local is None:
                        REQUESTS.labels(status="unknown_user").inc()
                        raise HTTPException(status_code=404, detail="user is not in the model vocabulary")
                    REQUESTS.labels(status="no_fallback").inc()
                    raise HTTPException(status_code=503, detail="popularity fallback is not available")
                seen = runtime["seen"].get(local, set()) if local is not None else set()
                items = [runtime["item_ids"][item] for item in fallback if item not in seen][:limit]
                REQUESTS.labels(status="fallback").inc()
                return {
                    "user_idx": user_idx,
                    "items": items,
                    "model": "popularity_fallback",
                }
            cache_key = (local, limit)
            redis_client = runtime.get("redis")
            redis_key = f"recommendations:{runtime['cache_version']}:{local}:{limit}"
            if redis_client is not None:
                try:
                    cached = redis_client.get(redis_key)
                    if cached:
                        cached_result = _decode_cached_recommendation(
                            cached,
                            user_idx=user_idx,
                            limit=limit,
                            runtime=runtime,
                            local_user_idx=local,
                        )
                        if cached_result is not None:
                            CACHE_HITS.labels(backend="redis").inc()
                            REQUESTS.labels(status="ok").inc()
                            return cached_result
                        CACHE_ERRORS.inc()
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

    @app.post("/explain/{user_idx}")
    def explain_recommendations(user_idx: int, limit: int = Query(default=5, ge=1, le=10)) -> dict:
        """Explain recommendations with Bedrock when configured; recommendations remain model-backed."""
        runtime = app.state.runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        local = runtime["user_to_local"].get(user_idx)
        if local is None:
            raise HTTPException(status_code=404, detail="user is not in the model vocabulary")
        try:
            recommendation = recommend(user_idx, limit)
            model_id = os.getenv("BEDROCK_MODEL_ID")
            if not model_id:
                raise HTTPException(status_code=503, detail="Bedrock explanations are not configured")
            import boto3

            client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
            response = client.converse(
                modelId=model_id,
                system=[{"text": "Explain recommendations cautiously using only the provided item identifiers. Do not claim knowledge of user preferences."}],
                messages=[{"role": "user", "content": [{"text": json.dumps({"items": recommendation["items"]})}]}],
                inferenceConfig={"maxTokens": 180, "temperature": 0.2},
            )
            explanation = response["output"]["message"]["content"][0]["text"]
            return {**recommendation, "explanation": explanation, "explanation_model": model_id}
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=502, detail="Bedrock explanation request failed") from error

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
