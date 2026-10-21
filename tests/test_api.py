"""Small request-level tests for recommendation serving."""

import json
import hashlib
import asyncio
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import pytest
import torch

from src.api.app import create_app
from src.models.train_two_tower import TrainingConfig, _save_checkpoint
from src.models.two_tower import TwoTowerRecommender


async def _request(app, path: str, method: str = "GET") -> tuple[int, bytes]:
    """Send one small HTTP request directly through the ASGI application."""
    status_code = 500
    body = bytearray()

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    parsed = urlsplit(path)
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": parsed.path, "raw_path": parsed.path.encode(),
        "query_string": parsed.query.encode(), "root_path": "", "headers": [],
        "client": ("127.0.0.1", 12345), "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    return status_code, bytes(body)


@pytest.fixture
def model_bundle(tmp_path):
    model = TwoTowerRecommender(num_users=2, num_items=3, embedding_dim=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.item_embedding.weight.copy_(torch.tensor([[5.0, 0.0], [3.0, 0.0], [0.0, 4.0]]))
    directory = tmp_path / "model"
    directory.mkdir()
    _save_checkpoint(
        directory / "best_model.pt", model, TrainingConfig(embedding_dim=2),
        best_epoch=1, best_validation_loss=0.5, model_user_ids=[77, 88],
        item_mapping_path=None,
    )
    np.save(directory / "item_vectors.npy", model.item_embedding.weight.detach().numpy())
    (directory / "item_ids.json").write_text(json.dumps(["item-a", "item-b", "item-c"]))
    (directory / "user_ids.json").write_text(json.dumps([77, 88]))
    (directory / "seen_items.json").write_text(json.dumps({"0": [0], "1": [2]}))
    (directory / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_type": "two_tower_id_embeddings",
                "model_version": "sha256:abc123",
                "pipeline_run_id": "pipeline-123",
                "training_mlflow_run_id": "train-456",
                "evaluation_mlflow_run_id": "eval-789",
                "trained_at_utc": "2026-10-05T12:00:00+00:00",
                "seed": 42,
            }
        )
    )
    files = ("best_model.pt", "item_vectors.npy", "item_ids.json", "user_ids.json", "seen_items.json", "model_metadata.json")
    manifest = {"files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_api_serves_known_user_hides_seen_and_rejects_unknown(model_bundle) -> None:
    app = create_app(model_bundle)
    async def run():
        async with app.router.lifespan_context(app):
            status, body = await _request(app, "/health")
            assert status == 200 and json.loads(body) == {"status": "ok"}
            assert (await _request(app, "/ready"))[0] == 200
            status, body = await _request(app, "/model-info")
            assert status == 200
            info = json.loads(body)
            assert info["model_version"] == "sha256:abc123"
            assert info["pipeline_run_id"] == "pipeline-123"
            assert info["training_mlflow_run_id"] == "train-456"
            assert info["evaluation_mlflow_run_id"] == "eval-789"
            assert info["seed"] == 42
            assert info["users"] == 2 and info["items"] == 3
            status, body = await _request(app, "/recommendations/77?limit=2")
            assert status == 200
            assert json.loads(body)["items"] == ["item-b", "item-c"]
            status, body = await _request(app, "/recommendations/999")
            assert status == 404
            assert (await _request(app, "/recommendations/77?limit=101"))[0] == 422
            status, body = await _request(app, "/metrics")
            assert status == 200 and b"recommendation_requests_total" in body

    asyncio.run(run())


def test_api_falls_back_when_redis_is_down(model_bundle) -> None:
    class BrokenRedis:
        def get(self, _key):
            raise ConnectionError("redis stopped")

        def setex(self, *_args):
            raise ConnectionError("redis stopped")

        def close(self):
            pass

    app = create_app(model_bundle)
    async def run():
        async with app.router.lifespan_context(app):
            app.state.runtime["redis"] = BrokenRedis()
            status, body = await _request(app, "/recommendations/77?limit=3")
            assert status == 200
            assert json.loads(body)["items"] == ["item-b", "item-c"]
            status, body = await _request(app, "/metrics")
            assert status == 200 and b"recommendation_cache_errors_total" in body

    asyncio.run(run())


def test_api_configures_short_redis_connect_and_read_timeouts(
    model_bundle, monkeypatch
) -> None:
    options = {}

    class FakeRedisClient:
        @staticmethod
        def from_url(url, **kwargs):
            options.update(kwargs)
            return SimpleNamespace(close=lambda: None)

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedisClient))
    monkeypatch.setenv("REDIS_URL", "redis://cache.invalid:6379/0")
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            assert app.state.runtime["redis"] is not None

    asyncio.run(run())
    assert options["socket_connect_timeout"] == 0.2
    assert options["socket_timeout"] == 0.2
    assert options["retry_on_timeout"] is False


def test_api_uses_popularity_for_unknown_user_when_bundle_has_fallback(model_bundle) -> None:
    (model_bundle / "popularity_item_indices.json").write_text(json.dumps([2, 1, 0]))
    manifest = json.loads((model_bundle / "manifest.json").read_text())
    manifest["files"]["popularity_item_indices.json"] = hashlib.sha256(
        (model_bundle / "popularity_item_indices.json").read_bytes()
    ).hexdigest()
    (model_bundle / "manifest.json").write_text(json.dumps(manifest))
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            status, body = await _request(app, "/recommendations/999?limit=2")
            assert status == 200
            assert json.loads(body) == {
                "user_idx": 999,
                "items": ["item-c", "item-b"],
                "model": "popularity_fallback",
            }

    asyncio.run(run())


def test_api_rejects_non_integer_external_user_ids(model_bundle) -> None:
    """The current contract is an integer model index, not a stable external ID."""
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            status, body = await _request(app, "/recommendations/customer-77")
            assert status == 422
            assert b"Input should be a valid integer" in body

    asyncio.run(run())


def test_api_rejects_limits_above_service_cap(model_bundle) -> None:
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            status, _ = await _request(app, "/recommendations/77?limit=101")
            assert status == 422
            status, body = await _request(app, "/recommendations/77?limit=100")
            assert status == 200
            assert len(json.loads(body)["items"]) <= 100

    asyncio.run(run())


def test_api_returns_bounded_empty_list_when_user_has_seen_catalog(model_bundle) -> None:
    """Sparse or exhausted histories must not leak seen items or overrun the catalog."""
    seen_path = model_bundle / "seen_items.json"
    seen_path.write_text(json.dumps({"0": [0, 1, 2], "1": [2]}))
    manifest = json.loads((model_bundle / "manifest.json").read_text())
    manifest["files"]["seen_items.json"] = hashlib.sha256(seen_path.read_bytes()).hexdigest()
    (model_bundle / "manifest.json").write_text(json.dumps(manifest))
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            status, body = await _request(app, "/recommendations/77?limit=100")
            assert status == 200
            assert json.loads(body)["items"] == []

    asyncio.run(run())


def test_api_is_not_ready_when_model_bundle_cannot_be_loaded(tmp_path) -> None:
    app = create_app(tmp_path / "missing-model")

    async def run():
        with pytest.raises(FileNotFoundError):
            async with app.router.lifespan_context(app):
                pass
        status, body = await _request(app, "/ready")
        assert status == 503
        assert json.loads(body)["detail"] == "model is not loaded"

    asyncio.run(run())


def test_api_recommends_deterministically_on_toy_fixture(model_bundle) -> None:
    """A fixed fixture is a mechanics check, never evidence of user relevance."""
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            first = await _request(app, "/recommendations/77?limit=2")
            second = await _request(app, "/recommendations/77?limit=2")
            assert first[0] == second[0] == 200
            assert json.loads(first[1])["items"] == json.loads(second[1])["items"]
            assert json.loads(first[1])["items"] == ["item-b", "item-c"]

    asyncio.run(run())


def test_api_uses_popularity_policy_for_known_user_and_filters_seen(model_bundle):
    metadata_path = model_bundle / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["serving_policy"] = "popularity"
    metadata_path.write_text(json.dumps(metadata))
    (model_bundle / "popularity_item_indices.json").write_text(json.dumps([0, 2, 1]))
    manifest = json.loads((model_bundle / "manifest.json").read_text())
    manifest["files"]["model_metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest["files"]["popularity_item_indices.json"] = hashlib.sha256(
        (model_bundle / "popularity_item_indices.json").read_bytes()
    ).hexdigest()
    (model_bundle / "manifest.json").write_text(json.dumps(manifest))
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            status, body = await _request(app, "/recommendations/77?limit=2")
            assert status == 200
            assert json.loads(body)["items"] == ["item-c", "item-b"]
            assert json.loads(body)["model"] == "popularity_fallback"

    asyncio.run(run())


def test_api_rejects_manifest_missing_required_checksum(model_bundle):
    (model_bundle / "manifest.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(ValueError, match="missing required"):
        from src.api.app import load_runtime

        load_runtime(model_bundle)


def test_bedrock_explanation_uses_selected_recommendations(model_bundle, monkeypatch):
    calls = {}

    class FakeBedrockRuntime:
        def converse(self, **kwargs):
            calls.update(kwargs)
            return {"output": {"message": {"content": [{"text": "These are ranked suggestions."}]}}}

    monkeypatch.setenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            client=lambda service, region_name: (
                FakeBedrockRuntime()
                if service == "bedrock-runtime" and region_name == "us-west-2"
                else pytest.fail("unexpected Bedrock client configuration")
            )
        ),
    )
    app = create_app(model_bundle)

    async def run():
        async with app.router.lifespan_context(app):
            status, body = await _request(app, "/explain/77?limit=2", method="POST")
            assert status == 200
            result = json.loads(body)
            assert result["items"] == ["item-b", "item-c"]
            assert result["explanation"] == "These are ranked suggestions."
            assert result["explanation_model"] == "amazon.nova-micro-v1:0"

    asyncio.run(run())
    assert calls["messages"][0]["content"][0]["text"] == json.dumps(
        {"items": ["item-b", "item-c"]}
    )
