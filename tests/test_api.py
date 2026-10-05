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


async def _request(app, path: str) -> tuple[int, bytes]:
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
        "method": "GET", "scheme": "http", "path": parsed.path, "raw_path": parsed.path.encode(),
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
            assert (await _request(app, "/recommendations/999"))[0] == 404
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
