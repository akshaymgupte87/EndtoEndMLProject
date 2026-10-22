"""Tests for the bounded pairwise two-tower training loop."""

import torch
import pytest

from src.pipeline.batch_job import export_serving_artifacts

from src.models.train_two_tower import (
    TrainingConfig,
    load_two_tower_checkpoint,
    train_two_tower,
)


def test_training_loop_saves_and_reloads_best_checkpoint(tmp_path) -> None:
    train_pairs = [(0, 0), (0, 1), (1, 2), (1, 3)]
    validation_pairs = [(0, 4), (1, 5)]
    checkpoint_path = tmp_path / "best_model.pt"
    config = TrainingConfig(
        embedding_dim=8,
        batch_size=4,
        epochs=6,
        learning_rate=0.03,
        weight_decay=0.0,
        seed=13,
    )

    result = train_two_tower(
        train_pairs,
        validation_pairs,
        num_users=2,
        num_items=8,
        config=config,
        model_user_ids=[101, 202],
        item_mapping_path="item_mapping",
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert len(result.history) == config.epochs
    assert 1 <= result.best_epoch <= config.epochs
    best_record = min(result.history, key=lambda row: row["validation_loss"])
    assert result.best_epoch == best_record["epoch"]
    assert result.best_validation_loss == best_record["validation_loss"]
    assert torch.isfinite(torch.tensor(result.best_validation_loss))

    restored, metadata = load_two_tower_checkpoint(checkpoint_path)
    probe_users = torch.tensor([0, 1], dtype=torch.long)
    probe_items = torch.tensor([6, 7], dtype=torch.long)
    assert torch.equal(
        result.model.score(probe_users, probe_items),
        restored.score(probe_users, probe_items),
    )
    assert metadata["model_user_ids"] == [101, 202]
    assert metadata["item_mapping_path"] == "item_mapping"
    assert metadata["best_epoch"] == result.best_epoch


@pytest.mark.parametrize(
    ("train_pairs", "validation_pairs", "num_users", "num_items", "error"),
    [
        ([(2, 0)], [(0, 1)], 2, 3, "training user index is outside"),
        ([(0, 0)], [(0, 3)], 1, 3, "validation item index is outside"),
        ([(0, 0)], [(1, 1)], 2, 3, "every validation user must have training"),
        ([(0, 0)], [(0, 0)], 1, 3, "must not overlap"),
        ([(0.5, 0)], [(0, 1)], 1, 3, "indices must be integers"),
        ([(0, 0)], [(0, True)], 1, 3, "indices must be integers"),
    ],
)
def test_training_rejects_invalid_pair_boundaries(
    train_pairs, validation_pairs, num_users, num_items, error
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        train_two_tower(
            train_pairs,
            validation_pairs,
            num_users=num_users,
            num_items=num_items,
            config=TrainingConfig(embedding_dim=2, batch_size=2, epochs=1),
        )


@pytest.mark.parametrize("config", [
    TrainingConfig(learning_rate=float("nan")),
    TrainingConfig(learning_rate=float("inf")),
    TrainingConfig(weight_decay=float("nan")),
    TrainingConfig(weight_decay=float("inf")),
    TrainingConfig(learning_rate=True),
    TrainingConfig(weight_decay="0.1"),
    TrainingConfig(embedding_dim=1.5),
])
def test_training_config_rejects_non_finite_or_non_integer_values(config) -> None:
    with pytest.raises((TypeError, ValueError)):
        config.validate()


@pytest.mark.parametrize("fallback", [[True], [1.5], [-1], [8], [1, 1]])
def test_serving_export_rejects_invalid_popularity_indices_before_spark(
    tmp_path, monkeypatch, fallback
) -> None:
    from src.pipeline import batch_job
    from src.models.two_tower import TwoTowerRecommender

    model = TwoTowerRecommender(num_users=1, num_items=3, embedding_dim=2)
    monkeypatch.setattr(
        batch_job,
        "load_two_tower_checkpoint",
        lambda *_args, **_kwargs: (model, {"model_user_ids": [10]}),
    )

    class FakeSparkSession:
        class builder:
            @staticmethod
            def appName(*_args):
                return FakeSparkSession.builder

            @staticmethod
            def master(*_args):
                pytest.fail("Spark should not start for invalid fallback indices")

    class FakeSparkModule:
        SparkSession = FakeSparkSession

    monkeypatch.setitem(__import__("sys").modules, "pyspark.sql", FakeSparkModule)
    with pytest.raises(ValueError, match="unique catalog indices"):
        export_serving_artifacts(
            tmp_path / "data", tmp_path / "output", tmp_path / "checkpoint.pt",
            popularity_item_indices=fallback,
        )
