"""Tests for the bounded pairwise two-tower training loop."""

import torch

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
