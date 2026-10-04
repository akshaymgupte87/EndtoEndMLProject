"""Run a tiny synthetic two-tower training demo without Spark or Amazon data."""

from __future__ import annotations

import torch

from src.models.train_two_tower import TrainingConfig, train_two_tower


def main() -> None:
    # Two small groups share interaction patterns. The held-out pair for each
    # user is an item seen by another user in the same group.
    train_pairs = [
        (0, 0), (0, 2),
        (1, 0), (1, 1),
        (2, 3), (2, 5),
        (3, 3), (3, 4),
    ]
    validation_pairs = [(0, 1), (1, 2), (2, 4), (3, 5)]
    item_names = [
        "shared-group-A",
        "group-A-item-1",
        "group-A-item-2",
        "shared-group-B",
        "group-B-item-1",
        "group-B-item-2",
        "unobserved-item",
    ]
    result = train_two_tower(
        train_pairs,
        validation_pairs,
        num_users=4,
        num_items=len(item_names),
        config=TrainingConfig(
            embedding_dim=8,
            batch_size=8,
            epochs=20,
            learning_rate=0.03,
            weight_decay=0.0,
            seed=42,
        ),
    )

    print(
        "\nSynthetic demonstration only; sampled validation BPR loss is not "
        "a product-quality metric."
    )
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best sampled validation BPR loss: {result.best_validation_loss:.6f}")
    result.model.eval()
    all_items = torch.arange(len(item_names), dtype=torch.long)
    with torch.inference_mode():
        for user_idx in range(result.model.num_users):
            users = torch.full_like(all_items, user_idx)
            scores = result.model.score(users, all_items)
            top_items = torch.argsort(scores, descending=True)[:3].tolist()
            recommendations = ", ".join(item_names[item_idx] for item_idx in top_items)
            print(f"user {user_idx} top 3: {recommendations}")


if __name__ == "__main__":
    main()
