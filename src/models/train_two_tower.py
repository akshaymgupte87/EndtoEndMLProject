"""Train an ID-only two-tower recommender on a bounded user sample."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StructField, StructType
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.models.sampling import UniformNegativeSampler, build_seen_items
from src.models.two_tower import PairwiseTrainingDataset, TwoTowerRecommender, bpr_loss


@dataclass(frozen=True)
class TrainingConfig:
    embedding_dim: int = 16
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if self.embedding_dim < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("embedding_dim, batch_size, and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")


@dataclass
class TrainingResult:
    model: TwoTowerRecommender
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float


def _fixed_validation_dataset(
    train_seen: dict[int, set[int]],
    validation_pairs: list[tuple[int, int]],
    num_items: int,
    seed: int,
) -> TensorDataset:
    """Create repeatable validation triples, excluding each target from negatives."""
    validation_seen = {user_idx: set(items) for user_idx, items in train_seen.items()}
    for user_idx, item_idx in validation_pairs:
        validation_seen.setdefault(user_idx, set()).add(item_idx)
    sampler = UniformNegativeSampler(num_items, validation_seen, seed=seed)
    user_indices: list[int] = []
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for user_idx, positive_item_idx in validation_pairs:
        user_indices.append(user_idx)
        positive_indices.append(positive_item_idx)
        negative_indices.append(sampler.sample(user_idx))
    return TensorDataset(
        torch.tensor(user_indices, dtype=torch.long),
        torch.tensor(positive_indices, dtype=torch.long),
        torch.tensor(negative_indices, dtype=torch.long),
    )


def _evaluate_pairwise_loss(
    model: TwoTowerRecommender,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for users, positive_items, negative_items in loader:
            users = users.to(device)
            positive_items = positive_items.to(device)
            negative_items = negative_items.to(device)
            positive_scores, negative_scores = model(users, positive_items, negative_items)
            loss = bpr_loss(positive_scores, negative_scores)
            batch_size = users.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
    return total_loss / total_examples


def _save_checkpoint(
    path: Path,
    model: TwoTowerRecommender,
    config: TrainingConfig,
    best_epoch: int,
    best_validation_loss: float,
    model_user_ids: list[int],
    item_mapping_path: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_config": {
            "num_users": model.num_users,
            "num_items": model.num_items,
            "embedding_dim": model.embedding_dim,
        },
        "training_config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "model_user_ids": model_user_ids,
        "item_mapping_path": item_mapping_path,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
    }
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_two_tower_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[TwoTowerRecommender, dict[str, Any]]:
    """Load the best model checkpoint and its small JSON-compatible metadata."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    model_config = payload["model_config"]
    model = TwoTowerRecommender(**model_config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata


def train_two_tower(
    train_pairs: list[tuple[int, int]],
    validation_pairs: list[tuple[int, int]],
    num_users: int,
    num_items: int,
    config: TrainingConfig,
    *,
    model_user_ids: list[int] | None = None,
    item_mapping_path: str | None = None,
    checkpoint_path: Path | None = None,
) -> TrainingResult:
    """Fit on training triples and restore the epoch with lowest validation loss."""
    config.validate()
    if not train_pairs or not validation_pairs:
        raise ValueError("both training and validation pairs must be non-empty")
    if model_user_ids is None:
        model_user_ids = list(range(num_users))
    if len(model_user_ids) != num_users or len(set(model_user_ids)) != num_users:
        raise ValueError("model_user_ids must contain one unique source ID per model user")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)

    train_seen = build_seen_items(train_pairs)
    train_sampler = UniformNegativeSampler(num_items, train_seen, seed=config.seed)
    train_dataset = PairwiseTrainingDataset(train_pairs, train_sampler)
    shuffle_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
    )

    validation_dataset = _fixed_validation_dataset(
        train_seen, validation_pairs, num_items, seed=config.seed + 1
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False
    )

    model = TwoTowerRecommender(num_users, num_items, config.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for users, positive_items, negative_items in train_loader:
            users = users.to(device)
            positive_items = positive_items.to(device)
            negative_items = negative_items.to(device)
            optimizer.zero_grad(set_to_none=True)
            positive_scores, negative_scores = model(users, positive_items, negative_items)
            loss = bpr_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()
            batch_size = users.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        mean_train_loss = total_loss / total_examples
        validation_loss = _evaluate_pairwise_loss(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch={epoch} train_loss={mean_train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    if checkpoint_path is not None:
        _save_checkpoint(
            checkpoint_path,
            model,
            config,
            best_epoch,
            best_validation_loss,
            model_user_ids,
            item_mapping_path,
        )
    return TrainingResult(model, history, best_epoch, best_validation_loss)


def _read_sampled_pairs(
    spark: SparkSession,
    data_path: Path,
    max_users: int,
    max_train_pairs: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], int]:
    """Read full histories for a seeded subset of train/validation users."""
    train = spark.read.parquet(str(data_path / "train")).select("user_idx", "item_idx")
    validation = spark.read.parquet(str(data_path / "validation")).select(
        "user_idx", "item_idx"
    )
    item_mapping = spark.read.parquet(str(data_path / "item_mapping")).select("item_idx")
    item_stats = item_mapping.agg(
        F.count("*").alias("count"),
        F.countDistinct("item_idx").alias("distinct_count"),
        F.min("item_idx").alias("minimum"),
        F.max("item_idx").alias("maximum"),
    ).first()
    num_items = int(item_stats["count"])
    if (
        num_items < 2
        or item_stats["distinct_count"] != num_items
        or item_stats["minimum"] != 0
        or item_stats["maximum"] != num_items - 1
    ):
        raise ValueError("item_mapping must contain contiguous, unique indices starting at zero")

    eligible_users = (
        train.select("user_idx")
        .distinct()
        .join(validation.select("user_idx").distinct(), "user_idx", "inner")
        .orderBy(F.rand(seed))
        .limit(max_users)
        .collect()
    )
    candidate_ids = [int(row.user_idx) for row in eligible_users]
    if not candidate_ids:
        raise ValueError("no users occur in both encoded train and validation splits")
    candidate_schema = StructType([StructField("user_idx", LongType(), nullable=False)])
    candidate_frame = spark.createDataFrame([(user_id,) for user_id in candidate_ids], candidate_schema)

    counts = {
        int(row.user_idx): int(row["count"])
        for row in train.join(candidate_frame, "user_idx")
        .groupBy("user_idx")
        .count()
        .collect()
    }
    selected_ids: list[int] = []
    selected_rows = 0
    for user_id in candidate_ids:
        user_rows = counts[user_id]
        if selected_rows + user_rows <= max_train_pairs:
            selected_ids.append(user_id)
            selected_rows += user_rows
    if not selected_ids:
        raise ValueError(
            "no complete user history fits --max-train-pairs; increase the limit"
        )

    selected_schema = StructType([StructField("user_idx", LongType(), nullable=False)])
    selected_frame = spark.createDataFrame([(user_id,) for user_id in selected_ids], selected_schema)
    train_rows = (
        train.join(selected_frame, "user_idx", "inner")
        .select("user_idx", "item_idx")
        .collect()
    )
    validation_rows = (
        validation.join(selected_frame, "user_idx", "inner")
        .select("user_idx", "item_idx")
        .collect()
    )

    source_to_local = {source_id: local_id for local_id, source_id in enumerate(selected_ids)}
    train_pairs = [
        (source_to_local[int(row.user_idx)], int(row.item_idx)) for row in train_rows
    ]
    validation_pairs = [
        (source_to_local[int(row.user_idx)], int(row.item_idx)) for row in validation_rows
    ]
    return train_pairs, validation_pairs, selected_ids, num_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Encoded two-tower data directory")
    parser.add_argument("--output", type=Path, required=True, help="Model artifact directory")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--max-train-pairs", type=int, default=100_000)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    args = parser.parse_args()
    if args.max_users < 1 or args.max_train_pairs < 1 or args.shuffle_partitions < 1:
        parser.error("max-users, max-train-pairs, and shuffle-partitions must be positive")

    config = TrainingConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    config.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.appName("amazon-recommender-two-tower-train")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        train_pairs, validation_pairs, model_user_ids, num_items = _read_sampled_pairs(
            spark,
            args.data,
            args.max_users,
            args.max_train_pairs,
            args.seed,
        )
        result = train_two_tower(
            train_pairs,
            validation_pairs,
            num_users=len(model_user_ids),
            num_items=num_items,
            config=config,
            model_user_ids=model_user_ids,
            item_mapping_path=str(args.data / "item_mapping"),
            checkpoint_path=args.output / "best_model.pt",
        )
        (args.output / "history.json").write_text(
            json.dumps(
                {
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "train_pairs": len(train_pairs),
                    "validation_pairs": len(validation_pairs),
                    "model_users": len(model_user_ids),
                    "catalog_items": num_items,
                    "epochs": result.history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Best epoch: {result.best_epoch}")
        print(f"Best validation BPR loss: {result.best_validation_loss:.6f}")
        print(f"Checkpoint written to: {args.output / 'best_model.pt'}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
