"""Evaluate a saved two-tower model with full-catalog ranking metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from math import log2
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StructField, StructType
from torch import Tensor

from src.models.experiment_tracking import log_experiment_run
from src.models.train_two_tower import load_two_tower_checkpoint
from src.models.two_tower import TwoTowerRecommender


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be positive")


def recommend_top_k(
    model: TwoTowerRecommender,
    user_indices: Sequence[int],
    seen_items_by_user: Mapping[int, Set[int]],
    k: int,
    batch_size: int = 128,
    candidate_items: Sequence[int] | None = None,
) -> dict[int, list[int]]:
    """Rank all or selected catalog items for model-local users.

    User indices here address rows in this particular model checkpoint. Seen
    item sets must contain model catalog indices and should come from training
    history only. The function returns item indices, not original product IDs.
    """
    _validate_k(k)
    if k > model.num_items:
        raise ValueError("k cannot exceed the number of catalog items")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    users = list(user_indices)
    for user_idx in users:
        if isinstance(user_idx, bool) or not isinstance(user_idx, Integral):
            raise TypeError("user indices must be integers")
        if user_idx < 0 or user_idx >= model.num_users:
            raise ValueError("user index is outside this model's user table")
    if len(set(users)) != len(users):
        raise ValueError("user_indices must not contain duplicates")

    catalog = list(range(model.num_items)) if candidate_items is None else list(candidate_items)
    if not catalog or len(catalog) != len(set(catalog)):
        raise ValueError("candidate_items must be non-empty and unique")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in catalog):
        raise TypeError("candidate item indices must be integers")
    if any(item < 0 or item >= model.num_items for item in catalog):
        raise ValueError("candidate item index is outside the model catalog")
    device = next(model.parameters()).device
    item_indices = torch.tensor(catalog, dtype=torch.long, device=device)
    catalog_positions = {item_id: position for position, item_id in enumerate(catalog)}
    recommendations: dict[int, list[int]] = {}
    model.eval()
    with torch.inference_mode():
        item_vectors = model.encode_items(item_indices)
        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            user_vectors = model.encode_users(user_tensor)
            scores = user_vectors @ item_vectors.T

            for row_idx, user_idx in enumerate(batch_users):
                seen_items = seen_items_by_user.get(user_idx, frozenset())
                for item_idx in seen_items:
                    if isinstance(item_idx, bool) or not isinstance(item_idx, Integral):
                        raise TypeError("seen item indices must be integers")
                    if item_idx < 0 or item_idx >= model.num_items:
                        raise ValueError("seen item index is outside the item table")
                seen_positions = [catalog_positions[item] for item in seen_items if item in catalog_positions]
                if seen_positions:
                    scores[row_idx, seen_positions] = -torch.inf
                available = len(catalog) - len(seen_positions)
                result_size = min(k, available)
                if result_size == 0:
                    recommendations[user_idx] = []
                else:
                    top_indices = torch.topk(scores[row_idx], k=result_size).indices
                    recommendations[user_idx] = [catalog[position] for position in top_indices.cpu().tolist()]
    return recommendations


def ranking_metrics_at_k(
    ranked_items_by_user: Mapping[int, Sequence[int]],
    relevant_items_by_user: Mapping[int, Set[int]],
    k: int,
) -> dict[str, float | int]:
    """Compute macro-averaged Recall@K, Precision@K, and binary NDCG@K."""
    _validate_k(k)
    evaluated_users = [
        user_idx for user_idx, relevant in relevant_items_by_user.items() if relevant
    ]
    if not evaluated_users:
        raise ValueError("at least one user must have a relevant item")

    recall_sum = 0.0
    precision_sum = 0.0
    ndcg_sum = 0.0
    for user_idx in evaluated_users:
        relevant = set(relevant_items_by_user[user_idx])
        ranked = list(ranked_items_by_user.get(user_idx, ()))[:k]
        hits = 0
        dcg = 0.0
        found: set[int] = set()
        for rank, item_idx in enumerate(ranked, start=1):
            if item_idx in relevant and item_idx not in found:
                found.add(item_idx)
                hits += 1
                dcg += 1.0 / log2(rank + 1)

        ideal_hits = min(len(relevant), k)
        ideal_dcg = sum(
            1.0 / log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        recall_sum += hits / len(relevant)
        precision_sum += hits / k
        ndcg_sum += dcg / ideal_dcg if ideal_dcg else 0.0

    denominator = len(evaluated_users)
    return {
        "users": denominator,
        "recall": recall_sum / denominator,
        "precision": precision_sum / denominator,
        "ndcg": ndcg_sum / denominator,
    }


def recommend_popular_top_k(
    user_indices: Sequence[int],
    popularity_order: Sequence[int],
    seen_items_by_user: Mapping[int, Set[int]],
    k: int,
    candidate_items: Sequence[int] | None = None,
) -> dict[int, list[int]]:
    """Return the same global popularity ranking with each user's seen items removed."""
    _validate_k(k)
    recommendations: dict[int, list[int]] = {}
    allowed = None if candidate_items is None else set(candidate_items)
    for user_idx in user_indices:
        seen_items = seen_items_by_user.get(user_idx, frozenset())
        ranked: list[int] = []
        for item_idx in popularity_order:
            if allowed is not None and item_idx not in allowed:
                continue
            if item_idx not in seen_items:
                ranked.append(item_idx)
                if len(ranked) == k:
                    break
        recommendations[user_idx] = ranked
    return recommendations


def _metrics_by_cohort(
    ranked_by_method: Mapping[str, Mapping[int, Sequence[int]]],
    relevant_items_by_user: Mapping[int, Set[int]],
    train_seen_by_user: Mapping[int, Set[int]],
    train_item_counts: Mapping[int, int],
    ks: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Compute validation metrics by training-history and target-popularity bands."""
    history_cohorts: dict[str, set[int]] = {
        "sparse_1_to_5_train_interactions": set(),
        "medium_6_to_20_train_interactions": set(),
        "active_over_20_train_interactions": set(),
    }
    for user_idx, relevant in relevant_items_by_user.items():
        if not relevant:
            continue
        history_size = len(train_seen_by_user.get(user_idx, set()))
        if history_size <= 5:
            cohort = "sparse_1_to_5_train_interactions"
        elif history_size <= 20:
            cohort = "medium_6_to_20_train_interactions"
        else:
            cohort = "active_over_20_train_interactions"
        history_cohorts[cohort].add(user_idx)

    popularity_cohorts: dict[str, dict[int, set[int]]] = {
        "tail_0_to_2_sample_train_interactions": defaultdict(set),
        "mid_3_to_10_sample_train_interactions": defaultdict(set),
        "popular_over_10_sample_train_interactions": defaultdict(set),
    }
    for user_idx, relevant in relevant_items_by_user.items():
        for item_idx in relevant:
            count = train_item_counts.get(item_idx, 0)
            if count <= 2:
                cohort = "tail_0_to_2_sample_train_interactions"
            elif count <= 10:
                cohort = "mid_3_to_10_sample_train_interactions"
            else:
                cohort = "popular_over_10_sample_train_interactions"
            popularity_cohorts[cohort][user_idx].add(item_idx)

    result: dict[str, dict[str, Any]] = {}
    for cohort, user_ids in history_cohorts.items():
        if not user_ids:
            continue
        result[cohort] = {
            "users": len(user_ids),
            "metrics_at_k": {
                method: {
                    str(k): ranking_metrics_at_k(
                        {user: ranked.get(user, ()) for user in user_ids},
                        {user: relevant_items_by_user[user] for user in user_ids},
                        k,
                    )
                    for k in ks
                }
                for method, ranked in ranked_by_method.items()
            },
        }

    for cohort, targets_by_user in popularity_cohorts.items():
        if not targets_by_user:
            continue
        user_ids = set(targets_by_user)
        result[cohort] = {
            "users": len(user_ids),
            "targets": sum(map(len, targets_by_user.values())),
            "metrics_at_k": {
                method: {
                    str(k): ranking_metrics_at_k(
                        {user: ranked.get(user, ()) for user in user_ids},
                        targets_by_user,
                        k,
                    )
                    for k in ks
                }
                for method, ranked in ranked_by_method.items()
            },
        }
    return result


def _load_evaluation_examples(
    spark: SparkSession,
    data_path: Path,
    model_user_ids: list[int],
    split: str = "validation",
) -> tuple[dict[int, set[int]], dict[int, set[int]], int, dict[int, int]]:
    """Load training histories and targets from one explicitly selected split."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if not model_user_ids or len(set(model_user_ids)) != len(model_user_ids):
        raise ValueError("checkpoint must contain unique source user IDs")
    source_to_local = {
        source_id: local_id for local_id, source_id in enumerate(model_user_ids)
    }
    user_schema = StructType([StructField("user_idx", LongType(), nullable=False)])
    selected_users = spark.createDataFrame(
        [(user_idx,) for user_idx in model_user_ids], user_schema
    )
    train = spark.read.parquet(str(data_path / "train")).select("user_idx", "item_idx")
    targets = spark.read.parquet(str(data_path / split)).select(
        "user_idx", "item_idx"
    )
    train_rows = (
        train.join(selected_users, "user_idx", "inner")
        .select("user_idx", "item_idx")
        .collect()
    )
    target_rows = (
        targets.join(selected_users, "user_idx", "inner")
        .select("user_idx", "item_idx")
        .collect()
    )

    train_seen: dict[int, set[int]] = defaultdict(set)
    train_item_counts: dict[int, int] = defaultdict(int)
    for row in train_rows:
        local_user = source_to_local[int(row.user_idx)]
        item_idx = int(row.item_idx)
        train_seen[local_user].add(item_idx)
        train_item_counts[item_idx] += 1

    relevant_items: dict[int, set[int]] = defaultdict(set)
    skipped_seen_validation_rows = 0
    for row in target_rows:
        local_user = source_to_local[int(row.user_idx)]
        item_idx = int(row.item_idx)
        if item_idx in train_seen[local_user]:
            skipped_seen_validation_rows += 1
        else:
            relevant_items[local_user].add(item_idx)
    return (
        dict(train_seen),
        dict(relevant_items),
        skipped_seen_validation_rows,
        dict(train_item_counts),
    )


def _load_validation_examples(
    spark: SparkSession,
    data_path: Path,
    model_user_ids: list[int],
) -> tuple[dict[int, set[int]], dict[int, set[int]], int, dict[int, int]]:
    """Backward-compatible validation-only loader."""
    return _load_evaluation_examples(spark, data_path, model_user_ids, "validation")


def evaluate_validation(
    checkpoint_path: Path,
    data_path: Path,
    ks: Sequence[int],
    batch_size: int = 128,
    shuffle_partitions: int = 32,
    candidate_item_indices: Sequence[int] | None = None,
    split: str = "validation",
) -> dict[str, Any]:
    """Evaluate one explicitly chosen holdout split for checkpoint users."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if not ks:
        raise ValueError("provide at least one K value")
    for k in ks:
        _validate_k(k)
    if len(set(ks)) != len(ks):
        raise ValueError("K values must be unique")
    if shuffle_partitions < 1:
        raise ValueError("shuffle_partitions must be positive")

    model, metadata = load_two_tower_checkpoint(checkpoint_path, device="cpu")
    model_user_ids = metadata["model_user_ids"]
    if len(model_user_ids) != model.num_users:
        raise ValueError("checkpoint user ID list does not match model user table")

    spark = (
        SparkSession.builder.appName("amazon-recommender-two-tower-evaluate")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        (
            train_seen,
            relevant_items,
            skipped_seen_rows,
            train_item_counts,
        ) = _load_evaluation_examples(spark, data_path, model_user_ids, split)
    finally:
        spark.stop()

    if not relevant_items:
        raise ValueError(f"no {split} targets remain after excluding training items")
    users_to_rank = sorted(relevant_items)
    max_k = max(ks)
    ranked = recommend_top_k(
        model,
        users_to_rank,
        train_seen,
        k=max_k,
        batch_size=batch_size,
        candidate_items=candidate_item_indices,
    )
    popularity_order = sorted(
        range(model.num_items), key=lambda item_idx: (-train_item_counts.get(item_idx, 0), item_idx)
    )
    popularity_ranked = recommend_popular_top_k(
        users_to_rank, popularity_order, train_seen, max_k,
        candidate_items=candidate_item_indices,
    )
    ranked_by_method = {
        "two_tower": ranked,
        "popularity_baseline_on_same_user_sample": popularity_ranked,
    }
    candidate_item_set = (
        set(range(model.num_items))
        if candidate_item_indices is None
        else set(candidate_item_indices)
    )
    return {
        "split": split,
        "protocol": "shared candidate catalog; each user's training items are removed",
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "model_users": model.num_users,
        "users_evaluated": len(users_to_rank),
        "catalog_items": model.num_items,
        "candidate_items": model.num_items if candidate_item_indices is None else len(candidate_item_indices),
        "validation_targets_evaluated": sum(len(items) for items in relevant_items.values()),
        "validation_targets_outside_candidate_catalog": sum(
            item not in candidate_item_set
            for items in relevant_items.values() for item in items
        ) if candidate_item_indices is not None else 0,
        "validation_rows_skipped_because_item_was_seen_in_training": skipped_seen_rows,
        "metrics_at_k": {
            "two_tower": {
                str(k): ranking_metrics_at_k(ranked, relevant_items, k) for k in ks
            },
            "popularity_baseline_on_same_user_sample": {
                str(k): ranking_metrics_at_k(popularity_ranked, relevant_items, k)
                for k in ks
            },
        },
        "metrics_by_cohort": _metrics_by_cohort(
            ranked_by_method, relevant_items, train_seen, train_item_counts, ks
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Encoded two-tower data directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON metrics output path")
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help="Holdout split to read; use test only once after model selection is frozen.",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-items", type=Path, help="JSON candidate list from comparison cohort")
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--mlflow-experiment", default="two-tower-recommender")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    candidate_items = None
    if args.candidate_items:
        candidate_items = json.loads(args.candidate_items.read_text(encoding="utf-8"))
        if not isinstance(candidate_items, list):
            parser.error("--candidate-items must contain a JSON array")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in candidate_items
        ):
            parser.error("--candidate-items must contain non-negative integer item indices")
    result = evaluate_validation(
        args.checkpoint,
        args.data,
        args.k,
        batch_size=args.batch_size,
        shuffle_partitions=args.shuffle_partitions,
        candidate_item_indices=candidate_items,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"{args.split.capitalize()} metrics written to: {args.output}")
    logged_metrics: dict[str, float] = {}
    for method, metrics_by_k in result["metrics_at_k"].items():
        for k, values in metrics_by_k.items():
            for metric_name in ("recall", "precision", "ndcg"):
                logged_metrics[f"{method}.{metric_name}_at_{k}"] = float(
                    values[metric_name]
                )
    for cohort, cohort_result in result["metrics_by_cohort"].items():
        for method, metrics_by_k in cohort_result["metrics_at_k"].items():
            for k, values in metrics_by_k.items():
                for metric_name in ("recall", "precision", "ndcg"):
                    logged_metrics[
                        f"{cohort}.{method}.{metric_name}_at_{k}"
                    ] = float(values[metric_name])
    run_id = log_experiment_run(
        experiment_name=args.mlflow_experiment,
        run_name=f"evaluate-{args.split}-{args.checkpoint.parent.name}",
        parameters={
            "checkpoint": str(args.checkpoint),
            "data_path": str(args.data),
            "users_evaluated": result["users_evaluated"],
            "catalog_items": result["catalog_items"],
            "k_values": ",".join(map(str, args.k)),
        },
        metrics=logged_metrics,
        artifacts=[args.output],
        tags={"stage": f"{args.split}_evaluation", "model": "two_tower"},
        tracking_uri=args.mlflow_tracking_uri,
    )
    (args.output.parent / "evaluation_mlflow_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_name": args.mlflow_experiment,
                "tracking_uri": args.mlflow_tracking_uri,
                "split": args.split,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"MLflow run: {run_id}")


if __name__ == "__main__":
    main()
