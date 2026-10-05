"""Train an XGBoost pairwise ranker and compare it on sampled candidates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StructField, StructType

from src.models.evaluate_two_tower import ranking_metrics_at_k
from src.models.train_two_tower import load_two_tower_checkpoint

FEATURE_COLUMNS = ["item_log_count", "history_affinity"]


def sample_unseen_items(
    seen_items: set[int],
    num_items: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample unique catalog items absent from one user's training history."""
    if num_items < 1 or count < 0:
        raise ValueError("num_items must be positive and count must be non-negative")
    if any(item < 0 or item >= num_items for item in seen_items):
        raise ValueError("seen item index is outside the catalog")
    available = num_items - len(seen_items)
    if count > available:
        raise ValueError("requested more negatives than unseen catalog items")
    chosen: set[int] = set()
    while len(chosen) < count:
        item = rng.randrange(num_items)
        if item not in seen_items:
            chosen.add(item)
    return sorted(chosen)


def build_labeled_pairs(
    positive_pairs: list[tuple[int, int]],
    seen_by_user: dict[int, set[int]],
    num_items: int,
    negatives_per_positive: int,
    seed: int,
) -> pd.DataFrame:
    """Build a grouped training table from positives and unseen-item samples."""
    if not positive_pairs:
        raise ValueError("positive_pairs must not be empty")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    positives_by_user: dict[int, set[int]] = defaultdict(set)
    for user_idx, item_idx in positive_pairs:
        if item_idx not in seen_by_user.get(user_idx, set()):
            raise ValueError("every positive must appear in that user's training history")
        positives_by_user[user_idx].add(item_idx)

    rng = random.Random(seed)
    rows: list[tuple[int, int, int]] = []
    for user_idx in sorted(positives_by_user):
        positives = sorted(positives_by_user[user_idx])
        rows.extend((user_idx, item_idx, 1) for item_idx in positives)
        available_negatives = num_items - len(seen_by_user[user_idx])
        if available_negatives < 1:
            raise ValueError(f"user {user_idx} has no unseen items for negative sampling")
        negatives = sample_unseen_items(
            seen_by_user[user_idx],
            num_items,
            min(len(positives) * negatives_per_positive, available_negatives),
            rng,
        )
        rows.extend((user_idx, item_idx, 0) for item_idx in negatives)
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])


def build_validation_candidates(
    relevant_by_user: dict[int, set[int]],
    seen_by_user: dict[int, set[int]],
    num_items: int,
    negatives_per_user: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    """Create a deterministic sampled candidate list without held-out leakage."""
    if negatives_per_user < 1:
        raise ValueError("negatives_per_user must be positive")
    rng = random.Random(seed)
    rows: list[tuple[int, int, int]] = []
    evaluable_relevant: dict[int, set[int]] = {}
    for user_idx in sorted(relevant_by_user):
        targets = set(relevant_by_user[user_idx]) - seen_by_user.get(user_idx, set())
        if not targets:
            continue
        evaluable_relevant[user_idx] = targets
        rows.extend((user_idx, item_idx, 1) for item_idx in sorted(targets))
        forbidden = seen_by_user.get(user_idx, set()) | targets
        negatives = sample_unseen_items(
            forbidden,
            num_items,
            min(negatives_per_user, num_items - len(forbidden)),
            rng,
        )
        rows.extend((user_idx, item_idx, 0) for item_idx in negatives)
    if not evaluable_relevant:
        raise ValueError("no validation targets remain outside training histories")
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"]), evaluable_relevant


def add_training_only_features(
    pairs: pd.DataFrame,
    item_counts: dict[int, int],
    seen_by_user: dict[int, set[int]],
    cooccurrence_counts: dict[tuple[int, int], int],
    num_users: int,
    num_items: int,
    *,
    leave_one_out: bool = False,
) -> pd.DataFrame:
    """Build popularity and user-history co-occurrence features from train."""
    if not {"user_idx", "item_idx", "label"}.issubset(pairs.columns):
        raise ValueError("pairs must contain user_idx, item_idx, and label")
    result = pairs.copy()
    users = result["user_idx"].astype("int64")
    items = result["item_idx"].astype("int64")
    if users.empty or users.min() < 0 or users.max() >= num_users:
        raise ValueError("user index is outside the model user table")
    if items.min() < 0 or items.max() >= num_items:
        raise ValueError("item index is outside the model item table")
    result["item_log_count"] = items.map(lambda value: math.log1p(item_counts.get(value, 0)))
    labels = result["label"].astype("int8").to_numpy()
    affinities: list[float] = []
    for user_idx, item_idx, label in zip(users, items, labels):
        history = seen_by_user.get(int(user_idx), set())
        affinity = sum(
            cooccurrence_counts.get((min(int(item_idx), other), max(int(item_idx), other)), 0)
            for other in history
            if other != item_idx
        )
        if leave_one_out and label == 1:
            # Remove the current user from every target/history co-occurrence.
            affinity -= sum(1 for other in history if other != item_idx)
        affinities.append(math.log1p(max(affinity, 0)))
    result["history_affinity"] = affinities
    result["user_idx"] = users.to_numpy()
    result["item_idx"] = items.to_numpy()
    return result


def _load_cohort_rows(
    data_path: Path,
    source_user_ids: list[int],
    shuffle_partitions: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Collect only the selected users' mapped train and validation rows."""
    if shuffle_partitions < 1:
        raise ValueError("shuffle_partitions must be positive")
    source_to_local = {source_id: local_id for local_id, source_id in enumerate(source_user_ids)}
    if not source_user_ids or len(source_to_local) != len(source_user_ids):
        raise ValueError("checkpoint must contain unique source user IDs")

    spark = (
        SparkSession.builder.appName("amazon-recommender-xgboost-reranker")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        mapping = spark.read.parquet(str(data_path / "item_mapping")).select("item_idx")
        item_stats = mapping.agg(
            {"item_idx": "count"}
        ).first()
        num_items = int(item_stats[0])
        schema = StructType([StructField("user_idx", LongType(), nullable=False)])
        selected = spark.createDataFrame([(value,) for value in source_user_ids], schema)
        train_rows = (
            spark.read.parquet(str(data_path / "train"))
            .select("user_idx", "item_idx")
            .join(selected, "user_idx", "inner")
            .collect()
        )
        validation_rows = (
            spark.read.parquet(str(data_path / "validation"))
            .select("user_idx", "item_idx")
            .join(selected, "user_idx", "inner")
            .collect()
        )
    finally:
        spark.stop()

    train_pairs = [
        (source_to_local[int(row.user_idx)], int(row.item_idx)) for row in train_rows
    ]
    validation_pairs = [
        (source_to_local[int(row.user_idx)], int(row.item_idx)) for row in validation_rows
    ]
    return train_pairs, validation_pairs, num_items


def _feature_maps(
    train_pairs: list[tuple[int, int]],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    item_counts: dict[int, int] = defaultdict(int)
    for user_idx, item_idx in train_pairs:
        seen[user_idx].add(item_idx)
        item_counts[item_idx] += 1
    return dict(seen), dict(item_counts)


def _cooccurrence_counts(
    seen_by_user: dict[int, set[int]],
) -> dict[tuple[int, int], int]:
    """Count item pairs present in the same training user's history."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for history in seen_by_user.values():
        for first, second in itertools.combinations(sorted(history), 2):
            counts[(first, second)] += 1
    return dict(counts)


def _ranked_items(frame: pd.DataFrame, scores: Any) -> dict[int, list[int]]:
    scored = pd.DataFrame(
        {
            "user_idx": frame["user_idx"].astype("int64").to_numpy(),
            "item_idx": frame["item_idx"].astype("int64").to_numpy(),
            "score": scores,
        }
    )
    return {
        int(user_idx): group.sort_values(
            ["score", "item_idx"], ascending=[False, True], kind="stable"
        )["item_idx"].astype(int).tolist()
        for user_idx, group in scored.groupby("user_idx", sort=True)
    }


def run_experiment(
    data_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    negatives_per_positive: int = 2,
    validation_negatives: int = 99,
    estimators: int = 150,
    seed: int = 42,
    ks: tuple[int, ...] = (10, 20),
    shuffle_partitions: int = 32,
) -> dict[str, Any]:
    """Train the reranker on train only; compare both rankers on validation."""
    if estimators < 1 or not ks or any(k < 1 for k in ks):
        raise ValueError("estimators and K values must be positive")
    tower, metadata = load_two_tower_checkpoint(checkpoint_path, device="cpu")
    source_user_ids = metadata["model_user_ids"]
    train_pairs, validation_pairs, num_items = _load_cohort_rows(
        data_path, source_user_ids, shuffle_partitions
    )
    if num_items != tower.num_items or len(source_user_ids) != tower.num_users:
        raise ValueError("checkpoint dimensions do not match the encoded dataset")
    seen, item_counts = _feature_maps(train_pairs)
    cooccurrence_counts = _cooccurrence_counts(seen)
    positives = sorted(set(train_pairs))
    relevant: dict[int, set[int]] = defaultdict(set)
    for user_idx, item_idx in validation_pairs:
        if item_idx not in seen.get(user_idx, set()):
            relevant[user_idx].add(item_idx)
    relevant = dict(relevant)
    if not positives or not relevant:
        raise ValueError("training positives and validation targets are both required")

    train_frame = build_labeled_pairs(
        positives, seen, num_items, negatives_per_positive, seed
    )
    train_frame = add_training_only_features(
        train_frame,
        item_counts,
        seen,
        cooccurrence_counts,
        tower.num_users,
        tower.num_items,
        leave_one_out=True,
    ).sort_values(["user_idx", "item_idx"], kind="stable").reset_index(drop=True)
    train_qid = train_frame["user_idx"].astype("int64").to_numpy()
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=4,
    )
    ranker.fit(
        train_frame[FEATURE_COLUMNS],
        train_frame["label"].astype("int32"),
        qid=train_qid,
        verbose=False,
    )

    validation_frame, relevant = build_validation_candidates(
        relevant, seen, num_items, validation_negatives, seed + 1
    )
    validation_features = add_training_only_features(
        validation_frame,
        item_counts,
        seen,
        cooccurrence_counts,
        tower.num_users,
        tower.num_items,
    )
    xgb_ranked = _ranked_items(
        validation_features,
        ranker.predict(validation_features[FEATURE_COLUMNS]),
    )
    user_tensor = torch.tensor(
        validation_frame["user_idx"].to_numpy(), dtype=torch.long
    )
    item_tensor = torch.tensor(
        validation_frame["item_idx"].to_numpy(), dtype=torch.long
    )
    tower.eval()
    with torch.inference_mode():
        tower_scores = tower.score(user_tensor, item_tensor).cpu().numpy()
    tower_ranked = _ranked_items(validation_frame, tower_scores)

    report: dict[str, Any] = {
        "split": "validation",
        "test_split_read": False,
        "protocol": "same user-specific sampled candidates for XGBoost and two-tower; not full-catalog metrics",
        "seed": seed,
        "training_users": len({user for user, _ in positives}),
        "training_positive_pairs": len(positives),
        "training_candidate_rows": len(train_frame),
        "validation_users": len(relevant),
        "validation_targets": sum(map(len, relevant.values())),
        "validation_negative_samples_per_user": validation_negatives,
        "validation_candidate_rows": len(validation_frame),
        "validation_candidates_per_user_average": len(validation_frame) / len(relevant),
        "feature_columns": FEATURE_COLUMNS,
        "cooccurrence_pairs": len(cooccurrence_counts),
        "feature_importance": {
            name: float(value)
            for name, value in zip(FEATURE_COLUMNS, ranker.feature_importances_)
        },
        "metrics_at_k": {
            "xgboost_ranker": {
                str(k): ranking_metrics_at_k(xgb_ranked, relevant, k) for k in ks
            },
            "two_tower_same_candidates": {
                str(k): ranking_metrics_at_k(tower_ranked, relevant, k) for k in ks
            },
        },
        "limitations": [
            "The ranker uses item popularity and training-history co-occurrence; item text and metadata are not features.",
            "Uniform sampled candidates are cheaper than catalog-wide retrieval and produce different metrics.",
            "This is an offline ranking experiment; the API does not load or serve the XGBoost model.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    ranker.save_model(output_dir / "xgboost_ranker.json")
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Encoded dataset directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Two-tower checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--validation-negatives", type=int, default=99)
    parser.add_argument("--estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    if min(args.negatives_per_positive, args.validation_negatives, args.estimators) < 1:
        parser.error("negative counts and estimators must be positive")
    try:
        run_experiment(
            args.data,
            args.checkpoint,
            args.output,
            negatives_per_positive=args.negatives_per_positive,
            validation_negatives=args.validation_negatives,
            estimators=args.estimators,
            seed=args.seed,
            ks=tuple(args.k),
            shuffle_partitions=args.shuffle_partitions,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
