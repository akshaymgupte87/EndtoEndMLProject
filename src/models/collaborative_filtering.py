"""Train and evaluate an implicit-feedback Spark ALS recommender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SparkSession, functions as F

from src.models.evaluate_two_tower import (
    _load_validation_examples,
    ranking_metrics_at_k,
    recommend_popular_top_k,
)
from src.models.experiment_tracking import log_experiment_run


def recommend_top_k_als(
    user_factors: dict[int, np.ndarray],
    item_factors: dict[int, np.ndarray],
    user_indices: Sequence[int],
    seen_items_by_user: dict[int, set[int]],
    k: int,
    batch_size: int = 128,
    candidate_items: Sequence[int] | None = None,
) -> dict[int, list[int]]:
    """Score known ALS item vectors in batches and hide each user's train items."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not user_factors or not item_factors:
        raise ValueError("ALS factors must contain at least one user and item")
    dimensions = {len(vector) for vector in (*user_factors.values(), *item_factors.values())}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("user and item factors must have one shared non-zero dimension")

    dimension = dimensions.pop()
    selected_items = sorted(item_factors) if candidate_items is None else list(candidate_items)
    if not selected_items or len(selected_items) != len(set(selected_items)):
        raise ValueError("candidate items must be non-empty and unique")
    missing_items = set(selected_items) - item_factors.keys()
    if missing_items:
        raise ValueError(f"ALS has no factor for {len(missing_items)} candidate items")
    item_ids = np.asarray(selected_items, dtype=np.int64)
    item_matrix = np.stack([item_factors[int(item_id)] for item_id in item_ids])
    results: dict[int, list[int]] = {}
    users = list(user_indices)
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        vectors: list[np.ndarray] = []
        for user_id in batch:
            if user_id not in user_factors:
                raise ValueError(f"ALS has no learned factor for user {user_id}")
            vector = np.asarray(user_factors[user_id], dtype=np.float32)
            if vector.shape != (dimension,):
                raise ValueError("user factor has an invalid shape")
            vectors.append(vector)
        scores = np.stack(vectors) @ item_matrix.T
        for row, user_id in enumerate(batch):
            seen = seen_items_by_user.get(user_id, set())
            available_mask = ~np.isin(item_ids, np.fromiter(seen, dtype=np.int64))
            candidate_ids = item_ids[available_mask]
            candidate_scores = scores[row, available_mask]
            count = min(k, len(candidate_ids))
            if count == 0:
                results[user_id] = []
                continue
            if count < len(candidate_ids):
                selected = np.argpartition(candidate_scores, -count)[-count:]
                selected = selected[np.argsort(-candidate_scores[selected], kind="stable")]
            else:
                selected = np.argsort(-candidate_scores, kind="stable")
            results[user_id] = candidate_ids[selected].tolist()
    return results


def _factor_map(frame: Any) -> dict[int, np.ndarray]:
    return {
        int(row.id): np.asarray(row.features, dtype=np.float32)
        for row in frame.select("id", "features").collect()
    }


def _localize_user_factors(
    source_factors: dict[int, np.ndarray], user_indices: Sequence[int]
) -> dict[int, np.ndarray]:
    """Align source ALS user IDs with the local row IDs used by evaluation."""
    missing = [source_id for source_id in user_indices if source_id not in source_factors]
    if missing:
        raise ValueError(f"ALS has no training factor for {len(missing)} validation users")
    return {
        local_id: source_factors[source_id]
        for local_id, source_id in enumerate(user_indices)
    }


def _archive_model(model_dir: Path, archive_path: Path) -> Path:
    """Package Spark's directory model as a single MLflow-loggable artifact."""
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(model_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(model_dir.parent))
    return archive_path


def fit_implicit_als(
    train: Any,
    rank: int = 16,
    max_iter: int = 10,
    reg_param: float = 0.1,
    alpha: float = 1.0,
    seed: int = 42,
) -> ALSModel:
    """Fit implicit ALS using interaction counts as confidence weights."""
    if rank < 1 or max_iter < 1 or reg_param < 0 or alpha <= 0:
        raise ValueError("rank/max_iter must be positive, reg_param non-negative, alpha positive")
    ratings = train.groupBy("user_idx", "item_idx").agg(F.count("*").cast("float").alias("rating"))
    estimator = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        implicitPrefs=True,
        coldStartStrategy="drop",
        nonnegative=True,
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        alpha=alpha,
        seed=seed,
    )
    return estimator.fit(ratings)


def evaluate_als_validation(
    spark: SparkSession,
    model: ALSModel,
    data_path: Path,
    user_indices: list[int],
    training_frame: Any,
    candidate_items: list[int] | None = None,
    ks: Sequence[int] = (10, 20),
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluate ALS against popularity for supplied users using validation only."""
    if not user_indices or len(set(user_indices)) != len(user_indices):
        raise ValueError("provide a non-empty list of unique evaluation users")
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("provide positive integer K values")
    train_seen, relevant, skipped_seen, _ = _load_validation_examples(
        spark, data_path, user_indices
    )
    users = sorted(user for user, targets in relevant.items() if targets)
    if not users:
        raise ValueError("no validation targets remain for ALS evaluation")
    source_user_factors = _factor_map(model.userFactors)
    # Validation helper IDs are local row numbers; Spark ALS factor IDs are source user_idx values.
    user_factors = _localize_user_factors(source_user_factors, user_indices)
    item_factors = _factor_map(model.itemFactors)
    train_item_counts = {
        int(row.item_idx): int(row.interactions)
        for row in training_frame
        .groupBy("item_idx")
        .agg(F.count("*").alias("interactions"))
        .collect()
    }
    max_k = max(ks)
    als_rankings = recommend_top_k_als(
        user_factors,
        item_factors,
        users,
        train_seen,
        max_k,
        batch_size,
        candidate_items=candidate_items,
    )
    selected_candidates = sorted(item_factors) if candidate_items is None else list(candidate_items)
    candidate_set = set(selected_candidates)
    if not selected_candidates or len(candidate_set) != len(selected_candidates):
        raise ValueError("candidate catalog must be non-empty and contain unique item IDs")
    popularity_order = sorted(
        selected_candidates,
        key=lambda item: (-train_item_counts.get(item, 0), item),
    )
    popular_rankings = recommend_popular_top_k(users, popularity_order, train_seen, max_k)
    metrics = {
        name: {str(k): ranking_metrics_at_k(ranked, relevant, k) for k in ks}
        for name, ranked in {
            "implicit_als": als_rankings,
            "popularity_baseline": popular_rankings,
        }.items()
    }
    return {
        "split": "validation",
        "protocol": "shared training-scope candidate catalog; each user's train items are removed",
        "users_evaluated": len(users),
        "validation_targets_evaluated": sum(len(relevant[user]) for user in users),
        "candidate_items": len(selected_candidates),
        "validation_targets_outside_candidate_catalog": sum(
            item not in candidate_set
            for user in users for item in relevant[user]
        ),
        "validation_rows_skipped_because_item_was_seen_in_training": skipped_seen,
        "als_user_factors": len(user_factors),
        "als_item_factors": len(item_factors),
        "validation_targets_without_als_item_factor": sum(
            item not in item_factors for user in users for item in relevant[user]
        ),
        "metrics_at_k": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for model and report")
    parser.add_argument("--user-ids", type=Path, required=True, help="JSON list from the shared comparison cohort")
    parser.add_argument("--candidate-items", type=Path, help="JSON list from the shared comparison cohort")
    parser.add_argument("--train-scope", choices=("full", "cohort"), default="full")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--reg-param", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--mlflow-experiment", default="two-tower-recommender")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    user_ids = json.loads(args.user_ids.read_text(encoding="utf-8"))
    if not isinstance(user_ids, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in user_ids):
        parser.error("--user-ids must point to a JSON array of integer user_idx values")
    candidate_items = None
    if args.candidate_items:
        candidate_items = json.loads(args.candidate_items.read_text(encoding="utf-8"))
        if not isinstance(candidate_items, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in candidate_items
        ):
            parser.error("--candidate-items must contain a JSON array of non-negative integer item_idx values")
    if args.train_scope == "cohort" and candidate_items is None:
        parser.error("--candidate-items is required when --train-scope=cohort")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("amazon-recommender-implicit-als")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        train = spark.read.parquet(str(args.data / "train"))
        if args.train_scope == "cohort":
            user_schema = train.select("user_idx").schema
            cohort_frame = spark.createDataFrame([(value,) for value in user_ids], user_schema)
            train = train.join(cohort_frame, "user_idx", "inner")
            item_schema = train.select("item_idx").schema
            candidate_frame = spark.createDataFrame(
                [(value,) for value in candidate_items], item_schema
            )
            train = train.join(candidate_frame, "item_idx", "inner")
        training_interactions = train.count()
        model = fit_implicit_als(
            train, args.rank, args.max_iter, args.reg_param, args.alpha, args.seed
        )
        model_dir = args.output / "als_model"
        model.write().save(str(model_dir))
        report = evaluate_als_validation(
            spark,
            model,
            args.data,
            user_ids,
            train,
            candidate_items,
            args.k,
            args.batch_size,
        )
        report["training_scope"] = args.train_scope
        report["training_interactions"] = training_interactions
        report["hyperparameters"] = {
            "rank": args.rank,
            "max_iter": args.max_iter,
            "reg_param": args.reg_param,
            "alpha": args.alpha,
            "seed": args.seed,
        }
        report_path = args.output / "validation_metrics.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        model_archive = _archive_model(model_dir, args.output / "als_model.zip")
        print(json.dumps(report, indent=2))
        logged = {
            f"{method}.{metric}_at_{k}": float(values[metric])
            for method, metrics in report["metrics_at_k"].items()
            for k, values in metrics.items()
            for metric in ("recall", "precision", "ndcg")
        }
        run_id = log_experiment_run(
            experiment_name=args.mlflow_experiment,
            run_name=f"implicit-als-rank-{args.rank}-seed-{args.seed}",
            parameters={
                **report["hyperparameters"],
                "training_scope": args.train_scope,
                "training_interactions": training_interactions,
                "users_evaluated": report["users_evaluated"],
                "candidate_items": report["candidate_items"],
            },
            metrics=logged,
            artifacts=[report_path, model_archive],
            tags={"stage": "validation_evaluation", "model": "implicit_als"},
            tracking_uri=args.mlflow_tracking_uri,
        )
        print(f"MLflow run: {run_id}")
        print(f"ALS model and validation report saved under: {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
