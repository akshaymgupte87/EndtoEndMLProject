"""Create a repeatable user cohort and candidate catalog for model comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import LongType, StructField, StructType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Encoded train/validation data")
    parser.add_argument("--output", type=Path, required=True, help="New versioned cohort directory")
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--max-train-pairs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    args = parser.parse_args()
    if min(args.max_users, args.max_train_pairs, args.shuffle_partitions) < 1:
        parser.error("user, interaction, and shuffle-partition limits must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("amazon-recommender-comparison-cohort")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        train = spark.read.parquet(str(args.data / "train")).select("user_idx", "item_idx")
        validation = spark.read.parquet(str(args.data / "validation")).select("user_idx", "item_idx")
        eligible = (
            train.select("user_idx").distinct()
            .join(validation.select("user_idx").distinct(), "user_idx", "inner")
            .orderBy(F.rand(args.seed))
            .limit(args.max_users)
            .collect()
        )
        candidate_ids = [int(row.user_idx) for row in eligible]
        if not candidate_ids:
            raise ValueError("no users have both training and validation interactions")

        schema = StructType([StructField("user_idx", LongType(), nullable=False)])
        candidate_frame = spark.createDataFrame([(value,) for value in candidate_ids], schema)
        counts = {
            int(row.user_idx): int(row.interactions)
            for row in train.join(candidate_frame, "user_idx")
            .groupBy("user_idx")
            .agg(F.count("*").alias("interactions"))
            .collect()
        }
        cohort: list[int] = []
        pair_count = 0
        for user_idx in candidate_ids:
            next_count = counts[user_idx]
            if pair_count + next_count <= args.max_train_pairs:
                cohort.append(user_idx)
                pair_count += next_count
        if not cohort:
            raise ValueError("no complete user history fits --max-train-pairs")

        selected = spark.createDataFrame([(value,) for value in cohort], schema)
        train_rows = (
            train.join(selected, "user_idx", "inner")
            .select("user_idx", "item_idx")
            .collect()
        )
        validation_rows = (
            validation.join(selected, "user_idx", "inner")
            .select("user_idx", "item_idx")
            .dropDuplicates(["user_idx", "item_idx"])
            .collect()
        )
        candidate_items = sorted({int(row.item_idx) for row in train_rows})
        candidate_item_set = set(candidate_items)
        covered_targets = sum(int(row.item_idx) in candidate_item_set for row in validation_rows)

        cohort_path = args.output / "cohort.json"
        candidate_path = args.output / "candidate_items.json"
        cohort_bytes = json.dumps(cohort, separators=(",", ":")).encode("utf-8")
        candidate_bytes = json.dumps(candidate_items, separators=(",", ":")).encode("utf-8")
        cohort_path.write_bytes(cohort_bytes)
        candidate_path.write_bytes(candidate_bytes)
        manifest = {
            "selection_seed": args.seed,
            "max_users": args.max_users,
            "max_train_pairs": args.max_train_pairs,
            "users": len(cohort),
            "training_interactions": len(train_rows),
            "candidate_items": len(candidate_items),
            "validation_targets": len(validation_rows),
            "validation_targets_in_candidate_catalog": covered_targets,
            "validation_targets_outside_candidate_catalog": len(validation_rows) - covered_targets,
            "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
            "candidate_items_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        print(f"Cohort files written to: {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
