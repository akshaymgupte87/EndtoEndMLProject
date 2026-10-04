"""Train and evaluate an implicit-feedback item popularity baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def rank_popular_items(train: DataFrame, top_k: int) -> DataFrame:
    """Rank items by training interaction count, breaking ties by item ID."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return (
        train.groupBy("item_id")
        .agg(F.count("*").alias("interaction_count"))
        .orderBy(F.desc("interaction_count"), F.asc("item_id"))
        .limit(top_k)
    )


def evaluate_hit_rate(
    recommendations: DataFrame, held_out: DataFrame
) -> dict[str, int | float]:
    """Measure whether each user's held-out item is in the global top-K list."""
    held_out_items = held_out.select("user_id", "item_id").dropDuplicates(
        ["user_id", "item_id"]
    )
    recommendation_items = recommendations.select("item_id").distinct()
    user_hits = (
        held_out_items.join(recommendation_items, "item_id", "left_semi")
        .select("user_id")
        .distinct()
        .count()
    )
    users = held_out_items.select("user_id").distinct().count()
    return {
        "users": users,
        "hits": user_hits,
        "hit_rate": user_hits / users if users else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared dataset directory")
    parser.add_argument("--k", type=int, default=10, help="Number of popular items")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    spark = (
        SparkSession.builder.appName("amazon-recommender-popularity")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        train = spark.read.parquet(str(args.data / "train"))
        validation = spark.read.parquet(str(args.data / "validation"))
        test = spark.read.parquet(str(args.data / "test"))
        recommendations = rank_popular_items(train, args.k)

        print(f"Top {args.k} items by training interaction count:")
        recommendations.show(args.k, truncate=False)
        validation_metrics = evaluate_hit_rate(recommendations, validation)
        test_metrics = evaluate_hit_rate(recommendations, test)
        print(f"Validation hit rate@{args.k}: {validation_metrics}")
        print(f"Test hit rate@{args.k}: {test_metrics}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
