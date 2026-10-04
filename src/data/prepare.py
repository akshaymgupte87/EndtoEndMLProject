"""Prepare a manageable Amazon Reviews 2023 interaction dataset.

The script intentionally keeps the first iteration small and local. It accepts
JSONL review files, filters sparse users/items, and creates chronological
train/validation/test splits suitable for recommender evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Review JSONL path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional input row limit")
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=32,
        help="Spark shuffle partitions for local runs",
    )
    return parser.parse_args()


def load_reviews(spark: SparkSession, path: Path, limit: int | None) -> DataFrame:
    reviews = spark.read.json(str(path))
    if limit is not None:
        reviews = reviews.limit(limit)

    required = {"user_id", "parent_asin", "timestamp"}
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    return (
        reviews.select(
            F.col("user_id").cast("string"),
            F.col("parent_asin").cast("string").alias("item_id"),
            F.col("timestamp").cast("long"),
            F.col("rating").cast("double"),
        )
        .where("user_id IS NOT NULL AND item_id IS NOT NULL AND timestamp IS NOT NULL")
        .dropDuplicates(["user_id", "item_id", "timestamp"])
    )


def filter_interactions(
    interactions: DataFrame, min_user: int, min_item: int
) -> DataFrame:
    users = (
        interactions.groupBy("user_id")
        .count()
        .where(F.col("count") >= min_user)
        .select("user_id")
    )
    items = (
        interactions.groupBy("item_id")
        .count()
        .where(F.col("count") >= min_item)
        .select("item_id")
    )
    return interactions.join(users, "user_id").join(items, "item_id")


def rank_interactions(interactions: DataFrame) -> DataFrame:
    order = Window.partitionBy("user_id").orderBy("timestamp", "item_id")
    users = Window.partitionBy("user_id")
    return (
        interactions
        .withColumn("position", F.row_number().over(order))
        .withColumn("n", F.count("*").over(users))
    )


def split_ranked(ranked: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    train = ranked.where(F.col("position") < F.col("n") - 1).drop("position", "n")
    validation = ranked.where(F.col("position") == F.col("n") - 1).drop("position", "n")
    test = ranked.where(F.col("position") == F.col("n")).drop("position", "n")
    return train, validation, test


def temporal_split(interactions: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    return split_ranked(rank_interactions(interactions))


def write_dataset(interactions: DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    interactions.write.mode("overwrite").parquet(str(output / "interactions"))
    # The saved dataset cuts off the expensive JSON/filter/join lineage.
    prepared = interactions.sparkSession.read.parquet(str(output / "interactions"))
    ranked = rank_interactions(prepared).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        for name, split in zip(("train", "validation", "test"), split_ranked(ranked)):
            split.write.mode("overwrite").parquet(str(output / name))
    finally:
        ranked.unpersist(blocking=True)


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("amazon-recommender-prepare")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        reviews = load_reviews(spark, args.input, args.limit).persist(StorageLevel.MEMORY_AND_DISK)
        interactions = filter_interactions(
            reviews, args.min_user_interactions, args.min_item_interactions
        )
        print(f"Writing prepared interactions to: {args.output}")
        write_dataset(interactions, args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

