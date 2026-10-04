"""Build deterministic training-only ID mappings for the two-tower model."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared split directory")
    parser.add_argument("--output", type=Path, required=True, help="Encoded output directory")
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=32,
        help="Spark shuffle partitions for local runs",
    )
    return parser.parse_args()


def build_id_mapping(frame: DataFrame, source_column: str, index_column: str) -> DataFrame:
    """Map sorted distinct string IDs to stable zero-based integer indices."""
    ids = (
        frame.select(source_column)
        .where(f"{source_column} IS NOT NULL")
        .distinct()
        .orderBy(source_column)
    )
    indexed = ids.rdd.zipWithIndex().map(lambda row_and_index: (row_and_index[0][0], row_and_index[1]))
    schema = StructType(
        [
            StructField(source_column, StringType(), nullable=False),
            StructField(index_column, LongType(), nullable=False),
        ]
    )
    return frame.sparkSession.createDataFrame(indexed, schema)


def encode_split(
    frame: DataFrame, user_mapping: DataFrame, item_mapping: DataFrame
) -> tuple[DataFrame, int]:
    """Encode known IDs and return the number of rows excluded as out of vocabulary."""
    source = frame.select("user_id", "item_id")
    mapped = (
        source.join(user_mapping, "user_id", "inner")
        .join(item_mapping, "item_id", "inner")
        .select("user_idx", "item_idx")
    )
    excluded = source.count() - mapped.count()
    return mapped, excluded


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("amazon-recommender-prepare-tower-data")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        train = spark.read.parquet(str(args.data / "train"))
        validation = spark.read.parquet(str(args.data / "validation"))
        test = spark.read.parquet(str(args.data / "test"))

        # Fit the vocabularies on train only. Held-out IDs never define model rows.
        user_mapping = build_id_mapping(train, "user_id", "user_idx")
        item_mapping = build_id_mapping(train, "item_id", "item_idx")

        args.output.mkdir(parents=True, exist_ok=True)
        user_mapping.write.mode("overwrite").parquet(str(args.output / "user_mapping"))
        item_mapping.write.mode("overwrite").parquet(str(args.output / "item_mapping"))

        for split_name, split in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            encoded, excluded = encode_split(split, user_mapping, item_mapping)
            encoded.write.mode("overwrite").parquet(str(args.output / split_name))
            print(f"{split_name}: excluded {excluded} rows with IDs absent from training")

        print(f"Encoded model data written to: {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
