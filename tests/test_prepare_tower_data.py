"""Spark behavior tests for two-tower ID mapping and split encoding."""

from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from src.models.prepare_tower_data import build_id_mapping, encode_split


def test_build_id_mapping_is_sorted_distinct_and_zero_based(spark: SparkSession) -> None:
    frame = spark.createDataFrame(
        [("user-b",), ("user-a",), ("user-b",), (None,)],
        "user_id string",
    )

    mapping = build_id_mapping(frame, "user_id", "user_idx")

    assert [(row.user_id, row.user_idx) for row in mapping.collect()] == [
        ("user-a", 0),
        ("user-b", 1),
    ]
    assert isinstance(mapping.schema["user_idx"].dataType, LongType)


def test_encode_split_drops_pairs_with_unknown_user_or_item(
    spark: SparkSession,
) -> None:
    train = spark.createDataFrame(
        [("u1", "i1"), ("u2", "i2")],
        "user_id string, item_id string",
    )
    user_mapping = build_id_mapping(train, "user_id", "user_idx")
    item_mapping = build_id_mapping(train, "item_id", "item_idx")
    held_out = spark.createDataFrame(
        [
            ("u1", "i1"),  # known user and item
            ("new-user", "i1"),  # unknown user
            ("u1", "new-item"),  # unknown item
            ("u1", "i1"),  # duplicate known pair is preserved
        ],
        "user_id string, item_id string",
    )

    encoded, excluded = encode_split(held_out, user_mapping, item_mapping)

    assert [(row.user_idx, row.item_idx) for row in encoded.collect()] == [
        (0, 0),
        (0, 0),
    ]
    assert excluded == 2
