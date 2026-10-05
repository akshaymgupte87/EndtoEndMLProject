"""Small unit tests for the first preprocessing iteration."""

from pyspark.sql import SparkSession

from src.data.prepare import filter_interactions, temporal_split


def test_filter_interactions_removes_sparse_entities(spark: SparkSession) -> None:
    rows = [
        ("u1", "i1", 1),
        ("u1", "i2", 2),
        ("u2", "i1", 3),
        ("u3", "i3", 4),
    ]
    frame = spark.createDataFrame(rows, ["user_id", "item_id", "timestamp"])
    result = filter_interactions(frame, min_user=2, min_item=2)
    assert {(row.user_id, row.item_id) for row in result.collect()} == {
        ("u1", "i1")
    }


def test_temporal_split_uses_last_two_interactions(spark: SparkSession) -> None:
    rows = [("u1", "i1", 1), ("u1", "i2", 2), ("u1", "i3", 3)]
    frame = spark.createDataFrame(rows, ["user_id", "item_id", "timestamp"])
    train, validation, test = temporal_split(frame)
    assert [row.item_id for row in train.collect()] == ["i1"]
    assert [row.item_id for row in validation.collect()] == ["i2"]
    assert [row.item_id for row in test.collect()] == ["i3"]
