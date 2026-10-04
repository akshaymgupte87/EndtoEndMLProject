"""Small unit tests for the first preprocessing iteration."""

from pyspark.sql import SparkSession

from src.data.prepare import filter_interactions, temporal_split


def spark_session() -> SparkSession:
    return (
        SparkSession.builder.master("local[2]")
        .appName("recommender-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def test_filter_interactions_removes_sparse_entities() -> None:
    spark = spark_session()
    try:
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
    finally:
        spark.stop()


def test_temporal_split_uses_last_two_interactions() -> None:
    spark = spark_session()
    try:
        rows = [("u1", "i1", 1), ("u1", "i2", 2), ("u1", "i3", 3)]
        frame = spark.createDataFrame(rows, ["user_id", "item_id", "timestamp"])
        train, validation, test = temporal_split(frame)
        assert [row.item_id for row in train.collect()] == ["i1"]
        assert [row.item_id for row in validation.collect()] == ["i2"]
        assert [row.item_id for row in test.collect()] == ["i3"]
    finally:
        spark.stop()
