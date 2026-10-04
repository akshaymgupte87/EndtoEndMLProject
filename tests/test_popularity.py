"""Tests for the implicit-feedback popularity baseline."""

import pytest
from pyspark.sql import SparkSession

from src.models.popularity import evaluate_hit_rate, rank_popular_items


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("popularity-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_rank_popular_items_orders_counts_and_breaks_ties(spark: SparkSession) -> None:
    train = spark.createDataFrame(
        [
            ("u1", "b"),
            ("u2", "b"),
            ("u3", "a"),
            ("u4", "a"),
            ("u5", "c"),
        ],
        ["user_id", "item_id"],
    )

    rows = rank_popular_items(train, top_k=2).collect()

    assert [(row.item_id, row.interaction_count) for row in rows] == [("a", 2), ("b", 2)]


def test_rank_popular_items_rejects_non_positive_k(spark: SparkSession) -> None:
    train = spark.createDataFrame([], "user_id string, item_id string")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        rank_popular_items(train, top_k=0)


def test_hit_rate_counts_each_user_once(spark: SparkSession) -> None:
    recommendations = spark.createDataFrame([("popular",)], ["item_id"])
    held_out = spark.createDataFrame(
        [
            ("u1", "popular"),
            ("u1", "popular"),
            ("u1", "unpopular"),
            ("u2", "unpopular"),
        ],
        ["user_id", "item_id"],
    )

    assert evaluate_hit_rate(recommendations, held_out) == {
        "users": 2,
        "hits": 1,
        "hit_rate": 0.5,
    }


def test_hit_rate_handles_empty_held_out_data(spark: SparkSession) -> None:
    recommendations = spark.createDataFrame([("popular",)], ["item_id"])
    held_out = spark.createDataFrame([], "user_id string, item_id string")

    assert evaluate_hit_rate(recommendations, held_out) == {
        "users": 0,
        "hits": 0,
        "hit_rate": 0.0,
    }
