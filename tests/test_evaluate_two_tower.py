"""Tests for full-catalog ranking and top-K metric calculations."""

import math

import pytest
import torch

from src.models.evaluate_two_tower import (
    _load_evaluation_examples,
    _load_validation_examples,
    _metrics_by_cohort,
    ranking_metrics_at_k,
    recommend_top_k,
    recommend_popular_top_k,
)
from src.models.two_tower import TwoTowerRecommender


def test_ranking_metrics_at_k_uses_macro_user_averages() -> None:
    ranked = {0: [4, 2, 1], 1: [2, 3, 4]}
    relevant = {0: {2}, 1: {1, 4}}

    metrics = ranking_metrics_at_k(ranked, relevant, k=2)

    assert metrics["users"] == 2
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.25)
    assert metrics["ndcg"] == pytest.approx(1 / (2 * math.log2(3)))


def test_ranking_metrics_rejects_empty_relevance() -> None:
    with pytest.raises(ValueError, match="at least one user"):
        ranking_metrics_at_k({0: [1]}, {0: set()}, k=10)


def test_recommend_top_k_excludes_each_users_training_items() -> None:
    model = TwoTowerRecommender(num_users=2, num_items=5, embedding_dim=1)
    with torch.no_grad():
        model.user_embedding.weight.fill_(1.0)
        model.item_embedding.weight.copy_(torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0]]))

    ranked = recommend_top_k(
        model,
        user_indices=[0, 1],
        seen_items_by_user={0: {0, 2}, 1: set()},
        k=2,
        batch_size=1,
    )

    assert ranked == {0: [1, 3], 1: [0, 1]}
    assert not ({0, 2} & set(ranked[0]))


def test_recommend_top_k_respects_shared_candidate_catalog() -> None:
    model = TwoTowerRecommender(num_users=1, num_items=5, embedding_dim=1)
    with torch.no_grad():
        model.user_embedding.weight.fill_(1.0)
        model.item_embedding.weight.copy_(torch.tensor([[10.0], [9.0], [8.0], [7.0], [6.0]]))

    ranked = recommend_top_k(
        model,
        user_indices=[0],
        seen_items_by_user={0: {3}},
        k=2,
        candidate_items=[1, 3, 4],
    )

    assert ranked == {0: [1, 4]}


def test_popularity_ranking_also_excludes_each_users_training_items() -> None:
    ranked = recommend_popular_top_k(
        user_indices=[0, 1],
        popularity_order=[0, 1, 2, 3],
        seen_items_by_user={0: {0, 1}, 1: set()},
        k=2,
    )

    assert ranked == {0: [2, 3], 1: [0, 1]}


def test_popularity_ranking_respects_shared_candidate_catalog() -> None:
    ranked = recommend_popular_top_k(
        user_indices=[0],
        popularity_order=[0, 1, 2, 3],
        seen_items_by_user={0: set()},
        k=2,
        candidate_items=[2, 3],
    )

    assert ranked == {0: [2, 3]}


def test_metrics_by_cohort_reports_user_history_and_target_popularity_groups() -> None:
    ranked = {
        "two_tower": {0: [3, 4], 1: [4, 2], 2: [8, 9]},
        "popularity": {0: [4, 3], 1: [2, 4], 2: [8, 9]},
    }
    relevant = {0: {3}, 1: {4}, 2: {8}}
    seen = {0: {0, 1}, 1: set(range(10)), 2: set(range(21))}
    item_counts = {3: 1, 4: 4, 8: 11}

    metrics = _metrics_by_cohort(ranked, relevant, seen, item_counts, [1, 2])

    assert metrics["sparse_1_to_5_train_interactions"]["users"] == 1
    assert metrics["medium_6_to_20_train_interactions"]["users"] == 1
    assert metrics["active_over_20_train_interactions"]["users"] == 1
    assert metrics["tail_0_to_2_sample_train_interactions"]["targets"] == 1
    assert metrics["mid_3_to_10_sample_train_interactions"]["targets"] == 1
    assert metrics["popular_over_10_sample_train_interactions"]["targets"] == 1
    assert metrics["tail_0_to_2_sample_train_interactions"]["metrics_at_k"][
        "two_tower"
    ]["1"]["recall"] == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_recommend_top_k_rejects_non_positive_k(k: int) -> None:
    model = TwoTowerRecommender(num_users=1, num_items=2, embedding_dim=1)
    with pytest.raises(ValueError, match="k must be positive"):
        recommend_top_k(model, [0], {0: set()}, k)


def test_recommend_top_k_rejects_out_of_range_seen_item() -> None:
    model = TwoTowerRecommender(num_users=1, num_items=2, embedding_dim=1)
    with pytest.raises(ValueError, match="seen item index"):
        recommend_top_k(model, [0], {0: {2}}, k=1)


def test_validation_loader_keeps_selected_users_and_skips_train_seen_targets(
    spark, tmp_path
) -> None:
    data_path = tmp_path / "encoded"
    data_path.mkdir()
    spark.createDataFrame(
        [(10, 0), (10, 1), (20, 2), (30, 4)],
        "user_idx long, item_idx long",
    ).write.parquet(str(data_path / "train"))
    spark.createDataFrame(
        [(10, 1), (10, 3), (20, 4), (30, 0)],
        "user_idx long, item_idx long",
    ).write.parquet(str(data_path / "validation"))
    # No test directory is created: validation loading must not read test data.

    train_seen, relevant, skipped, item_counts = _load_validation_examples(
        spark,
        data_path,
        model_user_ids=[10, 20],
    )

    assert train_seen == {0: {0, 1}, 1: {2}}
    assert relevant == {0: {3}, 1: {4}}
    assert skipped == 1
    assert item_counts == {0: 1, 1: 1, 2: 1}


def test_evaluation_loader_reads_only_the_requested_holdout_split(spark, tmp_path) -> None:
    data_path = tmp_path / "encoded"
    data_path.mkdir()
    spark.createDataFrame(
        [(10, 0), (10, 1)], "user_idx long, item_idx long"
    ).write.parquet(str(data_path / "train"))
    spark.createDataFrame(
        [(10, 2)], "user_idx long, item_idx long"
    ).write.parquet(str(data_path / "validation"))
    spark.createDataFrame(
        [(10, 3)], "user_idx long, item_idx long"
    ).write.parquet(str(data_path / "test"))

    _, validation_targets, _, _ = _load_evaluation_examples(
        spark, data_path, [10], split="validation"
    )
    _, test_targets, _, _ = _load_evaluation_examples(
        spark, data_path, [10], split="test"
    )

    assert validation_targets == {0: {2}}
    assert test_targets == {0: {3}}
    with pytest.raises(ValueError, match="split must be"):
        _load_evaluation_examples(spark, data_path, [10], split="train")
