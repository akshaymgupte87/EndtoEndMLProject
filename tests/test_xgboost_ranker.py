"""Tests for bounded pandas candidate construction and XGBoost ranking."""

import random

import pandas as pd
import pytest
import xgboost as xgb

from src.models.xgboost_ranker import (
    FEATURE_COLUMNS,
    _cooccurrence_counts,
    add_training_only_features,
    build_labeled_pairs,
    build_validation_candidates,
    sample_unseen_items,
)


def test_sample_unseen_items_is_repeatable_unique_and_excludes_history() -> None:
    first = sample_unseen_items({0, 1}, 20, 5, random.Random(17))
    second = sample_unseen_items({0, 1}, 20, 5, random.Random(17))

    assert first == second
    assert len(first) == len(set(first)) == 5
    assert not ({0, 1} & set(first))


def test_sample_unseen_items_rejects_impossible_request() -> None:
    with pytest.raises(ValueError, match="more negatives"):
        sample_unseen_items({0, 1, 2}, 3, 1, random.Random(1))


def test_training_table_contains_only_train_positives_and_unseen_negatives() -> None:
    seen = {0: {0, 1}, 1: {2}}
    pairs = [(0, 0), (0, 1), (1, 2)]

    table = build_labeled_pairs(pairs, seen, 8, negatives_per_positive=2, seed=7)

    assert table.groupby("user_idx")["label"].sum().to_dict() == {0: 2, 1: 1}
    for row in table.itertuples(index=False):
        if row.label == 0:
            assert row.item_idx not in seen[row.user_idx]


def test_validation_candidates_exclude_train_history_and_all_targets() -> None:
    relevant = {0: {2, 3}, 1: {4}}
    seen = {0: {0, 1}, 1: {0, 2}}

    frame, retained_targets = build_validation_candidates(
        relevant, seen, 12, negatives_per_user=5, seed=3
    )

    assert retained_targets == relevant
    assert frame.groupby("user_idx").size().to_dict() == {0: 7, 1: 6}
    for user_idx, group in frame.groupby("user_idx"):
        negatives = set(group.loc[group.label == 0, "item_idx"])
        assert not (negatives & seen[user_idx])
        assert not (negatives & relevant[user_idx])


def test_cooccurrence_counts_are_built_from_unique_user_histories() -> None:
    counts = _cooccurrence_counts({0: {0, 1, 2}, 1: {0, 1}})

    assert counts == {(0, 1): 2, (0, 2): 1, (1, 2): 1}


def test_pandas_features_use_training_counts_and_leave_one_out_affinity() -> None:
    pairs = pd.DataFrame(
        [(0, 1, 1), (0, 2, 0)], columns=["user_idx", "item_idx", "label"]
    )
    seen = {0: {0, 1, 4}}
    cooccurrence = {(0, 1): 2, (1, 4): 1}

    result = add_training_only_features(
        pairs,
        {1: 9, 2: 0},
        seen,
        cooccurrence,
        num_users=3,
        num_items=5,
        leave_one_out=True,
    )

    assert result["item_log_count"].tolist() == pytest.approx([2.3026, 0.0], abs=1e-4)
    assert result["history_affinity"].tolist() == pytest.approx([0.6931, 0.0], abs=1e-4)
    assert pd.api.types.is_integer_dtype(result["user_idx"])
    assert pd.api.types.is_integer_dtype(result["item_idx"])
    assert FEATURE_COLUMNS == ["item_log_count", "history_affinity"]


def test_xgboost_ranker_fits_grouped_candidate_rows() -> None:
    rows = []
    for user_idx in range(3):
        for item_idx in range(4):
            rows.append((user_idx, item_idx, int(item_idx == user_idx)))
    frame = pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])
    seen = {0: {0}, 1: {1}, 2: {2}}
    features = add_training_only_features(
        frame,
        {0: 2, 1: 2, 2: 2, 3: 0},
        seen,
        {},
        3,
        4,
    ).sort_values(["user_idx", "item_idx"], kind="stable")
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=3,
        max_depth=2,
        tree_method="hist",
        n_jobs=1,
    )

    ranker.fit(
        features[FEATURE_COLUMNS],
        features["label"],
        qid=features["user_idx"].astype("int64").to_numpy(),
        verbose=False,
    )

    assert ranker.predict(features[FEATURE_COLUMNS]).shape == (len(features),)
