from __future__ import annotations

from zipfile import ZipFile

import numpy as np
import pytest

from src.models.collaborative_filtering import (
    _archive_model,
    _localize_user_factors,
    recommend_top_k_als,
)
from src.models.summarize_comparison import mean_and_sample_std


def test_als_top_k_uses_factor_similarity_and_removes_seen_items() -> None:
    users = {
        4: np.asarray([1.0, 0.0], dtype=np.float32),
        9: np.asarray([0.0, 1.0], dtype=np.float32),
    }
    items = {
        2: np.asarray([0.1, 0.9], dtype=np.float32),
        5: np.asarray([0.8, 0.2], dtype=np.float32),
        7: np.asarray([0.7, 0.3], dtype=np.float32),
    }

    ranked = recommend_top_k_als(users, items, [4, 9], {4: {5}, 9: {2}}, k=2)

    assert ranked == {4: [7, 2], 9: [7, 5]}


def test_als_top_k_rejects_missing_user_factor() -> None:
    with pytest.raises(ValueError, match="no learned factor"):
        recommend_top_k_als(
            {4: np.asarray([1.0, 0.0], dtype=np.float32)},
            {2: np.asarray([0.1, 0.9], dtype=np.float32)},
            [9],
            {},
            k=1,
        )


def test_model_archive_contains_saved_spark_model_files(tmp_path) -> None:
    model_dir = tmp_path / "als_model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text("{}", encoding="utf-8")

    archive_path = _archive_model(model_dir, tmp_path / "als_model.zip")

    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["als_model/metadata.json"]


def test_localize_user_factors_maps_source_ids_to_evaluation_rows() -> None:
    factors = {
        81: np.asarray([1.0, 0.0], dtype=np.float32),
        305: np.asarray([0.0, 1.0], dtype=np.float32),
    }

    localized = _localize_user_factors(factors, [305, 81])

    assert localized == {0: factors[305], 1: factors[81]}


def test_mean_and_sample_std_summarizes_seed_metrics() -> None:
    mean, sample_std = mean_and_sample_std([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert sample_std == pytest.approx(1.0)
