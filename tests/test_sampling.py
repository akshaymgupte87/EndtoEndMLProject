"""Tests for training-only pairwise negative sampling."""

from unittest.mock import Mock

import pytest

from src.models.sampling import UniformNegativeSampler, build_seen_items


def test_build_seen_items_deduplicates_positive_pairs() -> None:
    assert build_seen_items([(0, 2), (0, 2), (0, 1), (1, 2)]) == {
        0: {1, 2},
        1: {2},
    }


def test_sampler_never_returns_a_training_positive() -> None:
    seen = build_seen_items([(0, 0), (0, 1), (1, 2)])
    sampler = UniformNegativeSampler(num_items=4, seen_items_by_user=seen, seed=7)

    for _ in range(100):
        assert sampler.sample(0) in {2, 3}
        assert sampler.sample(1) in {0, 1, 3}


def test_sampler_is_repeatable_with_the_same_seed() -> None:
    seen = build_seen_items([(0, 0), (0, 1)])
    first = UniformNegativeSampler(num_items=5, seen_items_by_user=seen, seed=19)
    second = UniformNegativeSampler(num_items=5, seen_items_by_user=seen, seed=19)

    assert [first.sample(0) for _ in range(20)] == [second.sample(0) for _ in range(20)]


def test_sampler_rejects_a_user_who_has_seen_the_full_catalog() -> None:
    with pytest.raises(ValueError, match="full item catalog"):
        UniformNegativeSampler(num_items=2, seen_items_by_user={0: {0, 1}})


def test_sampler_rejects_unknown_users() -> None:
    sampler = UniformNegativeSampler(num_items=2, seen_items_by_user={0: {0}})

    with pytest.raises(KeyError, match="no training interactions"):
        sampler.sample(1)


@pytest.mark.parametrize("value", [0.5, "0", True])
def test_build_seen_items_rejects_non_integer_indices(value: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        build_seen_items([(value, 0)])
    with pytest.raises(TypeError, match="integer"):
        build_seen_items([(0, value)])


@pytest.mark.parametrize("pair", [(-1, 0), (0, -1)])
def test_build_seen_items_rejects_negative_indices(pair: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_seen_items([pair])


@pytest.mark.parametrize("count", [1.5, True])
def test_sampler_rejects_non_integer_catalog_size(count: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        UniformNegativeSampler(count, {0: {0}})


@pytest.mark.parametrize("history", [{0: set()}, {-1: {0}}, {0: {-1}}, {0: {3}}])
def test_sampler_rejects_invalid_history(history: dict[int, set[int]]) -> None:
    with pytest.raises(ValueError):
        UniformNegativeSampler(3, history)


@pytest.mark.parametrize("history", [{0.5: {0}}, {0: {0.5}}, {0: {True}}])
def test_sampler_rejects_non_integer_history(history: dict) -> None:
    with pytest.raises(TypeError, match="integer"):
        UniformNegativeSampler(3, history)


def test_sampler_keeps_a_read_only_snapshot_of_history() -> None:
    history = {0: {0}}
    sampler = UniformNegativeSampler(2, history)
    history[0].add(1)
    history.clear()

    assert sampler.seen_items_by_user == {0: frozenset({0})}
    assert sampler.sample(0) == 1
    with pytest.raises(TypeError):
        sampler.seen_items_by_user[0] = frozenset({0, 1})


@pytest.mark.parametrize("rank, expected", [(0, 1), (1, 4), (2, 6)])
def test_dense_fallback_maps_each_available_rank_uniformly(
    monkeypatch: pytest.MonkeyPatch, rank: int, expected: int
) -> None:
    sampler = UniformNegativeSampler(7, {0: {0, 2, 3, 5}})
    draw = Mock(side_effect=[0] * 64 + [rank])
    monkeypatch.setattr(sampler.rng, "randrange", draw)

    assert sampler.sample(0) == expected
    assert draw.call_count == 65
    draw.assert_called_with(3)


@pytest.mark.parametrize("user", [-1, 0.0, True])
def test_sample_rejects_invalid_user_indices(user: object) -> None:
    sampler = UniformNegativeSampler(2, {0: {0}, 1: {0}})
    with pytest.raises((TypeError, ValueError)):
        sampler.sample(user)
