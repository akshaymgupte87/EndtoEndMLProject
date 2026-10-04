"""Tests for the pairwise dataset and ID-embedding two-tower model."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.models.sampling import UniformNegativeSampler
from src.models.two_tower import PairwiseTrainingDataset, TwoTowerRecommender, bpr_loss


def test_pairwise_dataset_returns_valid_training_triples() -> None:
    seen = {0: {0, 1}, 1: {1}}
    sampler = UniformNegativeSampler(num_items=4, seen_items_by_user=seen, seed=2)
    dataset = PairwiseTrainingDataset([(0, 0), (1, 1)], sampler)

    users, positives, negatives = zip(
        *(dataset[index] for index in range(len(dataset)))
    )

    assert torch.stack(users).tolist() == [0, 1]
    assert torch.stack(positives).tolist() == [0, 1]
    assert torch.stack(negatives)[0].item() in {2, 3}
    assert torch.stack(negatives)[1].item() in {0, 2, 3}


def test_pairwise_dataset_rejects_empty_pairs() -> None:
    sampler = UniformNegativeSampler(num_items=2, seen_items_by_user={0: {0}})

    with pytest.raises(ValueError, match="positive_pairs must not be empty"):
        PairwiseTrainingDataset([], sampler)


def test_two_tower_returns_one_dot_product_score_per_pair() -> None:
    model = TwoTowerRecommender(num_users=3, num_items=5, embedding_dim=4)
    users = torch.tensor([0, 1], dtype=torch.long)
    positives = torch.tensor([2, 3], dtype=torch.long)
    negatives = torch.tensor([1, 4], dtype=torch.long)

    positive_scores, negative_scores = model(users, positives, negatives)

    assert positive_scores.shape == (2,)
    assert negative_scores.shape == (2,)
    assert torch.isfinite(positive_scores).all()
    assert torch.isfinite(negative_scores).all()


def test_bpr_loss_is_lower_when_positives_score_higher() -> None:
    good_loss = bpr_loss(torch.tensor([3.0]), torch.tensor([-1.0]))
    bad_loss = bpr_loss(torch.tensor([-1.0]), torch.tensor([3.0]))

    assert good_loss < bad_loss


def test_bpr_loss_rejects_mismatched_batch_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        bpr_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


@pytest.mark.parametrize("pairs", [[(0,)], [(0, 0, 1)], [0]])
def test_dataset_rejects_malformed_pairs(pairs: list) -> None:
    sampler = UniformNegativeSampler(3, {0: {0}})
    with pytest.raises(ValueError, match="exactly two"):
        PairwiseTrainingDataset(pairs, sampler)


@pytest.mark.parametrize("pair", [(0.5, 0), (0, 0.5), (True, 0), (0, False)])
def test_dataset_does_not_silently_coerce_indices(pair: tuple) -> None:
    sampler = UniformNegativeSampler(3, {0: {0}, 1: {0}})
    with pytest.raises(TypeError, match="integer"):
        PairwiseTrainingDataset([pair], sampler)


@pytest.mark.parametrize("pair", [(-1, 0), (0, -1), (0, 3), (1, 0), (0, 1)])
def test_dataset_rejects_pairs_outside_sampler_history(pair: tuple[int, int]) -> None:
    sampler = UniformNegativeSampler(3, {0: {0}})
    with pytest.raises(ValueError):
        PairwiseTrainingDataset([pair], sampler)


def test_dataloader_workers_have_distinct_reproducible_negative_streams() -> None:
    def load_negatives() -> list[list[int]]:
        sampler = UniformNegativeSampler(100, {0: {0}}, seed=7)
        dataset = PairwiseTrainingDataset([(0, 0)] * 16, sampler)
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            generator=torch.Generator().manual_seed(123),
        )
        return [negatives.tolist() for _, _, negatives in loader]

    first = load_negatives()
    assert first[0] != first[1]
    assert first == load_negatives()
    assert all(0 < item < 100 for batch in first for item in batch)


def test_scores_equal_known_dot_products() -> None:
    model = TwoTowerRecommender(2, 3, embedding_dim=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 3.0]]))
        model.item_embedding.weight.copy_(
            torch.tensor([[4.0, 5.0], [2.0, -1.0], [1.0, 0.0]])
        )

    positive, negative = model(
        torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 0])
    )

    torch.testing.assert_close(positive, torch.tensor([14.0, -5.0]))
    torch.testing.assert_close(negative, torch.tensor([1.0, 11.0]))


def test_scores_reject_accidental_broadcasting() -> None:
    model = TwoTowerRecommender(2, 3)
    with pytest.raises(ValueError, match="same shape"):
        model.score(torch.tensor([0, 1]), torch.tensor([[0], [1]]))


@pytest.mark.parametrize("dimensions", [(True, 3, 2), (2, 3.5, 2), (2, 3, 0)])
def test_model_rejects_invalid_dimensions(dimensions: tuple) -> None:
    with pytest.raises((TypeError, ValueError)):
        TwoTowerRecommender(*dimensions)


@pytest.mark.parametrize(
    "bad_scores",
    [torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([float("inf")])],
)
def test_bpr_rejects_empty_and_nonfinite_scores(bad_scores: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        bpr_loss(bad_scores, torch.zeros_like(bad_scores))
    with pytest.raises(ValueError):
        bpr_loss(torch.zeros_like(bad_scores), bad_scores)


def test_bpr_rejects_integer_scores() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        bpr_loss(torch.tensor([1]), torch.tensor([0]))


def test_bpr_is_finite_for_large_finite_score_differences() -> None:
    loss = bpr_loss(torch.tensor([-1000.0, 1000.0]), torch.tensor([1000.0, -1000.0]))
    torch.testing.assert_close(loss, torch.tensor(1000.0))


def test_gradient_step_increases_positive_margin_and_reduces_loss() -> None:
    model = TwoTowerRecommender(1, 2, embedding_dim=1)
    with torch.no_grad():
        model.user_embedding.weight.fill_(1.0)
        model.item_embedding.weight.copy_(torch.tensor([[0.5], [1.0]]))
    indices = (torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    positive, negative = model(*indices)
    before_margin = (positive - negative).detach()
    loss = bpr_loss(positive, negative)
    loss.backward()

    assert model.user_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad[0].item() < 0
    assert model.item_embedding.weight.grad[1].item() > 0
    with torch.no_grad():
        for parameter in model.parameters():
            assert torch.isfinite(parameter.grad).all()
            parameter.add_(parameter.grad, alpha=-0.1)

    after_positive, after_negative = model(*indices)
    assert (after_positive - after_negative).item() > before_margin.item()
    assert bpr_loss(after_positive, after_negative).item() < loss.item()
