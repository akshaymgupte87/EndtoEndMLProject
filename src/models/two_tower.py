"""ID-embedding two-tower model and pairwise training dataset."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, get_worker_info

from src.models.sampling import UniformNegativeSampler, _validate_index


class PairwiseTrainingDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Return (user, positive item, sampled negative item) training triples.

    With DataLoader workers, seed its generator to reproduce the worker streams.
    Without workers, the sampler's seed determines the negative sequence.
    """

    def __init__(
        self,
        positive_pairs: Sequence[tuple[int, int]],
        negative_sampler: UniformNegativeSampler,
    ) -> None:
        if len(positive_pairs) == 0:
            raise ValueError("positive_pairs must not be empty")
        history = negative_sampler.seen_items_by_user
        for pair in positive_pairs:
            try:
                user_idx, item_idx = pair
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "each positive pair must contain exactly two indices"
                ) from error
            _validate_index(user_idx, "user index")
            _validate_index(item_idx, "item index")
            if item_idx >= negative_sampler.num_items:
                raise ValueError("positive item index is outside the sampler catalog")
            if user_idx not in history or item_idx not in history[user_idx]:
                raise ValueError(
                    "every positive pair must be present in the sampler's training history"
                )
        self.positive_pairs = torch.tensor(positive_pairs, dtype=torch.long)
        self.negative_sampler = negative_sampler
        self._worker_seed: int | None = None

    def __len__(self) -> int:
        return self.positive_pairs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        worker = get_worker_info()
        if worker is not None and self._worker_seed != worker.seed:
            # random.Random instances are copied into each worker. Reseeding
            # avoids identical negative streams; DataLoader's generator controls
            # reproducibility when num_workers > 0.
            self.negative_sampler.rng.seed(worker.seed)
            self._worker_seed = worker.seed
        user_idx, positive_item_idx = self.positive_pairs[index].tolist()
        negative_item_idx = self.negative_sampler.sample(user_idx)
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(positive_item_idx, dtype=torch.long),
            torch.tensor(negative_item_idx, dtype=torch.long),
        )


class TwoTowerRecommender(nn.Module):
    """Learn user and item vectors and score pairs with their dot product."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (num_users, num_items, embedding_dim)
        ):
            raise TypeError("num_users, num_items, and embedding_dim must be integers")
        if num_users < 1 or num_items < 1 or embedding_dim < 1:
            raise ValueError("num_users, num_items, and embedding_dim must be positive")

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def encode_users(self, user_indices: Tensor) -> Tensor:
        """Look up vectors for user indices."""
        return self.user_embedding(user_indices)

    def encode_items(self, item_indices: Tensor) -> Tensor:
        """Look up vectors for item indices."""
        return self.item_embedding(item_indices)

    def score(self, user_indices: Tensor, item_indices: Tensor) -> Tensor:
        """Compute dot-product scores for aligned user/item index tensors."""
        if user_indices.shape != item_indices.shape:
            raise ValueError("user_indices and item_indices must have the same shape")
        user_vectors = self.encode_users(user_indices)
        item_vectors = self.encode_items(item_indices)
        return (user_vectors * item_vectors).sum(dim=-1)

    def forward(
        self,
        user_indices: Tensor,
        positive_item_indices: Tensor,
        negative_item_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return positive and negative scores for a batch of training triples."""
        positive_scores = self.score(user_indices, positive_item_indices)
        negative_scores = self.score(user_indices, negative_item_indices)
        return positive_scores, negative_scores


def load_two_tower_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[TwoTowerRecommender, dict[str, Any]]:
    """Load a model checkpoint without importing the Spark training runtime."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    model_config = payload["model_config"]
    model = TwoTowerRecommender(**model_config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata


def bpr_loss(positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
    """Numerically stable mean pairwise ranking loss."""
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive_scores and negative_scores must have the same shape")
    if positive_scores.numel() == 0:
        raise ValueError("score tensors must not be empty")
    if not positive_scores.is_floating_point() or not negative_scores.is_floating_point():
        raise TypeError("score tensors must contain floating-point values")
    if (
        not torch.isfinite(positive_scores).all()
        or not torch.isfinite(negative_scores).all()
    ):
        raise ValueError("score tensors must contain only finite values")
    return F.softplus(negative_scores - positive_scores).mean()
