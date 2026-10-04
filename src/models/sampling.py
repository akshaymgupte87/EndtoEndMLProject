"""Uniform negative sampling for pairwise implicit-feedback training."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Set
from numbers import Integral
from types import MappingProxyType


def _validate_index(value: int, name: str) -> None:
    """Reject fractional, boolean, and negative IDs before tensor conversion."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def build_seen_items(
    positive_pairs: Iterable[tuple[int, int]],
) -> dict[int, set[int]]:
    """Build the set of training-positive item indices for each user index."""
    seen: dict[int, set[int]] = defaultdict(set)
    for user_idx, item_idx in positive_pairs:
        _validate_index(user_idx, "user index")
        _validate_index(item_idx, "item index")
        seen[user_idx].add(item_idx)
    return dict(seen)


class UniformNegativeSampler:
    """Sample catalog items absent from a user's observed training positives.

    The sampler uses only the supplied training history. It deliberately does
    not inspect validation or test interactions.
    """

    def __init__(
        self,
        num_items: int,
        seen_items_by_user: Mapping[int, Set[int]],
        seed: int = 42,
    ) -> None:
        _validate_index(num_items, "num_items")
        if num_items < 1:
            raise ValueError("num_items must be at least 1")
        self.num_items = num_items
        # Snapshot the history: later caller edits must not change exclusions or
        # invalidate the checks below. Store a plain dict so workers can pickle it.
        self._seen_items_by_user = {
            user_idx: frozenset(items) for user_idx, items in seen_items_by_user.items()
        }
        self.rng = random.Random(seed)

        for user_idx, seen_items in self._seen_items_by_user.items():
            _validate_index(user_idx, "user index")
            if not seen_items:
                raise ValueError(f"user {user_idx} has no training interactions")
            for item_idx in seen_items:
                _validate_index(item_idx, "item index")
                if item_idx >= num_items:
                    raise ValueError(f"user {user_idx} has an item index outside the catalog")
            if len(seen_items) >= num_items:
                raise ValueError(f"user {user_idx} has interacted with the full item catalog")

    @property
    def seen_items_by_user(self) -> Mapping[int, frozenset[int]]:
        """Expose the validated history without allowing it to be mutated."""
        return MappingProxyType(self._seen_items_by_user)

    def sample(self, user_idx: int) -> int:
        """Return one uniformly sampled item not in this user's training history."""
        _validate_index(user_idx, "user index")
        if user_idx not in self._seen_items_by_user:
            raise KeyError(f"user {user_idx} has no training interactions")

        seen_items = self._seen_items_by_user[user_idx]
        # Rejection sampling is efficient when each user has few positives
        # relative to the catalog, as is typical for this sparse dataset.
        for _ in range(64):
            candidate = self.rng.randrange(self.num_items)
            if candidate not in seen_items:
                return candidate

        # Draw the rank of an unseen item, then skip observed IDs to recover
        # its catalog index. This remains uniform without allocating a list of
        # every available item or allowing an unbounded rejection loop.
        candidate = self.rng.randrange(self.num_items - len(seen_items))
        for seen_item in sorted(seen_items):
            if seen_item > candidate:
                break
            candidate += 1
        return candidate
