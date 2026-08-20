"""TCI Absolute Transfer Effect Dataset (Task 1).

Defines the data structure for absolute transfer value supervision.

Transfer Effect:
  Effect(m) = Y_m - Y_0

where:
  Y_0 = baseline outcome (no memory)
  Y_m = outcome with memory m

Values: -1 (harmful), 0 (neutral), +1 (beneficial)

This enables the critic to learn absolute transfer values, not just
relative rankings between memory pairs.

Forbidden:
  - Modifying router policy
  - Modifying candidate generation
  - Score fusion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TCIEffectExample:
    """One absolute transfer effect example.

    Attributes
    ----------
    memory_features : list[float]
        Encoded feature vector φ(m) for the memory.
    transfer_effect : int
        Absolute transfer effect τ(m) = Y_m - Y_0.
        Values: -1 (harmful), 0 (neutral), +1 (beneficial).
    effect_source : str
        Provenance: "observational" or "tci_intervention".
    contrast_type : str
        Type of perturbation that generated this example:
        "precondition", "environment_constraint", "capability", etc.
    perturbation_type : str
        Specific field that was perturbed.

    Notes
    -----
    Unlike TCIAugmentedExample (which uses relative ranking), this
    structure captures absolute transfer effect, enabling value
    prediction rather than just pairwise ranking.
    """

    memory_features: list[float]
    transfer_effect: int
    effect_source: str
    contrast_type: str
    perturbation_type: str

    def __post_init__(self) -> None:
        """Validate transfer_effect is in {-1, 0, +1}."""
        if self.transfer_effect not in (-1, 0, 1):
            raise ValueError(
                f"transfer_effect must be -1, 0, or +1; "
                f"got {self.transfer_effect}"
            )


@dataclass
class TCIEffectBatch:
    """Batch of absolute transfer effect examples.

    This is the interface consumed by the TCI value head during training.
    """

    examples: list[TCIEffectExample]

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    @property
    def features(self) -> np.ndarray:
        """Stack features into (n_examples, n_features) array."""
        if not self.examples:
            return np.zeros((0, 0))
        return np.array([ex.memory_features for ex in self.examples])

    @property
    def effects(self) -> np.ndarray:
        """Extract effect labels as (n_examples,) array."""
        return np.array([ex.transfer_effect for ex in self.examples])

    def effect_distribution(self) -> dict[int, int]:
        """Count examples per effect class."""
        counts = {effect: 0 for effect in (-1, 0, 1)}
        for ex in self.examples:
            counts[ex.transfer_effect] += 1
        return counts
