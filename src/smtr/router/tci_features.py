"""TCI structural feature encoder (Tasks 1, 5).

Combines memory card, receiver, and task structural features
into a fixed 128-dim vector for TCI ranker supervision.

Pipeline:
  memory_card  → MemoryCardFeatureEncoder  → h_m
  receiver     → ReceiverFeatureEncoder     → h_r
  task         → TaskFeatureEncoder         → h_t

  interaction  = h_r ⊙ h_m                 → h_rm

  concat       = [h_t, h_r, h_m, h_rm]     → h_full
  projection   = Linear(h_full_dim, 128)    → phi(m, r, t)

No memory_id. No perturbation_id. No digest. No hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smtr.router.memory_card_features import MemoryCardFeatureEncoder
from smtr.router.receiver_features import ReceiverFeatureEncoder
from smtr.router.task_features import TaskFeatureEncoder


# Target output dimension.
TCI_FEATURE_DIM: int = 128

# Forbidden tokens that must never appear in feature metadata.
_FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset({
    "memory_id",
    "agent_id",
    "task_id",
    "digest",
    "hash",
    "perturbation_id",
    "source_record_digest",
    "original_memory_digest",
    "perturbed_memory_digest",
})


@dataclass(frozen=True)
class TCIFeature:
    """Structural feature vector for one (memory, receiver, task) triple."""

    vector: list[float]
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.vector)


@dataclass
class TCIFeatureEncoder:
    """Encode (memory_card, receiver_context, task_context) → 128-dim vector.

    Parameters
    ----------
    feature_dim : target output dimension (default 128).
    seed : random seed for deterministic projection matrix.
    """

    feature_dim: int = TCI_FEATURE_DIM
    seed: int = 42

    _memory_encoder: MemoryCardFeatureEncoder = field(
        default_factory=MemoryCardFeatureEncoder,
    )
    _receiver_encoder: ReceiverFeatureEncoder = field(
        default_factory=ReceiverFeatureEncoder,
    )
    _task_encoder: TaskFeatureEncoder = field(
        default_factory=TaskFeatureEncoder,
    )
    _projection_matrix: np.ndarray | None = field(
        default=None, repr=False,
    )
    _projection_bias: np.ndarray | None = field(
        default=None, repr=False,
    )

    def __post_init__(self) -> None:
        """Initialize deterministic random projection."""
        self._init_projection()

    def _init_projection(self) -> None:
        """Build deterministic random projection matrix."""
        raw_dim = self._raw_feature_dim
        if raw_dim <= self.feature_dim:
            # No projection needed; use truncated identity.
            self._projection_matrix = np.eye(
                raw_dim, self.feature_dim
            )
            self._projection_bias = np.zeros(self.feature_dim)
        else:
            rng = np.random.RandomState(self.seed)
            # Random projection (Johnson-Lindenstrauss style).
            self._projection_matrix = (
                rng.randn(raw_dim, self.feature_dim)
                / np.sqrt(raw_dim)
            )
            self._projection_bias = np.zeros(self.feature_dim)

    @property
    def _raw_feature_dim(self) -> int:
        """Total concatenated raw dimension."""
        # h_t + h_r + h_m + h_rm (interaction has same dim as h_m)
        return (
            self._task_encoder.feature_dim
            + self._receiver_encoder.feature_dim
            + self._memory_encoder.feature_dim
            + self._memory_encoder.feature_dim  # interaction
        )

    @property
    def feature_names(self) -> list[str]:
        """Feature names for the projected 128-dim space."""
        return [f"tci_dim_{i}" for i in range(self.feature_dim)]

    def encode(
        self,
        memory_card: Any,
        receiver_context: dict[str, Any],
        task_context: dict[str, Any],
    ) -> TCIFeature:
        """Encode (memory, receiver, task) into a structural feature vector.

        Parameters
        ----------
        memory_card : MemoryRoutingCard or dict with card fields.
        receiver_context : dict with receiver context fields.
        task_context : dict with task context fields.

        Returns
        -------
        TCIFeature with 128-dim vector.
        """
        # Encode each component.
        h_t = np.array(
            self._task_encoder.encode(task_context), dtype=float
        )
        h_r = np.array(
            self._receiver_encoder.encode(receiver_context), dtype=float
        )
        h_m = np.array(
            self._memory_encoder.encode(memory_card), dtype=float
        )

        # Interaction: element-wise product of receiver × memory.
        # Truncate or pad to match dimensions.
        min_dim = min(len(h_r), len(h_m))
        h_rm = h_r[:min_dim] * h_m[:min_dim]
        # Pad interaction to memory dim if needed.
        if min_dim < len(h_m):
            h_rm = np.pad(h_rm, (0, len(h_m) - min_dim))

        # Concatenate: [task, receiver, memory, interaction].
        h_full = np.concatenate([h_t, h_r, h_m, h_rm])

        # Project to target dim.
        projected = h_full @ self._projection_matrix + self._projection_bias

        # Metadata (no identity fields).
        metadata: dict[str, Any] = {
            "perturbation_type": task_context.get("perturbation_type", ""),
            "changed_field": task_context.get("changed_field", ""),
        }
        # Verify no forbidden keys.
        for key in metadata:
            if key in _FORBIDDEN_METADATA_KEYS:
                raise ValueError(
                    f"Forbidden metadata key in TCIFeature: {key}"
                )

        return TCIFeature(
            vector=projected.tolist(),
            feature_names=self.feature_names,
            metadata=metadata,
        )

    def encode_pair(
        self,
        original_card: Any,
        perturbed_card: Any,
        receiver_context: dict[str, Any],
        task_context: dict[str, Any],
    ) -> tuple[TCIFeature, TCIFeature]:
        """Encode original and perturbed cards as a pair.

        Returns (original_feature, perturbed_feature).
        """
        f_orig = self.encode(original_card, receiver_context, task_context)
        f_pert = self.encode(perturbed_card, receiver_context, task_context)
        return f_orig, f_pert


def interaction_feature(
    receiver_feature: np.ndarray,
    memory_feature: np.ndarray,
) -> np.ndarray:
    """Compute receiver × memory interaction (element-wise product).

    Parameters
    ----------
    receiver_feature : shape (d_r,)
    memory_feature : shape (d_m,)

    Returns
    -------
    shape (min(d_r, d_m),) padded to d_m.
    """
    min_dim = min(len(receiver_feature), len(memory_feature))
    result = receiver_feature[:min_dim] * memory_feature[:min_dim]
    if min_dim < len(memory_feature):
        result = np.pad(result, (0, len(memory_feature) - min_dim))
    return result


def validate_no_identity_leakage(feature: TCIFeature) -> bool:
    """Check that no identity fields leaked into the feature.

    Returns True if clean, raises ValueError if leak detected.
    """
    for key in feature.metadata:
        if key in _FORBIDDEN_METADATA_KEYS:
            raise ValueError(
                f"Identity leakage detected in feature metadata: {key}"
            )
    # Check feature values are not NaN or inf.
    arr = np.array(feature.vector)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Feature contains NaN or Inf values")
    return True
