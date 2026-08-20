"""P2 intervention contrast data structures (Tasks 1-4).

Defines pairwise transfer-effect contrast between
original memory and transfer-critical perturbation.

Transfer effect:
  Effect(m) = Y_m - Y_0

Contrast direction:
  +1 if Effect(m) > Effect(m~)  (original better)
  -1 if Effect(m) < Effect(m~)  (perturbed better)
   0 if Effect(m) = Effect(m~)  (no ranking info)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InterventionContrast:
    """Pairwise transfer-effect contrast.

    Represents causal difference between
    original memory and transfer-critical perturbation.
    """

    perturbation_id: str

    task_id: str
    receiver_agent_id: str
    candidate_memory_id: str

    perturbation_type: str
    changed_field: str

    # observed outcomes
    y0: int
    y_original: int
    y_perturbed: int

    # transfer effects
    effect_original: int
    effect_perturbed: int

    # ranking direction
    contrast_direction: int

    # metadata
    source_record_digest: str
    original_memory_digest: str
    perturbed_memory_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterventionContrast:
        """Reconstruct from dict."""
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})


def compute_transfer_effect(y0: int, y_memory: int) -> int:
    """Compute transfer effect: Effect(m) = Y_m - Y_0.

    Returns:
      +1 if memory improves over baseline
       0 if memory has no effect
      -1 if memory degrades below baseline
    """
    return y_memory - y0


def compute_contrast_direction(
    effect_original: int,
    effect_perturbed: int,
) -> int:
    """Compute contrast direction between original and perturbed.

    Returns:
      +1 if Effect(m) > Effect(m~)  — original memory better
      -1 if Effect(m) < Effect(m~)  — perturbed memory better
       0 if Effect(m) = Effect(m~)  — no ranking information
    """
    if effect_original > effect_perturbed:
        return 1
    elif effect_original < effect_perturbed:
        return -1
    return 0


def is_valid_contrast(
    effect_original: int,
    effect_perturbed: int,
) -> bool:
    """Check if contrast carries ranking information.

    Returns True iff Effect(m) != Effect(m~).
    When effects are equal, there is no pairwise supervision signal.
    """
    return effect_original != effect_perturbed
