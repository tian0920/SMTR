"""P2 contrast type classification (Task 6).

Three contrast labels based on (Y_0, Y_m, Y_~m) triple:

  INDUCED_DAMAGE:     (1, 1, 0) — memory helps, perturbation breaks it
  RESCUE_DESTRUCTION: (0, 1, 0) — memory rescues, perturbation breaks it
  DAMAGE_REPAIR:      (1, 0, 1) — memory hurts, perturbation fixes it

All other triples map to NONE (no contrast signal).
"""

from __future__ import annotations

from enum import Enum


class ContrastType(Enum):
    """Classification of pairwise intervention contrast."""

    INDUCED_DAMAGE = "induced_damage"
    RESCUE_DESTRUCTION = "rescue_destruction"
    DAMAGE_REPAIR = "damage_repair"
    NONE = "none"


def classify_contrast(
    y0: int,
    y_original: int,
    y_perturbed: int,
) -> ContrastType:
    """Classify an intervention contrast by outcome triple.

    Rules:
      (1, 1, 0) -> INDUCED_DAMAGE
      (0, 1, 0) -> RESCUE_DESTRUCTION
      (1, 0, 1) -> DAMAGE_REPAIR
      otherwise -> NONE
    """
    triple = (y0, y_original, y_perturbed)
    if triple == (1, 1, 0):
        return ContrastType.INDUCED_DAMAGE
    if triple == (0, 1, 0):
        return ContrastType.RESCUE_DESTRUCTION
    if triple == (1, 0, 1):
        return ContrastType.DAMAGE_REPAIR
    return ContrastType.NONE
