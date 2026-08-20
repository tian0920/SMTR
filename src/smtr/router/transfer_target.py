"""Unified transfer target for the TCI-SMTR critic (Tasks 1-3).

Unifies observational supervision (Y) and TCI intervention supervision
(τ = Y_m - Y_0) into a single four-outcome label space.

Transfer Target Semantics
-------------------------

The critic output ``s_θ(m) = q10(m) - q01(m)`` is the **transfer utility
score** estimating E[Y_m - Y_0]:

  - ``positive_transfer`` (q10): P(τ(m) > 0) — memory improves outcome
  - ``negative_transfer`` (q01): P(τ(m) < 0) — memory degrades outcome
  - ``neutral_success``   (q11): P(Y_m=1, τ=0) — memory shares but no
    additional benefit
  - ``neutral_failure``   (q00): P(Y_m=0, τ=0) — memory shares but fails

Effect → Label Mapping
-----------------------
  effect=+1 → positive_transfer (reinforces q10)
  effect=-1 → negative_transfer (reinforces q01)
  effect= 0 → neutral_success   (anchors τ=0 to the neutral class)

This mapping allows the SAME critic to learn:
  - P(Y) from observational data (L_obs)
  - P(τ) from TCI effect data (L_τ)
  - τ(m) > τ(m̃) from TCI rank data (L_rank)

All three losses operate on the same four-class output. No separate
value head. No lambda. No weighting search.

Forbidden:
  - Modifying router policy
  - Modifying candidate generation
  - Score fusion
  - Adding lambda parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smtr.core.types import CandidateExposureInput


# Supervision source labels.
SOURCE_OBSERVATIONAL: str = "observational"
SOURCE_TCI_EFFECT: str = "tci_effect"
SOURCE_TCI_RANK: str = "tci_rank"

# Effect → four-outcome label mapping.
_EFFECT_TO_LABEL: dict[int, str] = {
    1: "positive_transfer",
    -1: "negative_transfer",
    0: "neutral_success",
}


@dataclass(frozen=True)
class TransferTarget:
    """One unified training example with provenance tracking.

    Attributes
    ----------
    input : CandidateExposureInput
        The routing input (receiver_state, candidate_card).
    observational_label : str
        Four-outcome label from observational data, or empty string
        if this example is TCI-only.
    transfer_effect : int
        Absolute transfer effect τ(m) = Y_m - Y_0 ∈ {-1, 0, +1},
        or 0 if this example is observational-only.
    supervision_source : str
        Provenance: ``observational``, ``tci_effect``, or ``tci_rank``.
    """

    input: CandidateExposureInput
    observational_label: str
    transfer_effect: int
    supervision_source: str = SOURCE_OBSERVATIONAL

    @property
    def unified_label(self) -> str:
        """Return the four-outcome label for critic training.

        For observational examples: use observational_label directly.
        For TCI effect examples: map effect → four-outcome label.
        For TCI rank examples: use observational_label (already set
        by the rank augmentation builder).
        """
        if self.supervision_source == SOURCE_TCI_EFFECT:
            return _EFFECT_TO_LABEL.get(
                self.transfer_effect, "neutral_success"
            )
        return self.observational_label


@dataclass
class TransferTargetBatch:
    """Batch of unified transfer targets.

    Tracks the composition of supervision sources for checkpoint
    metadata and ablation analysis.
    """

    targets: list[TransferTarget]

    @property
    def n_examples(self) -> int:
        return len(self.targets)

    @property
    def inputs(self) -> list[CandidateExposureInput]:
        return [t.input for t in self.targets]

    @property
    def labels(self) -> list[str]:
        return [t.unified_label for t in self.targets]

    def source_counts(self) -> dict[str, int]:
        """Count examples per supervision source."""
        counts: dict[str, int] = {}
        for t in self.targets:
            src = t.supervision_source
            counts[src] = counts.get(src, 0) + 1
        return counts

    def effect_distribution(self) -> dict[int, int]:
        """Count effect labels (only for tci_effect sources)."""
        dist: dict[int, int] = {-1: 0, 0: 0, 1: 0}
        for t in self.targets:
            if t.supervision_source == SOURCE_TCI_EFFECT:
                dist[t.transfer_effect] = dist.get(
                    t.transfer_effect, 0
                ) + 1
        return dist


def build_effect_targets(
    effect_batch: Any,
    *,
    tci_inputs: list[tuple[
        CandidateExposureInput,
        CandidateExposureInput,
        int,
        str,
    ]] | None = None,
) -> list[TransferTarget]:
    """Convert a TCIEffectBatch into TransferTarget list.

    Maps each effect example into the unified four-outcome space:
      effect=+1 → positive_transfer
      effect=-1 → negative_transfer
      effect= 0 → neutral_success

    Parameters
    ----------
    effect_batch : TCIEffectBatch with features and effects.
    tci_inputs : corresponding CandidateExposureInput pairs (same order
        as contrasts that generated the batch).

    Returns
    -------
    List of TransferTarget instances with supervision_source="tci_effect".
    """
    if effect_batch is None or effect_batch.n_examples == 0:
        return []

    targets: list[TransferTarget] = []

    # Build inputs from the batch examples' features.
    # The effect batch has pre-encoded features; we need the original
    # CandidateExposureInput objects. If tci_inputs is provided, we
    # reconstruct the mapping: each contrast produces 2 examples
    # (original, perturbed).
    if tci_inputs is not None:
        for (inp_orig, inp_pert, direction, ct), ex_pair in _zip_contrasts_with_effects(
            tci_inputs, effect_batch
        ):
            inp_o, eff_o, inp_p, eff_p = ex_pair
            targets.append(TransferTarget(
                input=inp_o,
                observational_label="",
                transfer_effect=eff_o,
                supervision_source=SOURCE_TCI_EFFECT,
            ))
            targets.append(TransferTarget(
                input=inp_p,
                observational_label="",
                transfer_effect=eff_p,
                supervision_source=SOURCE_TCI_EFFECT,
            ))
    else:
        # Fallback: create dummy inputs (won't be used for encoding).
        # This path is for testing only.
        pass

    return targets


def _zip_contrasts_with_effects(
    tci_inputs: list,
    effect_batch: Any,
) -> list:
    """Pair each TCI input with its effect values from the batch.

    Each contrast produces 2 effect examples:
      (original_effect, perturbed_effect).

    Returns list of (inp_orig, inp_pert, direction, ct),
                    (inp_orig, effect_orig, inp_pert, effect_pert).
    """
    examples = effect_batch.examples
    result = []
    idx = 0
    for tci_input in tci_inputs:
        if idx + 1 >= len(examples):
            break
        ex_orig = examples[idx]
        ex_pert = examples[idx + 1]
        result.append((
            tci_input,
            (
                tci_input[0], ex_orig.transfer_effect,
                tci_input[1], ex_pert.transfer_effect,
            ),
        ))
        idx += 2
    return result
