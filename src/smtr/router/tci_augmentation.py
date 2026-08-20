"""TCI augmentation data interface for the final integration.

Normalizes TCI intervention pairs into weighted training examples that
can be appended to observational training data.

Design decisions (Codex Task 2/3):
  - ``source_type`` distinguishes observational vs intervention supervision
    (tci_induced_damage, tci_rescue_destruction).
  - ``weight`` is fixed at 1.0 per example (obs_weight=1, tci_weight=1).
    No alpha hyperparameter is exposed.
  - Each TCI pair produces two binary classification examples:
      direction > 0: original → positive_transfer (q10),
                     perturbed → negative_transfer (q01)
      direction < 0: perturbed → positive_transfer (q10),
                     original → negative_transfer (q01)
      direction == 0: skipped (no supervision signal).

Forbidden:
  - Modifying critic architecture.
  - Modifying router policy.
  - Exposing alpha as a hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smtr.core.types import CandidateExposureInput


# Source type labels for provenance tracking.
SOURCE_OBSERVATIONAL: str = "observational"
SOURCE_TCI_INDUCED_DAMAGE: str = "tci_induced_damage"
SOURCE_TCI_RESCUE_DESTRUCTION: str = "tci_rescue_destruction"
SOURCE_TCI_DAMAGE_REPAIR: str = "tci_damage_repair"

# Mapping from contrast_type to source_type.
_CONTRAST_TO_SOURCE: dict[str, str] = {
    "precondition": SOURCE_TCI_INDUCED_DAMAGE,
    "environment_constraint": SOURCE_TCI_RESCUE_DESTRUCTION,
    "capability": SOURCE_TCI_DAMAGE_REPAIR,
}


@dataclass
class TCIAugmentedExample:
    """One TCI-derived training example.

    Attributes
    ----------
    features : CandidateExposureInput
        Routing input (receiver_state, candidate_card) used by the critic
        feature encoder.
    label : str
        Four-outcome taxonomy label: "positive_transfer" (q10) or
        "negative_transfer" (q01). TCI pairs always produce q10/q01
        labels because they encode relative ranking (better vs worse),
        not absolute outcome.
    weight : float
        Fixed at 1.0 (Task 3). No hyperparameter exposure.
    source_type : str
        Provenance label: ``observational``, ``tci_induced_damage``,
        ``tci_rescue_destruction``, or ``tci_damage_repair``.
    """

    features: CandidateExposureInput
    label: str
    weight: float = 1.0
    source_type: str = SOURCE_OBSERVATIONAL


@dataclass
class TCIAugmentedBatch:
    """Batch of TCI augmentation examples.

    This is the interface consumed by ``transfer_critic.fit()`` when
    ``tci_inputs`` is provided. The batch always uses weight=1.0 per
    example (no alpha, no normalization).
    """

    inputs: list[CandidateExposureInput]
    labels: list[str]
    source_types: list[str] = field(default_factory=list)

    @property
    def n_examples(self) -> int:
        return len(self.inputs)


def build_tci_augmentation_examples(
    tci_inputs: list[tuple[CandidateExposureInput,
                            CandidateExposureInput,
                            int,
                            str]],
) -> TCIAugmentedBatch:
    """Convert TCI pairs into weighted training examples (weight=1 fixed).

    Parameters
    ----------
    tci_inputs : list of (input_original, input_perturbed, direction,
                          contrast_type)
        direction > 0: original is better → q10; perturbed is worse → q01.
        direction < 0: perturbed is better → q10; original is worse → q01.
        direction == 0: skipped.
        contrast_type is used to derive ``source_type``
        (precondition/environment_constraint/capability).

    Returns
    -------
    TCIAugmentedBatch with inputs, labels, source_types. All weights=1.
    """
    inputs: list[CandidateExposureInput] = []
    labels: list[str] = []
    source_types: list[str] = []

    for (input_orig, input_pert, direction, ct) in tci_inputs:
        if direction == 0:
            continue
        src = _CONTRAST_TO_SOURCE.get(ct, SOURCE_TCI_INDUCED_DAMAGE)
        if direction > 0:
            # Original is better.
            inputs.append(input_orig)
            labels.append("positive_transfer")
            source_types.append(src)
            inputs.append(input_pert)
            labels.append("negative_transfer")
            source_types.append(src)
        else:
            # Perturbed is better.
            inputs.append(input_pert)
            labels.append("positive_transfer")
            source_types.append(src)
            inputs.append(input_orig)
            labels.append("negative_transfer")
            source_types.append(src)

    return TCIAugmentedBatch(
        inputs=inputs,
        labels=labels,
        source_types=source_types,
    )
