"""TCI pairwise training dataset (Tasks 11-12, extended for structural features).

Builds TCIPair records from InterventionContrast list.
Filter rule: only keep direction != 0 pairs.

No oversampling. No synthetic labels. No class balancing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from smtr.intervention.intervention_contrast import InterventionContrast


@dataclass(frozen=True)
class TCIPair:
    """One pairwise TCI training example.

    direction must be non-zero (+1 or -1).
    original_features and perturbed_features are optional:
    when empty, hash-based features are used as fallback.
    """

    perturbation_id: str
    task_id: str
    receiver_agent_id: str
    candidate_memory_id: str

    perturbation_type: str
    changed_field: str

    y0: int
    y_original: int
    y_perturbed: int

    effect_original: int
    effect_perturbed: int

    direction: int
    contrast_type: str

    # Optional structural features (empty = use hash fallback).
    original_features: tuple[float, ...] = ()
    perturbed_features: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TCIPair:
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})

    @property
    def has_structural_features(self) -> bool:
        """Whether this pair has precomputed structural features."""
        return (
            len(self.original_features) > 0
            and len(self.perturbed_features) > 0
        )


def build_tci_pairs(
    contrasts: list[InterventionContrast],
) -> list[TCIPair]:
    """Build TCI training pairs from intervention contrasts.

    Filter: only keep direction != 0.
    No oversampling. No synthetic labels. No class balancing.
    """
    from smtr.intervention.contrast_types import classify_contrast

    pairs: list[TCIPair] = []
    for c in contrasts:
        if c.contrast_direction == 0:
            continue
        ct = classify_contrast(c.y0, c.y_original, c.y_perturbed)
        pairs.append(
            TCIPair(
                perturbation_id=c.perturbation_id,
                task_id=c.task_id,
                receiver_agent_id=c.receiver_agent_id,
                candidate_memory_id=c.candidate_memory_id,
                perturbation_type=c.perturbation_type,
                changed_field=c.changed_field,
                y0=c.y0,
                y_original=c.y_original,
                y_perturbed=c.y_perturbed,
                effect_original=c.effect_original,
                effect_perturbed=c.effect_perturbed,
                direction=c.contrast_direction,
                contrast_type=ct.value,
            )
        )
    return pairs


def build_tci_pairs_with_features(
    contrasts: list[InterventionContrast],
    *,
    feature_encoder: Any,
    original_cards: dict[str, Any],
    perturbed_cards: dict[str, Any],
    receiver_contexts: dict[str, dict[str, Any]],
    task_contexts: dict[str, dict[str, Any]],
) -> list[TCIPair]:
    """Build TCI pairs with structural features precomputed.

    Parameters
    ----------
    contrasts : list of InterventionContrast.
    feature_encoder : TCIFeatureEncoder instance.
    original_cards : {memory_id: card_dict} original memory cards.
    perturbed_cards : {perturbation_id: card_dict} perturbed cards.
    receiver_contexts : {receiver_agent_id: context_dict}.
    task_contexts : {task_id: context_dict}.

    Returns
    -------
    List of TCIPair with original_features and perturbed_features.
    """
    from smtr.intervention.contrast_types import classify_contrast

    pairs: list[TCIPair] = []
    for c in contrasts:
        if c.contrast_direction == 0:
            continue
        ct = classify_contrast(c.y0, c.y_original, c.y_perturbed)

        # Look up context (fallback to empty dicts).
        recv_ctx = receiver_contexts.get(c.receiver_agent_id, {})
        task_ctx = dict(task_contexts.get(c.task_id, {}))
        # Inject perturbation metadata into task context for feature.
        task_ctx["perturbation_type"] = c.perturbation_type
        task_ctx["changed_field"] = c.changed_field

        orig_card = original_cards.get(c.candidate_memory_id, {})
        pert_card = perturbed_cards.get(c.perturbation_id, orig_card)

        # Encode structural features.
        f_orig, f_pert = feature_encoder.encode_pair(
            orig_card, pert_card, recv_ctx, task_ctx,
        )

        pairs.append(
            TCIPair(
                perturbation_id=c.perturbation_id,
                task_id=c.task_id,
                receiver_agent_id=c.receiver_agent_id,
                candidate_memory_id=c.candidate_memory_id,
                perturbation_type=c.perturbation_type,
                changed_field=c.changed_field,
                y0=c.y0,
                y_original=c.y_original,
                y_perturbed=c.y_perturbed,
                effect_original=c.effect_original,
                effect_perturbed=c.effect_perturbed,
                direction=c.contrast_direction,
                contrast_type=ct.value,
                original_features=tuple(f_orig.vector),
                perturbed_features=tuple(f_pert.vector),
            )
        )
    return pairs
