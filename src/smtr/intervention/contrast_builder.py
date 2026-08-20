"""P2 contrast builder (Task 5).

Converts PerturbationOutcomeRecord list into InterventionContrast list.
Discards contrasts where Effect(m) = Effect(m~) (no ranking signal).
"""

from __future__ import annotations

from smtr.intervention.intervention_contrast import (
    InterventionContrast,
    compute_contrast_direction,
    compute_transfer_effect,
    is_valid_contrast,
)
from smtr.intervention.perturbation_schema import PerturbationOutcomeRecord


def build_intervention_contrasts(
    outcomes: list[PerturbationOutcomeRecord],
) -> list[InterventionContrast]:
    """Build intervention contrasts from perturbation outcomes.

    For each outcome record:
      1. Compute effect_original = Y_original - Y_0
      2. Compute effect_perturbed = Y_perturbed - Y_0
      3. If effect_original == effect_perturbed → discard
      4. Otherwise → build InterventionContrast

    No oversampling. No synthetic labels. No class balancing.
    """
    contrasts: list[InterventionContrast] = []
    for rec in outcomes:
        y0 = int(rec.y0)
        y_original = int(rec.y_original)
        y_perturbed = int(rec.y_perturbed)

        effect_original = compute_transfer_effect(y0, y_original)
        effect_perturbed = compute_transfer_effect(y0, y_perturbed)

        if not is_valid_contrast(effect_original, effect_perturbed):
            continue

        direction = compute_contrast_direction(
            effect_original, effect_perturbed
        )

        contrasts.append(
            InterventionContrast(
                perturbation_id=rec.spec.perturbation_id,
                task_id=rec.task_id,
                receiver_agent_id=rec.receiver_agent_id,
                candidate_memory_id=rec.candidate_memory_id,
                perturbation_type=rec.spec.perturbation_type,
                changed_field=rec.spec.changed_field,
                y0=y0,
                y_original=y_original,
                y_perturbed=y_perturbed,
                effect_original=effect_original,
                effect_perturbed=effect_perturbed,
                contrast_direction=direction,
                source_record_digest=rec.spec.source_record_id,
                original_memory_digest=rec.spec.original_memory_digest,
                perturbed_memory_digest=rec.spec.perturbed_memory_digest,
            )
        )

    return contrasts
