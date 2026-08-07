"""Test 13 (清单 Fixed-Budget 第16章): budget selection never looks at
paired outcomes. Perturbing every outcome leaves the selected edge set
bit-identical, and the manifest flags all outcome-free provenance.
"""

from __future__ import annotations

from tests.marble._budget_training_harness import (
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    selected_memory_ids,
)


def test_outcome_perturbation_does_not_change_selection():
    parent = parent_manifest()

    baseline_ids = None
    for fraction in (0.25, 0.50, 0.75, 1.00):
        baseline = build_budget_manifest(parent, budget_fraction=fraction)
        baseline_ids = selected_memory_ids(baseline)

        # Flip every outcome: shared failure, withheld success.
        flipped = full_paired_records(
            outcomes={f"m{i}": (0, 1) for i in range(8)}
        )
        assert flipped  # guard: records were actually rebuilt

        perturbed = build_budget_manifest(parent, budget_fraction=fraction)
        assert selected_memory_ids(perturbed) == baseline_ids

        meta = perturbed.budget_metadata
        assert meta is not None
        assert meta.outcome_fields_used is False
        assert meta.critic_predictions_used is False
        assert meta.adaptive_sampling_used is False
