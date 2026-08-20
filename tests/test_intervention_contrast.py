"""Tests for P2 intervention contrast layer (Tasks 10).

Verifies:
  - Transfer effect computation.
  - Contrast direction.
  - Valid contrast filtering.
  - Contrast type classification.
  - Contrast builder.
  - TCI pair filtering.
"""

from __future__ import annotations

import pytest

from smtr.intervention.intervention_contrast import (
    InterventionContrast,
    compute_contrast_direction,
    compute_transfer_effect,
    is_valid_contrast,
)
from smtr.intervention.contrast_types import (
    ContrastType,
    classify_contrast,
)
from smtr.intervention.contrast_builder import (
    build_intervention_contrasts,
)
from smtr.intervention.perturbation_schema import (
    SCHEMA_VERSION,
    PerturbationOutcomeRecord,
    PerturbationSpec,
)
from smtr.router.tci_dataset import TCIPair, build_tci_pairs


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _make_spec(ptype: str = "precondition") -> PerturbationSpec:
    return PerturbationSpec(
        perturbation_id=f"pert_{ptype}",
        task_id="t1",
        receiver_agent_id="a1",
        candidate_memory_id="m1",
        perturbation_type=ptype,
        changed_field="precondition_tags",
        original_value="a",
        perturbed_value="b",
        source_record_id="e1",
        control_group_key="k",
        generation_seed=1,
        original_memory_digest="d1",
        perturbed_memory_digest="d2",
    )


def _make_outcome(
    y0: bool,
    y_original: bool,
    y_perturbed: bool,
    ptype: str = "precondition",
) -> PerturbationOutcomeRecord:
    return PerturbationOutcomeRecord(
        schema_version=SCHEMA_VERSION,
        spec=_make_spec(ptype),
        y0=y0,
        y_original=y_original,
        y_perturbed=y_perturbed,
        original_branch_id="b1",
        perturbed_branch_id="b2",
        task_id="t1",
        receiver_agent_id="a1",
        candidate_memory_id="m1",
        generation_seed=1,
        runtime_metadata={},
    )


# ──────────────────────────────────────────────────────────────
# Task 10: Test 1 — Induced Damage (1,1,0)
# ──────────────────────────────────────────────────────────────
class TestTransferEffect:
    def test_induced_damage(self) -> None:
        """Y0=1, Ym=1, Y~=0.

        effect_original = 1 - 1 = 0
        effect_perturbed = 0 - 1 = -1
        direction = +1 (original better)
        type = INDUCED_DAMAGE
        """
        y0, ym, yp = 1, 1, 0
        eff_orig = compute_transfer_effect(y0, ym)
        eff_pert = compute_transfer_effect(y0, yp)
        assert eff_orig == 0
        assert eff_pert == -1

        direction = compute_contrast_direction(eff_orig, eff_pert)
        assert direction == 1

        ct = classify_contrast(y0, ym, yp)
        assert ct == ContrastType.INDUCED_DAMAGE

    def test_rescue_destruction(self) -> None:
        """Y0=0, Ym=1, Y~=0.

        type = RESCUE_DESTRUCTION
        """
        ct = classify_contrast(0, 1, 0)
        assert ct == ContrastType.RESCUE_DESTRUCTION

    def test_damage_repair(self) -> None:
        """Y0=1, Ym=0, Y~=1.

        type = DAMAGE_REPAIR
        """
        ct = classify_contrast(1, 0, 1)
        assert ct == ContrastType.DAMAGE_REPAIR

    def test_no_contrast(self) -> None:
        """Y0=1, Ym=1, Y~=1.

        effect_original = 0, effect_perturbed = 0
        direction = 0 → discard
        """
        eff_orig = compute_transfer_effect(1, 1)
        eff_pert = compute_transfer_effect(1, 1)
        assert eff_orig == 0
        assert eff_pert == 0

        direction = compute_contrast_direction(eff_orig, eff_pert)
        assert direction == 0

        assert not is_valid_contrast(eff_orig, eff_pert)


# ──────────────────────────────────────────────────────────────
# Contrast direction tests
# ──────────────────────────────────────────────────────────────
class TestContrastDirection:
    def test_positive_direction(self) -> None:
        assert compute_contrast_direction(1, 0) == 1

    def test_negative_direction(self) -> None:
        assert compute_contrast_direction(0, 1) == -1

    def test_zero_direction(self) -> None:
        assert compute_contrast_direction(0, 0) == 0
        assert compute_contrast_direction(1, 1) == 0

    def test_is_valid(self) -> None:
        assert is_valid_contrast(1, 0) is True
        assert is_valid_contrast(0, 1) is True
        assert is_valid_contrast(0, 0) is False
        assert is_valid_contrast(1, 1) is False


# ──────────────────────────────────────────────────────────────
# Contrast builder tests
# ──────────────────────────────────────────────────────────────
class TestContrastBuilder:
    def test_build_from_induced_damage(self) -> None:
        """(1,1,0) should produce a contrast with direction=+1."""
        outcomes = [
            _make_outcome(y0=True, y_original=True, y_perturbed=False),
        ]
        contrasts = build_intervention_contrasts(outcomes)
        assert len(contrasts) == 1
        c = contrasts[0]
        assert c.effect_original == 0
        assert c.effect_perturbed == -1
        assert c.contrast_direction == 1

    def test_build_from_rescue_destruction(self) -> None:
        """(0,1,0) should produce a contrast with direction=+1."""
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False),
        ]
        contrasts = build_intervention_contrasts(outcomes)
        assert len(contrasts) == 1
        c = contrasts[0]
        assert c.effect_original == 1
        assert c.effect_perturbed == 0
        assert c.contrast_direction == 1

    def test_discard_no_contrast(self) -> None:
        """(1,1,1) should be discarded (no contrast signal)."""
        outcomes = [
            _make_outcome(y0=True, y_original=True, y_perturbed=True),
        ]
        contrasts = build_intervention_contrasts(outcomes)
        assert len(contrasts) == 0

    def test_discard_both_zero(self) -> None:
        """(0,0,0) should be discarded."""
        outcomes = [
            _make_outcome(y0=False, y_original=False, y_perturbed=False),
        ]
        contrasts = build_intervention_contrasts(outcomes)
        assert len(contrasts) == 0

    def test_mixed_keep_and_discard(self) -> None:
        """Mixed outcomes: only valid contrasts kept."""
        outcomes = [
            _make_outcome(y0=True, y_original=True, y_perturbed=False),  # keep
            _make_outcome(y0=False, y_original=False, y_perturbed=False),  # discard
            _make_outcome(y0=True, y_original=False, y_perturbed=True),  # keep
        ]
        contrasts = build_intervention_contrasts(outcomes)
        assert len(contrasts) == 2

    def test_contrast_serialization(self) -> None:
        """InterventionContrast round-trip serialization."""
        outcomes = [
            _make_outcome(y0=True, y_original=True, y_perturbed=False),
        ]
        contrasts = build_intervention_contrasts(outcomes)
        d = contrasts[0].to_dict()
        restored = InterventionContrast.from_dict(d)
        assert restored.perturbation_id == contrasts[0].perturbation_id
        assert restored.contrast_direction == contrasts[0].contrast_direction


# ──────────────────────────────────────────────────────────────
# TCI pair tests
# ──────────────────────────────────────────────────────────────
class TestTCIPairs:
    def test_build_tci_pairs_filters_direction_zero(self) -> None:
        """Only direction != 0 contrasts should become TCI pairs."""
        outcomes = [
            _make_outcome(y0=True, y_original=True, y_perturbed=False),  # dir=+1
            _make_outcome(y0=False, y_original=False, y_perturbed=False),  # discard
        ]
        contrasts = build_intervention_contrasts(outcomes)
        pairs = build_tci_pairs(contrasts)
        assert len(pairs) == 1
        assert pairs[0].direction != 0

    def test_tci_pair_has_contrast_type(self) -> None:
        """Each TCI pair must have a contrast_type label."""
        outcomes = [
            _make_outcome(y0=True, y_original=True, y_perturbed=False),
        ]
        contrasts = build_intervention_contrasts(outcomes)
        pairs = build_tci_pairs(contrasts)
        assert pairs[0].contrast_type == "induced_damage"

    def test_tci_pair_serialization(self) -> None:
        """TCIPair round-trip serialization."""
        outcomes = [
            _make_outcome(y0=True, y_original=True, y_perturbed=False),
        ]
        contrasts = build_intervention_contrasts(outcomes)
        pairs = build_tci_pairs(contrasts)
        d = pairs[0].to_dict()
        restored = TCIPair.from_dict(d)
        assert restored.direction == pairs[0].direction
        assert restored.contrast_type == pairs[0].contrast_type
