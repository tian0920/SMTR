"""Tests for TCI dataset (Task 12).

Verifies:
  - TCIPair construction and filtering.
  - direction != 0 enforcement.
  - Contrast type classification on pairs.
"""

from __future__ import annotations

import pytest

from smtr.intervention.contrast_types import classify_contrast, ContrastType
from smtr.intervention.intervention_contrast import InterventionContrast
from smtr.router.tci_dataset import TCIPair, build_tci_pairs


def _make_contrast(
    y0: int,
    y_original: int,
    y_perturbed: int,
    direction: int,
    ptype: str = "precondition",
) -> InterventionContrast:
    eff_orig = y_original - y0
    eff_pert = y_perturbed - y0
    return InterventionContrast(
        perturbation_id="pert_1",
        task_id="t1",
        receiver_agent_id="a1",
        candidate_memory_id="m1",
        perturbation_type=ptype,
        changed_field="precondition_tags",
        y0=y0,
        y_original=y_original,
        y_perturbed=y_perturbed,
        effect_original=eff_orig,
        effect_perturbed=eff_pert,
        contrast_direction=direction,
        source_record_digest="e1",
        original_memory_digest="d1",
        perturbed_memory_digest="d2",
    )


class TestBuildTCIPairs:
    def test_keeps_nonzero_direction(self) -> None:
        contrasts = [
            _make_contrast(1, 1, 0, direction=1),
            _make_contrast(0, 1, 0, direction=1),
        ]
        pairs = build_tci_pairs(contrasts)
        assert len(pairs) == 2
        assert all(p.direction != 0 for p in pairs)

    def test_discards_zero_direction(self) -> None:
        contrasts = [
            _make_contrast(1, 1, 0, direction=1),
            _make_contrast(0, 0, 0, direction=0),
        ]
        pairs = build_tci_pairs(contrasts)
        assert len(pairs) == 1

    def test_contrast_type_induced_damage(self) -> None:
        contrasts = [_make_contrast(1, 1, 0, direction=1)]
        pairs = build_tci_pairs(contrasts)
        assert pairs[0].contrast_type == "induced_damage"

    def test_contrast_type_rescue_destruction(self) -> None:
        contrasts = [_make_contrast(0, 1, 0, direction=1)]
        pairs = build_tci_pairs(contrasts)
        assert pairs[0].contrast_type == "rescue_destruction"

    def test_contrast_type_damage_repair(self) -> None:
        contrasts = [_make_contrast(1, 0, 1, direction=-1)]
        pairs = build_tci_pairs(contrasts)
        assert pairs[0].contrast_type == "damage_repair"

    def test_empty_contrasts(self) -> None:
        pairs = build_tci_pairs([])
        assert pairs == []

    def test_no_oversampling(self) -> None:
        """Input count must equal output count (no synthetic pairs)."""
        contrasts = [
            _make_contrast(1, 1, 0, direction=1, ptype="precondition"),
            _make_contrast(0, 1, 0, direction=1, ptype="required_tool"),
        ]
        pairs = build_tci_pairs(contrasts)
        assert len(pairs) == 2


class TestTCIPairSerialization:
    def test_to_dict_roundtrip(self) -> None:
        pair = TCIPair(
            perturbation_id="pert_1",
            task_id="t1",
            receiver_agent_id="a1",
            candidate_memory_id="m1",
            perturbation_type="precondition",
            changed_field="precondition_tags",
            y0=1,
            y_original=1,
            y_perturbed=0,
            effect_original=0,
            effect_perturbed=-1,
            direction=1,
            contrast_type="induced_damage",
        )
        d = pair.to_dict()
        restored = TCIPair.from_dict(d)
        assert restored == pair
