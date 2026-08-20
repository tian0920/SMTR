"""Tests for P2 perturbation analysis (清单 §20-§27).

Hand-constructed outcomes verifying:
  - FlipRate, HFR, BFR computations.
  - Baseline-conditioned flips.
  - Support gain.
  - Leakage audit.
"""

from __future__ import annotations

import pytest

from smtr.intervention.perturbation_analysis import (
    LeakageAuditResult,
    PerturbationMetrics,
    audit_perturbation_leakage,
    compute_baseline_conditioned_flips,
    compute_operator_level_metrics,
    compute_perturbation_metrics,
    compute_support_gain,
    format_results_table,
    validate_real_execution_records,
)
from smtr.intervention.perturbation_schema import (
    SCHEMA_VERSION,
    PerturbationOutcomeRecord,
    PerturbationSpec,
)


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


class TestComputePerturbationMetrics:
    def test_hand_constructed_4_records(self) -> None:
        # 4 records:
        # 1. y_orig=1, y_pert=0 → harmful flip
        # 2. y_orig=0, y_pert=1 → beneficial flip
        # 3. y_orig=1, y_pert=1 → no effect
        # 4. y_orig=0, y_pert=0 → no effect
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False),
            _make_outcome(y0=False, y_original=False, y_perturbed=True),
            _make_outcome(y0=False, y_original=True, y_perturbed=True),
            _make_outcome(y0=False, y_original=False, y_perturbed=False),
        ]
        m = compute_perturbation_metrics(outcomes)
        assert m.n_total == 4
        assert m.n_flip == 2
        assert m.n_harmful_flip == 1
        assert m.n_beneficial_flip == 1
        assert m.n_no_effect == 2
        assert m.flip_rate == pytest.approx(0.5)
        assert m.harmful_flip_rate == pytest.approx(0.25)
        assert m.beneficial_flip_rate == pytest.approx(0.25)

    def test_empty(self) -> None:
        m = compute_perturbation_metrics([])
        assert m.n_total == 0
        assert m.flip_rate == 0.0


class TestOperatorLevelMetrics:
    def test_split_by_operator(self) -> None:
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False,
                          ptype="precondition"),
            _make_outcome(y0=False, y_original=True, y_perturbed=True,
                          ptype="required_tool"),
        ]
        by_op = compute_operator_level_metrics(outcomes)
        assert "precondition" in by_op
        assert "required_tool" in by_op
        assert by_op["precondition"].n_harmful_flip == 1
        assert by_op["required_tool"].n_harmful_flip == 0


class TestBaselineConditionedFlips:
    def test_conditioned_on_y0(self) -> None:
        outcomes = [
            # Y_0=1 group:
            _make_outcome(y0=True, y_original=True, y_perturbed=False),
            _make_outcome(y0=True, y_original=True, y_perturbed=True),
            # Y_0=0 group:
            _make_outcome(y0=False, y_original=False, y_perturbed=True),
            _make_outcome(y0=False, y_original=False, y_perturbed=False),
        ]
        bc = compute_baseline_conditioned_flips(outcomes)
        assert bc.n_y0_one == 2
        assert bc.n_y0_zero == 2
        assert bc.flip_given_y0_one == 1
        assert bc.harmful_given_y0_one == 1
        assert bc.harmful_flip_rate_given_y0_one == pytest.approx(0.5)
        assert bc.flip_given_y0_zero == 1
        assert bc.beneficial_given_y0_zero == 1


class TestSupportGain:
    def test_gain_computation(self) -> None:
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False),
            _make_outcome(y0=False, y_original=True, y_perturbed=False),
            _make_outcome(y0=False, y_original=True, y_perturbed=True),
        ]
        sg = compute_support_gain(
            original_damage_positives=4,
            outcomes=outcomes,
        )
        assert sg.original_damage_positives == 4
        assert sg.new_harmful_flips == 2
        assert sg.relative_gain == pytest.approx(0.5)

    def test_zero_original_with_flips(self) -> None:
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False),
        ]
        sg = compute_support_gain(
            original_damage_positives=0,
            outcomes=outcomes,
        )
        assert sg.relative_gain == float("inf")

    def test_zero_original_no_flips(self) -> None:
        sg = compute_support_gain(
            original_damage_positives=0,
            outcomes=[],
        )
        assert sg.relative_gain == 0.0


class TestFormatResultsTable:
    def test_contains_key_sections(self) -> None:
        overall = compute_perturbation_metrics([
            _make_outcome(y0=False, y_original=True, y_perturbed=False),
        ])
        by_op = compute_operator_level_metrics([
            _make_outcome(y0=False, y_original=True, y_perturbed=False,
                          ptype="precondition"),
        ])
        support = compute_support_gain(
            original_damage_positives=2, outcomes=[
                _make_outcome(y0=False, y_original=True, y_perturbed=False),
            ],
        )
        table = format_results_table(overall, by_op, support)
        assert "P2-B Intervention Results" in table
        assert "precondition" in table
        assert "Overall" in table


class TestLeakageAudit:
    def test_clean_cards_pass(self) -> None:
        cards = [
            {"memory_id": "m1", "goal_summary": "do a thing"},
        ]
        result = audit_perturbation_leakage(cards)
        assert result.passed is True
        assert len(result.violations) == 0

    def test_forbidden_token_fails(self) -> None:
        cards = [
            {"memory_id": "m1", "goal_summary": "this is harmful transfer"},
        ]
        result = audit_perturbation_leakage(cards)
        assert result.passed is False
        assert any("harmful" in v for v in result.violations)

    def test_synthetic_token_fails(self) -> None:
        cards = [
            {"memory_id": "m1", "goal_summary": "synthetic data"},
        ]
        result = audit_perturbation_leakage(cards)
        assert result.passed is False

    def test_cross_contamination(self) -> None:
        perturbed = [{"memory_id": "m1", "goal_summary": "ok"}]
        original = [{"memory_id": "m1", "goal_summary": "ok"}]
        test_ids = {"m1"}
        result = audit_perturbation_leakage(
            perturbed,
            original_cards=original,
            test_memory_ids=test_ids,
        )
        assert result.passed is False
        assert any("contamination" in v for v in result.violations)

    def test_no_contamination_clean(self) -> None:
        perturbed = [{"memory_id": "m1", "goal_summary": "ok"}]
        original = [{"memory_id": "m1", "goal_summary": "ok"}]
        test_ids = {"m_test"}
        result = audit_perturbation_leakage(
            perturbed,
            original_cards=original,
            test_memory_ids=test_ids,
        )
        assert result.passed is True


class TestValidateRealExecution:
    def test_analysis_rejects_dry_run(self) -> None:
        """Dry-run outcomes must be rejected in causal analysis."""
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False),
        ]
        # Override runtime_metadata to dry_run=True.
        dry_outcome = PerturbationOutcomeRecord(
            schema_version=outcomes[0].schema_version,
            spec=outcomes[0].spec,
            y0=outcomes[0].y0,
            y_original=outcomes[0].y_original,
            y_perturbed=outcomes[0].y_perturbed,
            original_branch_id=outcomes[0].original_branch_id,
            perturbed_branch_id=outcomes[0].perturbed_branch_id,
            task_id=outcomes[0].task_id,
            receiver_agent_id=outcomes[0].receiver_agent_id,
            candidate_memory_id=outcomes[0].candidate_memory_id,
            generation_seed=outcomes[0].generation_seed,
            runtime_metadata={"dry_run": True},
        )
        with pytest.raises(ValueError, match="Dry-run outcomes"):
            validate_real_execution_records([dry_outcome])
