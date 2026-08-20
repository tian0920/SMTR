"""Tests for P2-B pilot quality gate and dry-run validation.

Verifies:
  - GREEN / YELLOW / RED gate logic.
  - Dry-run rejection in causal analysis.
  - Balanced operator selection.
  - Operator distribution computation.
"""

from __future__ import annotations

import pytest

from smtr.intervention.perturbation_analysis import (
    PilotGate,
    TripleCounts,
    compute_operator_distribution,
    compute_pilot_gate,
    validate_real_execution_records,
)
from smtr.intervention.perturbation_schema import (
    SCHEMA_VERSION,
    PerturbationOutcomeRecord,
    PerturbationSpec,
)
from smtr.intervention.perturbation_selector import (
    select_balanced_operator,
)
from smtr.intervention.transfer_perturbation import (
    OPERATOR_PRIORITY,
    PreconditionPerturbation,
    RequiredToolPerturbation,
    RequiredCapabilityPerturbation,
)


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
    dry_run: bool = False,
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
        runtime_metadata={"dry_run": dry_run},
    )


def _make_triple_counts(
    n000: int = 0,
    n001: int = 0,
    n010: int = 0,
    n011: int = 0,
    n100: int = 0,
    n101: int = 0,
    n110: int = 0,
    n111: int = 0,
) -> TripleCounts:
    return TripleCounts(
        n000=n000,
        n001=n001,
        n010=n010,
        n011=n011,
        n100=n100,
        n101=n101,
        n110=n110,
        n111=n111,
    )


# ──────────────────────────────────────────────────────────────
# Pilot gate tests
# ──────────────────────────────────────────────────────────────
class TestPilotGate:
    def test_green_gate(self) -> None:
        """N110=20, IDR=0.5, 2 operators with count >= 10."""
        tc = _make_triple_counts(n110=20, n111=20, n000=60)
        op_counts = {"precondition": 40, "required_tool": 60}
        gate = compute_pilot_gate(tc, op_counts)
        assert gate.gate == "GREEN"
        assert any("N110=20" in r for r in gate.reasons)

    def test_yellow_gate_low_n110(self) -> None:
        """N110=8, below GREEN threshold."""
        tc = _make_triple_counts(n110=8, n111=12, n000=80)
        op_counts = {"precondition": 50, "required_tool": 50}
        gate = compute_pilot_gate(tc, op_counts)
        assert gate.gate == "YELLOW"
        assert any("N110=8" in r for r in gate.reasons)

    def test_yellow_gate_single_operator(self) -> None:
        """N110=20 but only 1 operator has count >= 10."""
        tc = _make_triple_counts(n110=20, n111=10, n000=70)
        op_counts = {"precondition": 95, "required_tool": 5}
        gate = compute_pilot_gate(tc, op_counts)
        assert gate.gate == "YELLOW"
        assert any("operator" in r.lower() for r in gate.reasons)

    def test_red_gate_low_n110(self) -> None:
        """N110=2, below RED threshold."""
        tc = _make_triple_counts(n110=2, n111=98, n000=0)
        op_counts = {"precondition": 50, "required_tool": 50}
        gate = compute_pilot_gate(tc, op_counts)
        assert gate.gate == "RED"
        assert any("N110=2" in r for r in gate.reasons)

    def test_red_gate_low_flip_rate(self) -> None:
        """FlipRate < 0.05."""
        tc = _make_triple_counts(n000=97, n110=3, n111=0)
        op_counts = {"precondition": 100}
        gate = compute_pilot_gate(tc, op_counts)
        assert gate.gate == "RED"

    def test_green_requires_two_operators(self) -> None:
        """N110=30, IDR=1.0 but only 1 operator exists."""
        tc = _make_triple_counts(n110=30, n111=0, n000=70)
        op_counts = {"precondition": 100}
        gate = compute_pilot_gate(tc, op_counts)
        assert gate.gate != "GREEN"

    def test_pilot_gate_to_dict(self) -> None:
        gate = PilotGate(gate="GREEN", reasons=["N110=20 >= 15"])
        d = gate.to_dict()
        assert d["gate"] == "GREEN"
        assert isinstance(d["reasons"], list)


# ──────────────────────────────────────────────────────────────
# Dry-run validation tests
# ──────────────────────────────────────────────────────────────
class TestValidateRealExecution:
    def test_rejects_dry_run(self) -> None:
        """Dry-run outcomes must raise ValueError."""
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False,
                          dry_run=True),
        ]
        with pytest.raises(ValueError, match="Dry-run outcomes"):
            validate_real_execution_records(outcomes)

    def test_accepts_real_execution(self) -> None:
        """Real execution outcomes must pass."""
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=False,
                          dry_run=False),
        ]
        # Should not raise.
        validate_real_execution_records(outcomes)

    def test_mixed_rejects_if_any_dry(self) -> None:
        """Mixed list with any dry_run=True must fail."""
        outcomes = [
            _make_outcome(y0=False, y_original=True, y_perturbed=True,
                          dry_run=False),
            _make_outcome(y0=True, y_original=True, y_perturbed=False,
                          dry_run=True),
        ]
        with pytest.raises(ValueError, match="Dry-run outcomes"):
            validate_real_execution_records(outcomes)


# ──────────────────────────────────────────────────────────────
# Balanced operator selection tests
# ──────────────────────────────────────────────────────────────
class TestBalancedOperatorSelection:
    def test_selects_less_used_operator(self) -> None:
        """When precondition has 50 uses and tool has 2, select tool."""
        ops = [PreconditionPerturbation(), RequiredToolPerturbation()]
        usage = {"precondition": 50, "required_tool": 2}
        selected = select_balanced_operator(ops, usage)
        assert selected.name == "required_tool"

    def test_deterministic_same_input(self) -> None:
        """Same inputs must produce same output."""
        ops = [
            PreconditionPerturbation(),
            RequiredCapabilityPerturbation(),
        ]
        usage = {"precondition": 5, "required_capability": 5}
        r1 = select_balanced_operator(ops, usage)
        r2 = select_balanced_operator(ops, usage)
        assert r1.name == r2.name

    def test_priority_tiebreak(self) -> None:
        """When usage counts are equal, priority order wins."""
        ops = [
            RequiredToolPerturbation(),
            PreconditionPerturbation(),
        ]
        usage = {"precondition": 10, "required_tool": 10}
        selected = select_balanced_operator(ops, usage)
        # precondition is higher priority (lower index).
        assert selected.name == "precondition"

    def test_empty_usage_defaults_to_priority(self) -> None:
        """Empty usage counts → all 0 → priority order."""
        ops = [
            RequiredToolPerturbation(),
            PreconditionPerturbation(),
        ]
        usage: dict[str, int] = {}
        selected = select_balanced_operator(ops, usage)
        assert selected.name == "precondition"


# ──────────────────────────────────────────────────────────────
# Operator distribution tests
# ──────────────────────────────────────────────────────────────
class TestOperatorDistribution:
    def test_counts_per_operator(self) -> None:
        outcomes = [
            _make_outcome(False, True, False, ptype="precondition"),
            _make_outcome(False, True, False, ptype="precondition"),
            _make_outcome(False, True, False, ptype="required_tool"),
        ]
        dist = compute_operator_distribution(outcomes)
        assert dist == {"precondition": 2, "required_tool": 1}

    def test_empty(self) -> None:
        dist = compute_operator_distribution([])
        assert dist == {}


# ──────────────────────────────────────────────────────────────
# TripleCounts properties
# ──────────────────────────────────────────────────────────────
class TestTripleCountsProperties:
    def test_idr(self) -> None:
        tc = _make_triple_counts(n110=10, n111=10)
        assert tc.idr == pytest.approx(0.5)

    def test_idr_zero_eligible(self) -> None:
        tc = _make_triple_counts(n000=100)
        assert tc.idr == 0.0

    def test_rdr(self) -> None:
        tc = _make_triple_counts(n010=5, n011=15)
        assert tc.rdr == pytest.approx(0.25)

    def test_flip_rate(self) -> None:
        tc = _make_triple_counts(n010=10, n110=10, n000=80)
        assert tc.flip_rate == pytest.approx(0.2)

    def test_total(self) -> None:
        tc = _make_triple_counts(n000=10, n010=20, n110=30, n111=40)
        assert tc.total == 100

    def test_to_dict_structure(self) -> None:
        tc = _make_triple_counts(n110=5, n111=5, n010=3, n011=7, n000=80)
        d = tc.to_dict()
        assert "triple_counts" in d
        assert "induced_damage" in d
        assert "rescue_destruction" in d
        assert d["induced_damage"]["eligible"] == 10
        assert d["rescue_destruction"]["eligible"] == 10
