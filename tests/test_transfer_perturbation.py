"""Tests for P2 transfer perturbation operators (清单 §5-§11).

Each operator must:
  - Only modify ONE declared field.
  - Non-target fields remain byte-equivalent.
  - Deterministic under fixed seed.
  - No forbidden tokens in output.
"""

from __future__ import annotations

import random

import pytest

from smtr.core.types import AgentProfile, MemoryRoutingCard, ReceiverState
from smtr.intervention.transfer_perturbation import (
    FORBIDDEN_TOKENS,
    EnvironmentConstraintPerturbation,
    PreconditionPerturbation,
    ProcedureDependencyPerturbation,
    RequiredCapabilityPerturbation,
    RequiredToolPerturbation,
    build_perturbation_spec,
    get_all_operators,
    validate_single_factor_change,
)


def _make_card(**overrides: object) -> MemoryRoutingCard:
    base = dict(
        memory_id="mem_001",
        goal_summary="Step 1: inspect schema\nStep 2: run query\nStep 3: verify",
        task_tags=("database",),
        required_tools=("sql_read", "sql_write"),
        required_capabilities=("query_optimization",),
        execution_role_tags=("executor",),
        environment_constraints=("read_write_access",),
        precondition_tags=("schema_inspected",),
        procedure_type="etl",
        procedure_length_bucket="medium",
        read_write_scope="table_level",
    )
    base.update(overrides)
    return MemoryRoutingCard.model_validate(base)


def _make_receiver(**overrides: object) -> ReceiverState:
    base = dict(
        task_id="task_1",
        scenario="database",
        task_instruction="Run ETL pipeline",
        receiver=AgentProfile(
            agent_id="agent_a",
            role="executor",
            capabilities=("query_optimization", "schema_design"),
            tool_names=("sql_read", "sql_write", "file_read"),
        ),
        environment_signature=("read_write_access", "local_db"),
    )
    base.update(overrides)
    return ReceiverState.model_validate(base)


class TestRequiredToolPerturbation:
    def test_applicable_with_tools(self) -> None:
        op = RequiredToolPerturbation()
        card = _make_card()
        recv = _make_receiver()
        assert op.applicable(card, recv) is True

    def test_not_applicable_no_tools(self) -> None:
        op = RequiredToolPerturbation()
        card = _make_card(required_tools=())
        recv = _make_receiver()
        assert op.applicable(card, recv) is False

    def test_single_field_change(self) -> None:
        op = RequiredToolPerturbation()
        card = _make_card()
        recv = _make_receiver()
        rng = random.Random(42)
        result = op.perturb(card, recv, rng=rng)
        assert result.changed_field == "required_tools"
        # Only required_tools changed.
        assert result.card.task_tags == card.task_tags
        assert result.card.goal_summary == card.goal_summary
        assert result.card.required_capabilities == card.required_capabilities
        assert result.card.memory_id == card.memory_id

    def test_deterministic(self) -> None:
        op = RequiredToolPerturbation()
        card = _make_card()
        recv = _make_receiver()
        r1 = op.perturb(card, recv, rng=random.Random(99))
        r2 = op.perturb(card, recv, rng=random.Random(99))
        assert r1.perturbed_value == r2.perturbed_value
        assert r1.original_value == r2.original_value

    def test_no_forbidden_tokens(self) -> None:
        op = RequiredToolPerturbation()
        card = _make_card()
        recv = _make_receiver()
        rng = random.Random(7)
        result = op.perturb(card, recv, rng=rng)
        low = result.perturbed_value.lower()
        for tok in FORBIDDEN_TOKENS:
            assert tok not in low


class TestRequiredCapabilityPerturbation:
    def test_applicable(self) -> None:
        op = RequiredCapabilityPerturbation()
        card = _make_card()
        recv = _make_receiver()
        assert op.applicable(card, recv) is True

    def test_not_applicable_no_overlap(self) -> None:
        op = RequiredCapabilityPerturbation()
        card = _make_card(required_capabilities=("unknown_cap",))
        recv = _make_receiver()
        assert op.applicable(card, recv) is False

    def test_single_field_change(self) -> None:
        op = RequiredCapabilityPerturbation()
        card = _make_card()
        recv = _make_receiver()
        rng = random.Random(42)
        result = op.perturb(card, recv, rng=rng)
        assert result.changed_field == "required_capabilities"
        assert result.card.required_tools == card.required_tools


class TestPreconditionPerturbation:
    def test_always_applicable(self) -> None:
        op = PreconditionPerturbation()
        card = _make_card()
        recv = _make_receiver()
        assert op.applicable(card, recv) is True

    def test_single_field_change(self) -> None:
        op = PreconditionPerturbation()
        card = _make_card()
        recv = _make_receiver()
        rng = random.Random(42)
        result = op.perturb(card, recv, rng=rng)
        assert result.changed_field == "precondition_tags"
        assert result.card.required_tools == card.required_tools
        assert result.card.goal_summary == card.goal_summary

    def test_adds_precondition_when_empty(self) -> None:
        op = PreconditionPerturbation()
        card = _make_card(precondition_tags=())
        recv = _make_receiver()
        rng = random.Random(42)
        result = op.perturb(card, recv, rng=rng)
        assert len(result.card.precondition_tags) == 1
        assert result.original_value == ()


class TestEnvironmentConstraintPerturbation:
    def test_always_applicable(self) -> None:
        op = EnvironmentConstraintPerturbation()
        card = _make_card()
        recv = _make_receiver()
        assert op.applicable(card, recv) is True

    def test_single_field_change(self) -> None:
        op = EnvironmentConstraintPerturbation()
        card = _make_card()
        recv = _make_receiver()
        rng = random.Random(42)
        result = op.perturb(card, recv, rng=rng)
        assert result.changed_field == "environment_constraints"
        assert result.card.required_tools == card.required_tools

    def test_adds_when_empty(self) -> None:
        op = EnvironmentConstraintPerturbation()
        card = _make_card(environment_constraints=())
        recv = _make_receiver()
        rng = random.Random(42)
        result = op.perturb(card, recv, rng=rng)
        assert len(result.card.environment_constraints) == 1


class TestProcedureDependencyPerturbation:
    def test_applicable_multiline(self) -> None:
        op = ProcedureDependencyPerturbation()
        card = _make_card()
        recv = _make_receiver()
        assert op.applicable(card, recv) is True

    def test_not_applicable_single_line(self) -> None:
        op = ProcedureDependencyPerturbation()
        card = _make_card(goal_summary="Single step only")
        recv = _make_receiver()
        assert op.applicable(card, recv) is False

    def test_swaps_steps(self) -> None:
        op = ProcedureDependencyPerturbation()
        card = _make_card()
        recv = _make_receiver()
        rng = random.Random(42)
        result = op.perturb(card, recv, rng=rng)
        assert result.changed_field == "goal_summary"
        # Same lines, different order.
        orig_lines = sorted(card.goal_summary.split("\n"))
        pert_lines = sorted(result.card.goal_summary.split("\n"))
        assert orig_lines == pert_lines

    def test_requires_two_steps_raises(self) -> None:
        op = ProcedureDependencyPerturbation()
        card = _make_card(goal_summary="only one")
        recv = _make_receiver()
        with pytest.raises(ValueError, match="at least 2"):
            op.perturb(card, recv, rng=random.Random(1))


class TestValidateSingleFactorChange:
    def test_valid_change(self) -> None:
        from smtr.intervention.perturbation_schema import PerturbationSpec

        card = _make_card()
        new_card = card.model_copy(
            update={"required_tools": ("new_tool",)}
        )
        spec = PerturbationSpec(
            perturbation_id="p1",
            task_id="t1",
            receiver_agent_id="a1",
            candidate_memory_id="mem_001",
            perturbation_type="required_tool",
            changed_field="required_tools",
            original_value=("sql_read", "sql_write"),
            perturbed_value=("new_tool",),
            source_record_id="e1",
            control_group_key="k",
            generation_seed=1,
            original_memory_digest="a",
            perturbed_memory_digest="b",
        )
        # Should not raise (different memory_id check skipped since
        # model_copy preserves it — this test checks field-level logic).
        # Actually, validator requires different memory_id.
        # Let's give perturbed a different ID.
        new_card_diff_id = new_card.model_copy(
            update={"memory_id": "mem_002"}
        )
        validate_single_factor_change(card, new_card_diff_id, spec)

    def test_no_change_raises(self) -> None:
        from smtr.intervention.perturbation_schema import PerturbationSpec

        card = _make_card()
        same = card.model_copy(update={"memory_id": "mem_002"})
        spec = PerturbationSpec(
            perturbation_id="p1",
            task_id="t1",
            receiver_agent_id="a1",
            candidate_memory_id="mem_001",
            perturbation_type="required_tool",
            changed_field="required_tools",
            original_value="a",
            perturbed_value="b",
            source_record_id="e1",
            control_group_key="k",
            generation_seed=1,
            original_memory_digest="a",
            perturbed_memory_digest="b",
        )
        with pytest.raises(ValueError, match="no fields changed"):
            validate_single_factor_change(card, same, spec)

    def test_multiple_changes_raises(self) -> None:
        from smtr.intervention.perturbation_schema import PerturbationSpec

        card = _make_card()
        multi = card.model_copy(
            update={
                "required_tools": ("x",),
                "goal_summary": "changed",
                "memory_id": "mem_002",
            }
        )
        spec = PerturbationSpec(
            perturbation_id="p1",
            task_id="t1",
            receiver_agent_id="a1",
            candidate_memory_id="mem_001",
            perturbation_type="required_tool",
            changed_field="required_tools",
            original_value="a",
            perturbed_value="b",
            source_record_id="e1",
            control_group_key="k",
            generation_seed=1,
            original_memory_digest="a",
            perturbed_memory_digest="b",
        )
        with pytest.raises(ValueError, match="multiple fields changed"):
            validate_single_factor_change(card, multi, spec)


class TestBuildPerturbationSpec:
    def test_computes_digests_and_id(self) -> None:
        op = PreconditionPerturbation()
        card = _make_card()
        recv = _make_receiver()
        rng = random.Random(7)
        result = op.perturb(card, recv, rng=rng)

        spec = build_perturbation_spec(
            task_id="task_1",
            receiver_agent_id="agent_a",
            candidate_memory_id="mem_001",
            perturbed=result,
            original_card=card,
            source_record_id="edge_1",
            control_group_key="task_1::agent_a::42",
            generation_seed=42,
        )
        assert spec.perturbation_id.startswith("pert_")
        assert spec.original_memory_digest != spec.perturbed_memory_digest
        assert spec.perturbation_type == "precondition"


class TestGetAllOperators:
    def test_returns_five(self) -> None:
        ops = get_all_operators()
        assert len(ops) == 5

    def test_priority_order(self) -> None:
        ops = get_all_operators()
        names = [op.name for op in ops]
        assert names == [
            "precondition",
            "required_capability",
            "required_tool",
            "environment_constraint",
            "procedure_dependency",
        ]
