"""Tests for P2 perturbation schema (清单 §4)."""

from __future__ import annotations

import pytest

from smtr.intervention.perturbation_schema import (
    PERTURBATION_TYPES,
    SCHEMA_VERSION,
    PerturbationOutcomeRecord,
    PerturbationSpec,
    compute_memory_digest,
    compute_perturbation_id,
)


def _make_spec(**overrides: object) -> PerturbationSpec:
    base = dict(
        perturbation_id="pert_test_001",
        task_id="task_1",
        receiver_agent_id="agent_a",
        candidate_memory_id="mem_1",
        perturbation_type="required_tool",
        changed_field="required_tools",
        original_value="tool_a",
        perturbed_value="tool_b",
        source_record_id="edge_1",
        control_group_key="task_1::agent_a::42",
        generation_seed=42,
        original_memory_digest="abc",
        perturbed_memory_digest="def",
    )
    base.update(overrides)
    return PerturbationSpec(**base)  # type: ignore[arg-type]


class TestPerturbationSpec:
    def test_valid_types_accepted(self) -> None:
        for ptype in PERTURBATION_TYPES:
            spec = _make_spec(perturbation_type=ptype)
            assert spec.perturbation_type == ptype

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown perturbation_type"):
            _make_spec(perturbation_type="invalid_type")

    def test_roundtrip_dict(self) -> None:
        spec = _make_spec()
        d = spec.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        restored = PerturbationSpec.from_dict(d)
        assert restored == spec

    def test_frozen(self) -> None:
        spec = _make_spec()
        with pytest.raises(AttributeError):
            spec.task_id = "other"  # type: ignore[misc]

    def test_deterministic_id(self) -> None:
        s1 = _make_spec()
        s2 = _make_spec()
        assert compute_perturbation_id(s1) == compute_perturbation_id(s2)

    def test_id_changes_with_seed(self) -> None:
        s1 = _make_spec(generation_seed=1)
        s2 = _make_spec(generation_seed=2)
        assert compute_perturbation_id(s1) != compute_perturbation_id(s2)


class TestPerturbationOutcomeRecord:
    def _make_outcome(self, **kw: object) -> PerturbationOutcomeRecord:
        base = dict(
            schema_version=SCHEMA_VERSION,
            spec=_make_spec(),
            y0=False,
            y_original=True,
            y_perturbed=False,
            original_branch_id="branch_orig",
            perturbed_branch_id="branch_pert",
            task_id="task_1",
            receiver_agent_id="agent_a",
            candidate_memory_id="mem_1",
            generation_seed=42,
            runtime_metadata={"dry_run": True},
        )
        base.update(kw)
        return PerturbationOutcomeRecord(**base)  # type: ignore[arg-type]

    def test_roundtrip(self) -> None:
        rec = self._make_outcome()
        d = rec.to_dict()
        restored = PerturbationOutcomeRecord.from_dict(d)
        assert restored == rec

    def test_fields_preserved(self) -> None:
        rec = self._make_outcome(y0=True, y_original=False, y_perturbed=True)
        assert rec.y0 is True
        assert rec.y_original is False
        assert rec.y_perturbed is True


class TestMemoryDigest:
    def test_deterministic(self) -> None:
        card = {"a": 1, "b": [2, 3]}
        assert compute_memory_digest(card) == compute_memory_digest(card)

    def test_order_independent(self) -> None:
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert compute_memory_digest(d1) == compute_memory_digest(d2)

    def test_different_cards_different_digest(self) -> None:
        d1 = {"a": 1}
        d2 = {"a": 2}
        assert compute_memory_digest(d1) != compute_memory_digest(d2)
