"""Writer-agnostic pipeline tests (清单 Writer-Agnostic 18.2/18.3/18.5-18.7/18.15).

Covers candidate-ranking writer invariance, routing-card serialization,
requirement extraction, compatibility computation, outcome-leakage
absence and treatment-edge identity.
"""

from __future__ import annotations

import json

from smtr.core.types import MemoryProvenance, MemoryRoutingCard
from smtr.counterfactual.edge_keys import seed_pair_key, treatment_edge_key
from smtr.marble.real_data import (
    AgentTrajectorySlice,
    ExtractedMemory,
    RealDatabaseTrajectory,
    _memory_receiver_compatibility,
    _score_memory_for_recipient,
    build_cross_task_candidates,
    extract_procedural_memories,
)

DIAGNOSIS_ACTIONS = (
    {"tool": "sql_query", "sql": "SELECT count(*) FROM events"},
    {"tool": "inspect_schema"},
    {"tool": "read_log"},
)


def _trajectory() -> RealDatabaseTrajectory:
    return RealDatabaseTrajectory(
        trajectory_id="traj00000001",
        task_id="task_src",
        split="train",
        generation_seed=1,
        model_id="model-1",
        team_success=True,
        score=1.0,
        task_success=True,
        task_instruction="diagnose database latency",
        environment_signature=("read-only SQL",),
        agents=(
            AgentTrajectorySlice(
                agent_id="agent-a",
                agent_role="executor",
                agent_capabilities=("database_read",),
                tool_names=("sql_query",),
                actions=DIAGNOSIS_ACTIONS,
            ),
        ),
    )


def _memory() -> ExtractedMemory:
    memories = extract_procedural_memories([_trajectory()])
    assert len(memories) == 1
    return memories[0]


def _with_source_agent(mem: ExtractedMemory, *, agent_id: str, role: str) -> ExtractedMemory:
    """Same memory with only the provenance source agent replaced."""
    old = mem.payload.provenance
    provenance = MemoryProvenance(
        source_agent_id=agent_id,
        source_agent_role=role,  # type: ignore[arg-type]
        source_task_id=old.source_task_id,
        source_trajectory_id=old.source_trajectory_id,
        source_split=old.source_split,
        source_scenario=old.source_scenario,
    )
    return mem.model_copy(update={
        "payload": mem.payload.model_copy(update={"provenance": provenance}),
    })


def _recipient() -> dict:
    return {
        "task_id": "task_target",
        "agent_id": "recv1",
        "agent_role": "executor",
        "agent_capabilities": ["database_read", "log_analysis"],
        "tool_names": ["sql_query", "read_log"],
        "environment_signature": ["read-only SQL"],
        "instruction": "diagnose database latency",
    }


class TestCandidateRankingWriterInvariance:
    """清单 18.2: changing the provenance source agent changes nothing."""

    def test_score_invariant_under_provenance_change(self):
        mem_a = _memory()
        mem_b = _with_source_agent(mem_a, agent_id="agent-z", role="critic")
        score_a = _score_memory_for_recipient(mem_a, _recipient())
        score_b = _score_memory_for_recipient(mem_b, _recipient())
        assert score_a["score"] == score_b["score"]
        assert score_a["components"] == score_b["components"]

    def test_manifest_ranking_invariant_under_provenance_change(self):
        mem_a = _memory()
        mem_b = _with_source_agent(mem_a, agent_id="agent-z", role="critic")
        recipients = [_recipient()]
        manifest_a = build_cross_task_candidates(memories=[mem_a], recipients=recipients)
        manifest_b = build_cross_task_candidates(memories=[mem_b], recipients=recipients)
        assert manifest_a.model_dump(mode="json") == manifest_b.model_dump(mode="json")
        records_a = manifest_a.candidates[0].candidate_records
        records_b = manifest_b.candidates[0].candidate_records
        assert [(r.rank, r.score, r.candidate_source, r.memory_receiver_match_type)
                for r in records_a] == [
            (r.rank, r.score, r.candidate_source, r.memory_receiver_match_type)
            for r in records_b]


class TestRoutingCardSerialization:
    """清单 18.3: serialized routing cards carry no writer/provenance."""

    def test_card_json_free_of_writer_and_provenance_fields(self):
        card_json = json.dumps(
            _memory().routing_card.model_dump(mode="json"), sort_keys=True).lower()
        for forbidden in (
            "writer", "source_agent", "source_task", "source_trajectory",
        ):
            assert forbidden not in card_json, forbidden

    def test_provenance_still_auditable_in_payload(self):
        mem = _memory()
        assert mem.payload.provenance.source_agent_id == "agent-a"
        assert mem.payload.provenance.source_trajectory_id == "traj00000001"


class TestRequirementExtraction:
    """清单 18.5: requirements derived from the observed procedure."""

    def test_card_requirements_from_action_trajectory(self):
        card = _memory().routing_card
        assert card.required_tools == ("inspect_schema", "read_log", "sql_query")
        assert card.required_capabilities == (
            "database_read", "log_analysis", "schema_inspection")
        # inspect_schema (diagnosis) + read_log (monitoring) -> mixed.
        assert card.procedure_type == "mixed"
        assert card.read_write_scope == "read_only"

    def test_write_action_switches_scope(self):
        trajectory = _trajectory().model_copy(update={
            "agents": (
                AgentTrajectorySlice(
                    agent_id="agent-a",
                    agent_role="executor",
                    actions=(
                        {"tool": "sql_query", "sql": "UPDATE settings SET x = 1"},
                        {"tool": "read_log"},
                    ),
                ),
            ),
        })
        memories = extract_procedural_memories([trajectory])
        assert memories[0].routing_card.read_write_scope == "write"


def _requirement_card() -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id="mem1",
        goal_summary="Diagnose database issue",
        task_tags=("database",),
        required_tools=("sql_query", "read_log"),
        required_capabilities=("database_read", "log_analysis"),
        environment_constraints=("read-only SQL",),
    )


class TestCompatibilityComputation:
    """清单 18.6: compatibility from explicit requirements only."""

    def test_receiver_a_fully_satisfied_is_compatible(self):
        compat = _memory_receiver_compatibility(
            _requirement_card(),
            receiver_role="executor",
            receiver_capabilities={"database_read", "log_analysis"},
            receiver_tools={"sql_query", "read_log"},
            receiver_environment={"read-only SQL"},
        )
        assert compat["compatible"] is True
        assert compat["incompatible"] is False
        assert compat["tool_satisfaction"] == 1.0

    def test_receiver_b_missing_one_tool_is_partially_compatible(self):
        compat = _memory_receiver_compatibility(
            _requirement_card(),
            receiver_role="executor",
            receiver_capabilities={"database_read", "log_analysis"},
            receiver_tools={"sql_query"},
            receiver_environment={"read-only SQL"},
        )
        assert compat["compatible"] is False
        assert compat["incompatible"] is True
        assert 0.0 < compat["tool_satisfaction"] < 1.0

    def test_receiver_c_missing_everything_is_incompatible(self):
        compat = _memory_receiver_compatibility(
            _requirement_card(),
            receiver_role="planner",
            receiver_capabilities=set(),
            receiver_tools=set(),
            receiver_environment=set(),
        )
        assert compat["compatible"] is False
        assert compat["incompatible"] is True
        assert compat["tool_satisfaction"] == 0.0
        assert compat["capability_satisfaction"] == 0.0


class TestNoOutcomeLeakage:
    """清单 18.7: paired labels/outcomes never influence candidates."""

    def test_card_rejects_label_tokens(self):
        card = _memory().routing_card
        card_json = json.dumps(card.model_dump(mode="json"), sort_keys=True).lower()
        for forbidden in (
            "y_share", "y_withhold", "team_success", "payload",
            "ordered_steps", "raw_action_sequence",
        ):
            assert forbidden not in card_json, forbidden

    def test_candidates_independent_of_paired_labels(self):
        """Candidate construction takes no outcome input: manifests stay
        identical no matter which paired labels the downstream evaluation
        observes."""
        memories = [_memory()]
        recipients = [_recipient()]
        manifest_a = build_cross_task_candidates(memories=memories, recipients=recipients)
        manifest_b = build_cross_task_candidates(memories=memories, recipients=recipients)
        assert manifest_a.model_dump(mode="json") == manifest_b.model_dump(mode="json")
        # Cohort tagging is metadata-derived as well.
        records = manifest_a.candidates[0].candidate_records
        assert records, "expected at least one candidate record"
        for record in records:
            assert record.score_components
            assert record.candidate_source in {
                "semantic_top",
                "receiver_compatible",
                "receiver_incompatible_hard_negative",
                "cross_receiver_anchor",
            }


class TestTreatmentEdgeUnchanged:
    """清单 18.15: edge key = (task, receiver, memory); provenance-free."""

    def test_edge_key_ignores_source_agent_changes(self):
        record = {
            "task_id": "t1",
            "receiver_agent_id": "recv1",
            "candidate_memory_id": "mem1",
            "generation_seed": 7,
            "memory_source_trajectory_id": "trajA",
            "memory_source_task_id": "task_src",
            "memory_source_split": "train",
        }
        mutated = {
            **record,
            "memory_source_trajectory_id": "trajZ",
            "memory_source_task_id": "task_other",
            "memory_source_agent_id": "agent-z",
        }
        assert treatment_edge_key(record) == ("t1", "recv1", "mem1")
        assert treatment_edge_key(record) == treatment_edge_key(mutated)
        assert seed_pair_key(record) == seed_pair_key(mutated)
