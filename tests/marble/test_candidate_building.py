"""Tests for candidate building (Writer-Agnostic rewrite).

Compatibility is memory-requirement vs receiver-state satisfaction; writer
identity never participates (清单 Writer-Agnostic 第五章).
"""

from __future__ import annotations

import json

from smtr.core.types import (
    MemoryProvenance,
    MemoryRoutingCard,
    ProcedurePayload,
)
from smtr.marble.real_data import (
    ExtractedMemory,
    build_cross_task_candidates,
)


def _make_memory(
    memory_id: str,
    source_task: str,
    *,
    required_tools: tuple[str, ...] = ("sql_tool",),
    required_capabilities: tuple[str, ...] = ("sql",),
    execution_role_tags: tuple[str, ...] = (),
    environment_constraints: tuple[str, ...] = (),
) -> ExtractedMemory:
    provenance = MemoryProvenance(
        source_agent_id="w1",
        source_agent_role="unknown",
        source_task_id=source_task,
        source_trajectory_id=f"traj_{memory_id}",
        source_split="train",
        source_scenario="database",
    )
    return ExtractedMemory(
        memory_id=memory_id,
        payload=ProcedurePayload(
            memory_id=memory_id,
            procedure="1. Do something",
            provenance=provenance,
        ),
        routing_card=MemoryRoutingCard(
            memory_id=memory_id,
            goal_summary="Diagnose issue",
            task_tags=("database", "performance"),
            required_tools=required_tools,
            required_capabilities=required_capabilities,
            execution_role_tags=execution_role_tags,
            environment_constraints=environment_constraints,
        ),
    )


def test_exclude_same_task_memory():
    """Candidate generation must exclude memories from the same task."""
    mem = _make_memory("m1", source_task="task_A")
    recipients = [{"task_id": "task_A", "agent_id": "r1", "agent_role": "executor", "agent_capabilities": ["sql"]}]
    manifest = build_cross_task_candidates(memories=[mem], recipients=recipients, top_k=4)
    assert len(manifest.candidates[0].candidate_records) == 0


def test_only_train_source_memory():
    """Only train-source memories should be used (enforced by extraction)."""
    mem = _make_memory("m1", source_task="train_task")
    recipients = [{"task_id": "test_task", "agent_id": "r1", "agent_role": "executor", "agent_capabilities": []}]
    manifest = build_cross_task_candidates(memories=[mem], recipients=recipients, top_k=4)
    assert manifest.memory_source_split == "train"


def test_compatible_and_incompatible_match_types():
    """Candidates must include both requirement-compatible and
    requirement-incompatible memory-receiver pairs; writer identity is
    never consulted."""
    mem_match = _make_memory("m1", source_task="t1")
    mem_mismatch = _make_memory(
        "m2", source_task="t2",
        required_tools=("admin_console",),
        required_capabilities=("cluster_admin",),
        execution_role_tags=("planner",),
        environment_constraints=("maintenance window",),
    )
    recipients = [{
        "task_id": "target", "agent_id": "r1", "agent_role": "executor",
        "agent_capabilities": ["sql"], "tool_names": ["sql_tool"],
    }]
    manifest = build_cross_task_candidates(memories=[mem_match, mem_mismatch], recipients=recipients, top_k=4)
    records = manifest.candidates[0].candidate_records
    match_types = {r.memory_id: r.memory_receiver_match_type for r in records}
    assert match_types["m1"] == "compatible"
    assert match_types["m2"] == "incompatible"


def test_score_equals_mean_of_components():
    """Score must equal the plain mean of the metadata-only components
    (清单 Writer-Agnostic 5.3: weights are never tuned on outcomes)."""
    mem = _make_memory("m1", source_task="t1",
                       environment_constraints=("read-only SQL",))
    recipients = [{"task_id": "target", "agent_id": "r1", "agent_role": "executor",
                   "agent_capabilities": ["sql"], "tool_names": ["sql_tool"],
                   "instruction": "database performance",
                   "environment_signature": ["read-only SQL"]}]
    manifest = build_cross_task_candidates(memories=[mem], recipients=recipients, top_k=4)
    for rec in manifest.candidates[0].candidate_records:
        mean_components = sum(rec.score_components.values()) / len(rec.score_components)
        assert abs(rec.score - mean_components) < 1e-3
        # Writer identity never contributes to the score.
        assert not any("writer" in key for key in rec.score_components)


def test_candidate_manifest_no_payload():
    """Candidate manifest must not contain payload or provenance."""
    mem = _make_memory("m1", source_task="t1")
    recipients = [{"task_id": "target", "agent_id": "r1", "agent_role": "executor", "agent_capabilities": []}]
    manifest = build_cross_task_candidates(memories=[mem], recipients=recipients, top_k=4)
    manifest_json = json.dumps(manifest.model_dump(mode="json")).lower()
    assert "do something" not in manifest_json
    assert "payload" not in manifest_json
    assert "provenance" not in manifest_json
    assert "source_agent" not in manifest_json
