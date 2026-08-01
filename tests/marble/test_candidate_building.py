"""Tests for candidate building."""

from __future__ import annotations

import json

from smtr.core.types import AgentProfile, MemoryRoutingCard, ProcedurePayload
from smtr.marble.real_data import (
    ExtractedMemory,
    build_cross_task_candidates,
)


def _make_memory(memory_id: str, source_task: str, writer_role: str = "executor") -> ExtractedMemory:
    writer = AgentProfile(agent_id="w1", role=writer_role, capabilities=("sql",))
    return ExtractedMemory(
        memory_id=memory_id,
        payload=ProcedurePayload(
            memory_id=memory_id,
            procedure="1. Do something",
            writer=writer,
            source_task_id=source_task,
            source_scenario="database",
        ),
        routing_card=MemoryRoutingCard(
            memory_id=memory_id,
            goal_summary="Diagnose issue",
            task_tags=("database", "performance"),
            environment_constraints=("read-only SQL",),
            writer=writer,
            source_task_id=source_task,
            source_scenario="database",
            compatible_receiver_roles=("executor",),
            evidence_count=3,
            historical_success_count=2,
            historical_failure_count=1,
            historical_success_rate=0.67,
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


def test_matched_and_mismatched():
    """Candidates must include both matched and mismatched writer-receiver."""
    mem_match = _make_memory("m1", source_task="t1", writer_role="executor")
    mem_mismatch = _make_memory("m2", source_task="t2", writer_role="planner")
    recipients = [{"task_id": "target", "agent_id": "r1", "agent_role": "executor", "agent_capabilities": ["sql"]}]
    manifest = build_cross_task_candidates(memories=[mem_match, mem_mismatch], recipients=recipients, top_k=4)
    records = manifest.candidates[0].candidate_records
    match_types = {r.match_type for r in records}
    assert "matched_writer_receiver" in match_types
    assert "mismatched_writer_receiver" in match_types


def test_score_equals_weighted_sum():
    """Score must equal the sum of weighted components."""
    mem = _make_memory("m1", source_task="t1")
    recipients = [{"task_id": "target", "agent_id": "r1", "agent_role": "executor",
                   "agent_capabilities": ["sql"], "instruction": "database performance",
                   "environment_signature": ["read-only SQL"]}]
    manifest = build_cross_task_candidates(memories=[mem], recipients=recipients, top_k=4)
    for rec in manifest.candidates[0].candidate_records:
        weighted_sum = sum(v for k, v in rec.score_components.items() if k.endswith("_weighted"))
        assert abs(rec.score - weighted_sum) < 1e-4


def test_candidate_manifest_no_payload():
    """Candidate manifest must not contain payload."""
    mem = _make_memory("m1", source_task="t1")
    recipients = [{"task_id": "target", "agent_id": "r1", "agent_role": "executor", "agent_capabilities": []}]
    manifest = build_cross_task_candidates(memories=[mem], recipients=recipients, top_k=4)
    manifest_json = json.dumps(manifest.model_dump(mode="json")).lower()
    assert "procedure" not in manifest_json
    assert "payload" not in manifest_json
