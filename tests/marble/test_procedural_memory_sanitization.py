"""Phase 13: procedural memory sanitization tests.

Ensures injected memory payloads cannot replay answers / ground truth /
evaluator scores, and that only procedural content reaches execution.
"""

from __future__ import annotations

import pytest

from smtr.memory.procedural_sanitizer import (
    PayloadLeakageError,
    assert_clean_payload,
    audit_payload_leakage,
    sanitize_candidate,
)


def test_answer_and_score_fragments_are_removed():
    raw = (
        "Reusable strategy: anchor with a low opening offer. "
        "final answer: accept at $95. "
        "ground_truth: root cause is deadlock. "
        "score: 5 team_success: true"
    )
    out = sanitize_candidate(memory_id="m1", source_agent_id="agent2", raw_content=raw)
    low = out.procedural_content.lower()
    assert "reusable strategy" in low
    assert "final answer" not in low
    assert "ground_truth" not in low
    assert out.removed_fragments


def test_origin_task_id_reference_removed():
    raw = "Apply the diagnostic heuristic from task_id: 77 step by step."
    out = sanitize_candidate(
        memory_id="m1", source_agent_id="agent2", raw_content=raw, task_id="77"
    )
    assert "77" not in out.procedural_content


def test_audit_detects_leakage_patterns():
    leaked = "SELECT * FROM orders; result: 12 rows. hidden state: seed=1"
    hits = audit_payload_leakage(leaked)
    assert hits


def test_clean_procedural_payload_passes():
    clean = (
        "Procedure: first query the slow-query log, then isolate the "
        "contention window, finally coordinate buyer and seller turns."
    )
    assert audit_payload_leakage(clean) == []
    assert_clean_payload(clean)


def test_assert_raises_on_leakage():
    with pytest.raises(PayloadLeakageError):
        assert_clean_payload("The correct answer: restart service A")
