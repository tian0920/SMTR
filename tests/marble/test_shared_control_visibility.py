"""Runtime visibility rules for the shared control (清单 Shared-Control 第4章).

The shared control expects NO memory and forbids every candidate of its
group; any leak must invalidate the control with a leak-specific reason.
"""

from __future__ import annotations

from types import SimpleNamespace

from smtr.marble.branch_runner import _validate_control
from smtr.marble.runtime_visibility_audit import RuntimeVisibilityRecord
from smtr.marble.runtime_visibility_validator import RuntimeVisibilityValidator

FORBIDDEN = ("m1", "m2", "m3", "m4")


def _visibility_record(*, visible: tuple[str, ...]) -> RuntimeVisibilityRecord:
    return RuntimeVisibilityRecord(
        schema_version="runtime_visibility_v1",
        run_id="run_1",
        task_id="t1",
        scenario="database",
        method="shared_no_memory_control",
        branch="withhold",
        agent_id="r1",
        agent_role="executor",
        receiver_agent=True,
        turn_id=0,
        visible_memory_ids=visible,
        ordered_memory_digest="ord_digest",
        memory_payload_digest="payload_digest",
        system_prompt_digest="sys_digest",
        messages_digest="msg_digest",
        intervention_id=None,
        invocation_index=0,
        timestamp_utc="2026-01-01T00:00:00Z",
    )


def test_shared_control_with_no_visible_memory_is_verified():
    validation = RuntimeVisibilityValidator().validate(
        method="shared_control",
        branch="withhold",
        receiver_agent_ids=("r1",),
        expected_memory_ids=(),
        records=[_visibility_record(visible=())],
        forbidden_memory_ids=FORBIDDEN,
    )
    assert validation.visibility_verified is True
    assert validation.invalid_reason is None


def test_shared_control_leak_is_rejected_with_leak_reason():
    validation = RuntimeVisibilityValidator().validate(
        method="shared_control",
        branch="withhold",
        receiver_agent_ids=("r1",),
        expected_memory_ids=(),
        records=[_visibility_record(visible=("m2",))],
        forbidden_memory_ids=FORBIDDEN,
    )
    assert validation.visibility_verified is False
    assert validation.invalid_reason == "withhold_candidate_memory_leaked"


def test_validate_control_maps_leak_to_candidate_memory_leaked():
    audit = SimpleNamespace(
        real_engine_executed=True,
        outcome=SimpleNamespace(
            native_evaluator_executed=True,
            environment_valid=True,
        ),
        initial_logical_fingerprint={"combined_digest": "logical"},
        runtime_visibility_verified=False,
        runtime_visibility_invalid_reason="withhold_candidate_memory_leaked",
        cleanup_succeeded=True,
    )
    valid, reason = _validate_control(audit)
    assert valid is False
    assert reason == "candidate_memory_leaked"


def test_validate_control_accepts_clean_control():
    audit = SimpleNamespace(
        real_engine_executed=True,
        outcome=SimpleNamespace(
            native_evaluator_executed=True,
            environment_valid=True,
        ),
        initial_logical_fingerprint={"combined_digest": "logical"},
        runtime_visibility_verified=True,
        runtime_visibility_invalid_reason=None,
        cleanup_succeeded=True,
    )
    valid, reason = _validate_control(audit)
    assert valid is True
    assert reason is None
