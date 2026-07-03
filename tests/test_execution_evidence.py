import pytest

from smtr.memory.execution_evidence import build_context_fingerprint
from smtr.memory.schemas import ExecutionEvidence


def test_context_fingerprint_keeps_only_lightweight_facts() -> None:
    fingerprint = build_context_fingerprint(
        task_id="task-1",
        task_tags=["artifact"],
        receiver_agent_id="executor-1",
        receiver_role="executor",
        receiver_capabilities=["execution"],
        environment_observation={
            "resource_available": True,
            "inventory": ["secret", "list"],
            "full_prompt": {"nested": "not allowed"},
        },
        task_stage="executor",
        selected_memory_ids=["b", "a"],
        episode_id="episode-1",
    )

    assert fingerprint.environment_facts == {"resource_available": True}
    assert fingerprint.selected_memory_ids == ["b", "a"]
    assert fingerprint.selected_set_signature


def test_failed_execution_evidence_requires_failure_category() -> None:
    fingerprint = build_context_fingerprint(
        task_id="task-1",
        task_tags=[],
        receiver_agent_id="executor-1",
        receiver_role="executor",
        receiver_capabilities=[],
        environment_observation={},
        task_stage="executor",
        selected_memory_ids=[],
        episode_id="episode-1",
    )

    with pytest.raises(ValueError, match="failure_category"):
        ExecutionEvidence(
            memory_id="m",
            payload_version=1,
            context=fingerprint,
            execution_success=False,
            source="synthetic_toy",
        )
