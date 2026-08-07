"""Routing card source invariance (清单 Writer-Agnostic §16).

Two memories with identical procedures/routing cards but different
source agents must produce identical routing cards. Only provenance
fields may differ.
"""

from __future__ import annotations

from smtr.router.transfer_features import build_routing_card_from_pool_entry


def _pool_entry(
    *,
    memory_id: str,
    source_agent_id: str,
    source_task_id: str = "train_t1",
) -> dict:
    return {
        "memory_id": memory_id,
        "payload": {
            "procedure": "Step 1. Check pg_stat_statements.",
            "provenance": {
                "source_agent_id": source_agent_id,
                "source_task_id": source_task_id,
                "source_trajectory_id": f"traj_{source_agent_id}",
                "source_split": "train",
            },
        },
        "routing_card": {
            "goal_summary": "Guide database performance diagnosis",
            "task_tags": ["database", "performance"],
            "required_tools": ["pg_stat_statements"],
            "required_capabilities": ["database_diagnosis"],
            "execution_role_tags": ["executor"],
            "environment_constraints": [],
            "precondition_tags": [],
            "procedure_type": "diagnostic",
            "procedure_length_bucket": "short",
            "read_write_scope": "read_only",
            "evidence_count": 3,
        },
    }


def test_routing_card_ignores_source_agent_identity():
    entry_a = _pool_entry(memory_id="m1", source_agent_id="agent_alpha")
    entry_b = _pool_entry(memory_id="m1", source_agent_id="agent_beta")

    card_a = build_routing_card_from_pool_entry(entry_a)
    card_b = build_routing_card_from_pool_entry(entry_b)

    assert card_a == card_b


def test_routing_card_ignores_source_task_and_trajectory():
    entry_a = _pool_entry(
        memory_id="m1",
        source_agent_id="agent_a",
        source_task_id="task_100",
    )
    entry_b = _pool_entry(
        memory_id="m1",
        source_agent_id="agent_b",
        source_task_id="task_999",
    )

    card_a = build_routing_card_from_pool_entry(entry_a)
    card_b = build_routing_card_from_pool_entry(entry_b)

    assert card_a.goal_summary == card_b.goal_summary
    assert card_a.required_tools == card_b.required_tools
    assert card_a.required_capabilities == card_b.required_capabilities
    assert card_a.execution_role_tags == card_b.execution_role_tags
    assert card_a.procedure_type == card_b.procedure_type
    assert card_a.read_write_scope == card_b.read_write_scope
