"""Bridge MARBLE real paired records to SMTR critic training inputs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from smtr.counterfactual.schemas import (
    BranchOutcome,
    EvaluationGroupMetadata,
    PairedInterventionRecord,
    RoutingFeatureSnapshot,
    transfer_class_from_outcomes,
)
from smtr.marble.real_data import (
    ProceduralRoutingCard,
    RealPairedRecord,
    RealProceduralMemory,
)
from smtr.memory.execution_evidence import selected_set_signature
from smtr.memory.schemas import ContextFingerprint

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def marble_routing_card_to_snapshot(
    card: ProceduralRoutingCard,
    memory_id: str,
    *,
    active_payload_version: int = 1,
) -> RoutingFeatureSnapshot:
    """Convert a MARBLE ProceduralRoutingCard to a SMTR RoutingFeatureSnapshot."""
    forbidden: dict[str, str] = {}
    for index, risk in enumerate(card.known_risks):
        key = _slugify(risk) or f"risk_{index}"
        forbidden[key] = risk

    return RoutingFeatureSnapshot(
        memory_id=memory_id,
        active_payload_version=active_payload_version,
        goal_summary=card.goal_summary,
        task_tags=list(card.task_tags),
        precondition_summary=card.precondition_summary,
        postcondition_summary=card.expected_effect,
        required_environment_facts={},
        forbidden_environment_facts=forbidden,
        compatible_receiver_roles=["agent1", "agent2", "agent3", "agent4", "agent5"],
        compatible_receiver_capabilities=["database_diagnosis"],
        execution_success_alpha=1.0,
        execution_success_beta=1.0,
        execution_success_count=0,
        execution_failure_count=0,
        paired_positive_transfer_count=0,
        paired_negative_transfer_count=0,
        paired_neutral_transfer_count=0,
    )


def marble_context_fingerprint(
    *,
    recipient_task_id: str,
    task_meta: dict[str, Any],
    selected_memory_ids: list[str] | None = None,
    episode_id: str | None = None,
) -> ContextFingerprint:
    """Build a ContextFingerprint from MARBLE task metadata."""
    selected = selected_memory_ids or []
    environment_type = str(task_meta.get("environment_type") or "database")
    task_tags = _extract_task_tags(task_meta)

    return ContextFingerprint(
        task_id=str(recipient_task_id),
        task_tags=task_tags,
        receiver_agent_id="agent1",
        receiver_role="database_diagnostician",
        receiver_capabilities=["database_diagnosis"],
        environment_facts={"environment_type": environment_type},
        task_stage="execution",
        selected_memory_ids=selected,
        selected_set_signature=selected_set_signature(selected),
        episode_id=episode_id or f"marble_{recipient_task_id}",
    )


def marble_record_to_training_input(
    record: RealPairedRecord,
    memory: RealProceduralMemory,
    task_meta: dict[str, Any],
) -> PairedInterventionRecord:
    """Convert a MARBLE RealPairedRecord to a SMTR PairedInterventionRecord."""
    if not record.valid:
        raise ValueError(
            f"cannot convert invalid paired record {record.pair_id}: "
            f"failure_reason={record.failure_reason}"
        )
    if record.Y_share is None or record.Y_withhold is None:
        raise ValueError(f"paired record {record.pair_id} lacks Y_share or Y_withhold")

    candidate_snapshot = marble_routing_card_to_snapshot(
        card=memory.routing_card,
        memory_id=memory.memory_id,
    )
    context = marble_context_fingerprint(
        recipient_task_id=record.recipient_task_id,
        task_meta=task_meta,
        episode_id=record.pair_id,
    )

    share_success = bool(record.Y_share)
    withhold_success = bool(record.Y_withhold)
    share_outcome = _branch_outcome(
        success=share_success,
        memory_id=memory.memory_id,
        visible=share_success,
    )
    withhold_outcome = _branch_outcome(
        success=withhold_success,
        memory_id=memory.memory_id,
        visible=False,
    )
    transfer_class = transfer_class_from_outcomes(record.Y_share, record.Y_withhold)

    return PairedInterventionRecord(
        schema_version="1.3",
        record_id=record.pair_id,
        episode_id=record.pair_id,
        task_id=str(record.recipient_task_id),
        graph_node="marble_database_root_cause",
        receiver_agent_id="agent1",
        receiver_role="database_diagnostician",
        task_stage="execution",
        evaluation_group_metadata=EvaluationGroupMetadata(
            scenario_family="database",
            environment_regime="marble_db_docker",
            target_memory_family="procedural_database_diagnosis",
            data_source="marble_database",
        ),
        candidate_memory_id=memory.memory_id,
        candidate_payload_version=1,
        candidate_card_snapshot=candidate_snapshot,
        selected_before_card_snapshots=[],
        selected_before_payload_versions={},
        candidate_order=[memory.memory_id],
        target_index=0,
        selected_before=[],
        prefix_size=0,
        decision_context=context,
        memory_store_revision=0,
        memory_snapshot_digest=_digest_str(
            {"memory_id": memory.memory_id, "pair_id": record.pair_id}
        ),
        runtime_snapshot_digest=_digest_str(
            {"task_id": record.recipient_task_id, "seed": record.generation_seed}
        ),
        continuation_policy_name="marble_real_engine",
        continuation_policy_version="1",
        common_seed=record.generation_seed,
        share_outcome=share_outcome,
        withhold_outcome=withhold_outcome,
        y_share=record.Y_share,
        y_withhold=record.Y_withhold,
        transfer_class=transfer_class,
        data_source="marble_database",
        created_at=datetime.now(UTC),
    )


def _branch_outcome(
    *, success: bool, memory_id: str, visible: bool
) -> BranchOutcome:
    return BranchOutcome(
        team_success=success,
        team_reward=1.0 if success else 0.0,
        team_summary=f"marble_database_outcome_{'success' if success else 'failure'}",
        final_environment_observation={"environment_type": "database"},
        selected_memory_ids_by_agent={"agent1": [memory_id] if visible else []},
        router_trace=[],
        target_memory_visible_to_receiver=visible,
        selected_final_at_target_node=[memory_id] if visible else [],
    )


def _slugify(text: str) -> str:
    tokens = _TOKEN_RE.findall(text.lower())
    return "_".join(tokens[:4])


def _extract_task_tags(task_meta: dict[str, Any]) -> list[str]:
    tags = []
    scenario = task_meta.get("scenario") or task_meta.get("environment_type")
    if scenario:
        tags.append(str(scenario))
    root_causes = task_meta.get("root_causes")
    if isinstance(root_causes, list):
        for cause in root_causes[:4]:
            tags.append(f"cause:{_slugify(str(cause))}")
    return sorted(set(tags))


def _digest_str(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
