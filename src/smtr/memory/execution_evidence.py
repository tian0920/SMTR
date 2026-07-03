import hashlib
import json
from typing import Any

from smtr.memory.schemas import ContextFingerprint, FactValue


def selected_set_signature(memory_ids: list[str]) -> str:
    encoded = json.dumps(sorted(memory_ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def coerce_fact_value(value: Any) -> FactValue | None:
    if isinstance(value, str | bool | int | float):
        return value
    return None


def build_context_fingerprint(
    *,
    task_id: str,
    task_tags: list[str],
    receiver_agent_id: str,
    receiver_role: str,
    receiver_capabilities: list[str],
    environment_observation: dict[str, Any],
    task_stage: str,
    selected_memory_ids: list[str],
    episode_id: str,
    decision_index: int | None = None,
) -> ContextFingerprint:
    environment_facts = {
        key: fact
        for key, value in sorted(environment_observation.items())
        if (fact := coerce_fact_value(value)) is not None
    }
    return ContextFingerprint(
        task_id=task_id,
        task_tags=sorted(set(task_tags)),
        receiver_agent_id=receiver_agent_id,
        receiver_role=receiver_role,
        receiver_capabilities=sorted(set(receiver_capabilities)),
        environment_facts=environment_facts,
        task_stage=task_stage,
        selected_memory_ids=list(selected_memory_ids),
        selected_set_signature=selected_set_signature(selected_memory_ids),
        episode_id=episode_id,
        decision_index=decision_index,
    )
