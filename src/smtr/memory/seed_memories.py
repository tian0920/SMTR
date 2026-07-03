from collections.abc import Iterable

from smtr.memory.pool import SharedMemoryPool
from smtr.memory.repository import SharedMemoryRepository
from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload, utc_now


def _memory(
    *,
    memory_id: str,
    goal: str,
    goal_summary: str,
    task_tags: list[str],
    preconditions: list[str],
    steps: list[str],
    postconditions: list[str],
    required: dict[str, str | bool | int | float],
    forbidden: dict[str, str | bool | int | float],
    roles: list[str],
    capabilities: list[str],
) -> tuple[MemoryRoutingCard, ProcedurePayload]:
    now = utc_now()
    payload = ProcedurePayload(
        memory_id=memory_id,
        version=1,
        writer_agent_id="seed",
        source_episode_id="seed_memories",
        goal=goal,
        preconditions=preconditions,
        steps=steps,
        postconditions=postconditions,
        created_at=now,
    )
    card = MemoryRoutingCard(
        memory_id=memory_id,
        active_payload_version=1,
        goal_summary=goal_summary,
        task_tags=task_tags,
        precondition_summary="; ".join(preconditions),
        postcondition_summary="; ".join(postconditions),
        required_environment_facts=required,
        forbidden_environment_facts=forbidden,
        compatible_receiver_roles=roles,
        compatible_receiver_capabilities=capabilities,
        created_at=now,
        updated_at=now,
    )
    return card, payload


def build_seed_memories() -> list[tuple[MemoryRoutingCard, ProcedurePayload]]:
    return [
        _memory(
            memory_id="mem_plan_artifact_sequence",
            goal="Create an ordered plan for obtaining an artifact.",
            goal_summary="plan valid action sequence to obtain target artifact",
            task_tags=["artifact", "ordered-actions", "planning"],
            preconditions=["target_artifact is known", "valid_sequence is visible"],
            steps=["inspect valid action sequence", "emit ordered artifact acquisition plan"],
            postconditions=["plan contains gather open collect order"],
            required={"tag:artifact": True},
            forbidden={},
            roles=["planner"],
            capabilities=["planning", "sequence-design"],
        ),
        _memory(
            memory_id="mem_execute_tool_chain",
            goal="Execute a tool chain to collect a target artifact.",
            goal_summary="execute valid action sequence to obtain target artifact",
            task_tags=["artifact", "tool-chain", "execution", "ordered-actions"],
            preconditions=["plan actions are concrete", "resource_available=true"],
            steps=["apply gathered key action", "apply chest opening action", "collect artifact"],
            postconditions=["artifact is in inventory"],
            required={"resource_available": True, "tag:tool-chain": True},
            forbidden={},
            roles=["executor"],
            capabilities=["execution", "tool-use"],
        ),
        _memory(
            memory_id="mem_critic_success_check",
            goal="Judge task success from final environment state.",
            goal_summary="judge whether target artifact was obtained",
            task_tags=["artifact", "verification", "critic"],
            preconditions=["final observation is available"],
            steps=["read final inventory", "compare inventory with target artifact"],
            postconditions=["team success and reward are assigned"],
            required={"tag:verification": True},
            forbidden={},
            roles=["critic"],
            capabilities=["verification", "rewarding"],
        ),
        _memory(
            memory_id="mem_execute_tool_v1",
            goal="Run a version one tool procedure.",
            goal_summary=(
                "execute valid action sequence to obtain target artifact with tool version v1"
            ),
            task_tags=["artifact", "tool-chain", "ordered-actions", "v1"],
            preconditions=["tool_version=v1"],
            steps=["select v1 adapter", "call v1 artifact transfer"],
            postconditions=["v1 tool result is recorded"],
            required={"tool_version": "v1"},
            forbidden={},
            roles=["executor"],
            capabilities=["execution", "tool-use"],
        ),
        _memory(
            memory_id="mem_execute_tool_v2",
            goal="Run a version two tool procedure.",
            goal_summary=(
                "execute valid action sequence to obtain target artifact with tool version v2"
            ),
            task_tags=["artifact", "tool-chain", "ordered-actions", "v2"],
            preconditions=["tool_version=v2"],
            steps=["select v2 adapter", "call v2 artifact transfer"],
            postconditions=["v2 tool result is recorded"],
            required={"tool_version": "v2"},
            forbidden={},
            roles=["executor"],
            capabilities=["execution", "tool-use"],
        ),
        _memory(
            memory_id="mem_plan_requires_resource",
            goal="Plan only when a shared resource is available.",
            goal_summary="plan artifact acquisition when resource is available",
            task_tags=["artifact", "planning", "resource"],
            preconditions=["resource_available=true"],
            steps=["reserve available resource", "include resource use in plan"],
            postconditions=["plan references available resource"],
            required={"resource_available": True},
            forbidden={},
            roles=["planner"],
            capabilities=["planning", "resource-reasoning"],
        ),
        _memory(
            memory_id="mem_execute_unlocked_resource",
            goal="Execute artifact collection only when the resource is not locked.",
            goal_summary=(
                "execute valid action sequence to obtain target artifact when resource is unlocked"
            ),
            task_tags=["artifact", "execution", "resource", "ordered-actions"],
            preconditions=["resource_locked is not true"],
            steps=["confirm resource lock is absent", "execute unlocked resource action"],
            postconditions=["unlocked resource execution is complete"],
            required={"resource_available": True},
            forbidden={"resource_locked": True},
            roles=["executor"],
            capabilities=["execution", "resource-reasoning"],
        ),
        _memory(
            memory_id="mem_critic_artifact_role_mismatch",
            goal="Evaluate artifact task semantics from a critic perspective.",
            goal_summary="assess target artifact action sequence for correctness",
            task_tags=["artifact", "ordered-actions", "verification"],
            preconditions=["candidate plan or trace is available"],
            steps=["compare artifact sequence to expected order", "mark semantic violations"],
            postconditions=["critic verdict is produced"],
            required={"tag:artifact": True},
            forbidden={},
            roles=["critic"],
            capabilities=["verification"],
        ),
    ]


def build_seed_memory_pool() -> SharedMemoryPool:
    memories = build_seed_memories()
    return SharedMemoryPool(
        routing_cards=[card for card, _ in memories],
        payloads=[payload for _, payload in memories],
    )


def seed_repository(repository: SharedMemoryRepository) -> list[str]:
    existing = {card.memory_id for card in repository.get_routing_cards()}
    inserted: list[str] = []
    for card, payload in build_seed_memories():
        if card.memory_id in existing:
            continue
        repository.create_memory(payload, card)
        inserted.append(card.memory_id)
    return inserted


def memory_ids(memories: Iterable[tuple[MemoryRoutingCard, ProcedurePayload]]) -> list[str]:
    return [card.memory_id for card, _ in memories]
