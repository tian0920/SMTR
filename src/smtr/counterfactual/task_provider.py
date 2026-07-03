from dataclasses import dataclass
from typing import Literal

from smtr.counterfactual.schemas import EvaluationGroupMetadata
from smtr.memory.repository import SharedMemoryRepository
from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload, utc_now

ScenarioName = Literal[
    "positive",
    "negative",
    "neutral_success",
    "neutral_failure",
    "prefix_sensitive",
]


@dataclass(frozen=True)
class ToyTaskSpec:
    episode_id: str
    task_id: str
    task: str
    environment_observation: dict
    target_memory_id: str
    scenario: ScenarioName


class CounterfactualToyTaskProvider:
    def generate(self, *, scenario: ScenarioName, seed: int) -> ToyTaskSpec:
        task = f"Counterfactual {scenario} target artifact task"
        base_environment = {
            "location": "workbench",
            "inventory": [],
            "target_artifact": "target_artifact",
            "valid_sequence": ["gather_key", "open_chest", "collect_artifact"],
            "next_index": 0,
            "tags": ["artifact", "ordered-actions", "tool-chain", "verification", scenario],
            "tool_version": "v1",
            "resource_available": True,
            "resource_locked": False,
            "scenario": scenario,
            "last_error": None,
        }
        if scenario in {"positive", "neutral_failure", "prefix_sensitive"}:
            base_environment["default_sequence"] = ["wrong_action"]
        else:
            base_environment["default_sequence"] = [
                "gather_key",
                "open_chest",
                "collect_artifact",
            ]
        return ToyTaskSpec(
            episode_id=f"cf-{scenario}-{seed}",
            task_id=f"task-{scenario}",
            task=task,
            environment_observation=base_environment,
            target_memory_id=(
                "mem_cf_prefix_recover"
                if scenario == "prefix_sensitive"
                else f"mem_cf_{scenario}"
            ),
            scenario=scenario,
        )

    def ensure_memories(self, repository: SharedMemoryRepository) -> None:
        existing = {card.memory_id for card in repository.get_routing_cards()}
        for scenario, strategy in [
            ("positive", "recover"),
            ("negative", "destructive"),
            ("neutral_success", "irrelevant"),
            ("neutral_failure", "irrelevant"),
            ("prefix_sensitive", "recover"),
        ]:
            memory_id = (
                "mem_cf_prefix_recover"
                if scenario == "prefix_sensitive"
                else f"mem_cf_{scenario}"
            )
            if memory_id in existing:
                continue
            now = utc_now()
            payload = ProcedurePayload(
                memory_id=memory_id,
                version=1,
                writer_agent_id="counterfactual_task_provider",
                source_episode_id="counterfactual_seed",
                goal=f"Counterfactual {scenario} target artifact task",
                preconditions=[f"scenario={scenario}"],
                steps=[f"strategy: {strategy}", f"scenario marker: {scenario}"],
                postconditions=["planner strategy may change only when payload is visible"],
                created_at=now,
            )
            card = MemoryRoutingCard(
                memory_id=memory_id,
                active_payload_version=1,
                goal_summary=f"counterfactual {scenario} target artifact task",
                task_tags=["counterfactual", scenario, "artifact", "planning"],
                precondition_summary=f"scenario={scenario}",
                postcondition_summary="strategy may alter planner output",
                required_environment_facts={"scenario": scenario},
                forbidden_environment_facts={},
                compatible_receiver_roles=["planner"],
                compatible_receiver_capabilities=["planning"],
                created_at=now,
                updated_at=now,
            )
            repository.create_memory(payload, card)
        self._ensure_prefix_lock_memory(repository)

    def evaluation_metadata(
        self,
        *,
        scenario: ScenarioName,
        target_memory_id: str,
        selected_before: list[str],
        seed: int,
    ) -> EvaluationGroupMetadata:
        prefix_family = (
            "empty"
            if not selected_before
            else "lock-prefix"
            if "mem_prefix_lock" in selected_before
            else "redundant-prefix"
        )
        target_family = (
            "recover"
            if target_memory_id in {"mem_cf_positive", "mem_cf_prefix_recover"}
            else "destructive"
            if target_memory_id == "mem_cf_negative"
            else "neutral"
        )
        return EvaluationGroupMetadata(
            scenario_family=(
                "locked-recovery" if scenario == "prefix_sensitive" else str(scenario)
            ),
            environment_regime=["v1", "v2", "limited"][seed % 3],
            target_memory_family=target_family,
            prefix_structure_family=prefix_family,
            factor_combination_id=(
                f"{scenario}|{target_family}|{prefix_family}|{seed % 3}|{seed % 4}"
            ),
            surface_variant_id=f"variant-{seed % 5}",
            mechanism_group_id=f"{scenario}|{target_family}|{seed % 4}",
        )

    def _ensure_prefix_lock_memory(self, repository: SharedMemoryRepository) -> None:
        if "mem_prefix_lock" in {card.memory_id for card in repository.get_routing_cards()}:
            return
        now = utc_now()
        payload = ProcedurePayload(
            memory_id="mem_prefix_lock",
            version=1,
            writer_agent_id="counterfactual_task_provider",
            source_episode_id="counterfactual_seed",
            goal="A prefix memory that locks the target before recovery.",
            preconditions=["scenario=prefix_sensitive"],
            steps=["strategy: lock_target", "scenario marker: prefix_sensitive"],
            postconditions=["target recovery becomes blocked by action order"],
            created_at=now,
        )
        card = MemoryRoutingCard(
            memory_id="mem_prefix_lock",
            active_payload_version=1,
            goal_summary="counterfactual prefix sensitive target artifact task lock prefix",
            task_tags=["counterfactual", "prefix_sensitive", "artifact", "planning"],
            precondition_summary="scenario=prefix_sensitive",
            postcondition_summary="lock target strategy may block recovery",
            required_environment_facts={"scenario": "prefix_sensitive"},
            forbidden_environment_facts={},
            compatible_receiver_roles=["planner"],
            compatible_receiver_capabilities=["planning"],
            created_at=now,
            updated_at=now,
        )
        repository.create_memory(payload, card)
