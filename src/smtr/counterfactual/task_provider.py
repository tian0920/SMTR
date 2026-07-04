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
    "flip_pos_to_neg",
    "flip_neg_to_pos",
    "flip_neu_to_neg",
    "flip_neu_to_pos",
]


@dataclass(frozen=True)
class ToyTaskSpec:
    episode_id: str
    task_id: str
    task: str
    environment_observation: dict
    target_memory_id: str
    scenario: ScenarioName
    forced_prefix: tuple[str, ...] = ()


class CounterfactualToyTaskProvider:
    # Mapping from flip scenario to its target memory (reuses base scenario memories)
    _FLIP_TARGET_MAP: dict[str, str] = {
        "flip_pos_to_neg": "mem_cf_positive",
        "flip_neg_to_pos": "mem_cf_negative",
        "flip_neu_to_neg": "mem_cf_neutral_success",
        "flip_neu_to_pos": "mem_cf_neutral_failure",
    }
    # Mapping from flip scenario to the prefix memory that enables the flip
    _FLIP_PREFIX_MAP: dict[str, str] = {
        "flip_pos_to_neg": "mem_cf_block",
        "flip_neg_to_pos": "mem_cf_override",
        "flip_neu_to_neg": "mem_cf_block",
        "flip_neu_to_pos": "mem_cf_enable",
    }

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
            "environment_regime": ["v1", "v2", "limited"][seed % 3],
            # Hidden perturbation: affects FakeLLM planning but excluded from
            # the context fingerprint so the critic cannot exploit it.
            "perturbation_offset": seed % 7,
        }
        # Determine default_sequence: the outcome when only prefix memories are
        # visible (target withheld).  Each scenario has an intended "baseline"
        # outcome; flip scenarios need the OPPOSITE default from their base so
        # that the prefix-only branch produces the flipped result.
        _WRONG_DEFAULTS = {
            "positive",        # recover succeeds → default must fail
            "neutral_failure", # irrelevant → default must fail
            "prefix_sensitive",# recover succeeds → default must fail
            "flip_neg_to_pos", # base negative (valid default) → invert to wrong
            "flip_neu_to_neg", # base neutral_success (valid default) → invert
        }
        if scenario in _WRONG_DEFAULTS:
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
            target_memory_id=self._target_memory_for_scenario(scenario),
            scenario=scenario,
            forced_prefix=self._forced_prefix_for_scenario(scenario),
        )

    def _base_scenario(self, scenario: str) -> str:
        """Map flip scenarios to their base scenario for environment setup."""
        return {
            "flip_pos_to_neg": "positive",
            "flip_neg_to_pos": "negative",
            "flip_neu_to_neg": "neutral_success",
            "flip_neu_to_pos": "neutral_failure",
        }.get(scenario, scenario)

    def _target_memory_for_scenario(self, scenario: str) -> str:
        """Get target memory ID for any scenario including flip scenarios."""
        if scenario in self._FLIP_TARGET_MAP:
            return self._FLIP_TARGET_MAP[scenario]
        if scenario == "prefix_sensitive":
            return "mem_cf_prefix_recover"
        return f"mem_cf_{scenario}"

    def _forced_prefix_for_scenario(self, scenario: str) -> tuple[str, ...]:
        """Get forced prefix memories for flip scenarios."""
        if scenario in self._FLIP_PREFIX_MAP:
            return (self._FLIP_PREFIX_MAP[scenario],)
        return ()

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
        self._ensure_flip_prefix_memories(repository)

    def _ensure_flip_prefix_memories(self, repository: SharedMemoryRepository) -> None:
        """Ensure prefix memories exist for flip scenarios.

        These memories have strategies that interact with target memory strategies
        to produce outcome flips (e.g., recover + block → failure).
        """
        existing = {card.memory_id for card in repository.get_routing_cards()}
        flip_prefix_defs = [
            ("mem_cf_block", "block", "Counterfactual target artifact task block prefix"),
            ("mem_cf_conflict", "conflict", "Counterfactual target artifact task conflict prefix"),
            ("mem_cf_override", "override", "Counterfactual target artifact task override prefix"),
            ("mem_cf_amplify", "amplify", "Counterfactual target artifact task amplify prefix"),
            (
                "mem_cf_reinforce",
                "reinforce",
                "Counterfactual target artifact task reinforce prefix",
            ),
            ("mem_cf_enable", "enable", "Counterfactual target artifact task enable prefix"),
        ]
        for memory_id, strategy, goal in flip_prefix_defs:
            if memory_id in existing:
                continue
            now = utc_now()
            payload = ProcedurePayload(
                memory_id=memory_id,
                version=1,
                writer_agent_id="counterfactual_task_provider",
                source_episode_id="counterfactual_seed",
                goal=goal,
                preconditions=["scenario=flip"],
                steps=[f"strategy: {strategy}", "scenario marker: flip_prefix"],
                postconditions=["modulates target memory effect when co-selected"],
                created_at=now,
            )
            card = MemoryRoutingCard(
                memory_id=memory_id,
                active_payload_version=1,
                goal_summary=f"counterfactual flip prefix: {strategy}",
                task_tags=["counterfactual", "flip", "prefix", "planning"],
                precondition_summary="scenario=flip",
                postcondition_summary="modulates target memory effect",
                required_environment_facts={},
                forbidden_environment_facts={},
                compatible_receiver_roles=["planner"],
                compatible_receiver_capabilities=["planning"],
                created_at=now,
                updated_at=now,
            )
            repository.create_memory(payload, card)

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
            else "flip-prefix"
            if any(m.startswith("mem_cf_") and m not in {
                "mem_cf_positive", "mem_cf_negative",
                "mem_cf_neutral_success", "mem_cf_neutral_failure",
                "mem_cf_prefix_recover",
            } for m in selected_before)
            else "redundant-prefix"
        )
        target_family = (
            "recover"
            if target_memory_id in {"mem_cf_positive", "mem_cf_prefix_recover"}
            else "destructive"
            if target_memory_id == "mem_cf_negative"
            else "neutral"
        )
        # Use base scenario as scenario_family so that flip records share the
        # same family as their base scenario.  This prevents trivial 1:1
        # scenario_family → transfer_class mappings (T-14).
        scenario_family = {
            "prefix_sensitive": "locked-recovery",
        }.get(scenario, self._base_scenario(scenario))
        return EvaluationGroupMetadata(
            scenario_family=scenario_family,
            environment_regime=["v1", "v2", "limited"][seed % 3],
            target_memory_family=target_family,
            prefix_structure_family=prefix_family,
            factor_combination_id=(
                f"{scenario}|{target_family}|{prefix_family}|{seed % 3}|{seed % 4}"
            ),
            surface_variant_id=f"variant-{seed % 5}",
            mechanism_group_id=f"{self._base_scenario(scenario)}|{target_family}|{seed % 4}",
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
