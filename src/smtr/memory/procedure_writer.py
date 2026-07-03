import hashlib
import re
from typing import Any

from smtr.memory.schemas import FactValue, MemoryRoutingCard, ProcedurePayload, utc_now

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fact_value(value: Any) -> FactValue | None:
    if isinstance(value, str | bool | int | float):
        return value
    return None


def _keywords(text: str) -> list[str]:
    stop = {"a", "an", "and", "for", "of", "the", "to", "using", "with"}
    return sorted({token for token in TOKEN_RE.findall(text.lower()) if token not in stop})


def _memory_id(episode_id: str, task: str) -> str:
    digest = hashlib.sha256(f"{episode_id}:{task}".encode()).hexdigest()[:12]
    return f"mem_episode_{digest}"


class DeterministicProcedureWriter:
    def __init__(self, *, writer_agent_role: str = "executor") -> None:
        self.writer_agent_role = writer_agent_role

    def write_from_successful_episode(
        self,
        *,
        episode_id: str,
        writer_agent_id: str,
        task: str,
        planner_output: dict[str, Any],
        executor_output: dict[str, Any],
        initial_environment: dict[str, Any],
        final_environment: dict[str, Any],
    ) -> tuple[ProcedurePayload, MemoryRoutingCard]:
        if not self._is_successful(executor_output, final_environment):
            raise ValueError("DeterministicProcedureWriter only writes successful episodes")

        now = utc_now()
        required_facts = self._required_environment_facts(initial_environment)
        preconditions = [f"{key}={value}" for key, value in required_facts.items()]
        steps = self._steps(executor_output, planner_output)
        postconditions = self._postconditions(initial_environment, final_environment)
        memory_id = _memory_id(episode_id, task)

        payload = ProcedurePayload(
            memory_id=memory_id,
            version=1,
            writer_agent_id=writer_agent_id,
            source_episode_id=episode_id,
            goal=task,
            preconditions=preconditions,
            steps=steps,
            postconditions=postconditions,
            created_at=now,
        )
        tags = self._task_tags(task)
        card = MemoryRoutingCard(
            memory_id=memory_id,
            active_payload_version=1,
            goal_summary=" ".join(_keywords(task)[:12]),
            task_tags=tags,
            precondition_summary="; ".join(preconditions[:4]),
            postcondition_summary="; ".join(postconditions[:4]),
            required_environment_facts=required_facts,
            forbidden_environment_facts={},
            compatible_receiver_roles=[self.writer_agent_role],
            compatible_receiver_capabilities=[self.writer_agent_role, "toy-environment"],
            execution_success_alpha=1.0,
            execution_success_beta=1.0,
            created_at=now,
            updated_at=now,
        )
        return payload, card

    def _is_successful(
        self, executor_output: dict[str, Any], final_environment: dict[str, Any]
    ) -> bool:
        if executor_output.get("team_success") is True:
            return True
        target = final_environment.get("target_artifact")
        inventory = final_environment.get("inventory")
        return isinstance(inventory, list) and target in inventory

    def _required_environment_facts(self, environment: dict[str, Any]) -> dict[str, FactValue]:
        keys = ["resource_available", "tool_version", "resource_locked", "location"]
        facts: dict[str, FactValue] = {}
        for key in keys:
            value = _fact_value(environment.get(key))
            if value is not None:
                facts[key] = value
        tags = environment.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                facts[f"tag:{str(tag).lower()}"] = True
        return facts

    def _steps(
        self, executor_output: dict[str, Any], planner_output: dict[str, Any]
    ) -> list[str]:
        raw_actions = executor_output.get("actions") or planner_output.get("plan") or []
        steps: list[str] = []
        for action in raw_actions:
            if isinstance(action, dict):
                name = str(action.get("name", "unknown_action"))
                args = {
                    key: value
                    for key, value in sorted(action.items())
                    if key != "name" and _fact_value(value) is not None
                }
                suffix = ""
                if args:
                    suffix = " with " + ",".join(f"{key}={value}" for key, value in args.items())
                steps.append(f"call {name}{suffix}")
            else:
                steps.append(f"call {str(action)}")
        if not steps:
            steps.append("verify target artifact exists")
        return steps

    def _postconditions(
        self, initial_environment: dict[str, Any], final_environment: dict[str, Any]
    ) -> list[str]:
        postconditions: list[str] = []
        for key, final_value in sorted(final_environment.items()):
            if _fact_value(final_value) is not None and initial_environment.get(key) != final_value:
                postconditions.append(f"{key}={final_value}")
        initial_inventory = set(initial_environment.get("inventory", []))
        final_inventory = set(final_environment.get("inventory", []))
        for item in sorted(final_inventory - initial_inventory):
            postconditions.append(f"inventory_contains={item}")
        return postconditions or ["final environment changed consistently"]

    def _task_tags(self, task: str) -> list[str]:
        tokens = set(_keywords(task))
        tags = set(tokens)
        if "artifact" in tokens:
            tags.add("artifact")
        if {"sequence", "ordered", "plan"} & tokens:
            tags.add("ordered-actions")
        if {"execute", "action", "tool"} & tokens:
            tags.add("tool-chain")
        return sorted(tags)
