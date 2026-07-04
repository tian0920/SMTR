from copy import deepcopy
from typing import Any, Protocol


class EnvironmentAdapter(Protocol):
    def observe(self) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, snapshot: dict[str, Any]) -> None: ...

    def apply(self, action: dict[str, Any]) -> dict[str, Any]: ...


class ToyEnvironment:
    """Toy environment with context-mode support and resource tracking.

    The environment supports three context modes that affect action outcomes:
    - "normal": default behavior, actions succeed if they match the valid sequence
    - "resource_constrained": actions consume energy; insufficient energy causes failure
    - "adversarial": actions have a chance of failing even when correct

    Resource tracking adds energy and durability fields that are consumed by actions,
    enabling richer scenario variation beyond pure sequence matching.
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        context_mode: str = "normal",
        initial_energy: int = 100,
        initial_durability: int = 100,
    ) -> None:
        self.seed = seed
        self._state: dict[str, Any] = {
            "location": "workbench",
            "inventory": [],
            "target_artifact": "target_artifact",
            "valid_sequence": ["gather_key", "open_chest", "collect_artifact"],
            "next_index": 0,
            "tags": ["artifact", "ordered-actions", "tool-chain", "verification"],
            "tool_version": "v1",
            "resource_available": True,
            "resource_locked": False,
            "last_error": None,
            # Context mode and resource tracking (v2 additions)
            "context_mode": context_mode,
            "energy": initial_energy,
            "durability": initial_durability,
            "action_cost": 10,
            "failure_threshold": 30,
        }

    def observe(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def restore(self, snapshot: dict[str, Any]) -> None:
        self._state = deepcopy(snapshot)

    @classmethod
    def clone_from_snapshot(
        cls, snapshot: dict[str, Any], *, seed: int = 7
    ) -> "ToyEnvironment":
        env = cls(seed=seed)
        env.restore(snapshot)
        return env

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        action_name = str(action.get("name", ""))
        context_mode = self._state.get("context_mode", "normal")

        # Lock action works regardless of mode
        if action_name == "lock_target":
            self._state["resource_locked"] = True
            self._state["last_error"] = "target resource locked"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}

        # Resource locked state blocks all actions
        if self._state.get("resource_locked") is True:
            self._state["last_error"] = "resource is locked"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}

        # Resource-constrained mode: check energy before action
        if context_mode == "resource_constrained":
            action_cost = self._state.get("action_cost", 10)
            if self._state["energy"] < action_cost:
                self._state["last_error"] = "insufficient energy"
                return {
                    "ok": False,
                    "action": action_name,
                    "error": "insufficient energy",
                }

        # Check valid sequence
        expected = self._state["valid_sequence"][self._state["next_index"]]
        if action_name != expected:
            self._state["last_error"] = f"expected {expected}, got {action_name}"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}

        # Adversarial mode: correct action can still fail if durability is low
        if context_mode == "adversarial":
            durability = self._state.get("durability", 100)
            threshold = self._state.get("failure_threshold", 30)
            if durability < threshold:
                self._state["durability"] = max(0, durability - 20)
                self._state["last_error"] = (
                    "adversarial failure: action failed despite correct input"
                )
                return {
                    "ok": False,
                    "action": action_name,
                    "error": "adversarial failure",
                }

        # Action succeeds: update state
        self._state["next_index"] += 1
        if action_name == "gather_key":
            self._state["inventory"].append("key")
        elif action_name == "open_chest":
            self._state["location"] = "open_chest"
        elif action_name == "collect_artifact":
            self._state["inventory"].append(self._state["target_artifact"])

        # Consume resources in resource_constrained mode
        if context_mode == "resource_constrained":
            action_cost = self._state.get("action_cost", 10)
            self._state["energy"] = max(0, self._state["energy"] - action_cost)
            self._state["durability"] = max(0, self._state["durability"] - 5)

        self._state["last_error"] = None
        return {"ok": True, "action": action_name, "observation": self.observe()}
