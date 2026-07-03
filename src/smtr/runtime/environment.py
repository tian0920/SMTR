from copy import deepcopy
from typing import Any, Protocol


class EnvironmentAdapter(Protocol):
    def observe(self) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, snapshot: dict[str, Any]) -> None: ...

    def apply(self, action: dict[str, Any]) -> dict[str, Any]: ...


class ToyEnvironment:
    def __init__(self, *, seed: int = 7) -> None:
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
        }

    def observe(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def restore(self, snapshot: dict[str, Any]) -> None:
        self._state = deepcopy(snapshot)

    @classmethod
    def clone_from_snapshot(cls, snapshot: dict[str, Any], *, seed: int = 7) -> "ToyEnvironment":
        env = cls(seed=seed)
        env.restore(snapshot)
        return env

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        action_name = str(action.get("name", ""))
        if action_name == "lock_target":
            self._state["resource_locked"] = True
            self._state["last_error"] = "target resource locked"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}
        if self._state.get("resource_locked") is True:
            self._state["last_error"] = "resource is locked"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}
        expected = self._state["valid_sequence"][self._state["next_index"]]
        if action_name != expected:
            self._state["last_error"] = f"expected {expected}, got {action_name}"
            return {"ok": False, "action": action_name, "error": self._state["last_error"]}

        self._state["next_index"] += 1
        if action_name == "gather_key":
            self._state["inventory"].append("key")
        elif action_name == "open_chest":
            self._state["location"] = "open_chest"
        elif action_name == "collect_artifact":
            self._state["inventory"].append(self._state["target_artifact"])
        self._state["last_error"] = None
        return {"ok": True, "action": action_name, "observation": self.observe()}
