from typing import Any, Literal, Protocol


class FrozenContinuationPolicy(Protocol):
    policy_name: str
    policy_version: str

    def decide(
        self,
        *,
        candidate_id: str,
        candidate_position: int,
        target_index: int,
        selected_so_far: list[str],
        decision_context: dict[str, Any],
    ) -> Literal["share", "withhold"]: ...


class FrozenNoShareContinuationPolicy:
    policy_name = "FrozenNoShareContinuationPolicy"
    policy_version = "1"

    def decide(
        self,
        *,
        candidate_id: str,
        candidate_position: int,
        target_index: int,
        selected_so_far: list[str],
        decision_context: dict[str, Any],
    ) -> Literal["share", "withhold"]:
        del candidate_id, candidate_position, target_index, selected_so_far, decision_context
        return "withhold"
