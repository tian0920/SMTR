from typing import Any, Literal, Protocol

from smtr.policy.no_share_policy import FrozenNoShareContinuationPolicy

__all__ = ["FrozenContinuationPolicy", "FrozenNoShareContinuationPolicy"]


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
