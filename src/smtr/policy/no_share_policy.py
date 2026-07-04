from typing import Any, Literal

from smtr.policy.schemas import ContinuationPolicyManifest, with_fingerprint


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


def create_no_share_manifest() -> ContinuationPolicyManifest:
    return with_fingerprint(
        ContinuationPolicyManifest(
            policy_name="FrozenNoShareContinuationPolicy",
            policy_version="1",
            policy_kind="frozen_no_share",
        )
    )
