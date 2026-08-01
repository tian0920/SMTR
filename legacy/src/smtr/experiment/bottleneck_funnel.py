"""Target-decision bottleneck funnel."""

from smtr.experiment.schemas import ComparisonRunRecord


def compute_bottleneck_funnel(
    runs: list[ComparisonRunRecord],
    *,
    target_memory_id: str,
) -> dict[str, int]:
    """Count monotone stages for target decision opportunities."""
    counts = {
        "stage1_candidate": 0,
        "stage2_evaluated": 0,
        "stage3_shared": 0,
        "stage4_payload_visible": 0,
        "stage5_team_success": 0,
    }
    for run in runs:
        for invocation in run.invocations:
            stage1_ok = target_memory_id in invocation.candidate_memory_ids
            target_decision = next(
                (
                    decision
                    for decision in invocation.decisions
                    if decision.memory_id == target_memory_id
                ),
                None,
            )
            stage2_ok = stage1_ok and target_decision is not None
            stage3_ok = stage2_ok and target_decision.action == "share"
            stage4_ok = stage3_ok and target_memory_id in invocation.visible_payload_memory_ids
            stage5_ok = stage4_ok and run.team_success

            counts["stage1_candidate"] += int(stage1_ok)
            counts["stage2_evaluated"] += int(stage2_ok)
            counts["stage3_shared"] += int(stage3_ok)
            counts["stage4_payload_visible"] += int(stage4_ok)
            counts["stage5_team_success"] += int(stage5_ok)

    ordered = list(counts.values())
    if any(left < right for left, right in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("bottleneck funnel counts must be monotone nonincreasing")
    return counts
