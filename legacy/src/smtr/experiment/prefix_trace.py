"""Invocation-local prefix trace utilities."""

from pydantic import BaseModel, ConfigDict

from smtr.experiment.schemas import ComparisonRunRecord


class PrefixTraceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    method: str
    invocation_id: str
    receiver_agent_id: str
    target_memory_id: str
    required_prefix_memory_ids: list[str]
    required_prefix_in_same_invocation_candidates: bool
    required_prefix_before_target_in_same_traversal: bool
    required_prefix_selected_before_target: bool
    target_evaluated_under_required_prefix: bool
    target_action: str | None
    target_payload_visible_to_same_receiver: bool


def compute_prefix_traces(
    runs: list[ComparisonRunRecord],
    *,
    target_memory_id: str,
    required_prefix_memory_ids: list[str],
) -> list[PrefixTraceRecord]:
    """Compute prefix traces without crossing invocation or receiver boundaries."""
    records: list[PrefixTraceRecord] = []
    required = set(required_prefix_memory_ids)
    for run in runs:
        for invocation in run.invocations:
            target_decision = next(
                (
                    decision
                    for decision in invocation.decisions
                    if decision.memory_id == target_memory_id
                ),
                None,
            )
            if target_decision is None:
                continue
            candidates = set(invocation.candidate_memory_ids)
            before_target = {
                decision.memory_id
                for decision in invocation.decisions
                if decision.traversal_position < target_decision.traversal_position
            }
            selected_before = set(target_decision.selected_before_memory_ids)
            required_present = bool(required) and required <= candidates
            required_before = bool(required) and required <= before_target
            required_selected = bool(required) and required <= selected_before
            records.append(
                PrefixTraceRecord(
                    run_id=f"{run.base_episode_id}:{run.method}:{run.traversal_seed}",
                    method=run.method,
                    invocation_id=invocation.invocation_id,
                    receiver_agent_id=invocation.receiver_agent_id,
                    target_memory_id=target_memory_id,
                    required_prefix_memory_ids=list(required_prefix_memory_ids),
                    required_prefix_in_same_invocation_candidates=required_present,
                    required_prefix_before_target_in_same_traversal=required_before,
                    required_prefix_selected_before_target=required_selected,
                    target_evaluated_under_required_prefix=required_selected,
                    target_action=target_decision.action,
                    target_payload_visible_to_same_receiver=(
                        target_memory_id in invocation.visible_payload_memory_ids
                    ),
                )
            )
    return records
