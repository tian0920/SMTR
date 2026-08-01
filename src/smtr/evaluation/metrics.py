"""Cross-agent transfer evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_method_metrics(
    *,
    method: str,
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute paper-required metrics for one method.

    Args:
        method: method name
        decisions: list of router decision dicts with keys:
            candidate_memory_id, receiver_agent_id, receiver_role, writer_role, action
        paired_outcomes: list of paired record dicts with keys:
            candidate_memory_id, receiver_agent_id, share.team_success, withhold.team_success, label
    """
    n_total = len(decisions)
    n_share = sum(1 for d in decisions if d["action"] == "share")
    share_rate = n_share / max(1, n_total)

    # Build outcome lookup
    outcome_by_key: dict[tuple[str, str], dict] = {}
    for rec in paired_outcomes:
        key = (rec["candidate_memory_id"], rec.get("receiver_agent_id", ""))
        outcome_by_key[key] = rec

    # Compute transfer metrics
    positive_transfer = 0
    negative_transfer = 0
    harmful_rejected = 0
    team_success_count = 0

    for d in decisions:
        key = (d["candidate_memory_id"], d.get("receiver_agent_id", ""))
        rec = outcome_by_key.get(key)
        if rec is None:
            continue
        label = rec.get("label", "")
        if d["action"] == "share":
            if label == "positive_transfer":
                positive_transfer += 1
                team_success_count += 1
            elif label == "negative_transfer":
                negative_transfer += 1
            elif label == "neutral_success":
                team_success_count += 1
        else:  # withhold
            if label == "negative_transfer":
                harmful_rejected += 1

    n_with_outcome = sum(1 for d in decisions if (d["candidate_memory_id"], d.get("receiver_agent_id", "")) in outcome_by_key)
    positive_transfer_rate = positive_transfer / max(1, n_with_outcome)
    negative_transfer_rate = negative_transfer / max(1, n_with_outcome)
    harmful_exposure_rejection_rate = harmful_rejected / max(1, negative_transfer + harmful_rejected)
    team_success_rate = team_success_count / max(1, n_share)

    # Writer-receiver mismatch share rate
    mismatch_share = sum(
        1 for d in decisions
        if d["action"] == "share" and d.get("writer_role") != d.get("receiver_role")
    )
    mismatch_total = sum(
        1 for d in decisions
        if d.get("writer_role") != d.get("receiver_role")
    )
    writer_receiver_mismatch_share_rate = mismatch_share / max(1, mismatch_total)

    # Same memory different receiver decision count
    memory_decisions: dict[str, set[str]] = defaultdict(set)
    for d in decisions:
        memory_decisions[d["candidate_memory_id"]].add(d["action"])
    same_memory_different_receiver_decision_count = sum(
        1 for actions in memory_decisions.values() if len(actions) > 1
    )

    # Receiver-specific quarantine pair count
    quarantine_count = sum(
        1 for d in decisions
        if d["action"] == "withhold" and d.get("eta_hat", 0) > 0.2
    )

    return {
        "method": method,
        "team_success_rate": round(team_success_rate, 4),
        "share_rate": round(share_rate, 4),
        "positive_transfer_rate": round(positive_transfer_rate, 4),
        "negative_transfer_rate": round(negative_transfer_rate, 4),
        "harmful_exposure_rejection_rate": round(harmful_exposure_rejection_rate, 4),
        "writer_receiver_mismatch_share_rate": round(writer_receiver_mismatch_share_rate, 4),
        "same_memory_different_receiver_decision_count": same_memory_different_receiver_decision_count,
        "receiver_specific_quarantine_pair_count": quarantine_count,
        "local_positive_team_negative_count": None,  # MARBLE cannot reliably get local success
    }


def compute_writer_receiver_breakdown(
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Breakdown metrics by writer_role -> receiver_role."""
    outcome_by_key: dict[tuple[str, str], dict] = {}
    for rec in paired_outcomes:
        key = (rec["candidate_memory_id"], rec.get("receiver_agent_id", ""))
        outcome_by_key[key] = rec

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in decisions:
        w_role = d.get("writer_role", "unknown")
        r_role = d.get("receiver_role", "unknown")
        groups[(w_role, r_role)].append(d)

    results = []
    for (w_role, r_role), group_decisions in sorted(groups.items()):
        n = len(group_decisions)
        n_share = sum(1 for d in group_decisions if d["action"] == "share")
        neg_transfer = sum(
            1 for d in group_decisions
            if outcome_by_key.get((d["candidate_memory_id"], d.get("receiver_agent_id", "")), {}).get("label") == "negative_transfer"
        )
        results.append({
            "writer_role": w_role,
            "receiver_role": r_role,
            "count": n,
            "share_rate": round(n_share / max(1, n), 4),
            "negative_transfer_rate": round(neg_transfer / max(1, n), 4),
        })
    return results
