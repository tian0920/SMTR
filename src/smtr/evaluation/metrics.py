"""Cross-agent transfer evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_method_metrics(
    *,
    method: str,
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
    negative_risk_budget: float = 0.2,
) -> dict[str, Any]:
    """Compute paper-required metrics for one method.

    Args:
        method: method name
        decisions: list of router decision dicts with keys:
            candidate_memory_id, receiver_agent_id, receiver_role, writer_role, action,
            task_id, generation_seed
        paired_outcomes: list of paired record dicts
        negative_risk_budget: threshold for quarantine (not hardcoded)
    """
    n_total = len(decisions)
    n_share = sum(1 for d in decisions if d["action"] == "share")
    share_rate = n_share / max(1, n_total)

    # Build outcome lookup using full pair key: (task_id, seed, receiver, memory)
    outcome_by_key: dict[tuple[str, int, str, str], dict] = {}
    for rec in paired_outcomes:
        key = (
            str(rec.get("task_id", "")),
            int(rec.get("generation_seed", 0)),
            str(rec.get("receiver_agent_id", "")),
            str(rec.get("candidate_memory_id", "")),
        )
        outcome_by_key[key] = rec

    # Compute transfer metrics
    positive_transfer_total = 0
    positive_transfer_shared = 0
    negative_transfer_total = 0
    negative_transfer_shared = 0
    negative_transfer_withheld = 0
    shared_nonharmful = 0
    all_shared_with_pair = 0
    policy_success_count = 0
    policy_total = 0

    for d in decisions:
        key = (
            str(d.get("task_id", "")),
            int(d.get("generation_seed", 0)),
            str(d.get("receiver_agent_id", "")),
            str(d.get("candidate_memory_id", "")),
        )
        rec = outcome_by_key.get(key)
        if rec is None:
            continue
        label = rec.get("label", "")
        action = d["action"]
        policy_total += 1

        # Policy success: use the potential outcome matching the action
        if action == "share":
            policy_success = rec.get("share", {}).get("team_success", False)
        else:
            policy_success = rec.get("withhold", {}).get("team_success", False)
        if policy_success:
            policy_success_count += 1

        if label == "positive_transfer":
            positive_transfer_total += 1
            if action == "share":
                positive_transfer_shared += 1
                shared_nonharmful += 1
                all_shared_with_pair += 1
        elif label == "negative_transfer":
            negative_transfer_total += 1
            if action == "share":
                negative_transfer_shared += 1
                all_shared_with_pair += 1
            else:
                negative_transfer_withheld += 1
        elif label == "neutral_success":
            if action == "share":
                shared_nonharmful += 1
                all_shared_with_pair += 1
        elif label == "neutral_failure":
            if action == "share":
                shared_nonharmful += 1
                all_shared_with_pair += 1

    paired_policy_success_rate = policy_success_count / max(1, policy_total)
    positive_transfer_share_rate = positive_transfer_shared / max(1, positive_transfer_total)
    negative_transfer_exposure_rate = negative_transfer_shared / max(1, negative_transfer_total)
    negative_transfer_rejection_rate = negative_transfer_withheld / max(1, negative_transfer_total)
    safe_exposure_precision = shared_nonharmful / max(1, all_shared_with_pair)
    safe_exposure_recall = positive_transfer_shared / max(1, positive_transfer_total)
    decision_coverage = policy_total / max(1, n_total)

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

    # Same memory different receiver flip count (P2-3)
    memory_receiver_actions: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for d in decisions:
        memory_id = d["candidate_memory_id"]
        receiver_id = d.get("receiver_agent_id", "")
        action = d["action"]
        memory_receiver_actions[memory_id][receiver_id].add(action)

    same_memory_different_receiver_flip_count = 0
    for memory_id, receiver_map in memory_receiver_actions.items():
        if len(receiver_map) < 2:
            continue
        # Check if different receivers have different final actions
        receiver_final_actions: dict[str, str] = {}
        for receiver_id, actions in receiver_map.items():
            # Final action: share if share is in the set
            receiver_final_actions[receiver_id] = "share" if "share" in actions else "withhold"
        unique_actions = set(receiver_final_actions.values())
        if len(unique_actions) > 1:
            same_memory_different_receiver_flip_count += 1

    # Receiver-specific quarantine pair count (P2-4: use negative_risk_budget param)
    quarantine_count = sum(
        1 for d in decisions
        if d["action"] == "withhold" and d.get("eta_hat", 0) > negative_risk_budget
    )

    return {
        "method": method,
        "paired_policy_success_rate": round(paired_policy_success_rate, 4),
        "share_rate": round(share_rate, 4),
        "positive_transfer_share_rate": round(positive_transfer_share_rate, 4),
        "negative_transfer_exposure_rate": round(negative_transfer_exposure_rate, 4),
        "negative_transfer_rejection_rate": round(negative_transfer_rejection_rate, 4),
        "safe_exposure_precision": round(safe_exposure_precision, 4),
        "safe_exposure_recall": round(safe_exposure_recall, 4),
        "decision_coverage": round(decision_coverage, 4),
        "writer_receiver_mismatch_share_rate": round(writer_receiver_mismatch_share_rate, 4),
        "same_memory_different_receiver_flip_count": same_memory_different_receiver_flip_count,
        "receiver_specific_quarantine_pair_count": quarantine_count,
    }


def compute_writer_receiver_breakdown(
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Breakdown metrics by writer_role -> receiver_role."""
    outcome_by_key: dict[tuple[str, int, str, str], dict] = {}
    for rec in paired_outcomes:
        key = (
            str(rec.get("task_id", "")),
            int(rec.get("generation_seed", 0)),
            str(rec.get("receiver_agent_id", "")),
            str(rec.get("candidate_memory_id", "")),
        )
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
            if outcome_by_key.get(
                (str(d.get("task_id", "")), int(d.get("generation_seed", 0)),
                 str(d.get("receiver_agent_id", "")), str(d.get("candidate_memory_id", ""))),
                {},
            ).get("label") == "negative_transfer"
        )
        results.append({
            "writer_role": w_role,
            "receiver_role": r_role,
            "count": n,
            "share_rate": round(n_share / max(1, n), 4),
            "negative_transfer_rate": round(neg_transfer / max(1, n), 4),
        })
    return results
