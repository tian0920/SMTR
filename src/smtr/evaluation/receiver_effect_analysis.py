"""Receiver-effect analysis (清单第十二章).

Core analyses beyond the same-memory flip count:

* eligible memory count (same memory evaluated by >= 2 receivers);
* predicted decision flip rate across receivers;
* empirical effect-sign flip rate (tau_emp signs differ across receivers);
* receiver ranking quality (pairwise accuracy / Spearman / top-receiver);
* risk heterogeneity stratified by receiver role, writer-receiver role
  pair, capability overlap bucket and tool overlap bucket, with a
  task-relevance-stratified negative-transfer rate so "task irrelevant"
  is never conflated with receiver mismatch.

Empirical tau per (memory, receiver) is the mean of y_share - y_withhold
over the paired records; predicted tau/action comes from the router
traces of one method.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from smtr.router.transfer_features import _overlap_bucket

# (y_share, y_withhold) -> four-outcome label.
OUTCOME_TO_LABEL = {
    (0, 0): "neutral_failure",
    (0, 1): "negative_transfer",
    (1, 0): "positive_transfer",
    (1, 1): "neutral_success",
}


def record_seed(rec: dict[str, Any]) -> int:
    """Generation seed of a paired record (generation_seed or common_seed)."""
    if rec.get("generation_seed") is not None:
        return int(rec["generation_seed"])
    return int(rec.get("common_seed", 0))


def record_label(rec: dict[str, Any]) -> str:
    """Four-outcome label derived from the paired potential outcomes."""
    y_share = int(rec.get("y_share", 0))
    y_withhold = int(rec.get("y_withhold", 0))
    return OUTCOME_TO_LABEL[(y_share, y_withhold)]


def _empirical_tau_table(paired_records: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Mean y_share - y_withhold per (memory, receiver_agent)."""
    sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in paired_records:
        key = (str(rec.get("candidate_memory_id", "")), str(rec.get("receiver_agent_id", "")))
        sums[key].append(int(rec.get("y_share", 0)) - int(rec.get("y_withhold", 0)))
    return {key: float(np.mean(vals)) for key, vals in sums.items()}


def _receiver_decision_table(
    decisions: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Final predicted action/tau per (memory, receiver_agent) from traces."""
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for d in decisions:
        key = (str(d.get("candidate_memory_id", "")), str(d.get("receiver_agent_id", "")))
        entry = table.setdefault(key, {"action": "withhold", "tau_hat": float(d.get("tau_hat", 0.0))})
        if d.get("action") == "share":
            entry["action"] = "share"
        entry["tau_hat"] = float(d.get("tau_hat", entry["tau_hat"]))
    return table


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr, yr = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    if xr.std() == 0 or yr.std() == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def analyze_receiver_effect(
    *,
    decisions: list[dict[str, Any]],
    paired_records: list[dict[str, Any]],
    cards_by_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full receiver-effect report for one method's decision traces."""
    cards_by_id = cards_by_id or {}
    decision_table = _receiver_decision_table(decisions)
    tau_emp = _empirical_tau_table(paired_records)

    # --- eligible memories: same memory evaluated by >= 2 receivers ---
    memory_receivers: dict[str, set[str]] = defaultdict(set)
    for memory_id, receiver_id in decision_table:
        memory_receivers[memory_id].add(receiver_id)
    eligible = {m for m, rs in memory_receivers.items() if len(rs) >= 2}

    # --- 12.2 predicted decision flip rate ---
    decision_flip_count = 0
    for memory_id in eligible:
        actions = {
            decision_table[(memory_id, r)]["action"]
            for r in memory_receivers[memory_id]
            if (memory_id, r) in decision_table
        }
        if len(actions) > 1:
            decision_flip_count += 1

    # --- 12.3 empirical effect-sign flip rate + correct flip identification ---
    effect_sign_flip_count = 0
    correct_flip_identification = 0
    for memory_id in eligible:
        taus = [
            tau_emp[(memory_id, r)]
            for r in memory_receivers[memory_id]
            if (memory_id, r) in tau_emp
        ]
        empirical_flip = any(t > 0 for t in taus) and any(t < 0 for t in taus)
        if empirical_flip:
            effect_sign_flip_count += 1
        predicted_flip = decision_flip_count_for(memory_id, memory_receivers, decision_table)
        if empirical_flip == predicted_flip:
            correct_flip_identification += 1

    # --- 12.4 receiver ranking quality ---
    pairwise_total = pairwise_correct = 0
    top_total = top_correct = 0
    spearmans: list[float] = []
    for memory_id in eligible:
        receivers = sorted(
            r for r in memory_receivers[memory_id]
            if (memory_id, r) in decision_table and (memory_id, r) in tau_emp
        )
        if len(receivers) < 2:
            continue
        pred = np.array([decision_table[(memory_id, r)]["tau_hat"] for r in receivers])
        emp = np.array([tau_emp[(memory_id, r)] for r in receivers])
        for i in range(len(receivers)):
            for j in range(i + 1, len(receivers)):
                pairwise_total += 1
                if (pred[i] - pred[j]) * (emp[i] - emp[j]) >= 0:
                    pairwise_correct += 1
        spearmans.append(_spearman(pred, emp))
        top_total += 1
        if receivers[int(np.argmax(pred))] == receivers[int(np.argmax(emp))]:
            top_correct += 1

    # --- 12.5 risk heterogeneity ---
    risk_heterogeneity = _risk_heterogeneity(paired_records, cards_by_id)

    n_eligible = len(eligible)
    return {
        "eligible_memory_count": n_eligible,
        "predicted_decision_flip_count": decision_flip_count,
        "predicted_decision_flip_rate": round(decision_flip_count / max(1, n_eligible), 4),
        "empirical_effect_sign_flip_count": effect_sign_flip_count,
        "empirical_effect_sign_flip_rate": round(effect_sign_flip_count / max(1, n_eligible), 4),
        "correct_flip_identification_rate": round(
            correct_flip_identification / max(1, n_eligible), 4),
        "receiver_ranking": {
            "memories_ranked": top_total,
            "pairwise_receiver_ranking_accuracy": round(
                pairwise_correct / max(1, pairwise_total), 4),
            "mean_spearman_correlation": round(float(np.mean(spearmans)), 4) if spearmans else 0.0,
            "top_receiver_accuracy": round(top_correct / max(1, top_total), 4),
        },
        "risk_heterogeneity": risk_heterogeneity,
    }


def decision_flip_count_for(
    memory_id: str,
    memory_receivers: dict[str, set[str]],
    decision_table: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    """Whether predicted decisions differ across receivers for one memory."""
    actions = {
        decision_table[(memory_id, r)]["action"]
        for r in memory_receivers[memory_id]
        if (memory_id, r) in decision_table
    }
    return len(actions) > 1


def _task_relevance_bucket(rec: dict[str, Any], cards_by_id: dict[str, Any]) -> str:
    """Coarse task-relevance bucket from task_tags overlap in decision context."""
    ctx = rec.get("decision_context", {}) or {}
    task_tags = set(ctx.get("task_tags", ()) or ())
    card = cards_by_id.get(str(rec.get("candidate_memory_id", "")))
    card_tags = set(card.task_tags) if card is not None else set()
    if not task_tags or not card_tags:
        return "unknown"
    return "relevant" if task_tags & card_tags else "irrelevant"


def _risk_heterogeneity(
    paired_records: list[dict[str, Any]],
    cards_by_id: dict[str, Any],
) -> dict[str, Any]:
    """Negative-transfer rates stratified by receiver/writer structure."""

    def _rate_table(key_fn) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[int]] = defaultdict(list)
        for rec in paired_records:
            key = key_fn(rec)
            if key is None:
                continue
            buckets[key].append(1 if record_label(rec) == "negative_transfer" else 0)
        return {
            key: {
                "n": len(vals),
                "negative_transfer_rate": round(float(np.mean(vals)), 4),
            }
            for key, vals in sorted(buckets.items())
        }

    def _receiver_role(rec):
        return str(rec.get("receiver_role", "unknown"))

    def _writer_receiver_pair(rec):
        card = cards_by_id.get(str(rec.get("candidate_memory_id", "")))
        writer_role = card.writer.role if card is not None else "unknown"
        return f"{writer_role}->{rec.get('receiver_role', 'unknown')}"

    def _cap_bucket(rec):
        card = cards_by_id.get(str(rec.get("candidate_memory_id", "")))
        if card is None:
            return None
        ctx = rec.get("decision_context", {}) or {}
        r_caps = set(ctx.get("receiver_capabilities", ()) or ())
        return _overlap_bucket(set(card.writer.capabilities), r_caps)

    def _tool_bucket(rec):
        card = cards_by_id.get(str(rec.get("candidate_memory_id", "")))
        if card is None:
            return None
        ctx = rec.get("decision_context", {}) or {}
        r_tools = set(ctx.get("receiver_tool_names", ()) or ())
        return _overlap_bucket(set(card.writer.tool_names), r_tools)

    # Negative-transfer rate stratified by task relevance so "task
    # irrelevant" is never conflated with receiver mismatch.
    by_relevance = _rate_table(lambda rec: _task_relevance_bucket(rec, cards_by_id))

    return {
        "by_receiver_role": _rate_table(_receiver_role),
        "by_writer_receiver_role_pair": _rate_table(_writer_receiver_pair),
        "by_capability_overlap_bucket": _rate_table(_cap_bucket),
        "by_tool_overlap_bucket": _rate_table(_tool_bucket),
        "negative_transfer_rate_by_task_relevance": by_relevance,
    }
