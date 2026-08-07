"""Receiver-effect analysis (清单第十二章).

Core analyses beyond the same-memory flip count:

* eligible memory count (same memory evaluated by >= 2 receivers);
* predicted decision flip rate across receivers;
* empirical effect-sign flip rate (tau_emp signs differ across receivers);
* receiver ranking quality (pairwise accuracy / Spearman / top-receiver);
* risk heterogeneity stratified by receiver role and by memory-requirement
  satisfaction against the receiver (tools / capabilities / environment /
  execution role), plus procedure type and length bucket, with a
  task-relevance-stratified negative-transfer rate so "task irrelevant"
  is never conflated with receiver mismatch. Writer identity never enters
  any stratification (清单 Writer-Agnostic 第十三章).

Empirical tau per (memory, receiver) is the mean of y_share - y_withhold
over the paired records; predicted tau/action comes from the router
traces of one method.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label

# 清单 P0-12: anchor group key is (target_task_id, candidate_memory_id).
ReceiverEffectGroupKey = tuple[str, str]


def record_seed(rec: dict[str, Any]) -> int:
    """Generation seed of a paired record (generation_seed or common_seed)."""
    if rec.get("generation_seed") is not None:
        return int(rec["generation_seed"])
    return int(rec.get("common_seed", 0))


def record_label(rec: dict[str, Any]) -> str:
    """Four-outcome label derived from the canonical paired outcomes."""
    return paired_record_label(rec)


def _empirical_tau_table(paired_records: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Mean y_share - y_withhold per (memory, receiver_agent)."""
    sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in paired_records:
        key = (str(rec.get("candidate_memory_id", "")), str(rec.get("receiver_agent_id", "")))
        y_share, y_withhold = get_paired_outcomes(rec)
        sums[key].append(y_share - y_withhold)
    return {key: float(np.mean(vals)) for key, vals in sums.items()}


def _receiver_decision_table(
    decisions: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Final predicted action/tau per (memory, receiver_agent) from traces."""
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for d in decisions:
        key = (str(d.get("candidate_memory_id", "")), str(d.get("receiver_agent_id", "")))
        entry = table.setdefault(
            key, {"action": "withhold", "tau_hat": float(d.get("tau_hat", 0.0))}
        )
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


def build_receiver_effect_anchor_groups(
    paired_records: list[dict[str, Any]],
    *,
    min_seeds_per_receiver: int = 1,
) -> dict[ReceiverEffectGroupKey, dict[str, list[dict[str, Any]]]]:
    """Cross-receiver anchor groups (清单 P0-12).

    A group ``(target_task_id, candidate_memory_id)`` is kept only when the
    same task + memory combination is evaluated by at least two different
    receivers and every receiver carries at least ``min_seeds_per_receiver``
    valid seed records.
    """
    grouped: dict[ReceiverEffectGroupKey, dict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for rec in paired_records:
        key = (
            str(rec.get("task_id", "")),
            str(rec.get("candidate_memory_id", "")),
        )
        grouped[key][str(rec.get("receiver_agent_id", ""))].append(rec)
    anchors: dict[ReceiverEffectGroupKey, dict[str, list[dict[str, Any]]]] = {}
    for key, by_receiver in grouped.items():
        sufficient = {
            receiver: recs
            for receiver, recs in by_receiver.items()
            if len(recs) >= min_seeds_per_receiver
        }
        if len(sufficient) >= 2:
            anchors[key] = dict(sufficient)
    return anchors


_LABEL_TO_Q_INDEX = {
    "neutral_failure": 0,
    "negative_transfer": 1,
    "positive_transfer": 2,
    "neutral_success": 3,
}


def empirical_receiver_effects(
    paired_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Empirical receiver-specific effects per (task, memory, receiver).

    Label frequencies over the cell's seed records give q-hat_00/01/10/11;
    tau_hat = q10 - q01 and eta_hat = q01 (清单 P0-13).
    """
    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for rec in paired_records:
        cell = (
            str(rec.get("task_id", "")),
            str(rec.get("candidate_memory_id", "")),
            str(rec.get("receiver_agent_id", "")),
        )
        cells[cell].append(record_label(rec))
    effects: dict[tuple[str, str, str], dict[str, float]] = {}
    for cell, labels in cells.items():
        q = [0.0, 0.0, 0.0, 0.0]
        for label in labels:
            q[_LABEL_TO_Q_INDEX[label]] += 1.0
        q = [v / len(labels) for v in q]
        effects[cell] = {
            "q00": q[0],
            "q01": q[1],
            "q10": q[2],
            "q11": q[3],
            "tau_hat": q[2] - q[1],
            "eta_hat": q[1],
            "seed_count": float(len(labels)),
        }
    return effects


def analyze_receiver_effect_anchor_groups(
    anchor_groups: dict[ReceiverEffectGroupKey, dict[str, list[dict[str, Any]]]],
    effects: dict[tuple[str, str, str], dict[str, float]],
    *,
    epsilon_star: float,
    decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-anchor-group receiver-effect statistics (清单 P0-13).

    Reports the tau range across receivers, transfer-sign flip, harm-risk
    flip against the validation-selected epsilon_star, and (when method
    decisions are supplied) the SMTR decision flip for identical
    task + memory across receivers.
    """
    decision_cells = _decision_cell_table(decisions) if decisions else {}
    groups: dict[str, Any] = {}
    counts = {"transfer_sign_flip": 0, "harm_risk_flip": 0, "decision_flip": 0}
    for (task_id, memory_id), by_receiver in anchor_groups.items():
        taus: dict[str, float] = {}
        etas: dict[str, float] = {}
        for receiver in by_receiver:
            cell = (task_id, memory_id, receiver)
            if cell in effects:
                taus[receiver] = effects[cell]["tau_hat"]
                etas[receiver] = effects[cell]["eta_hat"]
        transfer_sign_flip = (
            any(t > 0 for t in taus.values()) and any(t < 0 for t in taus.values())
        )
        harm_risk_flip = (
            any(e <= epsilon_star for e in etas.values())
            and any(e > epsilon_star for e in etas.values())
        )
        actions = {
            decision_cells[(task_id, memory_id, receiver)]["action"]
            for receiver in by_receiver
            if (task_id, memory_id, receiver) in decision_cells
        }
        decision_flip = len(actions) > 1
        for flag, name in (
            (transfer_sign_flip, "transfer_sign_flip"),
            (harm_risk_flip, "harm_risk_flip"),
            (decision_flip, "decision_flip"),
        ):
            if flag:
                counts[name] += 1
        groups[f"{task_id}|{memory_id}"] = {
            "receivers": sorted(by_receiver),
            "tau_by_receiver": {r: round(v, 4) for r, v in sorted(taus.items())},
            "eta_by_receiver": {r: round(v, 4) for r, v in sorted(etas.items())},
            "delta_tau_range": (
                round(max(taus.values()) - min(taus.values()), 4) if taus else 0.0
            ),
            "transfer_sign_flip": transfer_sign_flip,
            "harm_risk_flip": harm_risk_flip,
            "decision_flip": decision_flip,
        }
    n_groups = max(1, len(anchor_groups))
    return {
        "epsilon_star": epsilon_star,
        "anchor_group_count": len(anchor_groups),
        "transfer_sign_flip_count": counts["transfer_sign_flip"],
        "transfer_sign_flip_rate": round(counts["transfer_sign_flip"] / n_groups, 4),
        "harm_risk_flip_count": counts["harm_risk_flip"],
        "harm_risk_flip_rate": round(counts["harm_risk_flip"] / n_groups, 4),
        "decision_flip_count": counts["decision_flip"],
        "decision_flip_rate": round(counts["decision_flip"] / n_groups, 4),
        "groups": groups,
    }


def compare_receiver_effect_methods(
    *,
    decisions_by_method: dict[str, list[dict[str, Any]]],
    paired_records: list[dict[str, Any]],
    epsilon_star: float,
    min_seeds_per_receiver: int = 1,
) -> dict[str, Any]:
    """Receiver-effect comparison table (清单 P0-14).

    Compares each method against the empirical receiver-specific effects on
    anchor groups: sign accuracy, receiver-specific decision accuracy,
    harmful-exposure rejection (overall and per receiver) and same-memory
    decision flip precision/recall against empirical transfer-sign flips.
    """
    anchor_groups = build_receiver_effect_anchor_groups(
        paired_records, min_seeds_per_receiver=min_seeds_per_receiver
    )
    effects = empirical_receiver_effects(paired_records)
    anchor_cells = {
        (task_id, memory_id, receiver)
        for (task_id, memory_id), by_receiver in anchor_groups.items()
        for receiver in by_receiver
    }
    table: dict[str, Any] = {}
    for method, decisions in decisions_by_method.items():
        cells = _decision_cell_table(decisions)
        sign_total = sign_correct = 0
        decision_total = decision_correct = 0
        flip_predicted = flip_empirical = flip_both = 0
        harmful_total = harmful_rejected = 0
        harmful_by_receiver: dict[str, list[int]] = defaultdict(list)
        for cell in anchor_cells:
            emp = effects.get(cell)
            pred = cells.get(cell)
            if emp is None:
                continue
            if pred is not None:
                sign_total += 1
                if (pred["tau_hat"] > 0) == (emp["tau_hat"] > 0):
                    sign_correct += 1
                optimal_share = emp["tau_hat"] > 0 and emp["eta_hat"] <= epsilon_star
                decision_total += 1
                if (pred["action"] == "share") == optimal_share:
                    decision_correct += 1
            harmful = record_cell_is_harmful(cell, effects)
            if pred is not None and harmful:
                harmful_total += 1
                rejected = pred["action"] != "share"
                harmful_rejected += int(rejected)
                harmful_by_receiver[cell[2]].append(int(rejected))
        # Same-memory decision flip precision/recall over anchor groups.
        for (task_id, memory_id), by_receiver in anchor_groups.items():
            emp_taus = [
                effects[(task_id, memory_id, r)]["tau_hat"]
                for r in by_receiver
                if (task_id, memory_id, r) in effects
            ]
            if len(emp_taus) < 2:
                continue
            emp_flip = any(t > 0 for t in emp_taus) and any(t < 0 for t in emp_taus)
            pred_actions = {
                cells[(task_id, memory_id, r)]["action"]
                for r in by_receiver
                if (task_id, memory_id, r) in cells
            }
            pred_flip = len(pred_actions) > 1
            flip_predicted += int(pred_flip)
            flip_empirical += int(emp_flip)
            flip_both += int(pred_flip and emp_flip)
        table[method] = {
            "anchor_group_count": len(anchor_groups),
            "receiver_effect_sign_accuracy": (
                round(sign_correct / sign_total, 4) if sign_total else 0.0
            ),
            "receiver_specific_decision_accuracy": (
                round(decision_correct / decision_total, 4) if decision_total else 0.0
            ),
            "harmful_exposure_rejection_rate": (
                round(harmful_rejected / harmful_total, 4) if harmful_total else 1.0
            ),
            "harmful_exposure_rejection_by_receiver": {
                receiver: round(float(np.mean(flags)), 4)
                for receiver, flags in sorted(harmful_by_receiver.items())
            },
            "same_memory_decision_flip_precision": (
                round(flip_both / flip_predicted, 4) if flip_predicted else 0.0
            ),
            "same_memory_decision_flip_recall": (
                round(flip_both / flip_empirical, 4) if flip_empirical else 0.0
            ),
        }
    return {
        "epsilon_star": epsilon_star,
        "min_seeds_per_receiver": min_seeds_per_receiver,
        "methods": table,
    }


def record_cell_is_harmful(
    cell: tuple[str, str, str],
    effects: dict[tuple[str, str, str], dict[str, float]],
) -> bool:
    """A cell is harmful when its empirical negative-transfer rate is > 0."""
    emp = effects.get(cell)
    return bool(emp and emp["q01"] > 0)


def _decision_cell_table(
    decisions: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Aggregate decision traces to (task, memory, receiver) cells.

    The cell action is ``share`` when at least half of the seed-level
    decisions share; tau_hat/eta_hat are the seed means (eta read from the
    trace's calibrated field).
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for d in decisions:
        cell = (
            str(d.get("task_id", "")),
            str(d.get("candidate_memory_id", "")),
            str(d.get("receiver_agent_id", "")),
        )
        grouped[cell].append(d)
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell, entries in grouped.items():
        share_rate = float(np.mean([d.get("action") == "share" for d in entries]))
        cells[cell] = {
            "action": "share" if share_rate >= 0.5 else "withhold",
            "tau_hat": float(np.mean([float(d.get("tau_hat", 0.0)) for d in entries])),
            "eta_hat": float(np.mean([
                float(d.get("eta_calibrated", d.get("eta_hat", 0.0))) for d in entries
            ])),
        }
    return cells


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


def _satisfaction_bucket(required: tuple[str, ...], available: set[str]) -> str:
    """Bucket a memory requirement set against receiver availability."""
    if not required:
        return "no_requirements"
    required_set = set(required)
    if required_set <= available:
        return "satisfied"
    return "partial" if required_set & available else "unsatisfied"


def _risk_heterogeneity(
    paired_records: list[dict[str, Any]],
    cards_by_id: dict[str, Any],
) -> dict[str, Any]:
    """Negative-transfer rates stratified by memory-requirement satisfaction.

    Strata are derived from the routing card's explicit requirements and
    the receiver's pre-execution state only; writer identity is never used
    (清单 Writer-Agnostic 第十三章).
    """

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

    def _card_of(rec: dict[str, Any]):
        return cards_by_id.get(str(rec.get("candidate_memory_id", "")))

    def _receiver_role(rec):
        return str(rec.get("receiver_role", "unknown"))

    def _tool_satisfaction(rec):
        card = _card_of(rec)
        if card is None:
            return None
        r_tools = set(rec.get("receiver_tool_names", []) or [])
        return _satisfaction_bucket(card.required_tools, r_tools)

    def _capability_satisfaction(rec):
        card = _card_of(rec)
        if card is None:
            return None
        r_caps = set(rec.get("receiver_capabilities", []) or [])
        return _satisfaction_bucket(card.required_capabilities, r_caps)

    def _environment_satisfaction(rec):
        card = _card_of(rec)
        if card is None:
            return None
        r_env = set(rec.get("environment_signature", []) or [])
        return _satisfaction_bucket(card.environment_constraints, r_env)

    def _execution_role_satisfaction(rec):
        card = _card_of(rec)
        if card is None:
            return None
        if not card.execution_role_tags:
            return "no_requirements"
        receiver_role = str(rec.get("receiver_role", "unknown"))
        return (
            "satisfied"
            if receiver_role in {str(r) for r in card.execution_role_tags}
            else "unsatisfied"
        )

    def _procedure_type(rec):
        card = _card_of(rec)
        return str(card.procedure_type) if card is not None else None

    def _procedure_length_bucket(rec):
        card = _card_of(rec)
        return str(card.procedure_length_bucket) if card is not None else None

    # Negative-transfer rate stratified by task relevance so "task
    # irrelevant" is never conflated with receiver mismatch.
    by_relevance = _rate_table(lambda rec: _task_relevance_bucket(rec, cards_by_id))

    return {
        "by_receiver_role": _rate_table(_receiver_role),
        "risk_by_tool_satisfaction": _rate_table(_tool_satisfaction),
        "risk_by_capability_satisfaction": _rate_table(_capability_satisfaction),
        "risk_by_environment_satisfaction": _rate_table(_environment_satisfaction),
        "risk_by_execution_role_satisfaction": _rate_table(_execution_role_satisfaction),
        "risk_by_procedure_type": _rate_table(_procedure_type),
        "risk_by_procedure_length_bucket": _rate_table(_procedure_length_bucket),
        "negative_transfer_rate_by_task_relevance": by_relevance,
    }
