"""Continual-transfer metrics for RIMA-Transfer (§34).

Computes routing-mode statistics, global-retrieval rates, known-memory
reuse, and transfer-state growth over a continual run.

These metrics are SEPARATE from the existing ``compute_memory_metrics()``
which operates on formal admission decisions.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

__all__ = [
    "compute_transfer_routing_metrics",
    "compute_transfer_cost",
    "build_curve_records",
    "compute_episode_metrics",
    "compute_continual_learning_metrics",
    "compute_three_way_cost",
]


def compute_transfer_routing_metrics(
    routing_diagnostics: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate routing diagnostics across all (task, receiver) pairs.

    Parameters
    ----------
    routing_diagnostics : list of per-receiver diagnostic dicts
        Each dict must contain keys produced by the continual runner:
        ``routing_mode``, ``global_retrieval_triggered``,
        ``selected_source``, ``transfer_state_size_after``, etc.

    Returns
    -------
    dict with routing-mode rates, retrieval rates, selection rates,
    candidate counts, transfer-state sizes, and reuse statistics.
    """
    diags = list(routing_diagnostics)
    n = len(diags)
    if n == 0:
        return {
            "n_diagnostics": 0,
            "exploit_only_rate": 0.0,
            "exploit_explore_rate": 0.0,
            "explore_only_rate": 0.0,
        }

    # --- Routing mode rates ---
    mode_counts = {"exploit_only": 0, "exploit_explore": 0, "explore_only": 0}
    for d in diags:
        mode = d.get("routing_mode")
        if mode in mode_counts:
            mode_counts[mode] += 1

    # --- Global retrieval rates ---
    triggered = sum(1 for d in diags if d.get("global_retrieval_triggered"))
    avoided = n - triggered

    # --- Selection source rates ---
    known_sel = sum(1 for d in diags if d.get("selected_source") == "known")
    global_sel = sum(1 for d in diags if d.get("selected_source") == "global")
    no_sel = sum(1 for d in diags if d.get("selected_memory_id") is None)
    with_sel = n - no_sel

    # --- Candidate counts ---
    known_counts = [d.get("n_known_candidates_considered", 0) for d in diags]
    global_counts = [d.get("n_global_candidates_considered", 0) for d in diags]

    # --- Transfer state sizes ---
    state_sizes_after = [d.get("transfer_state_size_after", 0) for d in diags]
    final_sizes: dict[str, int] = {}
    for d in diags:
        rid = d.get("receiver_id", "?")
        final_sizes[rid] = d.get("transfer_state_size_after", 0)

    # --- Known reuse rate = # selected from K_r / # tasks with selected ---
    known_reuse_rate = (known_sel / with_sel) if with_sel > 0 else 0.0

    # --- Distinct global explored ---
    distinct_global = set()
    for d in diags:
        for cid in d.get("global_candidate_ids", []):
            distinct_global.add(cid)

    # --- Critic calls ---
    total_known_calls = sum(known_counts)
    total_global_calls = sum(global_counts)

    def _safe_mean(vals: list) -> float | None:
        return sum(vals) / len(vals) if vals else None

    return {
        "n_diagnostics": n,
        "exploit_only_rate": mode_counts["exploit_only"] / n,
        "exploit_explore_rate": mode_counts["exploit_explore"] / n,
        "explore_only_rate": mode_counts["explore_only"] / n,
        "global_retrieval_trigger_rate": triggered / n,
        "avoided_global_retrieval_rate": avoided / n,
        "known_memory_selection_rate": known_sel / n,
        "global_memory_selection_rate": global_sel / n,
        "no_memory_fallback_rate": no_sel / n,
        "mean_known_candidates_scored_per_task": _safe_mean(known_counts),
        "mean_global_candidates_scored_per_task": _safe_mean(global_counts),
        "total_transfer_model_calls": total_known_calls + total_global_calls,
        "mean_transfer_state_size": _safe_mean(state_sizes_after),
        "final_transfer_state_size": final_sizes,
        "distinct_global_memories_explored": len(distinct_global),
        "known_memory_reuse_rate": known_reuse_rate,
    }


def build_curve_records(
    routing_diagnostics: Iterable[dict[str, Any]],
    task_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-task-position curve-ready records (§34, §35).

    Produces one record per task position with aggregated routing
    information across receivers, joined with the task score.

    Output fields per record::

        task_position
        global_retrieval_triggered   (any receiver triggered)
        selected_from_known          (any receiver selected known)
        transfer_state_size          (sum across receivers)
        task_score

    These records feed two key continual curves:

    * Global Retrieval Rate over Task Position  (expected ↓)
    * Known Transfer Reuse Rate over Task Position  (expected ↑)
    """
    diags = list(routing_diagnostics)
    records = list(task_records)

    # Index task scores by position
    score_by_pos: dict[int, float | None] = {}
    for r in records:
        pos = r.get("task_position")
        if pos is not None:
            score_by_pos[pos] = r.get("task_score")

    # Group diagnostics by task_position
    by_pos: dict[int, list[dict[str, Any]]] = {}
    for d in diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    curve: list[dict[str, Any]] = []
    for pos in sorted(by_pos):
        group = by_pos[pos]
        triggered = any(d.get("global_retrieval_triggered") for d in group)
        from_known = any(
            d.get("selected_source") == "known" for d in group
        )
        state_size = sum(
            d.get("transfer_state_size_after", 0) for d in group
        )
        curve.append({
            "task_position": pos,
            "global_retrieval_triggered": triggered,
            "selected_from_known": from_known,
            "transfer_state_size": state_size,
            "task_score": score_by_pos.get(pos),
        })

    return curve


def compute_transfer_cost(
    routing_diagnostics: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Online transfer cost block (§36).

    Counts critic inference calls and global-retrieval calls from
    routing diagnostics.  ``online_intervention_episodes`` is always 0
    for a formal run (interventions are forbidden).
    """
    diags = list(routing_diagnostics)
    known_calls = sum(d.get("n_known_candidates_considered", 0) for d in diags)
    global_calls = sum(d.get("n_global_candidates_considered", 0) for d in diags)
    retrieval_calls = sum(1 for d in diags if d.get("global_retrieval_triggered"))
    avoided = sum(1 for d in diags if not d.get("global_retrieval_triggered"))
    return {
        "online_transfer_cost": {
            "known_candidate_critic_calls": known_calls,
            "global_candidate_critic_calls": global_calls,
            "global_retrieval_calls": retrieval_calls,
            "global_retrieval_avoided": avoided,
            "online_intervention_episodes": 0,
        }
    }


# ---------------------------------------------------------------------------
# §17.2 — Episode-level metrics
# ---------------------------------------------------------------------------


def compute_episode_metrics(
    semantics_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Episode-level selection metrics (§17.2).

    Aggregates ``RoutingSemanticsLog`` entries across all tasks.

    Parameters
    ----------
    semantics_logs : iterable of routing semantics log dicts
        Each dict must contain: ``episode_selected_receiver``,
        ``episode_selected_memory``, ``episode_selected_lcb``,
        ``candidate_receivers_considered``,
        ``receiver_plans_generated``, ``joint_exposure_count``.

    Returns
    -------
    dict with episode-level selection statistics.
    """
    logs = list(semantics_logs)
    n = len(logs)
    if n == 0:
        return {
            "n_episodes": 0,
            "selection_rate": 0.0,
            "mean_candidate_receivers_considered": None,
            "mean_receiver_plans_generated": None,
            "joint_exposure_violations": 0,
        }

    selected = sum(
        1 for log in logs if log.get("episode_selected_memory") is not None
    )
    joint_violations = sum(
        1 for log in logs if log.get("joint_exposure_count", 0) != 0
    )

    candidate_receivers = [
        log.get("candidate_receivers_considered", 0) for log in logs
    ]
    plans_generated = [
        log.get("receiver_plans_generated", 0) for log in logs
    ]

    def _safe_mean(vals: list) -> float | None:
        return sum(vals) / len(vals) if vals else None

    return {
        "n_episodes": n,
        "selection_rate": selected / n,
        "mean_candidate_receivers_considered": _safe_mean(candidate_receivers),
        "mean_receiver_plans_generated": _safe_mean(plans_generated),
        "joint_exposure_violations": joint_violations,
    }


# ---------------------------------------------------------------------------
# §17.3 — Continual-learning metrics
# ---------------------------------------------------------------------------


def compute_continual_learning_metrics(
    *,
    causal_probe_count: int = 0,
    causal_probe_episode_count: int = 0,
    causal_observed_edge_count: int = 0,
    predicted_only_state_size: int = 0,
    causal_observed_state_size: int = 0,
    online_critic_refit_count: int = 0,
    critic_version: int = 0,
    online_causal_evidence_used: int = 0,
) -> dict[str, Any]:
    """True continual-learning metrics (§17.3).

    These metrics distinguish predicted-only state (critic inference)
    from causal-observed state (matched interventions) and track the
    online learning loop.

    All parameters default to 0 for a frozen run.
    """
    return {
        "causal_probe_count": causal_probe_count,
        "causal_probe_episode_count": causal_probe_episode_count,
        "causal_observed_edge_count": causal_observed_edge_count,
        "predicted_only_state_size": predicted_only_state_size,
        "causal_observed_state_size": causal_observed_state_size,
        "online_critic_refit_count": online_critic_refit_count,
        "critic_version": critic_version,
        "online_causal_evidence_used": online_causal_evidence_used,
    }


# ---------------------------------------------------------------------------
# §17.4 — Three-way cost breakdown
# ---------------------------------------------------------------------------


def compute_three_way_cost(
    routing_diagnostics: Iterable[dict[str, Any]],
    *,
    post_task_probe_expose_episodes: int = 0,
    post_task_probe_control_episodes: int = 0,
) -> dict[str, Any]:
    """Three-way cost breakdown (§17.4).

    Separates:

    * **Retrieval cost**: global_retrieval_calls, known_retrieval_calls
    * **Model cost**: known_critic_predictions, global_critic_predictions
    * **Environment-learning cost**: post_task_probe_expose_episodes,
      post_task_probe_control_episodes

    Do NOT just report “global retrieval count dropped” — the cost
    might have shifted to critic predictions or causal probes.
    """
    diags = list(routing_diagnostics)

    # Retrieval cost
    global_retrieval_calls = sum(
        1 for d in diags if d.get("global_retrieval_triggered")
    )
    known_retrieval_calls = len(diags)  # one known recall per (task, receiver)

    # Model cost
    known_critic_predictions = sum(
        d.get("n_known_candidates_considered", 0) for d in diags
    )
    global_critic_predictions = sum(
        d.get("n_global_candidates_considered", 0) for d in diags
    )

    return {
        "retrieval_cost": {
            "global_retrieval_calls": global_retrieval_calls,
            "known_retrieval_calls": known_retrieval_calls,
        },
        "model_cost": {
            "known_critic_predictions": known_critic_predictions,
            "global_critic_predictions": global_critic_predictions,
        },
        "environment_learning_cost": {
            "post_task_probe_expose_episodes": post_task_probe_expose_episodes,
            "post_task_probe_control_episodes": post_task_probe_control_episodes,
        },
    }
