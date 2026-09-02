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
