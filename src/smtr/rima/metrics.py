"""RIMA canonical metric system (Phase 23-26).

Metric hierarchy (paper Section: Evaluation Protocol):

* **Primary**: Official Task Score (normalized official MARBLE metric).
* **Secondary**: Coordination Score (diagnostic companion).
* **Long-term**: Cumulative Task Score, Late-stage Task Score.
* **Memory**: admission rate, cross-task reuse rate,
  receiver-specific reuse rate, harmful admission rate,
  receiver disagreement rate.
* **Critic**: tau prediction correlation, sign accuracy.
* **Cost**: reported SEPARATELY for the offline phase
  (intervention collection + critic training) and the online phase
  (frozen-critic inference only). Interventions never occur during
  formal inference/admission (Phase 25).

Diagnostic-only quantities (Phase 24) — ``memory_bank_size``,
``validated_memory_count``, ``candidate_count`` — are reported under a
dedicated ``diagnostic_only`` block and must never be used as primary
success metrics ("more memories" is not a claim; future team performance
is).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from smtr.rima.admission import AdmissionStatus

__all__ = [
    "PRIMARY_METRIC",
    "SECONDARY_METRIC",
    "DIAGNOSTIC_ONLY_METRICS",
    "compute_primary_metrics",
    "compute_longterm_metrics",
    "compute_memory_metrics",
    "compute_critic_quality_metrics",
    "compute_cost_report",
    "summarize_rima_run",
]

PRIMARY_METRIC = "official_task_score"
SECONDARY_METRIC = "coordination_score"

#: Quantities that may only ever appear as diagnostics (Phase 24).
DIAGNOSTIC_ONLY_METRICS = frozenset(
    {"memory_bank_size", "validated_memory_count", "candidate_count"}
)


def _safe_mean(values: Sequence[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0.0 or vy == 0.0:
        return None
    return cov / (vx * vy)


# ---------------------------------------------------------------------------
# Primary / secondary / long-term metrics (Phase 23)
# ---------------------------------------------------------------------------

def compute_primary_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Official Task Score statistics over valid outcomes.

    Invalid outcomes are fail-closed (delta/score = None): they are
    excluded from the mean and counted, never treated as zero.
    """
    records = list(records)
    valid = [r for r in records if r.get("is_valid") and r.get("task_score") is not None]
    invalid = [r for r in records if not r.get("is_valid")]
    scores = [float(r["task_score"]) for r in valid]
    return {
        "primary_metric": PRIMARY_METRIC,
        "mean_official_task_score": _safe_mean(scores),
        "n_tasks": len(records),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "valid_rate": (len(valid) / len(records)) if records else 0.0,
    }


def compute_longterm_metrics(
    records: Iterable[Mapping[str, Any]], late_stage_window: int = 5
) -> dict[str, Any]:
    """Cumulative and late-stage Official Task Score over valid records."""
    valid = [
        r
        for r in records
        if r.get("is_valid") and r.get("task_score") is not None
    ]
    valid.sort(key=lambda r: r.get("task_position", 0))
    scores = [float(r["task_score"]) for r in valid]
    tail = scores[-late_stage_window:] if scores else []
    return {
        "cumulative_task_score": sum(scores) if scores else 0.0,
        "late_stage_task_score": _safe_mean(tail),
        "late_stage_window": late_stage_window,
    }


# ---------------------------------------------------------------------------
# Memory metrics (Phase 23)
# ---------------------------------------------------------------------------

def compute_memory_metrics(decisions: Sequence[Any]) -> dict[str, Any]:
    """Memory lifecycle metrics from formal admission decisions.

    Self-transfer exclusions and invalid predictions are counted
    separately and never enter rate denominators (fail-closed).
    """
    formal = [
        d
        for d in decisions
        if d.status in (AdmissionStatus.ADMITTED, AdmissionStatus.REJECTED)
    ]
    admitted = [d for d in formal if d.status == AdmissionStatus.ADMITTED]
    n_formal = len(formal)

    # Cross-task reuse: an admitted memory reused on >=2 distinct tasks.
    tasks_by_memory: dict[str, set[str]] = {}
    receivers_by_memory: dict[str, set[str]] = {}
    decided_receivers_by_memory: dict[str, set[str]] = {}
    for d in formal:
        decided_receivers_by_memory.setdefault(d.memory_id, set()).add(d.receiver_id)
    for d in admitted:
        tasks_by_memory.setdefault(d.memory_id, set()).add(d.task_id)
        receivers_by_memory.setdefault(d.memory_id, set()).add(d.receiver_id)

    n_admitted_memories = len(tasks_by_memory)
    reused = [m for m, ts in tasks_by_memory.items() if len(ts) >= 2]
    cross_task_reuse_rate = (len(reused) / n_admitted_memories) if n_admitted_memories else None

    # Receiver-specific reuse: among memories admitted to >=1 receiver,
    # mean fraction of deciding receivers that actually admitted it.
    fractions = []
    for memory_id, admitted_rids in receivers_by_memory.items():
        decided = decided_receivers_by_memory.get(memory_id, set())
        if decided:
            fractions.append(len(admitted_rids & decided) / len(decided))
    receiver_specific_reuse_rate = _safe_mean(fractions)

    # Harmful admission rate: only measurable where mechanism evaluation
    # attached an observed delta (metadata recorded AFTER admission).
    observed = [
        float(d.metadata["observed_delta"])
        for d in admitted
        if d.metadata.get("observed_delta") is not None
    ]
    harmful = [x for x in observed if x < 0.0]
    harmful_admission_rate = (len(harmful) / len(observed)) if observed else None

    # Receiver disagreement: memories decided for >=2 receivers whose
    # admission status differs across receivers.
    status_by_memory: dict[str, set[str]] = {}
    for d in formal:
        status_by_memory.setdefault(d.memory_id, set()).add(d.status)
    multi = [s for s in status_by_memory.values() if len(s) > 1]
    receiver_disagreement_rate = (
        len(multi) / len(status_by_memory) if status_by_memory else None
    )

    return {
        "admission_rate": (len(admitted) / n_formal) if n_formal else None,
        "n_admitted": len(admitted),
        "n_rejected": len(formal) - len(admitted),
        "n_candidates_diagnostic": n_formal,
        "cross_task_reuse_rate": cross_task_reuse_rate,
        "receiver_specific_reuse_rate": receiver_specific_reuse_rate,
        "harmful_admission_rate": harmful_admission_rate,
        "receiver_disagreement_rate": receiver_disagreement_rate,
        "n_memories_with_observed_delta": len(observed),
    }


# ---------------------------------------------------------------------------
# Critic quality metrics (Phase 23; evaluated on mechanism-eval evidence)
# ---------------------------------------------------------------------------

def compute_critic_quality_metrics(
    evaluated_pairs: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """tau_hat vs observed_delta correlation and sign accuracy.

    ``evaluated_pairs`` come from mechanism evaluation (offline
    interventions only). Pairs with invalid outcomes are excluded and
    counted, never zero-filled.
    """
    pairs = list(evaluated_pairs)
    valid = [
        p
        for p in pairs
        if p.get("tau_hat") is not None and p.get("observed_delta") is not None
    ]
    taus = [float(p["tau_hat"]) for p in valid]
    deltas = [float(p["observed_delta"]) for p in valid]
    if valid:
        agree = sum(
            1 for t, d in zip(taus, deltas) if (t > 0.0) == (d > 0.0)
        )
        sign_accuracy = agree / len(valid)
    else:
        sign_accuracy = None
    return {
        "tau_prediction_correlation": _pearson(taus, deltas),
        "sign_accuracy": sign_accuracy,
        "n_evaluated_pairs": len(valid),
        "n_invalid_pairs_excluded": len(pairs) - len(valid),
    }


# ---------------------------------------------------------------------------
# Cost protocol (Phase 25-26)
# ---------------------------------------------------------------------------

def compute_cost_report(
    records: Iterable[Mapping[str, Any]],
    intervention_episodes: int = 0,
    critic_training_seconds: float | None = None,
) -> dict[str, Any]:
    """Separate offline intervention cost from online inference cost.

    Interventions (matched expose/withhold executions) happen ONLY for
    critic training / mechanism evaluation; the online formal path uses
    frozen-critic inference and must never trigger interventions.
    """
    records = list(records)
    wall = [float(r["wall_seconds"]) for r in records if r.get("wall_seconds") is not None]
    extra_tokens = sum(
        int(r.get("extra_tokens", 0) or 0) for r in records
    )
    return {
        "offline_intervention_cost": {
            "intervention_collection_episodes": int(intervention_episodes),
            "critic_training_seconds": critic_training_seconds,
        },
        "online_rima_inference_cost": {
            "n_formal_episodes": len(records),
            "mean_wall_seconds_per_task": _safe_mean(wall),
            "extra_tokens": extra_tokens,
            "intervention_episodes_in_formal_path": 0,
        },
    }


# ---------------------------------------------------------------------------
# One-shot summary used by the canonical runner
# ---------------------------------------------------------------------------

def summarize_rima_run(
    records: Iterable[Mapping[str, Any]],
    decisions: Sequence[Any] = (),
    pool_size: int | None = None,
    late_stage_window: int = 5,
) -> dict[str, Any]:
    """Canonical metric block for one continual run."""
    records = list(records)
    return {
        "primary": compute_primary_metrics(records),
        "longterm": compute_longterm_metrics(records, late_stage_window),
        "memory": compute_memory_metrics(decisions),
        "diagnostic_only": {
            "memory_bank_size": pool_size,
            "note": (
                "Diagnostic quantities only; memory quantity is never a "
                "primary success metric. The claim target is future team "
                "performance (Official Task Score)."
            ),
        },
    }
