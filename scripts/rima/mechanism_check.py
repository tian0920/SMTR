"""Phase 45A mechanism check: evaluate 5 mechanism metrics.

Usage::

    python scripts/rima/mechanism_check.py \
        results/rima_transfer/pilot/phase45a/bargaining__stream0__exec0__methodrima_transfer_adaptive

Reads tasks.jsonl, routing.jsonl (if present), critic_versions.jsonl
(if present) and evaluates the five mechanism metrics for Gate A.

Output: ``mechanism_check.json`` in the same directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _split_windows(n_tasks: int) -> tuple[range, range, range]:
    """Return (early, middle, late) index ranges.

    30 tasks: 1-10, 11-20, 21-30
    40 tasks: 1-13, 14-27, 28-40
    General: roughly thirds.
    """
    if n_tasks >= 36:
        # 40-task window: 1-13, 14-27, 28-40
        n_early = max(1, n_tasks // 3)
        n_late = max(1, n_tasks - n_tasks * 2 // 3)
        early_end = n_early
        late_start = n_tasks - n_late
    else:
        # 30-task window: 1-10, 11-20, 21-30
        third = max(1, n_tasks // 3)
        early_end = third
        late_start = n_tasks - third

    return (
        range(0, early_end),
        range(early_end, late_start),
        range(late_start, n_tasks),
    )


def _mean_of(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# ----- Metric A: Global Exploration -----

def metric_a_global_exploration(
    routing_diags: list[dict],
    early: range,
    late: range,
) -> dict[str, Any]:
    """Fraction of tasks where global retrieval was triggered."""
    by_pos: dict[int, list[dict]] = {}
    for d in routing_diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    def _global_rate(positions: range) -> float | None:
        triggered = 0
        total = 0
        for pos in positions:
            group = by_pos.get(pos, [])
            if group:
                total += 1
                if any(d.get("global_retrieval_triggered") for d in group):
                    triggered += 1
        return triggered / total if total > 0 else None

    early_rate = _global_rate(early)
    late_rate = _global_rate(late)

    if early_rate is not None and late_rate is not None:
        direction = "PASS" if late_rate < early_rate else "FAIL"
    else:
        direction = "INSUFFICIENT_DATA"

    return {
        "metric": "A_global_exploration",
        "early_rate": early_rate,
        "late_rate": late_rate,
        "direction": direction,
        "requirement": "late < early",
    }


# ----- Metric B: Known Reuse -----

def metric_b_known_reuse(
    routing_diags: list[dict],
    early: range,
    late: range,
) -> dict[str, Any]:
    """Fraction of tasks where selected memory came from 'known'."""
    by_pos: dict[int, list[dict]] = {}
    for d in routing_diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    def _known_rate(positions: range) -> float | None:
        known = 0
        total = 0
        for pos in positions:
            group = by_pos.get(pos, [])
            if group:
                total += 1
                if any(d.get("selected_source") == "known" for d in group):
                    known += 1
        return known / total if total > 0 else None

    early_rate = _known_rate(early)
    late_rate = _known_rate(late)

    if early_rate is not None and late_rate is not None:
        direction = "PASS" if late_rate > early_rate else "FAIL"
    else:
        direction = "INSUFFICIENT_DATA"

    return {
        "metric": "B_known_reuse",
        "early_rate": early_rate,
        "late_rate": late_rate,
        "direction": direction,
        "requirement": "late > early",
    }


# ----- Metric C: Causal Transfer State Growth -----

def metric_c_causal_state(
    routing_diags: list[dict],
    early: range,
    late: range,
) -> dict[str, Any]:
    """Sum of causal-observed state size across receivers.

    Uses ``transfer_state_causal_observed_after`` (edges with
    CAUSAL_OBSERVED evidence only). Deliberately does NOT fall back to
    ``transfer_state_size_after``: that field also counts PREDICTED_ONLY
    registrations and previously inflated |K^causal| when probe count
    was zero.
    """
    by_pos: dict[int, list[dict]] = {}
    for d in routing_diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    def _state_size(positions: range) -> float | None:
        sizes = []
        for pos in positions:
            group = by_pos.get(pos, [])
            if group and all(
                "transfer_state_causal_observed_after" in d for d in group
            ):
                sizes.append(sum(
                    d.get("transfer_state_causal_observed_after", 0)
                    for d in group
                ))
        return _mean_of(sizes)

    early_size = _state_size(early)
    late_size = _state_size(late)

    if early_size is not None and late_size is not None:
        direction = "PASS" if late_size > early_size else "FAIL"
    else:
        direction = "INSUFFICIENT_DATA"

    return {
        "metric": "C_causal_state_growth",
        "early_mean_size": early_size,
        "late_mean_size": late_size,
        "direction": direction,
        "requirement": "|K_causal_late| > |K_causal_early|",
    }


# ----- Metric D: Transfer MAE -----

def metric_d_transfer_mae(
    routing_diags: list[dict],
    task_records: list[dict],
    early: range,
    late: range,
) -> dict[str, Any]:
    """Transfer prediction MAE: |mu_pre_probe - tau_obs|.

    For each task where we have both a selected_mu (critic prediction)
    and a task_score (observed outcome), compute MAE.
    """
    # Build score lookup by position
    scores_by_pos: dict[int, float] = {}
    for r in task_records:
        pos = r.get("task_position")
        score = r.get("task_score")
        if pos is not None and score is not None:
            scores_by_pos[pos] = score

    # Build prediction lookup by position
    by_pos: dict[int, list[dict]] = {}
    for d in routing_diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    def _mae(positions: range) -> float | None:
        errors = []
        for pos in positions:
            group = by_pos.get(pos, [])
            obs = scores_by_pos.get(pos)
            if not group or obs is None:
                continue
            # Take best predicted mu across receivers
            mus = [
                d.get("selected_mu") for d in group
                if d.get("selected_mu") is not None
            ]
            if mus:
                best_mu = max(mus)
                errors.append(abs(best_mu - obs))
        return _mean_of(errors)

    early_mae = _mae(early)
    late_mae = _mae(late)

    if early_mae is not None and late_mae is not None:
        direction = "PASS" if late_mae < early_mae else "FAIL"
    else:
        direction = "INSUFFICIENT_DATA"

    return {
        "metric": "D_transfer_mae",
        "early_mae": early_mae,
        "late_mae": late_mae,
        "direction": direction,
        "requirement": "MAE_late < MAE_early",
    }


# ----- Metric E: Critic Chronology -----

def metric_e_critic_chronology(
    critic_versions: list[dict],
    n_tasks: int,
) -> dict[str, Any]:
    """All critic_trained_through_position < current_task_position."""
    violations = 0
    total_checks = 0

    for entry in critic_versions:
        trained_through = entry.get("trained_through")
        task_pos = entry.get("task_position")
        if trained_through is not None and task_pos is not None:
            total_checks += 1
            if trained_through >= task_pos:
                violations += 1

    if total_checks == 0:
        return {
            "metric": "E_critic_chronology",
            "violations": 0,
            "total_checks": 0,
            "direction": "INSUFFICIENT_DATA",
            "requirement": "trained_through < current_task_position",
        }

    direction = "PASS" if violations == 0 else "FAIL"
    return {
        "metric": "E_critic_chronology",
        "violations": violations,
        "total_checks": total_checks,
        "direction": direction,
        "requirement": "trained_through < current_task_position",
    }


# ----- Gate A Decision -----

def gate_a_decision(metrics: list[dict]) -> dict[str, Any]:
    """Determine GO / NO-GO / YELLOW for Phase 45A.

    GO: A=PASS, B=PASS, C=PASS (D can be YELLOW)
    NO-GO: A=FAIL AND B=FAIL
    YELLOW: mixed
    """
    results = {m["metric"]: m["direction"] for m in metrics}

    a_dir = results.get("A_global_exploration", "INSUFFICIENT_DATA")
    b_dir = results.get("B_known_reuse", "INSUFFICIENT_DATA")
    c_dir = results.get("C_causal_state_growth", "INSUFFICIENT_DATA")
    d_dir = results.get("D_transfer_mae", "INSUFFICIENT_DATA")
    e_dir = results.get("E_critic_chronology", "INSUFFICIENT_DATA")

    # Hard NO-GO: both exploration and reuse show no improvement
    if a_dir == "FAIL" and b_dir == "FAIL":
        verdict = "NO-GO"
        reason = "Global exploration and known reuse both show no improvement"
    # GO: core three pass
    elif a_dir == "PASS" and b_dir == "PASS" and c_dir == "PASS":
        if d_dir in ("PASS", "INSUFFICIENT_DATA"):
            verdict = "GO"
            reason = "All core metrics pass"
        else:
            verdict = "YELLOW"
            reason = "Core metrics pass but transfer MAE does not improve"
    # YELLOW: partial pass
    else:
        verdict = "YELLOW"
        reason = "Mixed results — review individual metrics"

    return {
        "verdict": verdict,
        "reason": reason,
        "metric_directions": results,
    }


# ----- Main -----

def mechanism_check(stream_dir: str | Path) -> dict:
    stream_dir = Path(stream_dir)
    task_records = _load_jsonl(stream_dir / "tasks.jsonl")
    routing_diags = _load_jsonl(stream_dir / "routing.jsonl")
    critic_versions = _load_jsonl(stream_dir / "critic_versions.jsonl")

    n_tasks = len(task_records)
    if n_tasks == 0:
        print("ERROR: No task records found")
        return {"error": "no task records"}

    early, middle, late = _split_windows(n_tasks)

    # Read run manifest
    manifest_path = stream_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Score trajectory
    scores = [r["task_score"] for r in task_records if "task_score" in r]

    result = {
        "stream_dir": str(stream_dir),
        "method": manifest.get("method", "unknown"),
        "n_tasks": n_tasks,
        "windows": {
            "early": f"1-{len(early)}",
            "middle": f"{len(early) + 1}-{len(early) + len(middle)}",
            "late": f"{len(early) + len(middle) + 1}-{n_tasks}",
        },
        "score_summary": {
            "early_mean": _mean_of([scores[i] for i in early if i < len(scores)]),
            "middle_mean": _mean_of([scores[i] for i in middle if i < len(scores)]),
            "late_mean": _mean_of([scores[i] for i in late if i < len(scores)]),
            "overall_mean": _mean_of(scores),
        },
        "routing_diagnostics_available": len(routing_diags) > 0,
        "critic_versions_available": len(critic_versions) > 0,
    }

    # Compute metrics
    metrics = []
    if routing_diags:
        metrics.append(metric_a_global_exploration(routing_diags, early, late))
        metrics.append(metric_b_known_reuse(routing_diags, early, late))
        metrics.append(metric_c_causal_state(routing_diags, early, late))
        metrics.append(metric_d_transfer_mae(routing_diags, task_records, early, late))
    else:
        # No routing diagnostics — cannot compute A-D
        for name in ["A_global_exploration", "B_known_reuse",
                      "C_causal_state_growth", "D_transfer_mae"]:
            metrics.append({
                "metric": name,
                "direction": "INSUFFICIENT_DATA",
                "note": "No routing.jsonl found",
            })

    # Metric E always available if critic_versions exists
    metrics.append(metric_e_critic_chronology(critic_versions, n_tasks))

    result["metrics"] = metrics
    result["gate_a"] = gate_a_decision(metrics)

    # Write output
    out_path = stream_dir / "mechanism_check.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"Mechanism Check for {stream_dir}")
    print(f"  Method: {result['method']}")
    print(f"  N tasks: {n_tasks}")
    print(f"  Windows: early={result['windows']['early']}, "
          f"middle={result['windows']['middle']}, "
          f"late={result['windows']['late']}")
    print()
    for m in metrics:
        print(f"  {m['metric']}: {m['direction']}")
        for k, v in m.items():
            if k not in ("metric", "direction"):
                print(f"    {k}: {v}")
    print()
    gate = result["gate_a"]
    print(f"  Gate A: {gate['verdict']}")
    print(f"  Reason: {gate['reason']}")
    print(f"\nWrote {out_path}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rima/mechanism_check.py <stream_dir>")
        sys.exit(1)
    mechanism_check(sys.argv[1])
