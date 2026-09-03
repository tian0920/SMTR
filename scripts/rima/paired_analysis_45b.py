"""Phase 45B paired analysis: Adaptive vs Frozen comparison.

Usage::

    python scripts/rima/paired_analysis_45b.py \
        results/rima_transfer/pilot/phase45b

Reads all completed streams from the Phase 45B output directory.
For each (scenario, stream_seed) pair, computes paired differences:

    Δ Score = Score_adaptive - Score_frozen
    Δ Late Score
    Δ Global Search Rate
    Δ Transfer MAE
    Δ Known Reuse Rate

Output: ``paired_analysis.json`` in the output directory.
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


def _load_summary(run_dir: Path) -> dict[str, Any] | None:
    """Load summary.json from a run directory."""
    path = run_dir / "summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _split_late(n_tasks: int) -> int:
    """Return the start index for 'late' tasks."""
    if n_tasks >= 36:
        return n_tasks - n_tasks // 3
    return n_tasks - n_tasks // 3


def _late_scores(task_records: list[dict]) -> list[float]:
    """Scores for the late window."""
    scores = [r["task_score"] for r in task_records if "task_score" in r]
    n = len(scores)
    late_start = _split_late(n)
    return scores[late_start:]


def _global_search_rate(routing_diags: list[dict], late: bool = False,
                         n_tasks: int = 0) -> float | None:
    """Fraction of tasks where global retrieval was triggered."""
    by_pos: dict[int, list[dict]] = {}
    for d in routing_diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    if late and n_tasks > 0:
        late_start = _split_late(n_tasks)
        positions = range(late_start, n_tasks)
    elif not late and n_tasks > 0:
        late_start = _split_late(n_tasks)
        positions = range(0, late_start // 2)
    else:
        positions = sorted(by_pos.keys())

    triggered = 0
    total = 0
    for pos in positions:
        group = by_pos.get(pos, [])
        if group:
            total += 1
            if any(d.get("global_retrieval_triggered") for d in group):
                triggered += 1
    return triggered / total if total > 0 else None


def _known_reuse_rate(routing_diags: list[dict], late: bool = False,
                       n_tasks: int = 0) -> float | None:
    """Fraction of tasks where selected_source == 'known'."""
    by_pos: dict[int, list[dict]] = {}
    for d in routing_diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    if late and n_tasks > 0:
        late_start = _split_late(n_tasks)
        positions = range(late_start, n_tasks)
    elif not late and n_tasks > 0:
        late_start = _split_late(n_tasks)
        positions = range(0, late_start // 2)
    else:
        positions = sorted(by_pos.keys())

    known = 0
    total = 0
    for pos in positions:
        group = by_pos.get(pos, [])
        if group:
            total += 1
            if any(d.get("selected_source") == "known" for d in group):
                known += 1
    return known / total if total > 0 else None


def _transfer_mae(routing_diags: list[dict], task_records: list[dict],
                   late: bool = False) -> float | None:
    """MAE = |mu_predicted - score_observed|."""
    scores_by_pos: dict[int, float] = {}
    for r in task_records:
        pos = r.get("task_position")
        score = r.get("task_score")
        if pos is not None and score is not None:
            scores_by_pos[pos] = score

    by_pos: dict[int, list[dict]] = {}
    for d in routing_diags:
        pos = d.get("task_position", -1)
        by_pos.setdefault(pos, []).append(d)

    n = max(scores_by_pos.keys(), default=0) + 1
    late_start = _split_late(n) if late else 0
    late_end = n if late else _split_late(n) // 2

    errors = []
    for pos in range(late_start, late_end):
        group = by_pos.get(pos, [])
        obs = scores_by_pos.get(pos)
        if not group or obs is None:
            continue
        mus = [d.get("selected_mu") for d in group
               if d.get("selected_mu") is not None]
        if mus:
            errors.append(abs(max(mus) - obs))

    return float(np.mean(errors)) if errors else None


def paired_analysis(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)

    # Find all run directories
    run_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "DONE").exists()
    ])

    # Group by (scenario, stream_seed)
    pairs: dict[tuple[str, int], dict[str, Path]] = {}
    for rd in run_dirs:
        name = rd.name
        # Parse: scenario__streamN__execN__methodXXX
        parts = name.split("__")
        if len(parts) < 4:
            continue
        scenario = parts[0]
        stream_seed = int(parts[1].replace("stream", ""))
        method = parts[3].replace("method", "")
        key = (scenario, stream_seed)
        pairs.setdefault(key, {})[method] = rd

    results: list[dict] = []
    for (scenario, ss), methods in sorted(pairs.items()):
        adaptive_dir = methods.get("rima_transfer_adaptive")
        frozen_dir = methods.get("rima_transfer_frozen")

        if not adaptive_dir or not frozen_dir:
            continue

        # Load task records
        a_tasks = _load_jsonl(adaptive_dir / "tasks.jsonl")
        f_tasks = _load_jsonl(frozen_dir / "tasks.jsonl")

        # Load routing diagnostics
        a_diags = _load_jsonl(adaptive_dir / "routing.jsonl")
        f_diags = _load_jsonl(frozen_dir / "routing.jsonl")

        n_a = len(a_tasks)
        n_f = len(f_tasks)

        # Scores
        a_scores = [r["task_score"] for r in a_tasks]
        f_scores = [r["task_score"] for r in f_tasks]
        a_late = _late_scores(a_tasks)
        f_late = _late_scores(f_tasks)

        pair: dict[str, Any] = {
            "scenario": scenario,
            "stream_seed": ss,
            "n_tasks_adaptive": n_a,
            "n_tasks_frozen": n_f,
            # Score comparison
            "score_adaptive_mean": float(np.mean(a_scores)) if a_scores else None,
            "score_frozen_mean": float(np.mean(f_scores)) if f_scores else None,
            "delta_score": (
                float(np.mean(a_scores) - np.mean(f_scores))
                if a_scores and f_scores else None
            ),
            # Late score
            "late_score_adaptive": float(np.mean(a_late)) if a_late else None,
            "late_score_frozen": float(np.mean(f_late)) if f_late else None,
            "delta_late_score": (
                float(np.mean(a_late) - np.mean(f_late))
                if a_late and f_late else None
            ),
            # Global search rate (late)
            "global_search_adaptive_late": _global_search_rate(
                a_diags, late=True, n_tasks=n_a,
            ),
            "global_search_frozen_late": _global_search_rate(
                f_diags, late=True, n_tasks=n_f,
            ),
            "delta_global_search_late": None,
            # Known reuse rate (late)
            "known_reuse_adaptive_late": _known_reuse_rate(
                a_diags, late=True, n_tasks=n_a,
            ),
            "known_reuse_frozen_late": _known_reuse_rate(
                f_diags, late=True, n_tasks=n_f,
            ),
            "delta_known_reuse_late": None,
            # Transfer MAE (late)
            "mae_adaptive_late": _transfer_mae(a_diags, a_tasks, late=True),
            "mae_frozen_late": _transfer_mae(f_diags, f_tasks, late=True),
            "delta_mae_late": None,
        }

        # Compute deltas where possible
        ags = pair["global_search_adaptive_late"]
        fgs = pair["global_search_frozen_late"]
        if ags is not None and fgs is not None:
            pair["delta_global_search_late"] = ags - fgs

        akr = pair["known_reuse_adaptive_late"]
        fkr = pair["known_reuse_frozen_late"]
        if akr is not None and fkr is not None:
            pair["delta_known_reuse_late"] = akr - fkr

        am = pair["mae_adaptive_late"]
        fm = pair["mae_frozen_late"]
        if am is not None and fm is not None:
            pair["delta_mae_late"] = am - fm

        results.append(pair)

    # Aggregate across pairs
    aggregate: dict[str, Any] = {}
    if results:
        delta_scores = [r["delta_score"] for r in results
                        if r["delta_score"] is not None]
        delta_late = [r["delta_late_score"] for r in results
                      if r["delta_late_score"] is not None]
        delta_mae = [r["delta_mae_late"] for r in results
                     if r["delta_mae_late"] is not None]
        delta_gs = [r["delta_global_search_late"] for r in results
                    if r["delta_global_search_late"] is not None]
        delta_kr = [r["delta_known_reuse_late"] for r in results
                    if r["delta_known_reuse_late"] is not None]

        aggregate = {
            "n_pairs": len(results),
            "mean_delta_score": float(np.mean(delta_scores)) if delta_scores else None,
            "mean_delta_late_score": float(np.mean(delta_late)) if delta_late else None,
            "mean_delta_mae_late": float(np.mean(delta_mae)) if delta_mae else None,
            "mean_delta_global_search_late": float(np.mean(delta_gs)) if delta_gs else None,
            "mean_delta_known_reuse_late": float(np.mean(delta_kr)) if delta_kr else None,
        }

    # Gate B decision
    gate_b: dict[str, Any] = {"verdict": "INSUFFICIENT_DATA"}
    if results:
        # Check: MAE_adaptive < MAE_frozen (late)
        mae_direction = (
            aggregate.get("mean_delta_mae_late") is not None
            and aggregate["mean_delta_mae_late"] < 0
        )
        # Check: Score_adaptive >= Score_frozen (late)
        score_direction = (
            aggregate.get("mean_delta_late_score") is not None
            and aggregate["mean_delta_late_score"] >= 0
        )
        # Check: Known reuse adaptive not lower than frozen
        reuse_direction = (
            aggregate.get("mean_delta_known_reuse_late") is not None
            and aggregate["mean_delta_known_reuse_late"] >= 0
        )

        if mae_direction and score_direction:
            gate_b = {"verdict": "GO", "reason": "MAE and score direction correct"}
        elif mae_direction or score_direction:
            gate_b = {"verdict": "YELLOW", "reason": "Partial signal — review"}
        else:
            # Check if adaptive ≈ frozen (very small differences)
            if aggregate.get("mean_delta_late_score") is not None:
                diff = abs(aggregate["mean_delta_late_score"])
                if diff < 0.02:
                    gate_b = {
                        "verdict": "YELLOW",
                        "reason": (
                            f"Adaptive ≈ Frozen (|Δ| = {diff:.4f}). "
                            "Causal refit not yet showing value."
                        ),
                    }
                else:
                    gate_b = {
                        "verdict": "NO-GO",
                        "reason": "Adaptive does not outperform Frozen",
                    }
            else:
                gate_b = {"verdict": "INSUFFICIENT_DATA", "reason": "No data"}

    doc = {
        "phase": "45B",
        "output_dir": str(output_dir),
        "pairs": results,
        "aggregate": aggregate,
        "gate_b": gate_b,
    }

    out_path = output_dir / "paired_analysis.json"
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)

    # Print summary
    print(f"Phase 45B Paired Analysis")
    print(f"  Pairs found: {len(results)}")
    print()
    for p in results:
        print(f"  {p['scenario']}/stream{p['stream_seed']}:")
        print(f"    Δ Score: {p.get('delta_score')}")
        print(f"    Δ Late Score: {p.get('delta_late_score')}")
        print(f"    Δ MAE late: {p.get('delta_mae_late')}")
        print(f"    Δ Global Search: {p.get('delta_global_search_late')}")
        print(f"    Δ Known Reuse: {p.get('delta_known_reuse_late')}")
    print()
    if aggregate:
        print(f"  Aggregate ({aggregate['n_pairs']} pairs):")
        for k, v in aggregate.items():
            if k != "n_pairs":
                print(f"    {k}: {v}")
    print()
    print(f"  Gate B: {gate_b.get('verdict')} — {gate_b.get('reason', '')}")
    print(f"\nWrote {out_path}")

    return doc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rima/paired_analysis_45b.py <output_dir>")
        sys.exit(1)
    paired_analysis(sys.argv[1])
