"""Generate timing_breakdown.json from a completed stream's tasks.jsonl.

Usage::

    python scripts/rima/generate_timing_breakdown.py \
        results/rima_transfer/pilot/mechanism/bargaining__stream0__exec0__methodrima_receiver

Reads event_timestamps from each task record and produces a breakdown
of wall time into stages:

    routing, execution (MARBLE engine), post_task_probe,
    critic_refit, memory_extraction, other.

Output: ``timing_breakdown.json`` in the same directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


STAGES = [
    ("routing", "routing_started", "routing_finished"),
    ("execution", "scored_execution_started", "scored_execution_finished"),
    ("post_task_probe", "post_task_probe_started", "post_task_probe_finished"),
    ("critic_refit", "critic_refit_started", "critic_refit_finished"),
    ("memory_extraction", "memory_extraction_started", "memory_extraction_finished"),
]


def _stage_duration(ev: dict, start_key: str, end_key: str) -> float | None:
    s = ev.get(start_key)
    e = ev.get(end_key)
    if s is not None and e is not None:
        return e - s
    return None


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def generate_breakdown(stream_dir: str | Path) -> dict:
    stream_dir = Path(stream_dir)
    tasks_path = stream_dir / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl not found in {stream_dir}")

    with open(tasks_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Per-stage durations
    stage_durations: dict[str, list[float]] = {s[0]: [] for s in STAGES}
    wall_times: list[float] = []

    for rec in records:
        ev = rec.get("event_timestamps", {})
        for stage_name, start_key, end_key in STAGES:
            d = _stage_duration(ev, start_key, end_key)
            if d is not None and d >= 0:
                stage_durations[stage_name].append(d)

        wt = rec.get("wall_seconds")
        if wt is not None:
            wall_times.append(wt)

    # Build breakdown
    breakdown: dict[str, dict] = {}
    total_accounted = 0.0

    for stage_name, durations in stage_durations.items():
        if not durations:
            breakdown[stage_name] = {
                "count": 0,
                "total_seconds": 0.0,
                "mean_seconds": 0.0,
                "p50": 0.0,
                "p95": 0.0,
            }
            continue
        total = sum(durations)
        total_accounted += total
        breakdown[stage_name] = {
            "count": len(durations),
            "total_seconds": round(total, 2),
            "mean_seconds": round(total / len(durations), 2),
            "p50": round(_percentile(durations, 50), 2),
            "p95": round(_percentile(durations, 95), 2),
        }

    # Wall time total
    total_wall = sum(wall_times) if wall_times else 0.0

    # Other (unaccounted)
    other_total = max(0.0, total_wall - total_accounted)
    breakdown["other"] = {
        "count": len(wall_times),
        "total_seconds": round(other_total, 2),
        "mean_seconds": round(other_total / max(1, len(wall_times)), 2),
        "p50": 0.0,
        "p95": 0.0,
    }

    breakdown["total_wall"] = {
        "count": len(wall_times),
        "total_seconds": round(total_wall, 2),
        "mean_seconds": round(total_wall / max(1, len(wall_times)), 2),
        "p50": round(_percentile(wall_times, 50), 2),
        "p95": round(_percentile(wall_times, 95), 2),
    }

    # Fraction table (where does time go?)
    fractions: dict[str, float] = {}
    if total_wall > 0:
        for stage_name in [s[0] for s in STAGES] + ["other"]:
            fractions[stage_name] = round(
                breakdown[stage_name]["total_seconds"] / total_wall, 3,
            )

    result = {
        "stream_dir": str(stream_dir),
        "n_tasks": len(records),
        "stages": breakdown,
        "fractions": fractions,
    }

    # Write output
    out_path = stream_dir / "timing_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  n_tasks: {len(records)}")
    print(f"  total_wall: {total_wall:.1f}s ({total_wall / 60:.1f} min)")
    print(f"  fractions: {json.dumps(fractions, indent=4)}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rima/generate_timing_breakdown.py <stream_dir>")
        sys.exit(1)
    generate_breakdown(sys.argv[1])
