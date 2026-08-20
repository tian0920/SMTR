"""Collect and analyze intervention records from existing MARBLE paired data.

This script loads existing paired records from real MARBLE engine runs,
extracts intervention data in a standardized format, and computes signal
statistics to verify that causal transfer signal exists.

Outputs:
  - data/intervention_records.jsonl
  - data/signal_statistics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_paired_records(path: Path) -> list[dict]:
    """Load paired records from JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _extract_intervention_record(record: dict) -> dict:
    """Extract a standardized intervention record from a paired record."""
    # Compute Y values
    y_expose = 1 if record.get("share", {}).get("team_success") else 0
    y_withhold = 1 if record.get("withhold", {}).get("team_success") else 0
    tau = y_expose - y_withhold

    return {
        "task": str(record.get("task_id", "")),
        "receiver": str(record.get("receiver_agent_id", "")),
        "memory_id": str(record.get("candidate_memory_id", "")),
        "generation_seed": int(record.get("generation_seed", 0)),
        "Y_expose": y_expose,
        "Y_withhold": y_withhold,
        "tau": tau,
        "valid": bool(record.get("valid", False)),
        "label": str(record.get("label", "unknown")),
        "scenario": str(record.get("scenario", "database")),
    }


def _compute_signal_statistics(records: list[dict]) -> dict:
    """Compute transfer signal statistics from intervention records."""
    # Filter to valid records only
    valid_records = [r for r in records if r["valid"]]

    n_total = len(valid_records)
    if n_total == 0:
        return {
            "total_pairs": 0,
            "valid_pairs": 0,
            "positive_transfer": 0,
            "negative_transfer": 0,
            "neutral": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
        }

    # Count by tau value
    positive = sum(1 for r in valid_records if r["tau"] > 0)
    negative = sum(1 for r in valid_records if r["tau"] < 0)
    neutral = sum(1 for r in valid_records if r["tau"] == 0)

    return {
        "total_pairs": len(records),
        "valid_pairs": n_total,
        "positive_transfer": positive,
        "negative_transfer": negative,
        "neutral": neutral,
        "positive_pct": round(positive / n_total, 4),
        "negative_pct": round(negative / n_total, 4),
        "neutral_pct": round(neutral / n_total, 4),
    }


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]

    print("=" * 60)
    print("MARBLE Feasibility Test — Intervention Collection")
    print("=" * 60)

    # Load paired records
    paired_path = _PROJECT_ROOT / data_cfg["paired_records_path"]
    print(f"\n  Loading paired records from: {paired_path}")
    records = _load_paired_records(paired_path)
    print(f"  Total records loaded: {len(records)}")

    # Extract intervention records
    print("\n  Extracting intervention records...")
    intervention_records = [_extract_intervention_record(r) for r in records]

    # Save to JSONL
    out_dir = _THIS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intervention_records.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for record in intervention_records:
            f.write(json.dumps(record) + "\n")
    print(f"  Saved: {out_path}")

    # Compute signal statistics
    print("\n  Computing signal statistics...")
    stats = _compute_signal_statistics(intervention_records)

    # Print summary
    print("\n  Signal Statistics (valid pairs only):")
    print(f"    Total pairs: {stats['total_pairs']}")
    print(f"    Valid pairs: {stats['valid_pairs']}")
    print(f"    Positive transfer (τ > 0): {stats['positive_transfer']} "
          f"({stats['positive_pct']:.1%})")
    print(f"    Negative transfer (τ < 0): {stats['negative_transfer']} "
          f"({stats['negative_pct']:.1%})")
    print(f"    Neutral (τ = 0): {stats['neutral']} "
          f"({stats['neutral_pct']:.1%})")

    # Save statistics
    stats_path = out_dir / "signal_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved: {stats_path}")

    # Verify acceptance criteria
    print("\n  Acceptance Criteria Check:")
    acceptance = config["acceptance"]

    pos_pass = stats["positive_pct"] >= acceptance["positive_transfer_min"]
    neg_pass = stats["negative_pct"] > acceptance["negative_transfer_min"]

    print(f"    Positive transfer >= {acceptance['positive_transfer_min']:.0%}: "
          f"{'PASS' if pos_pass else 'FAIL'} ({stats['positive_pct']:.1%})")
    print(f"    Negative transfer > {acceptance['negative_transfer_min']:.0%}: "
          f"{'PASS' if neg_pass else 'FAIL'} ({stats['negative_pct']:.1%})")

    print("\nDone.")


if __name__ == "__main__":
    main()
