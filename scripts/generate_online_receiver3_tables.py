"""Generate online MARBLE Receiver=3 paper tables.

Reads results from ``results/marble/receiver3/online/`` and
``results/marble/receiver3/online_contamination/`` to produce
LaTeX-compatible CSV tables.

Output: ``paper/tables/online_receiver3/``
  - ``main.csv``            — main experiment summary
  - ``domain_breakdown.csv`` — per-scenario breakdown
  - ``contamination.csv``    — contamination experiment summary
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


def load_episode_csv(path: Path) -> list[dict[str, Any]]:
    """Load episode metrics CSV into list of dicts."""
    if not path.exists():
        return []
    with path.open("r") as f:
        return list(csv.DictReader(f))


def generate_main_table(
    online_dir: Path,
    output_dir: Path,
) -> None:
    """Generate the main experiment summary table."""
    csv_path = online_dir / "episode_metrics.csv"
    rows = load_episode_csv(csv_path)
    if not rows:
        print(f"  WARNING: no data in {csv_path}")
        return

    methods = sorted(set(r["method"] for r in rows))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build summary
    table_rows: list[dict[str, Any]] = []
    for method in methods:
        m_rows = [r for r in rows if r["method"] == method]
        rewards = [float(r["team_reward"]) for r in m_rows]
        successes = [r["team_success"] == "True" for r in m_rows]
        n_injected = [int(r["n_injected"]) for r in m_rows]

        table_rows.append({
            "method": method,
            "n_episodes": len(m_rows),
            "mean_reward": f"{np.mean(rewards):.4f}",
            "std_reward": f"{np.std(rewards):.4f}",
            "success_rate": f"{np.mean(successes):.3f}",
            "mean_injected": f"{np.mean(n_injected):.1f}",
            "mean_validated": f"{np.mean([int(r['n_validated']) for r in m_rows]):.1f}",
            "mean_rejected": f"{np.mean([int(r['n_rejected']) for r in m_rows]):.1f}",
        })

    out_path = output_dir / "main.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"  Written: {out_path}")


def generate_domain_breakdown(
    online_dir: Path,
    output_dir: Path,
) -> None:
    """Generate per-scenario breakdown table."""
    csv_path = online_dir / "episode_metrics.csv"
    rows = load_episode_csv(csv_path)
    if not rows:
        return

    methods = sorted(set(r["method"] for r in rows))
    scenarios = sorted(set(r["scenario"] for r in rows))
    output_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for method in methods:
            m_rows = [
                r for r in rows
                if r["method"] == method and r["scenario"] == scenario
            ]
            if not m_rows:
                continue
            rewards = [float(r["team_reward"]) for r in m_rows]
            successes = [r["team_success"] == "True" for r in m_rows]

            table_rows.append({
                "scenario": scenario,
                "method": method,
                "n_episodes": len(m_rows),
                "mean_reward": f"{np.mean(rewards):.4f}",
                "std_reward": f"{np.std(rewards):.4f}",
                "success_rate": f"{np.mean(successes):.3f}",
            })

    out_path = output_dir / "domain_breakdown.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"  Written: {out_path}")


def generate_contamination_table(
    contamination_dir: Path,
    output_dir: Path,
) -> None:
    """Generate contamination experiment table."""
    csv_path = contamination_dir / "contamination_episodes.csv"
    rows = load_episode_csv(csv_path)
    if not rows:
        print(f"  WARNING: no data in {csv_path}")
        return

    methods = sorted(set(r["method"] for r in rows))
    output_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    for method in methods:
        m_rows = [r for r in rows if r["method"] == method]
        rewards = [float(r["team_reward"]) for r in m_rows]
        retained = [r["harmful_retained"] == "True" for r in m_rows]
        propagation = [int(r["harmful_propagation_depth"]) for r in m_rows]

        table_rows.append({
            "method": method,
            "n_episodes": len(m_rows),
            "harmful_retention_rate": f"{np.mean(retained):.3f}",
            "mean_propagation_depth": f"{np.mean(propagation):.2f}",
            "mean_reward": f"{np.mean(rewards):.4f}",
            "std_reward": f"{np.std(rewards):.4f}",
        })

    out_path = output_dir / "contamination.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"  Written: {out_path}")


def main() -> None:
    results_root = _PROJECT_ROOT / "results" / "marble" / "receiver3"
    online_dir = results_root / "online"
    contamination_dir = results_root / "online_contamination"
    output_dir = _PROJECT_ROOT / "paper" / "tables" / "online_receiver3"

    print("=== Generate Online Receiver=3 Tables ===")
    print(f"  online_dir: {online_dir}")
    print(f"  contamination_dir: {contamination_dir}")
    print(f"  output_dir: {output_dir}")
    print()

    generate_main_table(online_dir, output_dir)
    generate_domain_breakdown(online_dir, output_dir)
    generate_contamination_table(contamination_dir, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
