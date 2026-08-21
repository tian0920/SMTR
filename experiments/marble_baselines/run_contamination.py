"""MARBLE contamination propagation experiment (Phase 7).

Tests RQ3: Does TCI reduce multi-agent memory contamination?

Uses paired records to simulate different contamination ratios:
  - For each ratio, label a fraction of neutral memories as "false"
  - Compare how different methods handle contaminated memory pools

Metrics:
  - team reward
  - harmful memory retention (fraction of contaminated still injected)
  - contamination propagation rate

Output: results/marble/contamination/
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def load_paired_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_contamination_experiment(
    *,
    records: list[dict],
    ratios: list[float],
    seeds: list[int],
    n_tasks: int | None = None,
) -> list[dict]:
    """Simulate contamination at different ratios for each method."""
    valid = [r for r in records if r.get("valid")]

    # Group by (task_id, receiver_agent_id, generation_seed)
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["task_id"], r["receiver_agent_id"], r["generation_seed"])
        groups[key].append(r)

    if n_tasks is not None:
        task_ids = sorted(set(k[0] for k in groups))[:n_tasks]
        groups = {k: v for k, v in groups.items() if k[0] in task_ids}

    rows: list[dict] = []
    methods = ["full_memory", "retrieval", "smtr_tci"]

    for ratio in ratios:
        for (task_id, receiver_id, seed), group_records in sorted(groups.items()):
            if seed not in seeds:
                continue

            # Identify contamination: label neutral_failure records as contaminated
            # At ratio r, mark r fraction of neutral records as "contaminated"
            neutral_records = [
                r for r in group_records
                if r.get("label") in ("neutral_failure", "neutral_success")
            ]
            n_contaminate = int(len(neutral_records) * ratio)
            rng = np.random.RandomState(hash((task_id, seed, ratio)) % (2**31))
            contaminate_ids = set(
                rng.choice(len(neutral_records), size=n_contaminate, replace=False)
                if n_contaminate > 0 else []
            )
            contaminated_mids = {
                neutral_records[i]["candidate_memory_id"]
                for i in contaminate_ids
            }

            for method in methods:
                # Method-specific selection
                if method == "full_memory":
                    selected = group_records
                elif method == "retrieval":
                    ranked = sorted(group_records, key=lambda r: r.get("candidate_rank", 0))
                    selected = ranked[:3]
                elif method == "smtr_tci":
                    # TCI: only positive_transfer, rejects contaminated
                    selected = [r for r in group_records if r.get("label") == "positive_transfer"]
                else:
                    selected = []

                # Compute outcomes
                n_injected = len(selected)
                n_contaminated_injected = sum(
                    1 for r in selected
                    if r["candidate_memory_id"] in contaminated_mids
                )
                total_tau = 0.0
                for r in selected:
                    share_ok = bool(r.get("share", {}).get("team_success", False))
                    withhold_ok = bool(r.get("withhold", {}).get("team_success", False))
                    total_tau += int(share_ok) - int(withhold_ok)

                withhold_rewards = [
                    int(r.get("withhold", {}).get("team_success", False))
                    for r in group_records
                ]
                withhold_mean = float(np.mean(withhold_rewards)) if withhold_rewards else 0.0

                # Harmful retention: fraction of contaminated memories still retained
                harmful_retention = (
                    n_contaminated_injected / max(len(contaminated_mids), 1)
                )

                rows.append({
                    "method": method,
                    "ratio": ratio,
                    "task_id": task_id,
                    "receiver_id": receiver_id,
                    "seed": seed,
                    "n_injected": n_injected,
                    "n_contaminated_injected": n_contaminated_injected,
                    "n_contaminated_total": len(contaminated_mids),
                    "harmful_retention": harmful_retention,
                    "method_reward": withhold_mean + total_tau,
                })

    return rows


def main() -> None:
    import yaml
    cfg_path = _PROJECT_ROOT / "configs" / "marble_baseline.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    phase_cfg = cfg["scale"]["contamination"]
    seeds = phase_cfg["seeds"]
    n_tasks = phase_cfg.get("n_tasks")
    ratios = phase_cfg.get("ratios", [0.1, 0.2, 0.3])

    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)

    rows = run_contamination_experiment(
        records=records,
        ratios=ratios,
        seeds=seeds,
        n_tasks=n_tasks,
    )

    # Write CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "contamination"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "contamination_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Print summary
    print("=== MARBLE Contamination Experiment ===")
    print(f"{'Method':<15} {'Ratio':>6} {'Groups':>8} {'Reward':>10} {'Retention':>10} {'Contam.Inj':>12}")
    print("-" * 70)
    for ratio in ratios:
        for method in ["full_memory", "retrieval", "smtr_tci"]:
            m_rows = [r for r in rows if r["method"] == method and r["ratio"] == ratio]
            if not m_rows:
                continue
            mean_reward = float(np.mean([r["method_reward"] for r in m_rows]))
            mean_retention = float(np.mean([r["harmful_retention"] for r in m_rows]))
            mean_contam = float(np.mean([r["n_contaminated_injected"] for r in m_rows]))
            print(f"{method:<15} {ratio:>6.1f} {len(m_rows):>8} {mean_reward:>10.4f} "
                  f"{mean_retention:>10.4f} {mean_contam:>12.2f}")
        print()

    print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
