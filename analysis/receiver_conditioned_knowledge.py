"""Receiver-conditioned knowledge analysis (Phase 6).

Analyzes the core thesis of receiver-conditioned TCI:
  "The same memory can be useful for receiver A but useless for receiver B."

Computes:
  1. Decision disagreement: P(decision_i != decision_j)
  2. Delta variance: Var(delta(receiver))
  3. Selective transfer gain: memory useful for k-of-3 receivers
  4. Negative transfer prevented per receiver

Output: results/marble/receiver3/main/receiver_conditioned_analysis.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

RECEIVER_IDS = ["receiver_1", "receiver_2", "receiver_3"]


def main() -> None:
    detail_path = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "main" / "main_receiver_details.csv"
    if not detail_path.exists():
        print("ERROR: run main experiment first (experiments/marble_receiver3/run_main.py)")
        sys.exit(1)

    with detail_path.open() as f:
        details = list(csv.DictReader(f))

    # Only analyze smtr_receiver method
    smtr_details = [d for d in details if d["method"] == "smtr_receiver"]
    print(f"Loaded {len(smtr_details)} smtr_receiver detail rows")

    # Group by (task_id, seed, memory_id) → per-receiver outcomes
    memory_receiver_groups: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for d in smtr_details:
        key = (d["task_id"], d["seed"], d["memory_id"])
        memory_receiver_groups[key][d["receiver_id"]] = d

    analysis_rows: list[dict] = []
    disagreement_count = 0
    total_memories = 0
    delta_variances: list[float] = []
    useful_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}  # k-of-3
    neg_prevented_per_receiver: dict[str, int] = {r: 0 for r in RECEIVER_IDS}

    for (task_id, seed, memory_id), receiver_data in sorted(memory_receiver_groups.items()):
        total_memories += 1

        deltas: dict[str, float] = {}
        decisions: dict[str, str] = {}
        for rid in RECEIVER_IDS:
            if rid in receiver_data:
                d = receiver_data[rid]
                deltas[rid] = float(d["delta"])
                decisions[rid] = "validated" if float(d["delta"]) > 0 else "rejected"
            else:
                deltas[rid] = 0.0
                decisions[rid] = "rejected"

        # Decision disagreement: P(decision_i != decision_j)
        dec_list = list(decisions.values())
        n_pairs = len(dec_list) * (len(dec_list) - 1) // 2
        n_disagree = sum(
            1 for i in range(len(dec_list))
            for j in range(i + 1, len(dec_list))
            if dec_list[i] != dec_list[j]
        )
        disagreement = n_disagree / max(n_pairs, 1)
        if disagreement > 0:
            disagreement_count += 1

        # Delta variance
        delta_vals = list(deltas.values())
        delta_var = float(np.var(delta_vals)) if len(delta_vals) > 1 else 0.0
        delta_variances.append(delta_var)

        # Selective transfer: how many receivers found this useful?
        n_useful = sum(1 for v in delta_vals if v > 0)
        useful_counts[n_useful] = useful_counts.get(n_useful, 0) + 1

        # Negative transfer prevented: check full_memory for comparison
        for rid in RECEIVER_IDS:
            full_rows = [
                d for d in details
                if d["method"] == "full_memory"
                and d["task_id"] == task_id
                and d["seed"] == seed
                and d["memory_id"] == memory_id
                and d["receiver_id"] == rid
            ]
            if full_rows and float(full_rows[0]["delta"]) < 0:
                # Full memory would inject this harmful memory
                # SMTR-receiver correctly rejects it
                if decisions.get(rid) == "rejected":
                    neg_prevented_per_receiver[rid] += 1

        analysis_rows.append({
            "task_id": task_id,
            "seed": seed,
            "memory_id": memory_id,
            "delta_receiver_1": deltas.get("receiver_1", 0),
            "delta_receiver_2": deltas.get("receiver_2", 0),
            "delta_receiver_3": deltas.get("receiver_3", 0),
            "decision_receiver_1": decisions.get("receiver_1", "rejected"),
            "decision_receiver_2": decisions.get("receiver_2", "rejected"),
            "decision_receiver_3": decisions.get("receiver_3", "rejected"),
            "disagreement_rate": round(disagreement, 4),
            "delta_variance": round(delta_var, 6),
            "n_receivers_useful": n_useful,
        })

    # Write analysis CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "main"
    csv_path = output_dir / "receiver_conditioned_analysis.csv"
    if analysis_rows:
        fieldnames = list(analysis_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis_rows)
    print(f"Written: {csv_path} ({len(analysis_rows)} rows)")

    # Summary
    print()
    print("=" * 60)
    print("Receiver-Conditioned Knowledge Analysis Summary")
    print("=" * 60)
    print(f"Total unique (task, seed, memory) groups: {total_memories}")
    print(f"Memories with receiver disagreement: {disagreement_count} ({disagreement_count/max(total_memories,1)*100:.1f}%)")
    print(f"Mean disagreement rate: {np.mean([r['disagreement_rate'] for r in analysis_rows]):.4f}")
    print(f"Mean delta variance: {np.mean(delta_variances):.6f}")
    print()
    print("Selective transfer distribution (k-of-3 receivers useful):")
    for k in sorted(useful_counts.keys()):
        print(f"  k={k}: {useful_counts[k]} memories ({useful_counts[k]/max(total_memories,1)*100:.1f}%)")
    print()
    print("Negative transfer prevented per receiver:")
    for rid, count in neg_prevented_per_receiver.items():
        print(f"  {rid}: {count}")
    print()

    # Key metric: per-(memory, receiver) alignment of uniform TCI decisions
    # with the receiver-specific measured delta.
    #
    # NOTE on honesty of metrics:
    #   - Receiver-conditioned decisions are computed FROM the measured
    #     per-receiver delta, so "receiver alignment = 100%" is a
    #     self-consistency check BY CONSTRUCTION, not independent accuracy.
    #   - Uniform TCI alignment IS meaningful: it measures how often the
    #     aggregate-delta decision is wrong for an individual receiver
    #     (false accept = harmful injection risk; false reject = lost transfer).
    uniform_correct = 0
    uniform_false_accept = 0
    uniform_false_reject = 0
    receiver_self_consistent = 0
    for row in analysis_rows:
        deltas = [row["delta_receiver_1"], row["delta_receiver_2"], row["delta_receiver_3"]]
        mean_delta = np.mean(deltas)
        uniform_decision = "validated" if mean_delta > 0 else "rejected"

        for i, rid in enumerate(RECEIVER_IDS):
            actual_useful = deltas[i] > 0
            uniform_agrees = (uniform_decision == "validated") == actual_useful
            if uniform_agrees:
                uniform_correct += 1
            elif uniform_decision == "validated" and not actual_useful:
                uniform_false_accept += 1
            else:
                uniform_false_reject += 1
            # Self-consistency of receiver-conditioned decision (tautological)
            receiver_decision = row[f"decision_{rid}"]
            if (receiver_decision == "validated") == actual_useful:
                receiver_self_consistent += 1

    total_decisions = total_memories * 3
    print(f"Per-(memory, receiver) decision alignment:")
    print(f"  Uniform TCI alignment:        {uniform_correct}/{total_decisions} ({uniform_correct/max(total_decisions,1)*100:.1f}%)")
    print(f"  Uniform false accepts (harm): {uniform_false_accept} ({uniform_false_accept/max(total_decisions,1)*100:.1f}%)")
    print(f"  Uniform false rejects (loss): {uniform_false_reject} ({uniform_false_reject/max(total_decisions,1)*100:.1f}%)")
    print(f"  Receiver self-consistency:    {receiver_self_consistent}/{total_decisions} (100% by construction)")

    # Write summary JSON
    summary = {
        "total_memories": total_memories,
        "disagreement_count": disagreement_count,
        "disagreement_rate": float(np.mean([r["disagreement_rate"] for r in analysis_rows])),
        "mean_delta_variance": float(np.mean(delta_variances)),
        "selective_transfer": {str(k): v for k, v in useful_counts.items()},
        "negative_transfer_prevented": neg_prevented_per_receiver,
        "uniform_per_receiver_alignment": uniform_correct / max(total_decisions, 1),
        "uniform_false_accept": uniform_false_accept,
        "uniform_false_reject": uniform_false_reject,
        "receiver_self_consistency_by_construction": receiver_self_consistent / max(total_decisions, 1),
    }
    summary_path = output_dir / "receiver_conditioned_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten: {summary_path}")


if __name__ == "__main__":
    main()
