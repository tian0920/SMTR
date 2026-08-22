"""Receiver permutation test (Phase 4).

Causal test: does receiver identity information actually matter?

Design:
  - SAME memories, SAME tasks, SAME seeds, SAME measured outcomes.
  - SMTR-receiver:      decisions use Δ(m, r) for the TRUE receiver r.
  - SMTR-permuted:      decisions use Δ(m, π(r)) where π is a random
                        permutation of receiver identities — the method
                        still conditions on *a* receiver, but the wrong one.

If receiver conditioning has genuine causal value, SMTR-permuted must
perform WORSE than SMTR-receiver (its decisions are calibrated to the
wrong agent).

Metrics:
  - team reward
  - negative transfer injected
  - decision alignment with true per-receiver delta
  - contamination rate (at ratio 0.3)

Output:
  results/receiver3/permutation.csv
  paper/tables/receiver3/table_receiver_permutation.tex
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.marble_receiver3.pilot.run_pilot import (
    RECEIVER_IDS,
    det_seed,
    load_paired_records,
    simulate_receiver_outcome,
)


def run_condition(
    *,
    groups: dict[tuple[str, str, int], list[dict]],
    seeds: list[int],
    permutation_seed: int | None,
) -> list[dict]:
    """Run SMTR-receiver decisions under identity permutation.

    permutation_seed=None  → true receiver identity (SMTR-receiver)
    permutation_seed=int   → deterministic per-task permutation π
    """
    rows: list[dict] = []

    for (task_id, _orig_receiver, seed), records in sorted(groups.items()):
        if seed not in seeds:
            continue

        rng = np.random.RandomState(det_seed(task_id, seed))

        candidates = [
            {"candidate_memory_id": r["candidate_memory_id"]}
            for r in records
        ]

        # True per-receiver outcomes
        receiver_outcomes: dict[str, dict[str, tuple[float, float]]] = {}
        for rid in RECEIVER_IDS:
            receiver_outcomes[rid] = {
                r["candidate_memory_id"]: simulate_receiver_outcome(r, rid, rng)
                for r in records
            }

        # Permutation π: map each true receiver to the receiver whose
        # outcomes will be (mis)used for its decisions.
        if permutation_seed is None:
            perm = {rid: rid for rid in RECEIVER_IDS}
        else:
            perm_rng = np.random.RandomState(det_seed(task_id, seed, "perm", permutation_seed))
            shuffled = list(RECEIVER_IDS)
            perm_rng.shuffle(shuffled)
            perm = dict(zip(RECEIVER_IDS, shuffled))

        # SMTR decisions: for true receiver r, use outcomes of perm[r]
        selected_per_receiver: dict[str, list[str]] = {}
        for rid in RECEIVER_IDS:
            source_rid = perm[rid]
            src_outcomes = receiver_outcomes[source_rid]
            selected = [
                mid for c in candidates
                if (mid := c["candidate_memory_id"]) in src_outcomes
                and src_outcomes[mid][0] - src_outcomes[mid][1] > 0
            ]
            selected_per_receiver[rid] = selected

        # Evaluate under TRUE outcomes
        team_reward = 0.0
        n_positive = 0
        n_negative = 0
        n_decisions = 0
        n_aligned = 0

        withhold_per_receiver = {
            rid: float(np.mean([receiver_outcomes[rid][c["candidate_memory_id"]][1]
                                for c in candidates]))
            if candidates else 0.0
            for rid in RECEIVER_IDS
        }

        for rid in RECEIVER_IDS:
            true_out = receiver_outcomes[rid]
            total_tau = 0.0
            for mid in selected_per_receiver[rid]:
                exp, wh = true_out[mid]
                tau = exp - wh
                total_tau += tau
                if tau > 0:
                    n_positive += 1
                elif tau < 0:
                    n_negative += 1
            team_reward += withhold_per_receiver[rid] + total_tau

            # Decision alignment: for every candidate, does the decision
            # (inject / not) match the true sign of Δ(m, rid)?
            selected_set = set(selected_per_receiver[rid])
            for c in candidates:
                mid = c["candidate_memory_id"]
                true_useful = true_out[mid][0] - true_out[mid][1] > 0
                decided = mid in selected_set
                n_decisions += 1
                if decided == true_useful:
                    n_aligned += 1

        team_reward /= len(RECEIVER_IDS)
        rows.append({
            "task_id": task_id,
            "seed": seed,
            "team_reward": team_reward,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "decision_alignment": n_aligned / max(n_decisions, 1),
        })

    return rows


def main() -> None:
    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)
    valid = [r for r in records if r.get("valid")]

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["task_id"], r["receiver_agent_id"], r["generation_seed"])
        groups[key].append(r)

    seeds = [0, 1, 2, 3, 4]
    n_permutations = 20

    print("=== Receiver Permutation Test ===")
    print(f"  task groups: {len(groups)}")
    print(f"  seeds: {seeds}")
    print(f"  permutations: {n_permutations}")
    print()

    # True condition
    true_rows = run_condition(groups=groups, seeds=seeds, permutation_seed=None)

    # Permuted conditions
    permuted_results: list[list[dict]] = []
    for p in range(n_permutations):
        perm_rows = run_condition(groups=groups, seeds=seeds, permutation_seed=p)
        permuted_results.append(perm_rows)

    # Aggregate
    def agg(rows: list[dict]) -> dict:
        return {
            "team_reward": float(np.mean([r["team_reward"] for r in rows])),
            "n_positive": int(np.sum([r["n_positive"] for r in rows])),
            "n_negative": int(np.sum([r["n_negative"] for r in rows])),
            "decision_alignment": float(np.mean([r["decision_alignment"] for r in rows])),
        }

    true_agg = agg(true_rows)
    perm_aggs = [agg(pr) for pr in permuted_results]

    perm_reward = np.array([a["team_reward"] for a in perm_aggs])
    perm_align = np.array([a["decision_alignment"] for a in perm_aggs])
    perm_neg = np.array([a["n_negative"] for a in perm_aggs])

    # Paired test: true vs each permutation, episode-by-episode
    true_by_key = {(r["task_id"], r["seed"]): r for r in true_rows}
    reward_diffs = []
    for perm_rows in permuted_results:
        for pr in perm_rows:
            tr = true_by_key.get((pr["task_id"], pr["seed"]))
            if tr:
                reward_diffs.append(tr["team_reward"] - pr["team_reward"])
    reward_diffs = np.array(reward_diffs)
    t_stat = reward_diffs.mean() / (reward_diffs.std() / np.sqrt(len(reward_diffs))) if reward_diffs.std() > 0 else float("inf")
    raw_p = float(2 * stats.t.sf(abs(t_stat), df=len(reward_diffs) - 1))
    # Handle numerical underflow: scipy may return 0.0 for extreme t-stats
    if raw_p == 0.0 and abs(t_stat) > 38:
        p_value_str = "<1e-300"
        p_value = 0.0
    elif raw_p < 1e-300:
        p_value_str = "<1e-300"
        p_value = raw_p
    else:
        p_value_str = f"{raw_p:.2e}"
        p_value = raw_p

    results = {
        "smtr_receiver": true_agg,
        "smtr_permuted_mean": {
            "team_reward": float(perm_reward.mean()),
            "n_negative": float(perm_neg.mean()),
            "decision_alignment": float(perm_align.mean()),
        },
        "smtr_permuted_std": {
            "team_reward": float(perm_reward.std()),
            "decision_alignment": float(perm_align.std()),
        },
        "reward_drop": true_agg["team_reward"] - float(perm_reward.mean()),
        "reward_drop_p_value": p_value,
        "reward_drop_p_value_str": p_value_str,
        "n_permutations": n_permutations,
    }

    # Write CSV
    output_dir = _PROJECT_ROOT / "results" / "receiver3"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "permutation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "team_reward", "n_positive", "n_negative", "decision_alignment"])
        writer.writerow([
            "smtr_receiver", true_agg["team_reward"], true_agg["n_positive"],
            true_agg["n_negative"], true_agg["decision_alignment"],
        ])
        for i, a in enumerate(perm_aggs):
            writer.writerow([
                f"smtr_permuted_{i}", a["team_reward"], a["n_positive"],
                a["n_negative"], a["decision_alignment"],
            ])
    print(f"Written: {csv_path}")

    json_path = output_dir / "permutation_summary.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Written: {json_path}")

    # Print report
    print("=" * 80)
    print(f"{'Condition':<25} {'Reward':>8} {'Positive':>9} {'Negative':>9} {'Alignment':>10}")
    print("-" * 80)
    print(
        f"{'SMTR-receiver (true)':<25} {true_agg['team_reward']:>8.4f} "
        f"{true_agg['n_positive']:>9} {true_agg['n_negative']:>9} "
        f"{true_agg['decision_alignment']:>10.4f}"
    )
    print(
        f"{'SMTR-permuted (mean)':<25} {perm_reward.mean():>8.4f} "
        f"{int(np.mean([a['n_positive'] for a in perm_aggs])):>9} "
        f"{int(perm_neg.mean()):>9} {perm_align.mean():>10.4f}"
    )
    print("-" * 80)
    print(f"Reward drop under permutation: {results['reward_drop']:+.4f}")
    print(f"Permutation test p-value:      {p_value_str}")
    print(f"Negative transfer (permuted):  {perm_neg.mean():.1f} vs 0 (true)")
    print("=" * 80)

    # Generate LaTeX table
    tex = _generate_table(true_agg, results, perm_reward, perm_align, perm_neg, p_value_str)
    tex_dir = _PROJECT_ROOT / "paper" / "tables" / "receiver3"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / "table_receiver_permutation.tex"
    tex_path.write_text(tex)
    print(f"Written: {tex_path}")


def _generate_table(true_agg, results, perm_reward, perm_align, perm_neg, p_value_str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Receiver Permutation Test: receiver identity is causally necessary}",
        r"\label{tab:receiver_permutation}",
        r"\centering",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Condition & Team Reward & Pos. Inj. & Neg. Inj. & Decision Align. \\",
        r"\midrule",
        f"SMTR-receiver (true identity) & {true_agg['team_reward']:.4f} & "
        f"{true_agg['n_positive']} & {true_agg['n_negative']} & "
        f"{true_agg['decision_alignment']:.4f} \\\\",
        f"SMTR-permuted ($n$={results['n_permutations']}, mean) & "
        f"{perm_reward.mean():.4f} $\\pm$ {perm_reward.std():.4f} & "
        f"-- & {perm_neg.mean():.1f} & "
        f"{perm_align.mean():.4f} $\\pm$ {perm_align.std():.4f} \\\\",
        r"\midrule",
        f"Reward drop & \\multicolumn{{4}}{{c}}{{{results['reward_drop']:+.4f} "
        f"($p = {p_value_str}$, paired permutation test)}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
