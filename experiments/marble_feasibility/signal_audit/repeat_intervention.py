"""Audit 1: Intervention Stability Test.

Computes ICC (Intraclass Correlation Coefficient) from existing replicates
to determine if τ = Y₁ - Y₀ is a stable causal effect or just noise.

Uses paired records that share the same (task, memory, receiver) triple
but were executed with different generation_seeds.

Output: repeat_intervention.jsonl + stability summary.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_all_paired_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for raw in paths:
        p = _PROJECT_ROOT / raw
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


def _get_tau(record: dict) -> int:
    y1 = 1 if record.get("share", {}).get("team_success") else 0
    y0 = 1 if record.get("withhold", {}).get("team_success") else 0
    return y1 - y0


def _group_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("task_id", "")),
        record.get("candidate_memory_id", ""),
        record.get("receiver_agent_id", ""),
    )


def _compute_icc(groups: dict[tuple, list[int]]) -> dict:
    """One-way random-effects ICC from grouped τ values.

    ICC = (MSB - MSW) / (MSB + (k̄ - 1) * MSW)

    where MSB = between-group mean square, MSW = within-group mean square,
    k̄ = mean group size.
    """
    group_taus = [taus for taus in groups.values() if len(taus) >= 2]
    if not group_taus:
        return {"icc": 0.0, "n_groups": 0, "error": "no groups with >=2 records"}

    n_groups = len(group_taus)
    group_sizes = [len(g) for g in group_taus]
    k_bar = float(np.mean(group_sizes))
    grand_mean = float(np.mean([t for g in group_taus for t in g]))

    # Between-group sum of squares
    ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_taus)
    msb = ssb / max(n_groups - 1, 1)

    # Within-group sum of squares
    ssw = sum(sum((t - np.mean(g)) ** 2 for t in g) for g in group_taus)
    df_w = sum(len(g) - 1 for g in group_taus)
    msw = ssw / max(df_w, 1)

    denominator = msb + (k_bar - 1) * msw
    icc = (msb - msw) / denominator if denominator > 0 else 0.0

    # Agreement rate
    exact_agree = sum(1 for g in group_taus if len(set(g)) == 1)

    # Variance decomposition
    var_within = [float(np.var(g, ddof=0)) for g in group_taus]

    return {
        "icc": round(float(icc), 4),
        "n_groups": n_groups,
        "mean_group_size": round(k_bar, 2),
        "grand_mean_tau": round(grand_mean, 4),
        "msb": round(float(msb), 4),
        "msw": round(float(msw), 4),
        "exact_agreement": exact_agree,
        "exact_agreement_rate": round(exact_agree / n_groups, 4),
        "mean_within_group_variance": round(float(np.mean(var_within)), 4),
        "median_within_group_variance": round(float(np.median(var_within)), 4),
        "n_stable": exact_agree,
        "n_unstable": n_groups - exact_agree,
    }


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    stability_cfg = config["audit"]["stability"]

    print("=" * 60)
    print("Audit 1: Intervention Stability Test")
    print("=" * 60)

    # ── Load data ──
    all_records = _load_all_paired_records(data_cfg["all_paired_splits"])
    valid = [r for r in all_records if r.get("valid", False)]
    print(f"\n  Total records: {len(all_records)}, Valid: {len(valid)}")

    # ── Group by (task, memory, receiver) ──
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in valid:
        groups[_group_key(r)].append(r)

    multi_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"  Unique groups: {len(groups)}")
    print(f"  Groups with >=2 replicates: {len(multi_groups)}")

    # ── Compute τ for each replicate ──
    tau_groups: dict[tuple, list[int]] = {}
    for key, recs in multi_groups.items():
        taus = [_get_tau(r) for r in recs]
        tau_groups[key] = taus

    # ── Compute ICC ──
    icc_result = _compute_icc(tau_groups)
    icc = icc_result["icc"]

    print(f"\n  ICC (one-way random): {icc:.4f}")
    print(f"  Exact agreement rate: "
          f"{icc_result['exact_agreement']}/{icc_result['n_groups']} = "
          f"{icc_result['exact_agreement_rate']:.2%}")
    print(f"  Mean within-group variance: {icc_result['mean_within_group_variance']:.4f}")

    # ── Verdict ──
    pass_thr = stability_cfg["icc_pass_threshold"]
    fail_thr = stability_cfg["icc_fail_threshold"]

    if icc > pass_thr:
        verdict = "PASS"
        interpretation = "Effect is stable — signal exists."
    elif icc < fail_thr:
        verdict = "FAIL"
        interpretation = "Mainly environmental randomness — no stable signal."
    else:
        verdict = "BORDERLINE"
        interpretation = "Weak stability — signal may exist but is noisy."

    print(f"\n  Verdict: {verdict} (ICC={icc:.4f}, "
          f"pass>{pass_thr}, fail<{fail_thr})")
    print(f"  Interpretation: {interpretation}")

    # ── Show unstable groups ──
    unstable_examples = []
    for key, taus in tau_groups.items():
        if len(set(taus)) > 1:
            seeds = []
            for r in multi_groups[key]:
                seeds.append(r.get("generation_seed", -1))
            unstable_examples.append({
                "task_id": key[0],
                "memory_id": key[1],
                "receiver_id": key[2],
                "taus": taus,
                "seeds": seeds,
            })

    print(f"\n  Unstable groups: {len(unstable_examples)}")
    for ex in unstable_examples[:5]:
        print(f"    task={ex['task_id']}, mem={ex['memory_id'][:30]}, "
              f"taus={ex['taus']}, seeds={ex['seeds']}")

    # ── Save output ──
    out_dir = _THIS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # repeat_intervention.jsonl
    output_records = []
    for key, recs in multi_groups.items():
        taus = [_get_tau(r) for r in recs]
        seeds = [r.get("generation_seed", -1) for r in recs]
        output_records.append({
            "task_id": key[0],
            "memory_id": key[1],
            "receiver_id": key[2],
            "n_replicates": len(recs),
            "taus": taus,
            "seeds": seeds,
            "tau_mean": round(float(np.mean(taus)), 4),
            "tau_std": round(float(np.std(taus)), 4),
            "stable": len(set(taus)) == 1,
        })

    out_path = out_dir / "repeat_intervention.jsonl"
    with open(out_path, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\n  Saved: {out_path}")

    # stability_summary.json
    summary = {
        "audit": "intervention_stability",
        "icc": icc,
        "verdict": verdict,
        "interpretation": interpretation,
        "thresholds": {
            "pass": pass_thr,
            "fail": fail_thr,
        },
        "statistics": icc_result,
        "n_unstable_examples": len(unstable_examples),
        "unstable_examples_sample": unstable_examples[:10],
    }
    summary_path = out_dir / "stability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {verdict}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
