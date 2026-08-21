"""Multi-receiver heterogeneity sanity checks (Task 3).

Inputs: multi_receiver_paired_records.jsonl produced by
collect_multi_receiver.py. Each record holds
  { task, receiver, memory, Y_expose, Y_withhold }
with receiver genuinely varying across agent0/agent1/agent2.

Checks:
  1. Receiver variance — for the same (task, memory), compare
     tau(m, r1), tau(m, r2), tau(m, r3). Output receiver_effect_variance.
     Requirement: variance > 0 for at least one memory (heterogeneity).
  2. Global vs Receiver-conditioned — compare prediction quality of
     global tau(m) (mean across receivers) vs receiver-conditioned
     tau(m, r). Expect SMTR (receiver-conditioned) >= Global.

Outputs:
  - reports/receiver_sanity.json
  - reports/receiver_sanity.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
_FEASIBILITY_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _FEASIBILITY_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml


def _load_config() -> dict:
    cfg_path = _PROJECT_ROOT / "configs" / "marble_3receiver.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _tau(record: dict) -> int:
    y1 = 1 if record.get("share", {}).get("team_success") else 0
    y0 = 1 if record.get("withhold", {}).get("team_success") else 0
    return y1 - y0


def receiver_variance(records: list[dict]) -> dict:
    """Compute per-memory tau across receivers and overall variance.

    Returns {
      "per_memory": {(task, memory): {receiver: tau}},
      "receiver_effect_variance": float,  # mean across-memory var of tau
      "heterogeneous_memories": int,      # memories with var > 0
      "n_comparable_memories": int,       # memories observed for >=2 receivers
    }
    """
    # group by (task_id, memory_id) -> {receiver: [tau values]}
    groups: dict[tuple[str, str], dict[str, list[int]]] = {}
    for r in records:
        if not r.get("valid", False):
            continue
        key = (str(r.get("task_id", "")), str(r.get("candidate_memory_id", "")))
        recv = str(r.get("receiver_agent_id", ""))
        groups.setdefault(key, {}).setdefault(recv, []).append(_tau(r))

    per_memory: dict[str, dict] = {}
    variances: list[float] = []
    n_heterogeneous = 0
    n_comparable = 0

    for (task_id, memory_id), recv_map in sorted(groups.items()):
        mean_tau = {recv: float(np.mean(vals)) for recv, vals in recv_map.items()}
        if len(mean_tau) >= 2:
            n_comparable += 1
            v = float(np.var(list(mean_tau.values())))
            variances.append(v)
            if v > 0:
                n_heterogeneous += 1
        per_memory[f"{task_id}::{memory_id}"] = mean_tau

    overall_var = float(np.mean(variances)) if variances else 0.0
    return {
        "per_memory": per_memory,
        "receiver_effect_variance": overall_var,
        "heterogeneous_memories": n_heterogeneous,
        "n_comparable_memories": n_comparable,
    }


def global_vs_receiver_conditioned(records: list[dict]) -> dict:
    """Compare predictive accuracy of global tau(m) vs tau(m, r).

    Held-out style evaluation: for each record, predict its tau with
    (a) global mean tau across other receivers for the same (task, memory)
    (b) receiver-conditioned leave-one-out mean for the same (task, memory, receiver)
    Score = mean absolute prediction error (lower is better).
    SMTR (receiver-conditioned) wins if its error <= global error.
    """
    valid = [r for r in records if r.get("valid", False)]
    if len(valid) < 4:
        return {"skip": True, "reason": f"only {len(valid)} valid records"}

    # index taus by (task, memory) and (task, memory, receiver)
    idx_by_key: dict[tuple[str, str], list[int]] = {}
    idx_by_key_recv: dict[tuple[str, str, str], list[int]] = {}
    keyed_taus: list[tuple[tuple[str, str], str, int]] = []
    for r in valid:
        key = (str(r.get("task_id", "")), str(r.get("candidate_memory_id", "")))
        recv = str(r.get("receiver_agent_id", ""))
        tau = _tau(r)
        idx_by_key.setdefault(key, []).append(tau)
        idx_by_key_recv.setdefault((key[0], key[1], recv), []).append(tau)
        keyed_taus.append((key, recv, tau))

    global_errors: list[float] = []
    recv_errors: list[float] = []
    for key, recv, tau in keyed_taus:
        # global prediction: leave-one-out mean across ALL receivers
        all_taus = idx_by_key[key]
        if len(all_taus) > 1:
            glob_pred = (sum(all_taus) - tau) / (len(all_taus) - 1)
            global_errors.append(abs(tau - glob_pred))

        # receiver-conditioned: leave-one-out mean within same receiver
        same_recv = idx_by_key_recv[(key[0], key[1], recv)]
        if len(same_recv) > 1:
            recv_pred = (sum(same_recv) - tau) / (len(same_recv) - 1)
            recv_errors.append(abs(tau - recv_pred))

    global_mae = float(np.mean(global_errors)) if global_errors else None
    recv_mae = float(np.mean(recv_errors)) if recv_errors else None
    return {
        "skip": False,
        "global_mae": global_mae,
        "receiver_conditioned_mae": recv_mae,
        "smtr_beats_global": (
            recv_mae <= global_mae if (recv_mae is not None and global_mae is not None) else None
        ),
        "n_global_evaluated": len(global_errors),
        "n_receiver_evaluated": len(recv_errors),
    }


def main() -> None:
    print("=" * 60)
    print("Multi-Receiver Heterogeneity Sanity Checks (Task 3)")
    print("=" * 60)

    config = _load_config()
    data_dir = _PROJECT_ROOT / config["data"]["output_dir"]
    records_path = data_dir / "multi_receiver_paired_records.jsonl"

    if not records_path.exists():
        print(f"\n  No records found at {records_path}")
        print("  Run collect_multi_receiver.py first.")
        sys.exit(2)

    records = _load_records(records_path)
    valid = [r for r in records if r.get("valid", False)]
    receivers = sorted({str(r.get("receiver_agent_id")) for r in valid})
    tasks = sorted({str(r.get("task_id")) for r in valid})
    print(f"\n  Records: {len(records)} total, {len(valid)} valid")
    print(f"  Receivers: {receivers}")
    print(f"  Tasks: {tasks}")

    # ── Check 1: Receiver variance ──
    print("\n  ── Check 1: Receiver effect variance ──")
    var_result = receiver_variance(records)
    print(f"    Comparable memories (>=2 receivers): {var_result['n_comparable_memories']}")
    print(f"    Heterogeneous memories (var>0):      {var_result['heterogeneous_memories']}")
    print(f"    receiver_effect_variance:            {var_result['receiver_effect_variance']:.4f}")
    for key, recv_map in var_result["per_memory"].items():
        print(f"      {key}: {recv_map}")

    variance_ok = var_result["receiver_effect_variance"] > config["sanity"]["receiver_effect_variance_min"]
    heterogeneity_ok = var_result["heterogeneous_memories"] > 0

    # ── Check 2: Global vs Receiver-conditioned ──
    print("\n  ── Check 2: Global vs Receiver-conditioned ──")
    cmp_result = global_vs_receiver_conditioned(records)
    if cmp_result.get("skip"):
        print(f"    Skipped: {cmp_result['reason']}")
        smtr_beats_global = None
    else:
        g_mae = cmp_result["global_mae"]
        r_mae = cmp_result["receiver_conditioned_mae"]
        print(f"    Global MAE:               "
              f"{g_mae:.4f}" if g_mae is not None else "    Global MAE: n/a")
        print("    Receiver-conditioned MAE: "
              + (f"{r_mae:.4f}" if r_mae is not None else "n/a (single seed per receiver)"))
        smtr_beats_global = cmp_result["smtr_beats_global"]
        print(f"    SMTR beats global: "
              + ("inconclusive (need >=2 seeds/receiver)"
                 if smtr_beats_global is None else str(smtr_beats_global)))

    # ── Verdict ──
    checks = {
        "multi_receiver_data_collected": {
            "description": "Records span >= 2 receivers",
            "value": f"receivers={receivers}",
            "passed": len(receivers) >= 2,
        },
        "receiver_effect_variance_positive": {
            "description": "receiver_effect_variance > 0",
            "value": f"variance={var_result['receiver_effect_variance']:.4f}",
            "passed": variance_ok,
        },
        "heterogeneity_exists": {
            "description": "at least one memory shows tau(m,r1) != tau(m,r2)",
            "value": f"heterogeneous={var_result['heterogeneous_memories']}/"
                     f"{var_result['n_comparable_memories']}",
            "passed": heterogeneity_ok,
        },
        "smtr_beats_global": {
            "description": "receiver-conditioned tau(m,r) MAE <= global tau(m) MAE",
            "value": (
                "skipped (insufficient data)" if cmp_result.get("skip")
                else (
                    "receiver-conditioned MAE n/a (need >=2 seeds per receiver); "
                    f"global_mae={cmp_result['global_mae']:.4f}"
                ) if cmp_result.get("receiver_conditioned_mae") is None
                else f"recv_mae={cmp_result['receiver_conditioned_mae']:.4f}, "
                     f"global_mae={cmp_result['global_mae']:.4f}"
            ),
            "passed": None if smtr_beats_global is None else bool(smtr_beats_global),
        },
    }

    n_fail = sum(1 for c in checks.values() if c["passed"] is False)
    n_na = sum(1 for c in checks.values() if c["passed"] is None)
    verdict = "PASS" if n_fail == 0 else "FAIL"

    print(f"\n{'=' * 60}")
    print("Sanity Check Results")
    print("=" * 60)
    for name, check in checks.items():
        status = ("PASS" if check["passed"] is True
                  else ("FAIL" if check["passed"] is False else "N/A"))
        print(f"  [{status}] {check['description']}")
        print(f"         {check['value']}")
    if n_na:
        print(f"\n  ({n_na} check(s) inconclusive due to single-seed data)")
    print(f"\n  Verdict: {verdict}")

    # ── Reports ──
    reports_dir = _THIS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "verdict": verdict,
        "checks": checks,
        "receiver_variance": var_result,
        "global_vs_receiver": cmp_result,
        "data_summary": {
            "n_records": len(records),
            "n_valid": len(valid),
            "receivers": receivers,
            "tasks": tasks,
        },
    }
    json_path = reports_dir / "receiver_sanity.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {json_path}")

    md = ["# Multi-Receiver Heterogeneity Sanity Report\n"]
    md.append(f"**Verdict: {verdict}**\n")
    md.append(f"- Records: {len(records)} total, {len(valid)} valid")
    md.append(f"- Receivers: {receivers}")
    md.append(f"- Tasks: {tasks}\n")
    md.append("## Receiver Effect Variance\n")
    md.append(f"| (task, memory) | " + " | ".join(receivers) + " |")
    md.append("|---|" + "---|" * len(receivers))
    for key, recv_map in var_result["per_memory"].items():
        row = [f"{recv_map.get(rv, '-'):g}" if isinstance(recv_map.get(rv), float) else "-"
               for rv in receivers]
        md.append(f"| {key} | " + " | ".join(row) + " |")
    md.append(f"\n**receiver_effect_variance = {var_result['receiver_effect_variance']:.4f}**\n")
    md.append("## Global vs Receiver-conditioned\n")
    if cmp_result.get("skip"):
        md.append(f"Skipped: {cmp_result['reason']}\n")
    else:
        g_mae = cmp_result.get("global_mae")
        r_mae = cmp_result.get("receiver_conditioned_mae")
        md.append(f"- Global τ(m) MAE: " + (f"{g_mae:.4f}" if g_mae is not None else "n/a"))
        md.append("- Receiver-conditioned τ(m,r) MAE: "
                  + (f"{r_mae:.4f}" if r_mae is not None else "n/a (single seed per receiver)"))
        md.append(f"- SMTR beats global: {cmp_result.get('smtr_beats_global')}\n")
    md.append("## Checks\n")
    for name, check in checks.items():
        icon = "PASS" if check["passed"] is True else ("FAIL" if check["passed"] is False else "N/A")
        md.append(f"- [{icon}] {check['description']}: {check['value']}")
    md_path = reports_dir / "receiver_sanity.md"
    md_path.write_text("\n".join(md))
    print(f"  Saved: {md_path}")

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
