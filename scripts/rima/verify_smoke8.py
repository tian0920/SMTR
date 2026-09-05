"""8-task real smoke verification (pre-v3 gate).

Checks the five acceptance criteria from the P0 revision directive for
the 8-task adaptive smoke run (bargaining / frozen stream0):

1. probe_count >= 5
2. refit_count >= 1
3. After each refit at task p, the controller critic version used at
   every task > p equals the new (higher) version.
4. At least one same/similar candidate shows a prediction change
   between critic v1 and a later version. Primary evidence:
   refit_prediction_deltas.jsonl (refit-time v_old vs v_new
   predictions on the fixed probed-candidate set). Fallback: probe
   events with identical (memory_id, receiver_id) under two versions.
5. Runtime causal audit passes (audit_continual_run.py exit code 0).

LCB > 0 is NOT required.

Usage::

    python scripts/rima/verify_smoke8.py <run_dir> [--skip-audit]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_smoke8(run_dir: str | Path, skip_audit: bool = False) -> dict:
    run_dir = Path(run_dir)
    probe_events = _load_jsonl(run_dir / "probe_events.jsonl")
    critic_versions = _load_jsonl(run_dir / "critic_versions.jsonl")
    tasks = _load_jsonl(run_dir / "tasks.jsonl")
    routing = _load_jsonl(run_dir / "routing.jsonl")

    checks: dict[str, Any] = {}

    # --- Criterion 1: probe_count >= 5 -------------------------------
    n_probes = len(probe_events)
    checks["1_probe_count"] = {
        "value": n_probes,
        "required": ">= 5",
        "pass": n_probes >= 5,
    }

    # --- Criterion 2: refit_count >= 1 -------------------------------
    n_refits = len(critic_versions)
    checks["2_refit_count"] = {
        "value": n_refits,
        "required": ">= 1",
        "pass": n_refits >= 1,
    }

    # --- Criterion 3: controller version increases after refit -------
    # Controller version actually used per task (P0-6) from
    # routing.jsonl; fall back to task records if missing.
    ctrl_by_pos: dict[int, int] = {}
    for d in routing:
        pos = d.get("task_position")
        ver = d.get("controller_critic_version")
        if pos is not None and ver is not None:
            ctrl_by_pos[pos] = int(ver)
    if not ctrl_by_pos:
        for t in tasks:
            pos = t.get("task_position")
            ver = t.get("controller_critic_version")
            if pos is not None and ver is not None:
                ctrl_by_pos[pos] = int(ver)

    violations = []
    for entry in critic_versions:
        p = entry.get("task_position")
        new_v = entry.get("critic_version")
        if p is None or new_v is None:
            continue
        for pos in sorted(ctrl_by_pos):
            if pos > p and ctrl_by_pos[pos] < new_v:
                violations.append(
                    {"refit_at": p, "expected_version": new_v,
                     "task_position": pos,
                     "actual_version": ctrl_by_pos[pos]},
                )
    checks["3_version_increases_after_refit"] = {
        "controller_versions_by_task": ctrl_by_pos,
        "refits": critic_versions,
        "violations": violations,
        "pass": n_refits >= 1 and not violations
        and len(ctrl_by_pos) > 0,
    }

    # --- Criterion 4: v1 vs v2 prediction change on same candidate ---
    # Primary evidence: refit-time prediction-delta log (old vs new
    # critic evaluated on the SAME fixed probed-candidate set).
    refit_deltas = _load_jsonl(run_dir / "refit_prediction_deltas.jsonl")
    delta_changes = [
        r for r in refit_deltas
        if r.get("delta_mu") is not None
        and abs(float(r["delta_mu"])) > 1e-9
    ]

    # Fallback evidence: probe events re-evaluated under two versions.
    by_cand: dict[tuple[str, str], list[dict]] = {}
    for ev in probe_events:
        mid = ev.get("memory_id")
        rid = ev.get("receiver_id")
        mu = ev.get("predicted_mu_pre_probe")
        ver = ev.get("critic_version_pre_probe")
        if mid is None or rid is None or mu is None or ver is None:
            continue
        by_cand.setdefault((mid, rid), []).append(
            {"version": int(ver), "mu": float(mu),
             "task_position": ev.get("task_position")},
        )

    changes = []
    for (mid, rid), obs in by_cand.items():
        versions = {o["version"] for o in obs}
        if len(versions) < 2:
            continue
        base = next(o for o in obs if o["version"] == min(versions))
        for o in obs:
            if o["version"] != base["version"]:
                changes.append({
                    "memory_id": mid,
                    "receiver_id": rid,
                    "v1_version": base["version"],
                    "v1_mu": base["mu"],
                    "v2_version": o["version"],
                    "v2_mu": o["mu"],
                    "delta_mu": abs(o["mu"] - base["mu"]),
                })
    checks["4_prediction_change_v1_vs_v2"] = {
        "refit_delta_rows": len(refit_deltas),
        "refit_delta_changes": delta_changes[:10],
        "probe_event_changes": changes,
        "pass": len(delta_changes) > 0 or len(changes) > 0,
        "note": (
            None if (delta_changes or changes) else
            "No prediction change observed in refit_prediction_deltas "
            "or probe events; criterion not observable in this run."
        ),
    }

    # --- Criterion 5: runtime causal audit ---------------------------
    if skip_audit:
        checks["5_runtime_causal_audit"] = {
            "pass": None,
            "note": "skipped (--skip-audit)",
        }
    else:
        proc = subprocess.run(
            [
                sys.executable,
                "experiments/rima/audit_continual_run.py",
                "--input", str(run_dir),
            ],
            capture_output=True, text=True,
        )
        checks["5_runtime_causal_audit"] = {
            "exit_code": proc.returncode,
            "pass": proc.returncode == 0,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    overall = all(
        c.get("pass") is True for c in checks.values()
    )
    result = {
        "run_dir": str(run_dir),
        "overall": "PASS" if overall else "FAIL",
        "checks": checks,
    }

    out_path = run_dir / "smoke8_verification.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Smoke8 verification for {run_dir}")
    for name, c in checks.items():
        status = {True: "PASS", False: "FAIL", None: "N/A"}[
            c.get("pass")
        ]
        print(f"  [{status}] {name}")
        for k, v in c.items():
            if k not in ("pass",):
                text = str(v)
                print(f"      {k}: {text[:300]}")
    print(f"\n  OVERALL: {result['overall']}")
    print(f"Wrote {out_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rima/verify_smoke8.py <run_dir> "
              "[--skip-audit]")
        sys.exit(1)
    res = verify_smoke8(
        sys.argv[1], skip_audit="--skip-audit" in sys.argv,
    )
    sys.exit(0 if res["overall"] == "PASS" else 1)
