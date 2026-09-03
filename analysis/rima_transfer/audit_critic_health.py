"""Offline Critic Health Audit (Phase 13).

Audits a frozen bootstrap transfer critic against its training/intervention
records BEFORE any real MARBLE re-run. Reports:

* Dataset support: n_valid_edges, n_task_receiver_families,
  n_positive_edges, n_negative_edges.
* Prediction distribution: mu (min/q25/median/q75/max).
* Uncertainty: sigma (min/median/max).
* Bounds: LCB and UCB distributions (min/q25/median/q75/max).
* Offline accuracy: MAE, sign accuracy, sigma-error correlation,
  LCB coverage.

Usage::

    python analysis/rima_transfer/audit_critic_health.py \\
        --critic results/rima_transfer/critic/critic_receiver_bootstrap.joblib \\
        --records results/rima/stage_a/intervention_records.json \\
        --source-agents results/rima/stage_a/source_agents.json \\
        --output results/rima_transfer/critic/critic_health_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rima.train_critic import (  # noqa: E402
    load_records,
    record_to_example,
)
from smtr.router.official_score_transfer_critic import (  # noqa: E402
    BootstrapOfficialScoreTransferCritic,
)

# Phase 23-25 minimum support targets.
MIN_VALID_EDGES = 60
MIN_TASK_FAMILIES = 15
MIN_POSITIVE_EDGES = 15


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summary(values: list[float]) -> dict[str, float | None]:
    vals = sorted(values)
    return {
        "n": len(vals),
        "min": vals[0] if vals else None,
        "q25": _quantile(vals, 0.25),
        "median": _quantile(vals, 0.50),
        "q75": _quantile(vals, 0.75),
        "max": vals[-1] if vals else None,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return cov / math.sqrt(vx * vy)


def audit_critic_health(
    *,
    critic_path: str,
    records_path: str,
    source_agents_path: str | None,
    beta: float = 1.64,
) -> dict[str, Any]:
    critic = BootstrapOfficialScoreTransferCritic.load(critic_path)
    records = load_records(Path(records_path))
    source_agent_ids: dict[str, str] = {}
    if source_agents_path:
        with open(source_agents_path) as f:
            source_agent_ids = json.load(f)

    examples = [
        record_to_example(r, source_agent_ids=source_agent_ids)
        for r in records
    ]

    # ---- Dataset support ----
    valid = [
        ex for ex in examples
        if ex.official_expose_score is not None
        and ex.official_withhold_score is not None
    ]
    families = {(ex.task_id, ex.receiver_id) for ex in valid}
    taus = [
        (ex.official_expose_score - ex.official_withhold_score)
        for ex in valid
    ]
    n_positive = sum(1 for t in taus if t > 0)
    dataset_support = {
        "n_examples_total": len(examples),
        "n_valid_edges": len(valid),
        "n_task_receiver_families": len(families),
        "n_positive_edges": n_positive,
        "n_negative_edges": len(valid) - n_positive,
    }

    # ---- Predictions over all examples ----
    mus: list[float] = []
    sigmas: list[float] = []
    lcbs: list[float] = []
    ucbs: list[float] = []
    errors: list[float] = []
    abs_errors: list[float] = []
    sigma_vs_abs_err: list[tuple[float, float]] = []
    sign_correct = 0
    lcb_covered = 0
    n_accuracy_pairs = 0

    for ex in examples:
        dist = critic.predict_distribution(ex)
        if dist.mu_tau is None or dist.sigma_tau is None:
            continue
        mus.append(dist.mu_tau)
        sigmas.append(dist.sigma_tau)
        lcb = dist.mu_tau - beta * dist.sigma_tau
        ucb = dist.mu_tau + beta * dist.sigma_tau
        lcbs.append(lcb)
        ucbs.append(ucb)

        if (
            ex.official_expose_score is not None
            and ex.official_withhold_score is not None
        ):
            tau_obs = ex.official_expose_score - ex.official_withhold_score
            err = dist.mu_tau - tau_obs
            errors.append(err)
            abs_errors.append(abs(err))
            sigma_vs_abs_err.append((dist.sigma_tau, abs(err)))
            n_accuracy_pairs += 1
            # Sign accuracy: predicted direction matches observed direction
            # (treat tau_obs == 0 as non-positive).
            if (dist.mu_tau > 0) == (tau_obs > 0):
                sign_correct += 1
            # LCB coverage: observed tau should not fall below the
            # lower confidence bound.
            if tau_obs >= lcb:
                lcb_covered += 1

    offline_accuracy = {
        "n_pairs": n_accuracy_pairs,
        "mae": (sum(abs_errors) / n_accuracy_pairs) if n_accuracy_pairs else None,
        "sign_accuracy": (sign_correct / n_accuracy_pairs) if n_accuracy_pairs else None,
        "sigma_error_correlation": _pearson(
            [s for s, _ in sigma_vs_abs_err],
            [e for _, e in sigma_vs_abs_err],
        ),
        "lcb_coverage": (lcb_covered / n_accuracy_pairs) if n_accuracy_pairs else None,
        "beta_used": beta,
    }

    # ---- Phase 14 support verdict ----
    underpowered = (
        dataset_support["n_valid_edges"] < MIN_VALID_EDGES
        or dataset_support["n_task_receiver_families"] < MIN_TASK_FAMILIES
        or dataset_support["n_positive_edges"] < MIN_POSITIVE_EDGES
    )
    warnings_list: list[str] = []
    if underpowered:
        warnings_list.append("CRITIC_LOW_SUPPORT_WARNING")
    if dataset_support["n_positive_edges"] < 5:
        warnings_list.append("GAMMA_LOW_SUPPORT")

    return {
        "schema_version": "rima_critic_health_audit_v1",
        "critic_checkpoint": critic_path,
        "records": records_path,
        "dataset_support": dataset_support,
        "mu_distribution": _summary(mus),
        "sigma_distribution": _summary(sigmas),
        "lcb_distribution": _summary(lcbs),
        "ucb_distribution": _summary(ucbs),
        "offline_accuracy": offline_accuracy,
        "critic_status": "UNDERPOWERED" if underpowered else "OK",
        "support_targets": {
            "min_valid_edges": MIN_VALID_EDGES,
            "min_task_receiver_families": MIN_TASK_FAMILIES,
            "min_positive_edges": MIN_POSITIVE_EDGES,
        },
        "warnings": warnings_list,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline critic health audit")
    parser.add_argument(
        "--critic",
        default="results/rima_transfer/critic/critic_receiver_bootstrap.joblib",
    )
    parser.add_argument(
        "--records",
        default="results/rima/stage_a/intervention_records.json",
    )
    parser.add_argument(
        "--source-agents",
        default="results/rima/stage_a/source_agents.json",
    )
    parser.add_argument("--beta", type=float, default=1.64)
    parser.add_argument(
        "--output",
        default="results/rima_transfer/critic/critic_health_audit.json",
    )
    args = parser.parse_args(argv)

    report = audit_critic_health(
        critic_path=args.critic,
        records_path=args.records,
        source_agents_path=args.source_agents,
        beta=args.beta,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    for w in report["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
