"""Mechanism pilot analysis pipeline (Phase 26-33).

Reads raw JSONL output from pilot runs and produces:
  - 4 core curves: G(t), R(t), Transfer State Growth, Task Score
  - Early/Middle/Late decomposition
  - Paired statistical comparison (bootstrap CI)
  - Transfer prediction generalization
  - Uncertainty online audit
  - γ routing audit
  - Cost decomposition
  - Search transfer sanity

Usage::

    python analysis/rima_transfer/analyze_mechanism_pilot.py \\
        --runs-dir results/rima_transfer/pilot/runs \\
        --output-dir results/rima_transfer/pilot/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.rima.experiment_config import (  # noqa: E402
    ALL_METHOD_VARIANTS,
    compute_early_late_scores,
)

logger = logging.getLogger("rima.analysis")

__all__ = [
    "load_run_data",
    "analyze_method",
    "run_full_analysis",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@dataclass
class RunData:
    """All raw data from one pilot run."""

    run_id: str
    method: str
    scenario: str
    stream_seed: int
    execution_seed: int
    tasks: list[dict[str, Any]] = field(default_factory=list)
    routing: list[dict[str, Any]] = field(default_factory=list)
    probe_events: list[dict[str, Any]] = field(default_factory=list)
    critic_versions: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)


def load_run_data(run_dir: Path) -> RunData | None:
    """Load all JSONL data from a single run directory."""
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        return None
    with open(manifest) as f:
        m = json.load(f)
    summary: dict[str, Any] = {}
    sp = run_dir / "summary.json"
    if sp.exists():
        with open(sp) as f:
            summary = json.load(f)
    costs: dict[str, Any] = {}
    cp = run_dir / "costs.json"
    if cp.exists():
        with open(cp) as f:
            costs = json.load(f)
    return RunData(
        run_id=m.get("run_id", run_dir.name),
        method=m.get("method", "unknown"),
        scenario=m.get("scenario", "unknown"),
        stream_seed=m.get("stream_seed", 0),
        execution_seed=m.get("execution_seed", 0),
        tasks=_read_jsonl(run_dir / "tasks.jsonl"),
        routing=_read_jsonl(run_dir / "routing.jsonl"),
        probe_events=_read_jsonl(run_dir / "probe_events.jsonl"),
        critic_versions=_read_jsonl(run_dir / "critic_versions.jsonl"),
        summary=summary,
        costs=costs,
    )


def load_all_runs(runs_dir: Path) -> dict[str, list[RunData]]:
    """Load all runs, grouped by method."""
    runs: dict[str, list[RunData]] = defaultdict(list)
    if not runs_dir.is_dir():
        logger.warning("runs dir not found: %s", runs_dir)
        return runs
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        rd = load_run_data(d)
        if rd is not None:
            runs[rd.method].append(rd)
    return dict(runs)


# ---------------------------------------------------------------------------
# Curve 1: G(t) — Global Exploration Rate (Phase 26)
# ---------------------------------------------------------------------------


def compute_global_exploration_curve(
    routing: list[dict[str, Any]],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """P(global_retrieval_triggered) over normalized progress bins."""
    max_pos = max(
        (r.get("task_position", 0) for r in routing), default=0,
    )
    if max_pos == 0:
        return []
    bins: dict[int, list[bool]] = defaultdict(list)
    for r in routing:
        pos = r.get("task_position", 0)
        p = pos / max_pos if max_pos > 0 else 0
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append(bool(r.get("global_retrieval_triggered", False)))
    curve = []
    for b in range(n_bins):
        vals = bins.get(b, [])
        rate = sum(vals) / len(vals) if vals else 0.0
        curve.append({
            "bin": b,
            "progress_start": round(b / n_bins, 2),
            "progress_end": round((b + 1) / n_bins, 2),
            "global_retrieval_rate": round(rate, 4),
            "n_observations": len(vals),
        })
    return curve


# ---------------------------------------------------------------------------
# Curve 2: R(t) — Known Memory Reuse Rate (Phase 26)
# ---------------------------------------------------------------------------


def compute_known_reuse_curve(
    routing: list[dict[str, Any]],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """P(selected_source=known | selected) over normalized progress."""
    max_pos = max(
        (r.get("task_position", 0) for r in routing), default=0,
    )
    if max_pos == 0:
        return []
    bins: dict[int, list[bool]] = defaultdict(list)
    for r in routing:
        sel = r.get("selected_memory_id")
        if sel is None:
            continue
        pos = r.get("task_position", 0)
        p = pos / max_pos if max_pos > 0 else 0
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append(r.get("selected_source") == "known")
    curve = []
    for b in range(n_bins):
        vals = bins.get(b, [])
        rate = sum(vals) / len(vals) if vals else 0.0
        curve.append({
            "bin": b,
            "progress_start": round(b / n_bins, 2),
            "progress_end": round((b + 1) / n_bins, 2),
            "known_reuse_rate": round(rate, 4),
            "n_selections": len(vals),
        })
    return curve


# ---------------------------------------------------------------------------
# Curve 3: Transfer State Growth (Phase 26)
# ---------------------------------------------------------------------------


def compute_transfer_state_growth(
    routing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """|K_predicted| and |K_causal| over task positions."""
    # Group by task_position.
    by_pos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in routing:
        by_pos[r.get("task_position", 0)].append(r)
    curve = []
    for pos in sorted(by_pos):
        recs = by_pos[pos]
        # transfer_state_size_before is K_predicted (known candidates).
        sizes = [r.get("transfer_state_size_before", 0) for r in recs]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        # n_known_candidates_considered approximates |K_predicted|.
        known = [r.get("n_known_candidates_considered", 0) for r in recs]
        avg_known = sum(known) / len(known) if known else 0
        curve.append({
            "task_position": pos,
            "avg_transfer_state_size": round(avg_size, 2),
            "avg_known_candidates": round(avg_known, 2),
        })
    return curve


# ---------------------------------------------------------------------------
# Curve 4: Task Score with Early/Middle/Late (Phase 26-27)
# ---------------------------------------------------------------------------


def compute_score_curve(
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-task score curve with early/middle/late decomposition."""
    valid = [t for t in tasks if t.get("is_valid", True)]
    scores = [t.get("task_score", 0.0) for t in valid]
    if not scores:
        return {"scores": [], "n_tasks": 0}
    n = len(scores)
    early_end = max(1, n // 3)
    late_start = max(early_end, 2 * n // 3)
    return {
        "scores": scores,
        "n_tasks": n,
        "early_end": early_end,
        "late_start": late_start,
        **compute_early_late_scores(scores),
    }


# ---------------------------------------------------------------------------
# Phase 28: Paired Statistical Comparison
# ---------------------------------------------------------------------------


@dataclass
class PairedComparison:
    """Result of paired bootstrap comparison between two methods."""

    method_a: str
    method_b: str
    metric: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    n_pairs: int


def _bootstrap_ci(
    diffs: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean of differences."""
    import random

    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n = len(diffs)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(diffs) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(diffs) / n, lo, hi


def paired_comparison(
    runs_a: list[RunData],
    runs_b: list[RunData],
    method_a: str,
    method_b: str,
) -> list[PairedComparison]:
    """Bootstrap CI comparing two methods on matched keys."""
    # Build match index: (scenario, stream_seed, exec_seed) → RunData.
    idx_a: dict[tuple, RunData] = {}
    for r in runs_a:
        idx_a[(r.scenario, r.stream_seed, r.execution_seed)] = r
    idx_b: dict[tuple, RunData] = {}
    for r in runs_b:
        idx_b[(r.scenario, r.stream_seed, r.execution_seed)] = r

    common = set(idx_a.keys()) & set(idx_b.keys())
    if not common:
        return []

    score_diffs: list[float] = []
    late_diffs: list[float] = []
    global_diffs: list[float] = []

    for key in sorted(common):
        ra = idx_a[key]
        rb = idx_b[key]
        sa = compute_score_curve(ra.tasks)
        sb = compute_score_curve(rb.tasks)
        score_diffs.append(
            (sa.get("score_late", 0) or 0) - (sb.get("score_late", 0) or 0),
        )
        late_diffs.append(
            (sa.get("score_late", 0) or 0) - (sb.get("score_late", 0) or 0),
        )
        # Global retrieval rate difference.
        ga = [r for r in ra.routing if r.get("global_retrieval_triggered")]
        gb = [r for r in rb.routing if r.get("global_retrieval_triggered")]
        rate_a = len(ga) / len(ra.routing) if ra.routing else 0
        rate_b = len(gb) / len(rb.routing) if rb.routing else 0
        global_diffs.append(rate_a - rate_b)

    results: list[PairedComparison] = []
    for name, diffs in [
        ("late_score", score_diffs),
        ("delta_score", late_diffs),
        ("global_retrieval_rate", global_diffs),
    ]:
        mean, lo, hi = _bootstrap_ci(diffs)
        results.append(PairedComparison(
            method_a=method_a, method_b=method_b,
            metric=name, mean_diff=round(mean, 4),
            ci_lower=round(lo, 4), ci_upper=round(hi, 4),
            n_pairs=len(common),
        ))
    return results


# ---------------------------------------------------------------------------
# Phase 29: Transfer Prediction Generalization
# ---------------------------------------------------------------------------


def compute_transfer_prediction_metrics(
    probe_events: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """e_t = |μ_t - τ_obs_t| using pre-probe predictions.

    Reports MAE_early, MAE_late, ΔMAE, SignAcc early vs late.
    """
    if not probe_events:
        return {"n_probes": 0}

    n = len(probe_events)
    early_end = max(1, n // 3)
    late_start = max(early_end, 2 * n // 3)

    errors: list[float] = []
    sign_correct: list[bool] = []

    for pe in probe_events:
        mu = pe.get("mu_predicted")
        tau_obs = pe.get("observed_tau", 0.0)
        if mu is not None:
            errors.append(abs(mu - tau_obs))
            sign_correct.append((mu > 0) == (tau_obs > 0))
        else:
            # Fallback: no prediction available.
            errors.append(abs(tau_obs))
            sign_correct.append(tau_obs == 0)

    early_errors = errors[:early_end]
    late_errors = errors[late_start:] if late_start < n else []

    mae_early = (
        sum(early_errors) / len(early_errors) if early_errors else None
    )
    mae_late = (
        sum(late_errors) / len(late_errors) if late_errors else None
    )

    early_signs = sign_correct[:early_end]
    late_signs = sign_correct[late_start:] if late_start < n else []
    sign_acc_early = (
        sum(early_signs) / len(early_signs) if early_signs else None
    )
    sign_acc_late = (
        sum(late_signs) / len(late_signs) if late_signs else None
    )

    return {
        "n_probes": n,
        "mae_early": round(mae_early, 4) if mae_early is not None else None,
        "mae_late": round(mae_late, 4) if mae_late is not None else None,
        "delta_mae": (
            round(mae_late - mae_early, 4)
            if mae_early is not None and mae_late is not None else None
        ),
        "sign_acc_early": (
            round(sign_acc_early, 4) if sign_acc_early is not None else None
        ),
        "sign_acc_late": (
            round(sign_acc_late, 4) if sign_acc_late is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Phase 30: Uncertainty Online Audit
# ---------------------------------------------------------------------------


def compute_uncertainty_audit(
    probe_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """corr(σ_t, |μ_t - τ_t|) and LCB coverage: I[τ_obs ≥ μ - βσ]."""
    if not probe_events:
        return {"n_probes": 0}

    sigmas: list[float] = []
    abs_errors: list[float] = []
    lcb_covered: list[bool] = []

    for pe in probe_events:
        mu = pe.get("mu_predicted")
        sigma = pe.get("sigma_predicted")
        tau_obs = pe.get("observed_tau", 0.0)
        if mu is not None:
            abs_errors.append(abs(mu - tau_obs))
        if sigma is not None:
            sigmas.append(sigma)
            if mu is not None:
                beta = 1.64  # from transfer policy
                lcb_covered.append(tau_obs >= mu - beta * sigma)

    # Correlation.
    corr = None
    if len(sigmas) >= 3 and len(abs_errors) >= 3:
        n = min(len(sigmas), len(abs_errors))
        s = sigmas[:n]
        e = abs_errors[:n]
        mean_s = sum(s) / n
        mean_e = sum(e) / n
        cov = sum((si - mean_s) * (ei - mean_e) for si, ei in zip(s, e)) / n
        std_s = math.sqrt(sum((si - mean_s) ** 2 for si in s) / n)
        std_e = math.sqrt(sum((ei - mean_e) ** 2 for ei in e) / n)
        if std_s > 0 and std_e > 0:
            corr = cov / (std_s * std_e)

    coverage = (
        sum(lcb_covered) / len(lcb_covered) if lcb_covered else None
    )

    return {
        "n_probes": len(probe_events),
        "corr_sigma_abs_error": round(corr, 4) if corr is not None else None,
        "lcb_coverage": round(coverage, 4) if coverage is not None else None,
        "mean_sigma": (
            round(sum(sigmas) / len(sigmas), 4) if sigmas else None
        ),
    }


# ---------------------------------------------------------------------------
# Phase 31: γ Routing Audit
# ---------------------------------------------------------------------------


def compute_gamma_routing_audit(
    routing: list[dict[str, Any]],
) -> dict[str, Any]:
    """P(bestLCB ≤ δ), P(δ < bestLCB < γ), P(bestLCB ≥ γ)."""
    exploit_only = 0
    exploit_explore = 0
    explore_only = 0
    total = 0
    best_lcb_values: list[float] = []

    for r in routing:
        mode = r.get("routing_mode")
        if mode is None:
            continue
        total += 1
        lcb = r.get("best_known_lcb")
        if lcb is not None:
            best_lcb_values.append(lcb)
        if mode == "exploit_only":
            exploit_only += 1
        elif mode == "exploit_explore":
            exploit_explore += 1
        elif mode == "explore_only":
            explore_only += 1

    if total == 0:
        return {"n_routing": 0}

    p_exploit = exploit_only / total
    p_explore = explore_only / total

    result: dict[str, Any] = {
        "n_routing": total,
        "p_exploit_only": round(p_exploit, 4),
        "p_exploit_explore": round(exploit_explore / total, 4),
        "p_explore_only": round(p_explore, 4),
    }

    # Warnings.
    warnings: list[str] = []
    if p_exploit < 0.02:
        warnings.append(
            f"exploit_only_rate={p_exploit:.2%} < 2% — "
            f"critic may not be learning"
        )
    if p_exploit > 0.90:
        warnings.append(
            f"exploit_only_rate={p_exploit:.2%} > 90% — "
            f"insufficient exploration"
        )
    result["warnings"] = warnings

    return result


# ---------------------------------------------------------------------------
# Phase 32: Cost Decomposition
# ---------------------------------------------------------------------------


def compute_cost_decomposition(
    tasks: list[dict[str, Any]],
    costs: dict[str, Any],
) -> dict[str, Any]:
    """Three-way cost: retrieval, model, environment-learning."""
    n_tasks = len(tasks)
    if n_tasks == 0:
        return {}

    report = dict(costs) if costs else {}
    # Per-task amortized rates.
    total_retrieval = report.get("retrieval_cost", 0)
    total_model = report.get("model_cost", 0)
    total_env = report.get("environment_learning_cost", 0)

    return {
        "n_tasks": n_tasks,
        "retrieval_per_task": round(total_retrieval / n_tasks, 4),
        "model_per_task": round(total_model / n_tasks, 4),
        "env_learning_per_task": round(total_env / n_tasks, 4),
        "total_per_task": round(
            (total_retrieval + total_model + total_env) / n_tasks, 4,
        ),
        **report,
    }


# ---------------------------------------------------------------------------
# Phase 33: Search Transfer Sanity
# ---------------------------------------------------------------------------


def compute_search_transfer_sanity(
    routing: list[dict[str, Any]],
) -> dict[str, Any]:
    """C_critic = N_known-predict + N_global-predict, cross-method."""
    n_known = sum(
        r.get("n_known_candidates_considered", 0) for r in routing
    )
    n_global = sum(
        r.get("n_global_candidates_considered", 0) for r in routing
    )
    n_selected = sum(
        1 for r in routing if r.get("selected_memory_id") is not None
    )
    return {
        "n_known_predictions": n_known,
        "n_global_predictions": n_global,
        "n_critic_total": n_known + n_global,
        "n_selections": n_selected,
        "selection_rate": (
            round(n_selected / len(routing), 4) if routing else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Per-method analysis
# ---------------------------------------------------------------------------


def analyze_method(
    method: str,
    runs: list[RunData],
    output_dir: Path,
) -> dict[str, Any]:
    """Run full analysis for one method, writing outputs."""
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate across runs.
    all_routing: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    all_probes: list[dict[str, Any]] = []
    score_curves: list[dict[str, Any]] = []

    for run in runs:
        all_routing.extend(run.routing)
        all_tasks.extend(run.tasks)
        all_probes.extend(run.probe_events)
        score_curves.append(compute_score_curve(run.tasks))

    # --- Curve 1: G(t) ---
    g_curve = compute_global_exploration_curve(all_routing)

    # --- Curve 2: R(t) ---
    r_curve = compute_known_reuse_curve(all_routing)

    # --- Curve 3: Transfer State Growth ---
    ts_growth = compute_transfer_state_growth(all_routing)

    # --- Curve 4: Task Score ---
    # Aggregate scores across runs.
    all_scores: list[float] = []
    for sc in score_curves:
        all_scores.extend(sc.get("scores", []))
    agg_score = compute_early_late_scores(all_scores)

    # --- Phase 29-33 ---
    transfer_pred = compute_transfer_prediction_metrics(all_probes, all_tasks)
    uncertainty = compute_uncertainty_audit(all_probes)
    gamma_audit = compute_gamma_routing_audit(all_routing)
    cost_decomp = compute_cost_decomposition(
        all_tasks,
        runs[0].costs if runs else {},
    )
    search_sanity = compute_search_transfer_sanity(all_routing)

    result: dict[str, Any] = {
        "method": method,
        "n_runs": len(runs),
        "n_tasks_total": len(all_tasks),
        "n_routing_total": len(all_routing),
        "n_probes_total": len(all_probes),
        "score": agg_score,
        "global_exploration_curve": g_curve,
        "known_reuse_curve": r_curve,
        "transfer_state_growth": ts_growth,
        "transfer_prediction": transfer_pred,
        "uncertainty_audit": uncertainty,
        "gamma_routing": gamma_audit,
        "cost_decomposition": cost_decomp,
        "search_transfer_sanity": search_sanity,
    }

    # --- Write CSV outputs ---
    _write_curve_csv(method_dir / "global_exploration.csv", g_curve)
    _write_curve_csv(method_dir / "known_reuse.csv", r_curve)
    _write_curve_csv(method_dir / "transfer_state_growth.csv", ts_growth)

    # --- Write JSON ---
    with open(method_dir / "analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def _write_curve_csv(
    path: Path, curve: list[dict[str, Any]],
) -> None:
    if not curve:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)


# ---------------------------------------------------------------------------
# Full analysis orchestrator
# ---------------------------------------------------------------------------


def run_full_analysis(
    runs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the full analysis pipeline on all pilot runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_by_method = load_all_runs(runs_dir)

    if not runs_by_method:
        logger.warning("No runs found in %s", runs_dir)
        return {}

    logger.info(
        "Found %d methods: %s",
        len(runs_by_method), sorted(runs_by_method.keys()),
    )

    # Per-method analysis.
    method_results: dict[str, Any] = {}
    for method, runs in sorted(runs_by_method.items()):
        logger.info("Analyzing %s (%d runs)", method, len(runs))
        method_results[method] = analyze_method(method, runs, output_dir)

    # Paired comparisons (Phase 28).
    comparisons: list[dict[str, Any]] = []
    methods = sorted(runs_by_method.keys())
    # Compare adaptive against each other method.
    ref_method = "rima_transfer_adaptive"
    if ref_method in runs_by_method:
        for m in methods:
            if m == ref_method:
                continue
            comps = paired_comparison(
                runs_by_method[ref_method],
                runs_by_method[m],
                ref_method, m,
            )
            for c in comps:
                comparisons.append({
                    "method_a": c.method_a,
                    "method_b": c.method_b,
                    "metric": c.metric,
                    "mean_diff": c.mean_diff,
                    "ci_lower": c.ci_lower,
                    "ci_upper": c.ci_upper,
                    "n_pairs": c.n_pairs,
                })

    # Write comparisons.
    if comparisons:
        with open(output_dir / "paired_comparisons.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
            writer.writeheader()
            writer.writerows(comparisons)

    # Write full results.
    full_result = {
        "methods": method_results,
        "paired_comparisons": comparisons,
    }
    with open(output_dir / "full_analysis.json", "w") as f:
        json.dump(full_result, f, indent=2)

    logger.info("Analysis complete. Output in %s", output_dir)
    return full_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mechanism pilot analysis pipeline (Phase 26-33).",
    )
    parser.add_argument(
        "--runs-dir",
        default="results/rima_transfer/pilot/runs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rima_transfer/pilot/analysis",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    project_root = _PROJECT_ROOT
    runs_dir = project_root / args.runs_dir
    output_dir = project_root / args.output_dir

    result = run_full_analysis(runs_dir, output_dir)
    if not result:
        print("No data to analyze.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
