#!/usr/bin/env python3
"""Phase 23.4 core diagnostic for the 45A_v3 adaptive continual-transfer run.

Reads the five run jsonl artefacts and produces the Phase 24 "14 core
numbers", the Phase 24.1-24.8 decompositions, the Phase 25 four-layer
diagnostic gate, and the Phase 26 decision file.

Inputs (all inside ``--run-dir``)::

    tasks.jsonl
    routing.jsonl
    probe_events.jsonl
    critic_versions.jsonl
    refit_prediction_deltas.jsonl

Outputs (all inside ``--output-dir``)::

    45a_v3_diagnosis.json
    45a_v3_diagnosis.csv
    critic_version_diagnosis.csv
    critic_version_usage.csv
    probe_prediction_records.csv
    lcb_decomposition.csv
    45a_v3_diagnostic_decision.json

This script performs NO parameter tuning and NEVER touches the run
directory. It only reads and classifies (Phase 26: "不要自动调参数").

Transfer MAE is strictly prequential (Phase 24.1)::

    e_t = |predicted_mu_pre_probe - observed_tau|

and NEVER uses a post-refit prediction for the error of task ``t``.

Usage::

    python analysis/rima_transfer/diagnose_45a_v3.py \
        --run-dir "$RUN_DIR" \
        --output-dir "$RUN_ROOT/diagnosis"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

try:  # optional ranking metrics
    from sklearn.metrics import average_precision_score, roc_auc_score

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - sklearn always present in this env
    _HAVE_SKLEARN = False

NA_SINGLE_CLASS = "NA_SINGLE_CLASS"


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def _quantile(values: list[float], q: float) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.quantile(vals, q)) if vals else None


def _split_windows(n_tasks: int) -> tuple[range, range, range]:
    """Fixed thirds: 30 tasks -> 0-9 / 10-19 / 20-29 (Phase 24.2)."""
    third = max(1, n_tasks // 3)
    return range(0, third), range(third, n_tasks - third), range(n_tasks - third, n_tasks)


def _safe_corr(x: list[float], y: list[float]) -> dict[str, Any]:
    """Pearson + Spearman with degenerate-input guards."""
    if len(x) < 2 or len(y) < 2:
        return {"pearson": None, "spearman": None, "n": len(x)}
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    pear = float(pearsonr(xa, ya).statistic) if np.std(xa) > 0 and np.std(ya) > 0 else None
    spear = float(spearmanr(xa, ya).statistic) if np.std(xa) > 0 and np.std(ya) > 0 else None
    return {"pearson": pear, "spearman": spear, "n": len(x)}


# --------------------------------------------------------------------------
# Pair extraction (Phase 24.1 - strictly prequential)
# --------------------------------------------------------------------------

def _valid_pairs(probe_events: list[dict]) -> list[dict[str, Any]]:
    """Probe events carrying BOTH a pre-probe mu and an observed tau."""
    pairs: list[dict[str, Any]] = []
    for ev in probe_events:
        mu = ev.get("predicted_mu_pre_probe")
        tau = ev.get("observed_tau")
        if mu is None or tau is None:
            continue
        sigma = ev.get("predicted_sigma_pre_probe")
        lcb = ev.get("predicted_lcb_pre_probe")
        pairs.append(
            {
                "task_id": ev.get("task_id"),
                "task_position": ev.get("task_position"),
                "receiver_id": ev.get("receiver_id"),
                "memory_id": ev.get("memory_id"),
                "mu": float(mu),
                "tau": float(tau),
                "sigma": None if sigma is None else float(sigma),
                "lcb": None if lcb is None else float(lcb),
                "version": int(ev.get("critic_version_pre_probe", 1) or 1),
                "abs_error": abs(float(mu) - float(tau)),
            }
        )
    return pairs


# --------------------------------------------------------------------------
# Phase 24 - 14 core numbers
# --------------------------------------------------------------------------

def _core_numbers(
    pairs: list[dict],
    probe_events: list[dict],
    critic_versions: list[dict],
    task_records: list[dict],
    n_tasks: int,
) -> dict[str, Any]:
    early, middle, late = _split_windows(n_tasks)

    probe_count = len(probe_events)
    refit_count = len(critic_versions)
    version_numbers = [int(c["critic_version"]) for c in critic_versions if "critic_version" in c]
    selection_versions = [
        int(t["selection_critic_version"]) for t in task_records
        if t.get("selection_critic_version") is not None
    ]
    final_critic_version = max(version_numbers + selection_versions + [1])
    injection_count = int(sum(int(t.get("n_injected_total", 0) or 0) for t in task_records))

    taus = [p["tau"] for p in pairs]
    mus = [p["mu"] for p in pairs]
    lcbs = [p["lcb"] for p in pairs if p["lcb"] is not None]
    sigmas = [p["sigma"] for p in pairs if p["sigma"] is not None]

    n_observed_tau = len(taus)
    n_tau_gt_0 = sum(1 for t in taus if t > 0.0)
    n_tau_gt_005 = sum(1 for t in taus if t > 0.05)
    n_tau_gt_010 = sum(1 for t in taus if t > 0.10)

    mu_gt_0_rate = (sum(1 for m in mus if m > 0.0) / len(mus)) if mus else None
    lcb_gt_0_rate = (sum(1 for v in lcbs if v > 0.0) / len(lcbs)) if lcbs else None
    max_lcb = max(lcbs) if lcbs else None
    max_mu = max(mus) if mus else None

    def _mae(window: range) -> dict[str, Any]:
        win = [p for p in pairs if p["task_position"] is not None and p["task_position"] in window]
        mae = _mean([p["abs_error"] for p in win])
        return {"n": len(win), "mae": mae, "low_support": len(win) < 3}

    mae_early = _mae(early)
    mae_middle = _mae(middle)
    mae_late = _mae(late)

    corr_mu_tau = _safe_corr(mus, taus)
    abs_errors = [p["abs_error"] for p in pairs]
    corr_sigma_err = _safe_corr(sigmas, [p["abs_error"] for p in pairs if p["sigma"] is not None])

    return {
        "probe_count": probe_count,
        "refit_count": refit_count,
        "final_critic_version": final_critic_version,
        "injection_count": injection_count,
        "n_observed_tau": n_observed_tau,
        "n_tau_gt_0": n_tau_gt_0,
        "n_tau_gt_0.05": n_tau_gt_005,
        "n_tau_gt_0.10": n_tau_gt_010,
        "mu_gt_0_rate": mu_gt_0_rate,
        "lcb_gt_0_rate": lcb_gt_0_rate,
        "max_lcb": max_lcb,
        "max_mu": max_mu,
        "mae_early": mae_early,
        "mae_middle": mae_middle,
        "mae_late": mae_late,
        "corr_mu_tau": corr_mu_tau,
        "corr_sigma_abs_error": corr_sigma_err,
        "_abs_errors": abs_errors,
        "_sigmas": sigmas,
    }


# --------------------------------------------------------------------------
# Phase 24.3 / 24.6 - per critic version tables
# --------------------------------------------------------------------------

def _by_version_metrics(pairs: list[dict]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict]] = {}
    for p in pairs:
        grouped.setdefault(p["version"], []).append(p)

    out: dict[int, dict[str, Any]] = {}
    for v in sorted(grouped):
        gp = grouped[v]
        mu = np.array([p["mu"] for p in gp])
        tau = np.array([p["tau"] for p in gp])
        err = mu - tau
        n = len(gp)
        pred_pos = mu > 0.0
        obs_pos = tau > 0.0
        sign_acc = float(np.mean(pred_pos == obs_pos)) if n else None
        sigmas = [p["sigma"] for p in gp if p["sigma"] is not None]
        lcbs = [p["lcb"] for p in gp if p["lcb"] is not None]
        out[v] = {
            "n": n,
            "mae": float(np.mean(np.abs(err))) if n else None,
            "rmse": float(np.sqrt(np.mean(err**2))) if n else None,
            "sign_accuracy": sign_acc,
            "mu_gt_0_count": int(np.sum(pred_pos)),
            "tau_gt_0_count": int(np.sum(obs_pos)),
            # Phase 24.6 decomposition
            "mu_mean": _mean([p["mu"] for p in gp]),
            "mu_median": _median([p["mu"] for p in gp]),
            "sigma_mean": _mean(sigmas),
            "sigma_median": _median(sigmas),
            "lcb_mean": _mean(lcbs),
            "lcb_max": max(lcbs) if lcbs else None,
            "fraction_mu_gt_0": (int(np.sum(pred_pos)) / n) if n else None,
            "fraction_lcb_gt_0": (sum(1 for x in lcbs if x > 0.0) / len(lcbs)) if lcbs else None,
        }
    return out


# --------------------------------------------------------------------------
# Phase 24.4 - tau distribution
# --------------------------------------------------------------------------

def _tau_distribution(pairs: list[dict]) -> dict[str, Any]:
    taus = [p["tau"] for p in pairs]
    if not taus:
        return {"n": 0}
    arr = np.array(taus)
    return {
        "n": len(taus),
        "min": float(arr.min()),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
        "p_tau_gt_0": float(np.mean(arr > 0.0)),
        "p_tau_gt_0.05": float(np.mean(arr > 0.05)),
        "p_tau_gt_0.10": float(np.mean(arr > 0.10)),
    }


# --------------------------------------------------------------------------
# Phase 24.5 - mean vs uncertainty (Type A/B/C)
# --------------------------------------------------------------------------

def _lcb_type_decomposition(pairs: list[dict]) -> dict[str, Any]:
    """Type A: mu<=0 | Type B: mu>0 & LCB<=0 | Type C: LCB>0."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    type_a = sum(1 for p in pairs if p["mu"] <= 0.0)
    type_b = sum(
        1 for p in pairs
        if p["mu"] > 0.0 and (p["lcb"] is None or p["lcb"] <= 0.0)
    )
    type_c = sum(1 for p in pairs if p["lcb"] is not None and p["lcb"] > 0.0)
    # beta*sigma == mu - lcb (LCB is emitted as mu - beta*sigma)
    beta_sigma = [p["mu"] - p["lcb"] for p in pairs if p["lcb"] is not None]
    return {
        "n": n,
        "fraction_mu_le_0": type_a / n,
        "fraction_mu_gt_0_lcb_le_0": type_b / n,
        "fraction_lcb_gt_0": type_c / n,
        "count_type_a_mu_le_0": type_a,
        "count_type_b_mu_gt_0_lcb_le_0": type_b,
        "count_type_c_lcb_gt_0": type_c,
        "mean_mu": _mean([p["mu"] for p in pairs]),
        "mean_sigma": _mean([p["sigma"] for p in pairs if p["sigma"] is not None]),
        "mean_beta_sigma": _mean(beta_sigma),
    }


# --------------------------------------------------------------------------
# Phase 24.7 / 24.8 - ranking & uncertainty quality
# --------------------------------------------------------------------------

def _ranking_quality(pairs: list[dict]) -> dict[str, Any]:
    if len(pairs) < 2:
        return {"n": len(pairs), "spearman": None, "pearson": None,
                "sign_accuracy": None, "auroc": None, "ap": None}
    mu = np.array([p["mu"] for p in pairs])
    tau = np.array([p["tau"] for p in pairs])
    corr = _safe_corr([p["mu"] for p in pairs], [p["tau"] for p in pairs])
    pred_pos = mu > 0.0
    obs_pos = tau > 0.0
    sign_acc = float(np.mean(pred_pos == obs_pos))

    labels = (tau > 0.0).astype(int)
    both_classes = labels.sum() > 0 and labels.sum() < len(labels)
    if both_classes and _HAVE_SKLEARN:
        auroc: Any = float(roc_auc_score(labels, mu))
        ap: Any = float(average_precision_score(labels, mu))
    else:
        auroc = NA_SINGLE_CLASS if not both_classes else "NA_NO_SKLEARN"
        ap = NA_SINGLE_CLASS if not both_classes else "NA_NO_SKLEARN"

    return {
        "n": len(pairs),
        "spearman": corr["spearman"],
        "pearson": corr["pearson"],
        "sign_accuracy": sign_acc,
        "auroc_tau_gt_0": auroc,
        "ap_tau_gt_0": ap,
        "n_positive_labels": int(labels.sum()),
        "n_negative_labels": int(len(labels) - labels.sum()),
    }


def _uncertainty_quality(pairs: list[dict]) -> dict[str, Any]:
    sig_pairs = [p for p in pairs if p["sigma"] is not None]
    corr = _safe_corr([p["sigma"] for p in sig_pairs], [p["abs_error"] for p in sig_pairs])
    # Coverage = P(tau_obs >= mu - beta*sigma) = P(tau_obs >= lcb)
    cov_pairs = [p for p in pairs if p["lcb"] is not None]
    coverage = (
        float(np.mean([p["tau"] >= p["lcb"] for p in cov_pairs])) if cov_pairs else None
    )
    primary = corr["pearson"] if corr["pearson"] is not None else corr["spearman"]
    weak = primary is not None and primary <= 0.0
    return {
        "sigma_error_correlation": corr["pearson"],
        "sigma_error_spearman": corr["spearman"],
        "lcb_empirical_coverage": coverage,
        "n_sigma": len(sig_pairs),
        "n_coverage": len(cov_pairs),
        "flag": "UNCERTAINTY_SIGNAL_WEAK" if weak else "OK",
    }


# --------------------------------------------------------------------------
# Phase 23.2 (adaptive-specific) / 23.3 - version identity & usage
# --------------------------------------------------------------------------

def _version_audit(
    task_records: list[dict],
    critic_versions: list[dict],
    pairs: list[dict],
) -> dict[str, Any]:
    # controller == learner critic version, per task (Phase 23.2 extra check)
    mismatches = []
    for t in task_records:
        sel = t.get("selection_critic_version")
        ctrl = t.get("controller_critic_version")
        if sel is not None and ctrl is not None and int(sel) != int(ctrl):
            mismatches.append(
                {"task_position": t.get("task_position"),
                 "selection_critic_version": sel,
                 "controller_critic_version": ctrl}
            )

    # usage: number of tasks selecting with each version
    usage: dict[int, int] = {}
    for t in task_records:
        v = t.get("selection_critic_version")
        if v is not None:
            usage[int(v)] = usage.get(int(v), 0) + 1

    # probe-prediction usage per version (how many pre-probe preds each made)
    pred_usage: dict[int, int] = {}
    for p in pairs:
        pred_usage[p["version"]] = pred_usage.get(p["version"], 0) + 1

    selection_versions = [int(t["selection_critic_version"]) for t in task_records
                          if t.get("selection_critic_version") is not None]
    max_selection = max(selection_versions) if selection_versions else 1

    return {
        "controller_learner_version_mismatches": mismatches,
        "identity_invariant_pass": len(mismatches) == 0,
        "tasks_using_version": {f"v{k}": usage[k] for k in sorted(usage)},
        "predictions_by_version": {f"v{k}": pred_usage[k] for k in sorted(pred_usage)},
        "max_selection_critic_version": max_selection,
        "refit_count": len(critic_versions),
    }


def _learning_loop(
    pairs: list[dict],
    probe_events: list[dict],
    critic_versions: list[dict],
    task_records: list[dict],
    version_audit: dict[str, Any],
) -> dict[str, Any]:
    """Layer 1: probe>0, refit>0, and critic version changes future routing."""
    probe_count = len(probe_events)
    refit_count = len(critic_versions)

    # does a refit actually change routing of a LATER task?
    affects_routing = False
    evidence = []
    for cv in critic_versions:
        pos = cv.get("task_position")
        ver = int(cv.get("critic_version", 1))
        if pos is None:
            continue
        later_same = [
            t for t in task_records
            if t.get("task_position") is not None
            and t["task_position"] > pos
            and int(t.get("selection_critic_version", 0)) == ver
        ]
        if later_same:
            affects_routing = True
            evidence.append(
                {"refit_at_task_position": pos, "new_version": ver,
                 "n_later_tasks_using_new_version": len(later_same)}
            )

    loop_pass = probe_count > 0 and refit_count > 0 and affects_routing
    return {
        "probe_count": probe_count,
        "refit_count": refit_count,
        "version_changes_future_routing": affects_routing,
        "max_selection_critic_version_gt_1": version_audit["max_selection_critic_version"] > 1,
        "evidence": evidence,
        "verdict": "PASS" if loop_pass else "FAIL (IMPLEMENTATION NO-GO)",
        "pass": loop_pass,
    }


def _behavior_gate(routing: list[dict], n_tasks: int) -> dict[str, Any]:
    """Direction-only behaviour gate: G_L<G_E (exploration) & R_L>R_E (reuse)."""
    if not routing:
        return {"available": False, "verdict": "INSUFFICIENT_DATA"}
    early, _mid, late = _split_windows(n_tasks)

    by_pos: dict[int, list[dict]] = {}
    for d in routing:
        by_pos.setdefault(d.get("task_position", -1), []).append(d)

    def _rate(positions: range, key: str, value: Any = True) -> float | None:
        hit = tot = 0
        for pos in positions:
            grp = by_pos.get(pos, [])
            if grp:
                tot += 1
                if any(d.get(key) == value for d in grp):
                    hit += 1
        return hit / tot if tot else None

    def _global(positions: range) -> float | None:
        hit = tot = 0
        for pos in positions:
            grp = by_pos.get(pos, [])
            if grp:
                tot += 1
                if any(d.get("global_retrieval_triggered") for d in grp):
                    hit += 1
        return hit / tot if tot else None

    g_e, g_l = _global(early), _global(late)
    r_e, r_l = _rate(early, "selected_source", "known"), _rate(late, "selected_source", "known")

    g_dir = (g_l < g_e) if (g_e is not None and g_l is not None) else None
    r_dir = (r_l > r_e) if (r_e is not None and r_l is not None) else None

    if g_dir is None or r_dir is None:
        verdict = "INSUFFICIENT_DATA"
    elif g_dir and r_dir:
        verdict = "PASS"
    elif g_dir or r_dir:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "available": True,
        "global_exploration_early": g_e,
        "global_exploration_late": g_l,
        "known_reuse_early": r_e,
        "known_reuse_late": r_l,
        "G_late_lt_G_early": g_dir,
        "R_late_gt_R_early": r_dir,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# Phase 25 - four-layer diagnostic gate
# --------------------------------------------------------------------------

def _diagnostic_gate(
    core: dict[str, Any],
    ranking: dict[str, Any],
    uncertainty: dict[str, Any],
    learning: dict[str, Any],
    behavior: dict[str, Any],
    by_version: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    # Layer 2 - Transfer Signal
    n_obs = core["n_observed_tau"]
    pos_rate = (core["n_tau_gt_0"] / n_obs) if n_obs else 0.0
    layer2_scarce = (
        n_obs >= 1
        and (pos_rate < 0.25 or core["n_tau_gt_0.05"] < 3 or core["n_tau_gt_0.10"] <= 1)
    )
    layer2 = {
        "positive_rate": pos_rate,
        "n_tau_gt_0.05": core["n_tau_gt_0.05"],
        "n_tau_gt_0.10": core["n_tau_gt_0.10"],
        "verdict": "SIGNAL_SCARCITY" if layer2_scarce else "OK",
    }

    # Layer 3 - Mean Learnability (Phase 25: tau>0 not scarce AND corr(mu,tau)~0
    # AND SignAcc~0.5).  "SignAcc~0.5" denotes an UNINFORMATIVE mean predictor at
    # (or below) chance.  A worse-than-chance sign accuracy is at least as much a
    # mean failure as exactly 0.5, so the faithful encoding is an UPPER bound
    # ("not meaningfully better than chance"), NOT a two-sided band -- a two-sided
    # band would wrongly label a worse-than-random critic (e.g. 0.31) as learnable.
    spear = ranking.get("spearman")
    sign_acc = ranking.get("sign_accuracy")
    corr_mu_tau = core["corr_mu_tau"].get("spearman")
    primary_corr = spear if spear is not None else corr_mu_tau
    mean_failure = (
        core["n_tau_gt_0"] >= 3
        and primary_corr is not None and abs(primary_corr) < 0.2
        and sign_acc is not None and sign_acc <= 0.6
    )
    layer3 = {
        "corr_mu_tau_spearman": primary_corr,
        "sign_accuracy": sign_acc,
        "n_tau_gt_0": core["n_tau_gt_0"],
        "verdict": "CRITIC_MEAN_FAILURE" if mean_failure else "OK",
    }

    # Layer 4 - Uncertainty Bottleneck
    mu_rate = core["mu_gt_0_rate"] or 0.0
    lcb_rate = core["lcb_gt_0_rate"] or 0.0
    uncertainty_dom = (
        mu_rate >= 0.30
        and primary_corr is not None and primary_corr > 0.2
        and lcb_rate < 0.05
    )
    layer4 = {
        "mu_gt_0_rate": core["mu_gt_0_rate"],
        "lcb_gt_0_rate": core["lcb_gt_0_rate"],
        "corr_mu_tau_spearman": primary_corr,
        "verdict": "UNCERTAINTY_DOMINATED" if uncertainty_dom else "OK",
    }

    # sigma shrink across versions (Phase 27.D.1)
    sig_by_ver = [(v, by_version[v]["sigma_mean"]) for v in sorted(by_version)
                  if by_version[v]["sigma_mean"] is not None]
    sigma_shrinks = (
        len(sig_by_ver) >= 2 and sig_by_ver[-1][1] < sig_by_ver[0][1]
    )
    sigma_trend = {
        "sigma_mean_by_version": {f"v{v}": s for v, s in sig_by_ver},
        "sigma_shrinks_across_versions": sigma_shrinks,
    }

    return {
        "layer1_learning_loop": learning["verdict"],
        "layer2_transfer_signal": layer2,
        "layer3_mean_learnability": layer3,
        "layer4_uncertainty_bottleneck": layer4,
        "sigma_trend": sigma_trend,
        "_flags": {
            "layer2_scarce": layer2_scarce,
            "mean_failure": mean_failure,
            "uncertainty_dom": uncertainty_dom,
            "sigma_shrinks": sigma_shrinks,
        },
    }


# --------------------------------------------------------------------------
# Phase 26 / 27 - decision
# --------------------------------------------------------------------------

def _decide_next_action(
    core: dict[str, Any],
    gate: dict[str, Any],
    behavior: dict[str, Any],
    learning: dict[str, Any],
) -> tuple[str, str]:
    flags = gate["_flags"]

    if not learning["pass"]:
        return (
            "SIGNAL_REPLICATION",
            "Learning loop FAILED (IMPLEMENTATION NO-GO); diagnose wiring before "
            "any signal/critic branch. next_action placeholder pending loop fix.",
        )

    # Branch A - PROCEED_45B
    if (
        core["injection_count"] > 0
        and behavior.get("verdict") == "PASS"
        and learning["pass"]
    ):
        return ("PROCEED_45B", "Injection occurred and behaviour gate G_L<G_E & R_L>R_E.")

    # Branch B - SIGNAL_REPLICATION
    if flags["layer2_scarce"]:
        return ("SIGNAL_REPLICATION", "Transfer signal scarce; replicate probe labels first.")

    # Branch C - DIRECT_TAU_DIAGNOSIS
    if flags["mean_failure"]:
        return (
            "DIRECT_TAU_DIAGNOSIS",
            "Positive signal exists but mu cannot rank it; offline DirectTauCritic.",
        )

    # Branch D - uncertainty dominated
    if flags["uncertainty_dom"]:
        if flags["sigma_shrinks"]:
            return (
                "UNCERTAINTY_SUPPORT_EXPANSION",
                "mu ranks but LCB<=0; sigma shrinks yet insufficient "
                "-> expand uncertainty support.",
            )
        return (
            "STAGE_A_EXPANSION",
            "mu ranks but LCB<=0 and sigma does NOT shrink across refits "
            "-> expand Stage-A causal support.",
        )

    return (
        "STAGE_A_EXPANSION",
        "No single failure mode dominates; broad Stage-A causal support "
        "expansion is the safe default.",
    )


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def diagnose(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_records = _load_jsonl(run_dir / "tasks.jsonl")
    routing = _load_jsonl(run_dir / "routing.jsonl")
    probe_events = _load_jsonl(run_dir / "probe_events.jsonl")
    critic_versions = _load_jsonl(run_dir / "critic_versions.jsonl")
    refit_deltas = _load_jsonl(run_dir / "refit_prediction_deltas.jsonl")

    n_tasks = len(task_records)
    pairs = _valid_pairs(probe_events)

    core = _core_numbers(pairs, probe_events, critic_versions, task_records, n_tasks)
    core.pop("_abs_errors", None)
    core.pop("_sigmas", None)

    by_version = _by_version_metrics(pairs)
    tau_dist = _tau_distribution(pairs)
    lcb_types = _lcb_type_decomposition(pairs)
    ranking = _ranking_quality(pairs)
    uncertainty = _uncertainty_quality(pairs)
    version_audit = _version_audit(task_records, critic_versions, pairs)
    learning = _learning_loop(pairs, probe_events, critic_versions, task_records, version_audit)
    behavior = _behavior_gate(routing, n_tasks)
    gate = _diagnostic_gate(core, ranking, uncertainty, learning, behavior, by_version)
    next_action, rationale = _decide_next_action(core, gate, behavior, learning)

    # ---- Phase 26 decision file ----
    decision = {
        "learning_loop": "PASS" if learning["pass"] else "FAIL",
        "transfer_signal": gate["layer2_transfer_signal"]["verdict"],
        "mean_learnability": gate["layer3_mean_learnability"]["verdict"],
        "uncertainty": gate["layer4_uncertainty_bottleneck"]["verdict"],
        "behavior_gate": behavior.get("verdict", "INSUFFICIENT_DATA"),
        "next_action": next_action,
        "next_action_rationale": rationale,
        "identity_invariant_pass": version_audit["identity_invariant_pass"],
        "controller_learner_version_mismatches":
            version_audit["controller_learner_version_mismatches"],
        "note": "No parameters were tuned by this script (Phase 26).",
    }

    diagnosis = {
        "run_dir": str(run_dir),
        "n_tasks": n_tasks,
        "core_numbers": core,
        "tau_distribution": tau_dist,
        "lcb_type_decomposition": lcb_types,
        "by_critic_version": {f"v{v}": by_version[v] for v in sorted(by_version)},
        "ranking_quality": ranking,
        "uncertainty_quality": uncertainty,
        "version_audit": version_audit,
        "learning_loop": learning,
        "behavior_gate": behavior,
        "diagnostic_gate": {k: v for k, v in gate.items() if k != "_flags"},
        "refit_delta_records": len(refit_deltas),
        "max_abs_delta_mu": (
            max((abs(float(d["delta_mu"])) for d in refit_deltas
                 if d.get("delta_mu") is not None), default=None)
        ),
        "decision": decision,
    }

    # ---- write JSON ----
    (output_dir / "45a_v3_diagnosis.json").write_text(json.dumps(diagnosis, indent=2))
    (output_dir / "45a_v3_diagnostic_decision.json").write_text(json.dumps(decision, indent=2))

    # ---- 45a_v3_diagnosis.csv : 14 core numbers + 2 correlations ----
    core_rows = []
    for k in [
        "probe_count", "refit_count", "final_critic_version", "injection_count",
        "n_observed_tau", "n_tau_gt_0", "n_tau_gt_0.05", "n_tau_gt_0.10",
        "mu_gt_0_rate", "lcb_gt_0_rate", "max_lcb", "max_mu",
    ]:
        core_rows.append({"metric": k, "value": core.get(k)})
    core_rows.append({"metric": "mae_early", "value": core["mae_early"]["mae"],
                      "n": core["mae_early"]["n"]})
    core_rows.append({"metric": "mae_middle", "value": core["mae_middle"]["mae"],
                      "n": core["mae_middle"]["n"]})
    core_rows.append({"metric": "mae_late", "value": core["mae_late"]["mae"],
                      "n": core["mae_late"]["n"]})
    core_rows.append({"metric": "corr_mu_tau_spearman",
                      "value": core["corr_mu_tau"]["spearman"]})
    core_rows.append({"metric": "corr_mu_tau_pearson",
                      "value": core["corr_mu_tau"]["pearson"]})
    core_rows.append({"metric": "corr_sigma_abs_error_pearson",
                      "value": core["corr_sigma_abs_error"]["pearson"]})
    core_rows.append({"metric": "corr_sigma_abs_error_spearman",
                      "value": core["corr_sigma_abs_error"]["spearman"]})
    _write_csv(output_dir / "45a_v3_diagnosis.csv", core_rows,
               ["metric", "value", "n"])

    # ---- critic_version_diagnosis.csv (Phase 24.3 + 24.6) ----
    cv_rows = []
    for v in sorted(by_version):
        m = by_version[v]
        cv_rows.append({
            "critic_version": f"v{v}", "n": m["n"], "mae": m["mae"], "rmse": m["rmse"],
            "sign_accuracy": m["sign_accuracy"], "mu_gt_0": m["mu_gt_0_count"],
            "tau_gt_0": m["tau_gt_0_count"], "mu_mean": m["mu_mean"],
            "mu_median": m["mu_median"], "sigma_mean": m["sigma_mean"],
            "sigma_median": m["sigma_median"], "lcb_mean": m["lcb_mean"],
            "lcb_max": m["lcb_max"], "fraction_mu_gt_0": m["fraction_mu_gt_0"],
            "fraction_lcb_gt_0": m["fraction_lcb_gt_0"],
        })
    _write_csv(output_dir / "critic_version_diagnosis.csv", cv_rows,
               ["critic_version", "n", "mae", "rmse", "sign_accuracy", "mu_gt_0",
                "tau_gt_0", "mu_mean", "mu_median", "sigma_mean", "sigma_median",
                "lcb_mean", "lcb_max", "fraction_mu_gt_0", "fraction_lcb_gt_0"])

    # ---- critic_version_usage.csv (Phase 23.3) ----
    usage_rows = []
    all_versions = sorted(set(list(version_audit["tasks_using_version"].keys())))
    for v in all_versions:
        usage_rows.append({
            "critic_version": v,
            "n_tasks_using": version_audit["tasks_using_version"].get(v, ""),
            "n_probe_predictions": version_audit["predictions_by_version"].get(v, ""),
        })
    _write_csv(output_dir / "critic_version_usage.csv", usage_rows,
               ["critic_version", "n_tasks_using", "n_probe_predictions"])

    # ---- probe_prediction_records.csv ----
    probe_rows = []
    for p in pairs:
        probe_rows.append({
            "task_id": p["task_id"], "task_position": p["task_position"],
            "receiver_id": p["receiver_id"], "memory_id": p["memory_id"],
            "critic_version": p["version"], "mu_pre_probe": p["mu"],
            "sigma_pre_probe": p["sigma"], "lcb_pre_probe": p["lcb"],
            "observed_tau": p["tau"], "abs_error": p["abs_error"],
            "sign_match": int((p["mu"] > 0.0) == (p["tau"] > 0.0)),
            "tau_ge_lcb": int(p["lcb"] is not None and p["tau"] >= p["lcb"]),
        })
    _write_csv(output_dir / "probe_prediction_records.csv", probe_rows,
               ["task_id", "task_position", "receiver_id", "memory_id",
                "critic_version", "mu_pre_probe", "sigma_pre_probe", "lcb_pre_probe",
                "observed_tau", "abs_error", "sign_match", "tau_ge_lcb"])

    # ---- lcb_decomposition.csv (Phase 24.5) ----
    _write_csv(output_dir / "lcb_decomposition.csv", [lcb_types],
               ["n", "fraction_mu_le_0", "fraction_mu_gt_0_lcb_le_0",
                "fraction_lcb_gt_0", "count_type_a_mu_le_0",
                "count_type_b_mu_gt_0_lcb_le_0", "count_type_c_lcb_gt_0",
                "mean_mu", "mean_sigma", "mean_beta_sigma"])

    _print_report(diagnosis, decision)
    return diagnosis


def _print_report(d: dict[str, Any], decision: dict[str, Any]) -> None:
    core = d["core_numbers"]
    tau = d["tau_distribution"]
    rank = d["ranking_quality"]
    unc = d["uncertainty_quality"]
    beh = d["behavior_gate"]

    print("=" * 66)
    print("45A_v3 DIAGNOSIS")
    print("=" * 66)
    print("RUN")
    print(f"  tasks                 = {d['n_tasks']}")
    print(f"  probe_count           = {core['probe_count']}")
    print(f"  refit_count           = {core['refit_count']}")
    print(f"  final_critic_version  = {core['final_critic_version']}")
    print(f"  injection_count       = {core['injection_count']}")
    print("CAUSAL SIGNAL")
    print(f"  tau > 0    = {core['n_tau_gt_0']} / {core['n_observed_tau']}")
    print(f"  tau > 0.05 = {core['n_tau_gt_0.05']} / {core['n_observed_tau']}")
    print(f"  tau > 0.10 = {core['n_tau_gt_0.10']} / {core['n_observed_tau']}")
    print(f"  tau mean   = {_fmt(tau.get('mean'))}")
    print(f"  tau median = {_fmt(tau.get('median'))}")
    print(f"  tau max    = {_fmt(tau.get('max'))}")
    print("MODEL")
    print(f"  mu > 0 rate  = {_fmt(core['mu_gt_0_rate'])}")
    print(f"  LCB > 0 rate = {_fmt(core['lcb_gt_0_rate'])}")
    print(f"  max mu       = {_fmt(core['max_mu'])}")
    print(f"  max LCB      = {_fmt(core['max_lcb'])}")
    print(f"  mean sigma   = {_fmt(d['lcb_type_decomposition'].get('mean_sigma'))}")
    print("PREQUENTIAL")
    print(f"  MAE early  = {_fmt(core['mae_early']['mae'])} (n={core['mae_early']['n']}"
          f"{', LOW_SUPPORT' if core['mae_early']['low_support'] else ''})")
    print(f"  MAE middle = {_fmt(core['mae_middle']['mae'])} (n={core['mae_middle']['n']}"
          f"{', LOW_SUPPORT' if core['mae_middle']['low_support'] else ''})")
    print(f"  MAE late   = {_fmt(core['mae_late']['mae'])} (n={core['mae_late']['n']}"
          f"{', LOW_SUPPORT' if core['mae_late']['low_support'] else ''})")
    print("  MAE by version:")
    for v, m in d["by_critic_version"].items():
        print(f"    {v}: n={m['n']} mae={_fmt(m['mae'])} rmse={_fmt(m['rmse'])} "
              f"sign={_fmt(m['sign_accuracy'])} mu>0={m['mu_gt_0_count']} "
              f"tau>0={m['tau_gt_0_count']}")
    print(f"  corr(mu, tau)        = {_fmt(core['corr_mu_tau']['spearman'])} (spearman) "
          f"/ {_fmt(core['corr_mu_tau']['pearson'])} (pearson)")
    print(f"  corr(sigma, abs_err) = {_fmt(core['corr_sigma_abs_error']['pearson'])} (pearson) "
          f"/ {_fmt(core['corr_sigma_abs_error']['spearman'])} (spearman)")
    print(f"  ranking: spearman={_fmt(rank.get('spearman'))} "
          f"sign_acc={_fmt(rank.get('sign_accuracy'))} "
          f"auroc={rank.get('auroc_tau_gt_0')} ap={rank.get('ap_tau_gt_0')}")
    print(f"  uncertainty: sigma_err_corr={_fmt(unc.get('sigma_error_correlation'))} "
          f"coverage={_fmt(unc.get('lcb_empirical_coverage'))} flag={unc.get('flag')}")
    print("BEHAVIOR")
    print(f"  global exploration early/late = {_fmt(beh.get('global_exploration_early'))} / "
          f"{_fmt(beh.get('global_exploration_late'))} (G_L<G_E={beh.get('G_late_lt_G_early')})")
    print(f"  known reuse early/late        = {_fmt(beh.get('known_reuse_early'))} / "
          f"{_fmt(beh.get('known_reuse_late'))} (R_L>R_E={beh.get('R_late_gt_R_early')})")
    print(f"  behavior_gate                 = {beh.get('verdict')}")
    print("VERDICT")
    print(f"  learning loop = {decision['learning_loop']}")
    print(f"  signal        = {decision['transfer_signal']}")
    print(f"  mean learner  = {decision['mean_learnability']}")
    print(f"  uncertainty   = {decision['uncertainty']}")
    print(f"  identity inv  = {decision['identity_invariant_pass']}")
    print(f"  NEXT ACTION   = {decision['next_action']}")
    print(f"  rationale     = {decision['next_action_rationale']}")
    print("=" * 66)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 23.4 45A_v3 diagnostic")
    ap.add_argument("--run-dir", required=True, help="stream run directory containing the 5 jsonl")
    ap.add_argument("--output-dir", required=True, help="directory for diagnosis outputs")
    args = ap.parse_args()
    diagnose(Path(args.run_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
