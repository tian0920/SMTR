"""Opportunity-factorized validation metrics (Counterfactual Opportunity v1).

Head-level metrics:
  - Baseline: Brier, log loss, AUROC on Y_0
  - Rescue: PR-AUC, AUROC, Brier on Y_1 given Y_0=0
  - Damage: PR-AUC, AUROC, Brier on (1-Y_1) given Y_0=1

Edge-level causal metrics:
  - tau_mae, eta_mae, tau_spearman, eta_spearman
  - top1_empirical_tau_hit_rate, top1_tau_regret
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from smtr.counterfactual.edge_keys import (
    TreatmentEdgeKey,
    treatment_edge_key,
)
from smtr.marble.paired_outcomes import get_paired_outcomes


# ---------------------------------------------------------------------------
# Head-level metrics
# ---------------------------------------------------------------------------


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """AUROC if both classes present, else None."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Average precision (PR-AUC) if both classes present, else None."""
    from sklearn.metrics import average_precision_score

    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean((y - p)^2)."""
    return float(np.mean((y_true - y_prob) ** 2))


def _log_loss_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Binary log loss if both classes present, else None."""
    from sklearn.metrics import log_loss as sk_log_loss

    if len(np.unique(y_true)) < 2:
        return None
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(sk_log_loss(y_true, y_prob))


def compute_baseline_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, Any]:
    """Baseline head metrics on Y_0."""
    return {
        "brier": _brier(y_true, y_prob),
        "log_loss": _log_loss_binary(y_true, y_prob),
        "auroc": _safe_auroc(y_true, y_prob),
        "n_samples": len(y_true),
    }


def compute_rescue_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, Any]:
    """Rescue head metrics on Y_1 given Y_0=0.

    PR-AUC is the primary metric since positive rescue is likely sparse.
    """
    return {
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "auroc": _safe_auroc(y_true, y_prob),
        "brier": _brier(y_true, y_prob),
        "n_samples": len(y_true),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }


def compute_damage_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, Any]:
    """Damage head metrics on (1-Y_1) given Y_0=1.

    PR-AUC is the primary metric for negative-transfer detection.
    """
    return {
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "auroc": _safe_auroc(y_true, y_prob),
        "brier": _brier(y_true, y_prob),
        "n_samples": len(y_true),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }


# ---------------------------------------------------------------------------
# Edge-level causal metrics
# ---------------------------------------------------------------------------


def compute_edge_level_metrics(
    records: list[dict[str, Any]],
    predicted_probs: list[dict[str, float]],
) -> dict[str, Any]:
    """Edge-level tau/eta metrics aggregating seeds per treatment edge.

    Parameters
    ----------
    records:
        Paired record dicts (same length as predicted_probs).
    predicted_probs:
        Each entry has keys ``b_hat``, ``g_hat``, ``h_hat``.

    Returns
    -------
    Dict with tau_mae, eta_mae, tau_spearman, eta_spearman,
    top1_empirical_tau_hit_rate, top1_tau_regret.
    """
    # Group by edge.
    edge_empirical: dict[TreatmentEdgeKey, list[tuple[int, int]]] = defaultdict(list)
    edge_predicted: dict[TreatmentEdgeKey, list[dict[str, float]]] = defaultdict(list)

    for rec, probs in zip(records, predicted_probs):
        ek = treatment_edge_key(rec)
        y1, y0 = get_paired_outcomes(rec)
        edge_empirical[ek].append((y1, y0))
        edge_predicted[ek].append(probs)

    empirical_taus: list[float] = []
    predicted_taus: list[float] = []
    empirical_etas: list[float] = []
    predicted_etas: list[float] = []

    # Candidate-family grouping for ranking metrics.
    family_edges: dict[tuple[str, str], list[TreatmentEdgeKey]] = defaultdict(list)

    for ek in edge_empirical:
        outcomes = edge_empirical[ek]
        preds = edge_predicted[ek]
        n = len(outcomes)

        # Empirical q values.
        q00_emp = sum(1 for y1, y0 in outcomes if y0 == 0 and y1 == 0) / n
        q01_emp = sum(1 for y1, y0 in outcomes if y0 == 1 and y1 == 0) / n
        q10_emp = sum(1 for y1, y0 in outcomes if y0 == 0 and y1 == 1) / n
        q11_emp = sum(1 for y1, y0 in outcomes if y0 == 1 and y1 == 1) / n
        tau_emp = q10_emp - q01_emp
        eta_emp = q01_emp

        # Predicted q values (averaged over seeds).
        b_vals = [p["b_hat"] for p in preds]
        g_vals = [p["g_hat"] for p in preds]
        h_vals = [p["h_hat"] for p in preds]
        b_mean = float(np.mean(b_vals))
        g_mean = float(np.mean(g_vals))
        h_mean = float(np.mean(h_vals))
        q01_pred = b_mean * h_mean
        q10_pred = (1 - b_mean) * g_mean
        tau_pred = q10_pred - q01_pred
        eta_pred = q01_pred

        empirical_taus.append(tau_emp)
        predicted_taus.append(tau_pred)
        empirical_etas.append(eta_emp)
        predicted_etas.append(eta_pred)

        # Family grouping: (task_id, receiver_agent_id).
        task_id, receiver_id, _ = ek
        family_edges[(task_id, receiver_id)].append(ek)

    emp_tau = np.array(empirical_taus)
    pred_tau = np.array(predicted_taus)
    emp_eta = np.array(empirical_etas)
    pred_eta = np.array(predicted_etas)

    result: dict[str, Any] = {
        "n_edges": len(emp_tau),
        "tau_mae": float(np.mean(np.abs(emp_tau - pred_tau))) if len(emp_tau) else None,
        "eta_mae": float(np.mean(np.abs(emp_eta - pred_eta))) if len(emp_eta) else None,
    }

    # Spearman correlation.
    from scipy.stats import spearmanr

    if len(emp_tau) >= 2:
        tau_corr, _ = spearmanr(emp_tau, pred_tau)
        result["tau_spearman"] = float(tau_corr) if not np.isnan(tau_corr) else None
        eta_corr, _ = spearmanr(emp_eta, pred_eta)
        result["eta_spearman"] = float(eta_corr) if not np.isnan(eta_corr) else None
    else:
        result["tau_spearman"] = None
        result["eta_spearman"] = None

    # Candidate-family ranking metrics.
    hit_count = 0
    regret_sum = 0.0
    n_families = 0

    for family_key, edges in family_edges.items():
        if len(edges) < 2:
            continue
        n_families += 1
        # Empirical best tau.
        family_emp_taus = {
            ek: emp_tau[list(edge_empirical.keys()).index(ek)]
            for ek in edges
        }
        # Predicted best tau.
        family_pred_taus = {
            ek: pred_tau[list(edge_empirical.keys()).index(ek)]
            for ek in edges
        }
        best_empirical_edge = max(family_emp_taus, key=family_emp_taus.get)
        best_predicted_edge = max(family_pred_taus, key=family_pred_taus.get)

        max_emp_tau = family_emp_taus[best_empirical_edge]
        pred_selected_emp_tau = family_emp_taus[best_predicted_edge]

        if best_predicted_edge == best_empirical_edge:
            hit_count += 1
        regret_sum += max_emp_tau - pred_selected_emp_tau

    result["top1_empirical_tau_hit_rate"] = (
        hit_count / n_families if n_families > 0 else None
    )
    result["top1_tau_regret"] = (
        regret_sum / n_families if n_families > 0 else None
    )
    result["n_ranking_families"] = n_families

    return result
