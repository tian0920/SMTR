"""TCI offline evaluation metrics (Task 15).

Metrics:
  - Pairwise accuracy: P(d * (s_m - s_m~) > 0)
  - Pairwise margin:   E[d * (s_m - s_m~)]
  - Operator breakdown: per-operator accuracy and margin

Does NOT modify router decision rules.
Does NOT modify transfer_critic.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TCIMetrics:
    """Offline evaluation results for TCI ranker."""

    n_pairs: int = 0
    pairwise_accuracy: float = 0.0
    pairwise_margin: float = 0.0

    # Per-operator breakdown.
    operator_accuracy: dict[str, float] = field(default_factory=dict)
    operator_margin: dict[str, float] = field(default_factory=dict)
    operator_count: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "pairwise_accuracy": self.pairwise_accuracy,
            "pairwise_margin": self.pairwise_margin,
            "operator_accuracy": self.operator_accuracy,
            "operator_margin": self.operator_margin,
            "operator_count": self.operator_count,
        }


def evaluate_tci_ranker(
    score_original: np.ndarray,
    score_perturbed: np.ndarray,
    directions: np.ndarray,
    operator_types: list[str] | None = None,
) -> TCIMetrics:
    """Evaluate TCI ranker on offline pairs.

    Parameters
    ----------
    score_original : shape (n,) — s(m) scores
    score_perturbed : shape (n,) — s(m~) scores
    directions : shape (n,) — +1 or -1
    operator_types : optional list of operator names per pair

    Returns
    -------
    TCIMetrics with pairwise accuracy, margin, and operator breakdown.
    """
    n = len(directions)
    if n == 0:
        return TCIMetrics()

    margin = directions * (score_original - score_perturbed)
    correct = (margin > 0).astype(float)

    metrics = TCIMetrics(
        n_pairs=n,
        pairwise_accuracy=float(np.mean(correct)),
        pairwise_margin=float(np.mean(margin)),
    )

    # Per-operator breakdown.
    if operator_types is not None:
        op_names = sorted(set(operator_types))
        for op in op_names:
            mask = np.array([t == op for t in operator_types])
            cnt = int(np.sum(mask))
            metrics.operator_count[op] = cnt
            if cnt > 0:
                metrics.operator_accuracy[op] = float(
                    np.mean(correct[mask])
                )
                metrics.operator_margin[op] = float(
                    np.mean(margin[mask])
                )
            else:
                metrics.operator_accuracy[op] = 0.0
                metrics.operator_margin[op] = 0.0

    return metrics


def compute_regret(
    score_original: np.ndarray,
    score_perturbed: np.ndarray,
    directions: np.ndarray,
) -> float:
    """Compute regret: fraction of pairs where ranker picks wrong memory.

    Regret = P(d * (s_m - s_m~) < 0)
    """
    n = len(directions)
    if n == 0:
        return 0.0
    margin = directions * (score_original - score_perturbed)
    return float(np.mean(margin < 0))


def top1_transfer_effect_hit_rate(
    scores: np.ndarray,
    true_effects: np.ndarray,
) -> float:
    """Compute top-1 transfer-effect hit rate.

    For each query, check if the top-scored memory has the highest
    true transfer effect.

    Parameters
    ----------
    scores : shape (n_queries, n_candidates) — predicted scores
    true_effects : shape (n_queries, n_candidates) — true transfer effects

    Returns
    -------
    Hit rate: fraction of queries where argmax(score) == argmax(effect).
    """
    if len(scores) == 0:
        return 0.0
    predicted_top = np.argmax(scores, axis=1)
    true_top = np.argmax(true_effects, axis=1)
    return float(np.mean(predicted_top == true_top))


# ──────────────────────────────────────────────────────────────
# Generalization evaluation (Tasks 3, 4, 5)
# ──────────────────────────────────────────────────────────────

def evaluate_split_metrics(
    ranker: Any,
    train_pairs: list[Any],
    valid_pairs: list[Any],
    test_pairs: list[Any],
    *,
    feature_encoder: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate ranker on train / valid / test splits.

    If ``feature_encoder`` is None, uses synthetic random features
    keyed by perturbation_id for deterministic evaluation.

    Returns
    -------
    dict with keys "train", "valid", "test", each containing
    {"n": int, "accuracy": float, "margin": float}.
    """
    results: dict[str, dict[str, float]] = {}
    for name, pairs in [("train", train_pairs), ("valid", valid_pairs), ("test", test_pairs)]:
        if not pairs:
            results[name] = {"n": 0, "accuracy": 0.0, "margin": 0.0}
            continue

        s_orig, s_pert, dirs = _score_pairs(ranker, pairs, feature_encoder)
        margin = dirs * (s_orig - s_pert)
        accuracy = float(np.mean((margin > 0).astype(float)))
        results[name] = {
            "n": len(pairs),
            "accuracy": accuracy,
            "margin": float(np.mean(margin)),
        }
    return results


def _score_pairs(
    ranker: Any,
    pairs: list[Any],
    feature_encoder: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score pairs with a ranker and return (s_orig, s_pert, dirs).

    If pairs have ``has_structural_features``, use precomputed
    structural features directly (no encoder needed).

    Otherwise fall back to deterministic pseudo-random
    hash features from perturbation_id.
    """
    import hashlib

    n = len(pairs)
    feat_dim = ranker.config.feature_dim
    dirs = np.zeros(n)

    # Check if pairs have precomputed structural features.
    use_structural = (
        hasattr(pairs[0], "has_structural_features")
        and pairs[0].has_structural_features
    ) if n > 0 else False

    if use_structural:
        feat_orig = np.zeros((n, feat_dim))
        feat_pert = np.zeros((n, feat_dim))
        for i, p in enumerate(pairs):
            of = list(p.original_features)
            pf = list(p.perturbed_features)
            # Truncate or pad to feat_dim.
            feat_orig[i] = (of + [0.0] * feat_dim)[:feat_dim]
            feat_pert[i] = (pf + [0.0] * feat_dim)[:feat_dim]
            dirs[i] = float(p.direction)
    else:
        # Fallback: hash-based deterministic features.
        feat_orig = np.zeros((n, feat_dim))
        feat_pert = np.zeros((n, feat_dim))
        for i, p in enumerate(pairs):
            h_orig = hashlib.sha256(
                f"orig:{p.perturbation_id}".encode()
            ).digest()
            h_pert = hashlib.sha256(
                f"pert:{p.perturbation_id}".encode()
            ).digest()
            # SHA-256 gives 32 bytes; tile to fill feat_dim.
            raw_o = np.frombuffer(h_orig, dtype=np.uint8).astype(float)
            raw_p = np.frombuffer(h_pert, dtype=np.uint8).astype(float)
            reps = (feat_dim // len(raw_o)) + 1
            feat_orig[i] = np.tile(raw_o, reps)[:feat_dim]
            feat_pert[i] = np.tile(raw_p, reps)[:feat_dim]
            dirs[i] = float(p.direction)
        feat_orig = feat_orig / 255.0
        feat_pert = feat_pert / 255.0

    s_orig = ranker.score(feat_orig)
    s_pert = ranker.score(feat_pert)
    return s_orig, s_pert, dirs


def compute_margin_accuracy_curve(
    score_original: np.ndarray,
    score_perturbed: np.ndarray,
    directions: np.ndarray,
    *,
    bins: int = 5,
) -> dict[str, dict[str, float]]:
    """Compute accuracy by margin quantile.

    For each quantile bin of |d * (s_m - s_m~)|, report mean margin
    and accuracy. Validates that higher confidence predictions are
    more reliable.

    Returns
    -------
    dict keyed by bin label (e.g. "0-20%", "80-100%"), each with
    {"mean_margin": float, "accuracy": float, "n": int}.
    """
    n = len(directions)
    if n == 0:
        return {}

    margin = directions * (score_original - score_perturbed)
    abs_margin = np.abs(margin)

    sorted_indices = np.argsort(abs_margin)
    bin_size = max(1, n // bins)

    result: dict[str, dict[str, float]] = {}
    for b in range(bins):
        lo = b * bin_size
        hi = n if b == bins - 1 else (b + 1) * bin_size
        idx = sorted_indices[lo:hi]
        if len(idx) == 0:
            continue
        lo_pct = b * 100 // bins
        hi_pct = (b + 1) * 100 // bins
        label = f"{lo_pct}-{hi_pct}%"
        bin_margin = margin[idx]
        bin_correct = (bin_margin > 0).astype(float)
        result[label] = {
            "mean_margin": float(np.mean(abs_margin[idx])),
            "accuracy": float(np.mean(bin_correct)),
            "n": int(len(idx)),
        }

    return result


def evaluate_by_factor(
    pairs: list[Any],
    ranker: Any,
    *,
    feature_encoder: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate ranker broken down by perturbation_type (factor).

    Returns
    -------
    dict keyed by perturbation_type, each with
    {"n": int, "accuracy": float, "margin": float}.
    """
    # Group by perturbation_type.
    groups: dict[str, list[Any]] = {}
    for p in pairs:
        pt = p.perturbation_type
        groups.setdefault(pt, []).append(p)

    result: dict[str, dict[str, Any]] = {}
    for pt, pt_pairs in groups.items():
        s_orig, s_pert, dirs = _score_pairs(ranker, pt_pairs, feature_encoder)
        margin = dirs * (s_orig - s_pert)
        accuracy = float(np.mean((margin > 0).astype(float)))
        result[pt] = {
            "n": len(pt_pairs),
            "accuracy": accuracy,
            "margin": float(np.mean(margin)),
        }
    return result


def compute_feature_shift(
    pairs: list[Any],
) -> dict[str, float]:
    """Compute feature shift between original and perturbed cards.

    D = ||phi(m) - phi(m~)||_2

    Returns
    -------
    dict with mean_shift, median_shift, zero_shift_rate.

    A valid contrast must have zero_shift_rate < 1.0
    (i.e., features actually change after perturbation).
    """
    if not pairs:
        return {"mean_shift": 0.0, "median_shift": 0.0, "zero_shift_rate": 1.0}

    shifts: list[float] = []
    for p in pairs:
        if p.has_structural_features:
            of = np.array(list(p.original_features))
            pf = np.array(list(p.perturbed_features))
            shift = float(np.linalg.norm(of - pf))
            shifts.append(shift)
        else:
            # No structural features: cannot compute shift.
            shifts.append(0.0)

    arr = np.array(shifts)
    zero_count = int(np.sum(arr == 0.0))
    return {
        "mean_shift": float(np.mean(arr)),
        "median_shift": float(np.median(arr)),
        "zero_shift_rate": zero_count / len(arr) if len(arr) > 0 else 1.0,
        "n_pairs": len(pairs),
    }
