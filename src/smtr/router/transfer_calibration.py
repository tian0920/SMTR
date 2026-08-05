"""q01 risk calibration, calibration metrics and epsilon selection (清单第八章).

The SMTR decision rule depends on eta_hat <= epsilon, so probability
calibration of q01 (predicted negative-transfer probability) and a
validation-selected risk budget are required; overall classification
accuracy alone is not sufficient. The risk budget is never selected on the
test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from smtr.counterfactual.edge_keys import (
    TreatmentEdgeKey,
    group_records_by_edge,
)
from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES

LABELS = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
DEFAULT_EPSILONS = (0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class EdgeCalibrationExample:
    """One validation edge's calibration example (清单 P0-7).

    ``predicted_q01`` is the critic's mean predicted negative-transfer
    probability across the edge's seed records; ``empirical_eta`` is the
    observed negative-transfer rate ``N_e^01 / N_e`` on those seeds.
    """

    edge_key: TreatmentEdgeKey
    predicted_q01: float
    empirical_eta: float
    seed_count: int


def build_edge_calibration_examples(
    records: list[dict[str, Any]],
    predicted_q01: np.ndarray,
    labels: list[str],
) -> list[EdgeCalibrationExample]:
    """Aggregate validation records into per-edge calibration examples.

    Fitting on these examples instead of raw seed records avoids fitting the
    same edge prediction repeatedly across its seeds (清单 P0-7).
    """
    predicted_q01 = np.asarray(predicted_q01, dtype=float)
    if len(predicted_q01) != len(records) or len(labels) != len(records):
        raise ValueError("records, predicted_q01 and labels must align")
    examples: list[EdgeCalibrationExample] = []
    for edge_key, rows in group_records_by_edge(records).items():
        empirical_eta = float(
            np.mean([labels[i] == "negative_transfer" for i in rows])
        )
        examples.append(
            EdgeCalibrationExample(
                edge_key=edge_key,
                predicted_q01=float(predicted_q01[rows].mean()),
                empirical_eta=empirical_eta,
                seed_count=len(rows),
            )
        )
    return examples


class Q01Calibrator:
    """Calibrates predicted q01 against the negative-transfer indicator.

    Uses isotonic regression when enough validation points are available,
    otherwise falls back to Platt-style logistic calibration.
    """

    def __init__(self, *, min_isotonic_samples: int = 20) -> None:
        self.min_isotonic_samples = min_isotonic_samples
        self.method: str | None = None
        self._model = None

    def fit(
        self,
        q01: np.ndarray,
        y_negative: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> Q01Calibrator:
        """Fit on calibration targets.

        Targets may be binary seed indicators or continuous edge-level
        empirical rates (清单 P0-7); ``sample_weight`` then carries the
        edge seed counts so edges with more seeds weigh proportionally more.
        """
        q01 = np.asarray(q01, dtype=float)
        y_negative = np.asarray(y_negative, dtype=float)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if len(sample_weight) != len(q01):
                raise ValueError("sample_weight must align with q01")
        both_classes = len(np.unique(y_negative)) == 2
        if both_classes and len(q01) >= self.min_isotonic_samples:
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(q01, y_negative, sample_weight=sample_weight)
            self.method = "isotonic"
        else:
            self._model = LogisticRegression(max_iter=1000, solver="lbfgs")
            if both_classes:
                # Platt fallback needs class targets; threshold continuous
                # edge-level rates at 0.5 for the binary decision.
                self._model.fit(
                    q01.reshape(-1, 1),
                    (y_negative >= 0.5).astype(int),
                    sample_weight=sample_weight,
                )
                self.method = "platt"
            else:
                # Degenerate validation set: keep raw probabilities.
                self.method = "identity"
        return self

    def predict(self, q01: np.ndarray) -> np.ndarray:
        q01 = np.asarray(q01, dtype=float)
        if self._model is None or self.method == "identity":
            return q01
        if self.method == "isotonic":
            return np.clip(self._model.predict(q01), 0.0, 1.0)
        return self._model.predict_proba(q01.reshape(-1, 1))[:, 1]


def predicted_label(probs: np.ndarray) -> str:
    """Most likely label for one probability vector over LABELS order."""
    return LABELS[int(np.argmax(probs))]


def compute_four_class_metrics(
    labels_true: list[str],
    labels_pred: list[str],
) -> dict[str, Any]:
    """Accuracy, macro F1, per-class precision/recall, confusion matrix."""
    confusion = {true: {pred: 0 for pred in LABELS} for true in LABELS}
    for true, pred in zip(labels_true, labels_pred):
        confusion[true][pred] += 1
    per_class_precision, per_class_recall, f1s = {}, {}, []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_precision[label] = precision
        per_class_recall[label] = recall
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    accuracy = (
        sum(confusion[label][label] for label in LABELS) / len(labels_true)
        if labels_true
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion,
        "negative_transfer_recall": per_class_recall["negative_transfer"],
    }


def compute_probability_metrics(
    labels_true: list[str],
    probs: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Probability-quality metrics focused on q01 (negative-transfer risk)."""
    probs = np.asarray(probs, dtype=float)
    y_index = np.array([LABEL_TO_INDEX[label] for label in labels_true])
    n = len(labels_true)
    if n == 0:
        return {}

    # Multiclass log loss (clipped for stability).
    picked = probs[np.arange(n), y_index]
    log_loss = float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))

    # Multiclass Brier score.
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_index] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # q01-focused calibration.
    q01 = probs[:, LABEL_TO_INDEX["negative_transfer"]]
    y_neg = (y_index == LABEL_TO_INDEX["negative_transfer"]).astype(float)
    negative_transfer_brier = float(np.mean((q01 - y_neg) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    curve = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (q01 > lo) & (q01 <= hi) if b > 0 else (q01 >= lo) & (q01 <= hi)
        if not mask.any():
            continue
        mean_pred = float(q01[mask].mean())
        mean_obs = float(y_neg[mask].mean())
        weight = mask.mean()
        ece += weight * abs(mean_pred - mean_obs)
        curve.append({"bin": b, "mean_predicted_q01": mean_pred, "observed_rate": mean_obs, "weight": weight})

    return {
        "multiclass_log_loss": log_loss,
        "multiclass_brier_score": brier,
        "expected_calibration_error": float(ece),
        "negative_transfer_brier_score": negative_transfer_brier,
        "q01_calibration_curve": curve,
    }


def share_decisions(
    tau_hat: np.ndarray,
    q01_calibrated: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """SMTR-v1 decision rule: share iff tau_hat > 0 and eta_hat <= epsilon."""
    return (np.asarray(tau_hat) > 0) & (np.asarray(q01_calibrated) <= epsilon)


def risk_utility_curve(
    tau_hat: np.ndarray,
    q01_calibrated: np.ndarray,
    labels: list[str],
    *,
    epsilons=DEFAULT_EPSILONS,
) -> dict[str, dict[str, float]]:
    """Candidate-level risk-utility metrics for each risk budget epsilon."""
    tau_hat = np.asarray(tau_hat)
    q01_calibrated = np.asarray(q01_calibrated)
    y_share = np.array([LABEL_TO_OUTCOMES[lb][0] for lb in labels])
    is_negative = np.array([lb == "negative_transfer" for lb in labels])
    is_positive = np.array([lb == "positive_transfer" for lb in labels])

    curve: dict[str, dict[str, float]] = {}
    for epsilon in epsilons:
        shared = share_decisions(tau_hat, q01_calibrated, epsilon)
        n_shared = int(shared.sum())
        policy_outcome = np.where(shared, y_share, _withhold_outcomes(labels))
        exposure = float(is_negative[shared].mean()) if n_shared else 0.0
        curve[str(epsilon)] = {
            "epsilon": epsilon,
            "policy_success_rate": float(policy_outcome.mean()) if len(labels) else 0.0,
            "share_coverage": float(shared.mean()) if len(labels) else 0.0,
            "positive_transfer_recall": (
                float(shared[is_positive].mean()) if is_positive.any() else 0.0
            ),
            "negative_transfer_exposure_rate": exposure,
            "negative_transfer_rejection_rate": (
                float((~shared[is_negative]).mean()) if is_negative.any() else 0.0
            ),
            "safe_exposure_precision": (
                float(y_share[shared].mean()) if n_shared else 0.0
            ),
        }
    return curve


def select_epsilon(
    tau_hat: np.ndarray,
    q01_calibrated: np.ndarray,
    labels: list[str],
    *,
    epsilons=DEFAULT_EPSILONS,
    delta: float = 0.10,
) -> dict[str, Any]:
    """Select epsilon_star on validation data only.

    epsilon_star = argmax PolicyValue(epsilon) subject to
    NegativeTransferExposureRate(epsilon) <= delta. If no epsilon meets the
    risk constraint, the most conservative budget is returned.
    """
    curve = risk_utility_curve(tau_hat, q01_calibrated, labels, epsilons=epsilons)
    feasible = [
        point
        for point in curve.values()
        if point["negative_transfer_exposure_rate"] <= delta
    ]
    if feasible:
        best = max(feasible, key=lambda point: point["policy_success_rate"])
    else:
        best = min(curve.values(), key=lambda point: point["epsilon"])
    return {
        "epsilon_star": best["epsilon"],
        "risk_delta": delta,
        "selected_on": "validation",
        "validation_policy_value": best["policy_success_rate"],
        "validation_negative_transfer_exposure_rate": best["negative_transfer_exposure_rate"],
        "candidates_evaluated": curve,
    }


def _withhold_outcomes(labels: list[str]) -> np.ndarray:
    return np.array([LABEL_TO_OUTCOMES[lb][1] for lb in labels])
