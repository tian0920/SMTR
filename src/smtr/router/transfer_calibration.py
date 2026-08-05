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

from smtr.counterfactual.edge_keys import (
    TreatmentEdgeKey,
    group_records_by_edge,
)
from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES, paired_record_label

LABELS = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
DEFAULT_EPSILONS = (0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class EdgeCalibrationExample:
    """One validation edge's calibration example (清单 P0-2).

    Exactly one example per treatment edge: ``predicted_q01`` /
    ``predicted_tau`` come from a single critic prediction for the edge,
    and the empirical values are rates over the edge's valid seed records
    (``empirical_eta = N_e^01 / N_e``).
    """

    task_id: str
    receiver_agent_id: str
    candidate_memory_id: str

    predicted_q01: float
    predicted_tau: float

    empirical_eta: float
    empirical_share_success: float
    empirical_withhold_success: float

    valid_seed_count: int

    @property
    def edge_key(self) -> TreatmentEdgeKey:
        return (self.task_id, self.receiver_agent_id, self.candidate_memory_id)


def build_edge_calibration_examples(
    *,
    records: list[dict[str, Any]],
    predictions_by_edge: dict[TreatmentEdgeKey, dict[str, float]],
) -> list[EdgeCalibrationExample]:
    """Aggregate validation records into one calibration example per edge.

    ``predictions_by_edge`` maps each treatment edge to its single critic
    prediction ``{"predicted_q01": ..., "predicted_tau": ...}``; calling the
    critic repeatedly on an edge's seed records and treating the copies as
    independent calibration samples is forbidden (清单 P0-2).
    """
    examples: list[EdgeCalibrationExample] = []
    for edge_key, rows in group_records_by_edge(records).items():
        prediction = predictions_by_edge.get(edge_key)
        if prediction is None:
            raise ValueError(
                f"edge {edge_key} has records but no critic prediction; "
                "predictions must cover every validation edge"
            )
        edge_records = [records[i] for i in rows]
        n = len(edge_records)
        negative_count = sum(
            1
            for record in edge_records
            if _record_label(record) == "negative_transfer"
        )
        share_success_count = sum(
            bool(record["share"]["team_success"]) for record in edge_records
        )
        withhold_success_count = sum(
            bool(record["withhold"]["team_success"]) for record in edge_records
        )
        examples.append(
            EdgeCalibrationExample(
                task_id=str(edge_key[0]),
                receiver_agent_id=str(edge_key[1]),
                candidate_memory_id=str(edge_key[2]),
                predicted_q01=float(prediction["predicted_q01"]),
                predicted_tau=float(prediction["predicted_tau"]),
                empirical_eta=negative_count / n,
                empirical_share_success=share_success_count / n,
                empirical_withhold_success=withhold_success_count / n,
                valid_seed_count=n,
            )
        )
    return examples


def _record_label(record: dict[str, Any]) -> str:
    """Transfer label of one record, preferring the persisted label."""
    label = record.get("label")
    if label:
        return str(label)
    return paired_record_label(record)


class Q01Calibrator:
    """Calibrates predicted q01 against continuous edge-level empirical eta.

    Edge-level empirical negative-transfer rates are continuous
    (``0, 1/n, ..., 1``); the calibrator therefore never requires exactly
    two unique target values and never binarizes the targets (清单 P0-1).
    With at least ``min_edges_for_isotonic`` validation edges an isotonic
    regression is fitted; otherwise the calibrator reports
    ``insufficient_validation_edges`` and applies the identity map instead
    of pretending calibration succeeded.
    """

    def __init__(self, *, min_edges_for_isotonic: int = 20) -> None:
        self.min_edges_for_isotonic = min_edges_for_isotonic
        self.method: str = "unfitted"
        self.model = None
        self.calibration_status: str = "unfitted"
        self.n_edges: int = 0

    def fit(
        self,
        predicted_q01: np.ndarray,
        empirical_eta: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> Q01Calibrator:
        """Fit on one (predicted q01, empirical eta) pair per edge."""
        predicted_q01 = np.asarray(predicted_q01, dtype=float)
        empirical_eta = np.asarray(empirical_eta, dtype=float)
        if predicted_q01.ndim != 1:
            raise ValueError("predicted_q01 must be one-dimensional")
        if empirical_eta.ndim != 1:
            raise ValueError("empirical_eta must be one-dimensional")
        if len(predicted_q01) != len(empirical_eta):
            raise ValueError(
                "predicted_q01 and empirical_eta length mismatch"
            )
        if np.any(predicted_q01 < 0) or np.any(predicted_q01 > 1):
            raise ValueError("predicted_q01 must lie in [0, 1]")
        if np.any(empirical_eta < 0) or np.any(empirical_eta > 1):
            raise ValueError("empirical_eta must lie in [0, 1]")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if len(sample_weight) != len(predicted_q01):
                raise ValueError("sample_weight must align with predicted_q01")

        self.n_edges = int(len(predicted_q01))
        if len(predicted_q01) >= self.min_edges_for_isotonic:
            self.model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
                increasing=True,
            )
            self.model.fit(predicted_q01, empirical_eta, sample_weight=sample_weight)
            self.method = "isotonic"
            self.calibration_status = "fitted"
        else:
            # Not enough validation edges for a monotone fit; keep raw
            # probabilities and say so explicitly (清单 P0-1).
            self.model = None
            self.method = "identity"
            self.calibration_status = "insufficient_validation_edges"
        return self

    def transform(self, predicted_q01: np.ndarray) -> np.ndarray:
        """Calibrated q01, clipped to [0, 1]."""
        x = np.asarray(predicted_q01, dtype=float)
        if self.method == "isotonic":
            calibrated = self.model.predict(x)
        elif self.method == "identity":
            calibrated = x
        else:
            raise RuntimeError("calibrator is not fitted")
        return np.clip(calibrated, 0.0, 1.0)

    # Existing call sites use ``predict``; it is the same operation.
    def predict(self, predicted_q01: np.ndarray) -> np.ndarray:
        return self.transform(predicted_q01)


@dataclass(frozen=True)
class EdgeThresholdExample:
    """One validation edge's input for epsilon threshold selection (P0-4).

    The selection unit is the treatment edge: each edge contributes exactly
    one example regardless of how many seeds it was observed under.
    """

    edge_key: tuple[str, str, str]

    predicted_tau: float
    calibrated_eta: float

    empirical_share_success: float
    empirical_withhold_success: float
    empirical_negative_transfer_rate: float

    valid_seed_count: int


def build_edge_threshold_examples(
    examples: list[EdgeCalibrationExample],
    calibrator: Q01Calibrator,
) -> list[EdgeThresholdExample]:
    """Threshold-selection examples from calibration examples + calibrator."""
    predicted_q01 = np.array([ex.predicted_q01 for ex in examples])
    calibrated = calibrator.transform(predicted_q01)
    return [
        EdgeThresholdExample(
            edge_key=ex.edge_key,
            predicted_tau=ex.predicted_tau,
            calibrated_eta=float(calibrated[i]),
            empirical_share_success=ex.empirical_share_success,
            empirical_withhold_success=ex.empirical_withhold_success,
            empirical_negative_transfer_rate=ex.empirical_eta,
            valid_seed_count=ex.valid_seed_count,
        )
        for i, ex in enumerate(examples)
    ]


def select_epsilon_edge_level(
    *,
    examples: list[EdgeThresholdExample],
    candidate_epsilons: list[float],
    max_negative_exposure_rate: float | None = None,
) -> dict[str, Any]:
    """Select epsilon_star over validation treatment edges (清单 P0-4~7).

    Each candidate epsilon is scored with an edge-equal-weight policy value:
    shared edges contribute ``empirical_share_success``, withheld edges
    ``empirical_withhold_success``. Edges with more seeds never gain more
    weight. Candidates violating the negative-exposure constraint are
    dropped; ties on policy value prefer the smaller epsilon; if nothing
    is feasible the selection fails instead of relaxing the constraint.
    """
    if not examples:
        raise ValueError("no validation edges available for epsilon selection")

    rows: list[dict[str, Any]] = []
    for epsilon in candidate_epsilons:
        shared_flags = [
            ex.predicted_tau > 0.0 and ex.calibrated_eta <= epsilon
            for ex in examples
        ]
        edge_values = [
            ex.empirical_share_success if flag else ex.empirical_withhold_success
            for ex, flag in zip(examples, shared_flags)
        ]
        shared_examples = [
            ex for ex, flag in zip(examples, shared_flags) if flag
        ]
        if shared_examples:
            negative_exposure_rate = float(np.mean([
                ex.empirical_negative_transfer_rate for ex in shared_examples
            ]))
            mean_eta_shared = float(np.mean([
                ex.empirical_negative_transfer_rate for ex in shared_examples
            ]))
        else:
            negative_exposure_rate = 0.0
            mean_eta_shared = 0.0
        policy_value = float(np.mean(edge_values))
        rows.append({
            "epsilon": float(epsilon),
            "policy_value": policy_value,
            "validation_policy_value": policy_value,
            "negative_exposure_rate": negative_exposure_rate,
            "shared_edge_rate": len(shared_examples) / len(examples),
            "mean_empirical_eta_among_shared_edges": mean_eta_shared,
        })

    eligible = [
        row
        for row in rows
        if (
            max_negative_exposure_rate is None
            or row["negative_exposure_rate"] <= max_negative_exposure_rate
        )
    ]
    if not eligible:
        raise ValueError("no epsilon satisfies the validation risk constraint")
    # Max policy value first; ties prefer the smaller (safer) epsilon.
    best = max(eligible, key=lambda row: (row["policy_value"], -row["epsilon"]))
    return {
        "selection_unit": "treatment_edge",
        "validation_edge_count": len(examples),
        "candidate_rows": rows,
        "epsilon_star": best["epsilon"],
        "max_negative_exposure_rate": max_negative_exposure_rate,
        "selected_on": "validation",
        "validation_policy_value": best["policy_value"],
        "negative_exposure_rate": best["negative_exposure_rate"],
    }


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
