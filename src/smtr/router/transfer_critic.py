"""Four-outcome transfer critic for cross-agent memory exposure."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from smtr.core.types import (
    CandidateExposureInput,
    TransferPrediction,
    TransferPredictionDistribution,
)
from smtr.counterfactual.edge_keys import (
    TreatmentEdgeKey,
    treatment_edge_key,
)
from smtr.router.transfer_calibration import (
    DEFAULT_EPSILONS,
    Q01Calibrator,
    build_edge_calibration_examples,
    build_edge_threshold_examples,
    select_epsilon_edge_level,
)
from smtr.router.transfer_coverage import (
    count_outcome_edges,
    validate_transfer_label_coverage,
)
from smtr.router.transfer_features import HashingTransferFeatureEncoder

CLASS_ORDER = ["q00", "q01", "q10", "q11"]
LABEL_TO_INDEX = {
    "neutral_failure": 0,
    "negative_transfer": 1,
    "positive_transfer": 2,
    "neutral_success": 3,
}


class FourOutcomeTransferCritic:
    """Ensemble of logistic regression critics predicting four transfer outcomes.

    Outputs: q00=P(neutral_failure), q01=P(negative_transfer),
             q10=P(positive_transfer), q11=P(neutral_success)
    """

    def __init__(
        self,
        *,
        n_features: int = 512,
        n_bootstrap: int = 31,
        feature_block: str = "full",
        seed: int = 7,
    ) -> None:
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.feature_block = feature_block
        self.seed = seed
        self.encoder = HashingTransferFeatureEncoder(
            n_features=n_features, feature_block=feature_block
        )
        self.members: list[LogisticRegression] = []
        self._fitted = False
        self.coverage_report: dict[str, Any] | None = None
        self.q01_calibrator: Q01Calibrator | None = None
        self.epsilon_star: float | None = None
        self.risk_calibration: dict[str, Any] | None = None
        self.calibration_split: str | None = None
        self.epsilon_selection_split: str | None = None
        self.validation_edge_count: int | None = None
        self.calibration_unit: str | None = None
        self.epsilon_selection_unit: str | None = None
        self._edge_calibration_examples: list | None = None

    def fit(
        self,
        inputs: list[CandidateExposureInput],
        labels: list[str],
        *,
        coverage_mode: str = "pilot",
        sample_weights: np.ndarray | None = None,
        edge_clusters: dict | None = None,
    ) -> None:
        """Train bootstrap ensemble on paired record features.

        ``coverage_mode`` enforces four-outcome label coverage (清单第七章):
        ``formal`` requires all four classes, ``pilot`` requires at least
        positive_transfer and negative_transfer. Training without negative
        transfer always fails fast.

        Multi-seed treatment edges (清单 P0-5/P0-6): when ``edge_clusters``
        maps treatment edges to their seed-record rows, bootstrap members
        resample whole edges (all seeds of a drawn edge enter together);
        ``sample_weights`` should then be the edge-equal weights ``1/n_e``
        so every edge contributes equal total training weight. When
        sample weights are supplied they are the only weighting scheme
        (class balancing is disabled to avoid double weighting).
        """
        X = self.encoder.encode_batch(inputs)
        y = np.array([LABEL_TO_INDEX[lb] for lb in labels])
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=float)
            if len(sample_weights) != len(labels):
                raise ValueError(
                    "sample_weights must have one entry per training record"
                )
        if edge_clusters is not None:
            covered = {i for rows in edge_clusters.values() for i in rows}
            if covered != set(range(len(labels))):
                raise ValueError(
                    "edge_clusters must partition every training record row"
                )

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(
                "training data must contain at least two transfer outcome classes"
            )

        report = validate_transfer_label_coverage(labels, mode=coverage_mode)
        report.update(count_outcome_edges(inputs, labels))
        self.coverage_report = report

        required_classes = set(unique_classes.tolist())
        rng = np.random.default_rng(self.seed)
        self.members = []
        for _ in range(self.n_bootstrap):
            if edge_clusters is not None:
                idx = _edge_cluster_bootstrap_with_full_coverage(
                    y, edge_clusters, required_classes, rng
                )
            else:
                idx = _bootstrap_with_full_coverage(y, required_classes, rng)
            if idx is None:
                # Skip this member rather than fitting on a class-deficient
                # sample; zero-padding missing classes is forbidden.
                continue
            X_boot = X[idx]
            y_boot = y[idx]
            w_boot = None if sample_weights is None else sample_weights[idx]
            clf = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight=None if w_boot is not None else "balanced",
            )
            clf.fit(X_boot, y_boot, sample_weight=w_boot)
            self.members.append(clf)
        if not self.members:
            raise ValueError("no bootstrap member covered all required classes")
        self._fitted = True

    def predict(self, item: CandidateExposureInput) -> TransferPrediction:
        """Predict four-outcome distribution for a candidate exposure."""
        probs = self._member_probs(item).mean(axis=0)
        return TransferPrediction(
            q00_neutral_failure=float(probs[0]),
            q01_negative_transfer=float(probs[1]),
            q10_positive_transfer=float(probs[2]),
            q11_neutral_success=float(probs[3]),
        )

    def _member_probs(self, item: CandidateExposureInput) -> np.ndarray:
        """Per-bootstrap-member four-outcome probabilities, shape (M, 4)."""
        if not self._fitted:
            raise RuntimeError("critic not fitted")
        X = self.encoder.encode_one(item)
        member_probs = []
        for clf in self.members:
            p = clf.predict_proba(X)[0]
            # Align to 4 classes
            full_p = np.zeros(4)
            for i, c in enumerate(clf.classes_):
                full_p[int(c)] = p[i]
            member_probs.append(full_p)
        return np.asarray(member_probs)

    def predict_distribution(
        self, item: CandidateExposureInput
    ) -> TransferPredictionDistribution:
        """Ensemble-mean prediction plus bootstrap member uncertainty (清单第九章).

        tau_lower is the 0.10 quantile of member taus and eta_upper the
        0.90 quantile of member etas; member etas use the validation-fitted
        q01 calibrator when available so the risk bound matches the SMTR
        decision rule.
        """
        member_probs = self._member_probs(item)
        probs = member_probs.mean(axis=0)
        mean = TransferPrediction(
            q00_neutral_failure=float(probs[0]),
            q01_negative_transfer=float(probs[1]),
            q10_positive_transfer=float(probs[2]),
            q11_neutral_success=float(probs[3]),
        )
        member_tau = member_probs[:, 2] - member_probs[:, 1]
        member_eta = member_probs[:, 1]
        if self.q01_calibrator is not None:
            member_eta = self.q01_calibrator.predict(member_eta)
        return TransferPredictionDistribution(
            mean=mean,
            tau_std=float(member_tau.std()),
            eta_std=float(member_eta.std()),
            tau_lower=float(np.quantile(member_tau, 0.10)),
            eta_upper=float(np.quantile(member_eta, 0.90)),
        )

    def predict_batch(self, items: list[CandidateExposureInput]) -> list[TransferPrediction]:
        """Predict for a batch."""
        return [self.predict(item) for item in items]

    def predict_calibrated(self, item: CandidateExposureInput) -> TransferPrediction:
        """Prediction with calibrated negative-transfer risk (清单第三章).

        The SMTR decision rule compares the *calibrated* eta against the
        validation-selected epsilon_star; raw eta must never drive sharing.
        """
        raw = self.predict(item)
        return raw.model_copy(
            update={"eta_hat_calibrated": self.calibrate_negative_risk(raw.eta_hat_raw)}
        )

    def calibrate_negative_risk(self, eta_raw: float) -> float:
        """Calibrated q01 for one raw risk estimate.

        Falls back to the raw probability only when no validation-fitted
        calibrator exists (e.g. unit-test stubs); production checkpoints
        always carry one after ``calibrate_q01``.
        """
        if self.q01_calibrator is None:
            return float(eta_raw)
        return float(self.q01_calibrator.predict(np.array([float(eta_raw)]))[0])

    def calibrate_q01(
        self,
        inputs: list[CandidateExposureInput],
        labels: list[str],
        records: list[dict[str, Any]] | None = None,
        *,
        split_name: str = "validation",
        epsilons=DEFAULT_EPSILONS,
        delta: float = 0.10,
    ) -> dict[str, Any]:
        """Fit q01 calibration and select epsilon on validation treatment edges."""

        if not self._fitted:
            raise RuntimeError("critic not fitted")

        if split_name != "validation":
            raise ValueError(
                "q01 calibration and epsilon selection must use "
                f"the validation split, got {split_name!r}"
            )

        if records is None:
            raise ValueError(
                "edge-level calibration requires paired records; "
                "record-level calibration is not part of formal SMTR"
            )

        if len(inputs) != len(labels) or len(inputs) != len(records):
            raise ValueError(
                "inputs, labels and records must have identical lengths"
            )

        if not inputs:
            raise ValueError("validation data is empty")

        predictions = self.predict_batch(inputs)

        edge_predictions: dict[
            TreatmentEdgeKey,
            list[dict[str, float]],
        ] = defaultdict(list)

        for record, prediction in zip(records, predictions):
            edge_key = treatment_edge_key(record)

            edge_predictions[edge_key].append({
                "predicted_q01":
                    float(prediction.q01_negative_transfer),
                "predicted_tau":
                    float(
                        prediction.q10_positive_transfer
                        - prediction.q01_negative_transfer
                    ),
            })

        predictions_by_edge: dict[
            TreatmentEdgeKey,
            dict[str, float],
        ] = {}

        for edge_key, rows in edge_predictions.items():
            predictions_by_edge[edge_key] = {
                "predicted_q01": float(
                    np.mean([
                        row["predicted_q01"]
                        for row in rows
                    ])
                ),
                "predicted_tau": float(
                    np.mean([
                        row["predicted_tau"]
                        for row in rows
                    ])
                ),
            }

        calibration_examples = build_edge_calibration_examples(
            records=records,
            predictions_by_edge=predictions_by_edge,
        )

        if not calibration_examples:
            raise ValueError(
                "no validation treatment edges available for calibration"
            )

        # One edge is one calibration unit.
        calibration_weights = np.ones(
            len(calibration_examples),
            dtype=float,
        )

        self.q01_calibrator = Q01Calibrator().fit(
            np.asarray([
                example.predicted_q01
                for example in calibration_examples
            ], dtype=float),
            np.asarray([
                example.empirical_eta
                for example in calibration_examples
            ], dtype=float),
            sample_weight=calibration_weights,
        )

        threshold_examples = build_edge_threshold_examples(
            calibration_examples,
            self.q01_calibrator,
        )

        selection = select_epsilon_edge_level(
            examples=threshold_examples,
            candidate_epsilons=[
                float(value)
                for value in epsilons
            ],
            max_negative_exposure_rate=float(delta),
        )

        selection.update({
            "calibration_unit": "treatment_edge",
            "calibration_split": "validation",
            "calibration_method":
                self.q01_calibrator.method,
            "calibration_status":
                self.q01_calibrator.calibration_status,
            "calibration_edge_count":
                len(calibration_examples),
            "epsilon_selection_unit":
                "treatment_edge",
            "epsilon_selection_split":
                "validation",
            "validation_edge_count":
                len(calibration_examples),
            "risk_delta":
                float(delta),
        })

        self.epsilon_star = float(
            selection["epsilon_star"]
        )
        self.risk_calibration = selection
        self.calibration_split = "validation"
        self.epsilon_selection_split = "validation"
        self.validation_edge_count = len(
            calibration_examples
        )
        self.calibration_unit = "treatment_edge"
        self.epsilon_selection_unit = "treatment_edge"

        return selection

    def calibrated_q01(self, pred: TransferPrediction) -> float:
        """Calibrated negative-transfer probability for one prediction."""
        return self.calibrate_negative_risk(pred.q01_negative_transfer)

    def save(self, path: Path) -> None:
        """Save critic checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        import sklearn
        joblib.dump(
            {
                "members": self.members,
                "n_features": self.n_features,
                "n_bootstrap": self.n_bootstrap,
                "feature_block": self.feature_block,
                "seed": self.seed,
                "encoder": self.encoder,
                "schema_version": self.encoder.schema_version,
                "sklearn_version": sklearn.__version__,
                "method_version": "1.0",
                "coverage_report": self.coverage_report,
                "q01_calibrator": self.q01_calibrator,
                "epsilon_star": self.epsilon_star,
                "risk_calibration": self.risk_calibration,
                # 清单 P0-8: checkpoint must record where calibration and
                # epsilon selection happened.
                "calibration_split": self.calibration_split,
                "epsilon_selection_split": self.epsilon_selection_split,
                "validation_edge_count": self.validation_edge_count,
                "calibration_unit": self.calibration_unit,
                "calibration_method": (
                    self.q01_calibrator.method
                    if self.q01_calibrator is not None
                    else "unfitted"
                ),
                "calibration_status": (
                    self.q01_calibrator.calibration_status
                    if self.q01_calibrator is not None
                    else "unfitted"
                ),
                "calibration_edge_count": self.validation_edge_count,
                "epsilon_selection_unit": self.epsilon_selection_unit,
                "epsilon_validation_edge_count": self.validation_edge_count,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> FourOutcomeTransferCritic:
        """Load critic from checkpoint."""
        data = joblib.load(path)
        critic = cls(
            n_features=data["n_features"],
            n_bootstrap=data["n_bootstrap"],
            feature_block=data["feature_block"],
            seed=data["seed"],
        )
        critic.members = data["members"]
        critic.encoder = data["encoder"]
        critic.coverage_report = data.get("coverage_report")
        critic.q01_calibrator = data.get("q01_calibrator")
        critic.epsilon_star = data.get("epsilon_star")
        critic.risk_calibration = data.get("risk_calibration")
        critic.calibration_split = data.get("calibration_split")
        critic.epsilon_selection_split = data.get("epsilon_selection_split")
        critic.validation_edge_count = data.get("validation_edge_count")
        critic.calibration_unit = data.get("calibration_unit")
        critic.epsilon_selection_unit = data.get("epsilon_selection_unit")
        critic._fitted = True
        return critic


def _edge_cluster_bootstrap_with_full_coverage(
    y: np.ndarray,
    edge_clusters: dict,
    required_classes: set[int],
    rng: np.random.Generator,
    *,
    max_attempts: int = 10,
) -> np.ndarray | None:
    """Edge-cluster bootstrap: resample edges, keep all seeds together.

    Each draw selects treatment edges with replacement; drawing an edge
    adds *all* of its seed records, so one edge's seeds never split across
    a member. Returns None when no draw covers the required classes.
    """
    edges = list(edge_clusters.keys())
    for _ in range(max_attempts):
        chosen = rng.choice(len(edges), size=len(edges), replace=True)
        idx: list[int] = []
        for pos in chosen:
            idx.extend(edge_clusters[edges[pos]])
        idx_arr = np.asarray(idx, dtype=int)
        if required_classes.issubset(set(np.unique(y[idx_arr]).tolist())):
            return idx_arr
    return None


def _bootstrap_with_full_coverage(
    y: np.ndarray,
    required_classes: set[int],
    rng: np.random.Generator,
    *,
    max_attempts: int = 10,
) -> np.ndarray | None:
    """Resample until the bootstrap draw covers all required classes.

    Returns None after ``max_attempts`` failures so the caller can skip the
    member; padding missing classes with zero probability is forbidden.
    """
    for _ in range(max_attempts):
        idx = _stratified_bootstrap_indices(y, rng)
        if required_classes.issubset(set(np.unique(y[idx]).tolist())):
            return idx
    return None


def _stratified_bootstrap_indices(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified bootstrap ensuring all classes are represented."""
    sampled: list[int] = []
    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        sampled.extend(
            rng.choice(
                cls_indices,
                size=len(cls_indices),
                replace=True,
            ).tolist()
        )
    sampled_array = np.asarray(sampled)
    rng.shuffle(sampled_array)
    return sampled_array
