"""Four-outcome transfer critic for cross-agent memory exposure."""

from __future__ import annotations

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
from smtr.router.transfer_calibration import (
    DEFAULT_EPSILONS,
    Q01Calibrator,
    build_edge_calibration_examples,
    select_epsilon,
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
        """Calibrate q01 and select epsilon_star on validation edges only.

        Calibration fits one example per treatment edge (清单 P0-7): the
        critic's mean predicted q01 for the edge against the empirical
        negative-transfer rate ``N_e^01 / N_e``. epsilon_star is selected on
        the same validation edges only (清单 P0-8); fitting on the test
        split raises. The selected budget is persisted in the checkpoint and
        the test split must only read it, never re-select it.
        """
        if not self._fitted:
            raise RuntimeError("critic not fitted")
        if split_name == "test":
            raise ValueError(
                "q01 calibration and epsilon selection are forbidden on the "
                "test split (清单 P0-8)"
            )
        if records is not None and len(records) != len(inputs):
            raise ValueError("records must align with inputs")
        preds = self.predict_batch(inputs)
        q01 = np.array([p.q01_negative_transfer for p in preds])
        tau = np.array(
            [p.q10_positive_transfer - p.q01_negative_transfer for p in preds]
        )
        if records is not None:
            # Edge-level calibration: one example per treatment edge,
            # weighted by its seed count.
            examples = build_edge_calibration_examples(records, q01, labels)
            self.q01_calibrator = Q01Calibrator().fit(
                np.array([ex.predicted_q01 for ex in examples]),
                np.array([ex.empirical_eta for ex in examples]),
                sample_weight=np.array([ex.seed_count for ex in examples]),
            )
            self.validation_edge_count = len(examples)
        else:
            y_negative = np.array(
                [1 if lb == "negative_transfer" else 0 for lb in labels]
            )
            self.q01_calibrator = Q01Calibrator().fit(q01, y_negative)
            self.validation_edge_count = None
        q01_calibrated = self.q01_calibrator.predict(q01)
        selection = select_epsilon(
            tau, q01_calibrated, labels, epsilons=epsilons, delta=delta
        )
        selection["calibration_level"] = "edge" if records is not None else "record"
        selection["validation_edge_count"] = self.validation_edge_count
        self.epsilon_star = selection["epsilon_star"]
        self.risk_calibration = selection
        self.calibration_split = split_name
        self.epsilon_selection_split = split_name
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
