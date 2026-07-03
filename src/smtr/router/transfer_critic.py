from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    TransferPredictionInput,
    prediction_input_from_record,
)

CLASS_ORDER = ["q00", "q01", "q10", "q11"]
LABEL_TO_CLASS = {
    "neutral_failure": "q00",
    "negative": "q01",
    "positive": "q10",
    "neutral_success": "q11",
}


class TransferEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    q00_mean: float
    q01_mean: float
    q10_mean: float
    q11_mean: float
    tau_mean: float
    tau_lcb: float
    tau_ucb: float
    negative_risk_mean: float
    negative_risk_ucb: float
    support_distance: float
    support_threshold: float
    low_support: bool
    ensemble_size: int
    critic_version: str


class SmoothedClassPriorModel:
    def __init__(self, labels: list[int]) -> None:
        counts = np.ones(4)
        for label in labels:
            counts[label] += 1
        self.probs = counts / counts.sum()

    def predict_proba(self, x) -> np.ndarray:
        return np.tile(self.probs, (x.shape[0], 1))


class FourOutcomeTransferCritic:
    critic_version = "four_outcome_bootstrap_v1"

    def __init__(self, *, encoder: HashingTransferFeatureEncoder | None = None) -> None:
        self.encoder = encoder or HashingTransferFeatureEncoder()
        self.models: list[Any] = []
        self.bootstrap_seeds: list[int] = []
        self.support_threshold = 0.0
        self.neighbor_model: NearestNeighbors | None = None

    def fit(
        self,
        records: list[PairedInterventionRecord],
        *,
        seed: int,
        n_bootstrap: int = 31,
    ) -> "FourOutcomeTransferCritic":
        items = [prediction_input_from_record(record) for record in records]
        x = self.encoder.transform(items)
        y = np.array(
            [CLASS_ORDER.index(LABEL_TO_CLASS[record.transfer_class]) for record in records]
        )
        rng = np.random.default_rng(seed)
        self.models = []
        self.bootstrap_seeds = []
        for _ in range(n_bootstrap):
            bootstrap_seed = int(rng.integers(0, 2**31 - 1))
            self.bootstrap_seeds.append(bootstrap_seed)
            sample_rng = np.random.default_rng(bootstrap_seed)
            indices = sample_rng.integers(0, len(records), size=len(records))
            sample_y = y[indices]
            if len(set(sample_y.tolist())) < 2:
                self.models.append(SmoothedClassPriorModel(sample_y.tolist()))
                continue
            model = LogisticRegression(max_iter=2000, solver="lbfgs")
            model.fit(x[indices], sample_y)
            self.models.append(model)
        self._fit_support(x)
        return self

    def predict(self, item: TransferPredictionInput) -> TransferEstimate:
        x = self.encoder.transform([item])
        probs = np.vstack([self._predict_model(model, x)[0] for model in self.models])
        q_mean = probs.mean(axis=0)
        tau = probs[:, 2] - probs[:, 1]
        eta = probs[:, 1]
        tau_mean = float(q_mean[2] - q_mean[1])
        support_distance = self._support_distance(x)
        return TransferEstimate(
            q00_mean=float(q_mean[0]),
            q01_mean=float(q_mean[1]),
            q10_mean=float(q_mean[2]),
            q11_mean=float(q_mean[3]),
            tau_mean=tau_mean,
            tau_lcb=float(np.quantile(tau, 0.05)),
            tau_ucb=float(np.quantile(tau, 0.95)),
            negative_risk_mean=float(eta.mean()),
            negative_risk_ucb=float(np.quantile(eta, 0.95)),
            support_distance=float(support_distance),
            support_threshold=float(self.support_threshold),
            low_support=bool(support_distance > self.support_threshold),
            ensemble_size=len(self.models),
            critic_version=self.critic_version,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FourOutcomeTransferCritic":
        return joblib.load(path)

    def _predict_model(self, model, x) -> np.ndarray:
        raw = model.predict_proba(x)
        if isinstance(model, SmoothedClassPriorModel):
            return raw
        mapped = np.zeros((x.shape[0], 4))
        for column_index, class_index in enumerate(model.classes_):
            mapped[:, int(class_index)] = raw[:, column_index]
        row_sums = mapped.sum(axis=1)
        mapped[row_sums == 0] = 0.25
        return mapped / mapped.sum(axis=1, keepdims=True)

    def _fit_support(self, x) -> None:
        n_neighbors = 2 if x.shape[0] > 1 else 1
        self.neighbor_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        self.neighbor_model.fit(x)
        distances, _ = self.neighbor_model.kneighbors(x)
        nearest = distances[:, 1] if n_neighbors == 2 else distances[:, 0]
        self.support_threshold = float(np.quantile(nearest, 0.95))

    def _support_distance(self, x) -> float:
        if self.neighbor_model is None:
            return 0.0
        distances, _ = self.neighbor_model.kneighbors(x, n_neighbors=1)
        return float(distances[0, 0])
