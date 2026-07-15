"""Factual-success critic for the retained supervision ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    TransferPredictionInput,
    prediction_input_from_record,
)


@dataclass(frozen=True)
class FactualSuccessEstimate:
    """Point prediction for P(Y_share=1 | o,m,S)."""

    p_share_success: float


class FactualSuccessCheckpointMetadata(BaseModel):
    """Metadata for the FactualSuccess-SMTR checkpoint."""

    model_config = ConfigDict(frozen=True)

    checkpoint_schema_version: str = "factual_success_checkpoint_v1"
    critic_class: str = "FactualSuccessCritic"
    encoder_class: str = "HashingTransferFeatureEncoder"
    method_family: str = "factual_success_ablation"
    target: str = "share_success"
    supports_point_prediction: bool = False
    supports_factual_success_prediction: bool = True
    uses_selected_set: bool = True
    uses_pairwise_counterfactual_labels: bool = False
    uses_pairwise_interactions: bool = True
    feature_block: str = "full"
    training_records_sha256: str = ""
    split_manifest_sha256: str = ""
    training_seed: int = 0
    hashing_dimension: int = 512
    threshold: float = 0.5


class SmoothedBinaryPriorModel:
    def __init__(self, labels: list[int]) -> None:
        positives = 1 + sum(labels)
        total = 2 + len(labels)
        self.prob = positives / total

    def predict_proba(self, x) -> np.ndarray:
        return np.tile([1.0 - self.prob, self.prob], (x.shape[0], 1))


class FactualSuccessCritic:
    """Binary classifier trained only on factual share success labels."""

    critic_version = "factual_success_logistic_v1"

    def __init__(self, *, encoder: HashingTransferFeatureEncoder | None = None) -> None:
        self.encoder = encoder or HashingTransferFeatureEncoder(feature_block="full")
        self.model: Any | None = None
        self.checkpoint_metadata: FactualSuccessCheckpointMetadata | None = None

    def fit(
        self,
        records: list[PairedInterventionRecord],
        *,
        seed: int,
    ) -> FactualSuccessCritic:
        items = [prediction_input_from_record(record) for record in records]
        x = self.encoder.transform(items)
        y = np.array([int(record.y_share) for record in records])
        if len(set(y.tolist())) < 2:
            self.model = SmoothedBinaryPriorModel(y.tolist())
        else:
            model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
            model.fit(x, y)
            self.model = model
        return self

    def predict_factual_success(self, item: TransferPredictionInput) -> float:
        if self.model is None:
            raise ValueError("factual success critic is not fitted")
        x = self.encoder.transform([item])
        return float(self.model.predict_proba(x)[0, 1])

    def predict_point(self, item: TransferPredictionInput) -> FactualSuccessEstimate:
        return FactualSuccessEstimate(
            p_share_success=self.predict_factual_success(item),
        )

    def save(
        self,
        path: Path,
        *,
        metadata: FactualSuccessCheckpointMetadata,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_metadata = metadata
        joblib.dump(self, path)
        self._metadata_path(path).write_text(
            metadata.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        require_metadata: bool = True,
    ) -> FactualSuccessCritic:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"checkpoint {path} did not contain {cls.__name__}: "
                f"got {type(obj).__name__}"
            )
        metadata_path = cls._metadata_path(path)
        if metadata_path.exists():
            metadata = FactualSuccessCheckpointMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            obj._validate_metadata(metadata)
            obj.checkpoint_metadata = metadata
        elif require_metadata:
            raise ValueError(f"checkpoint metadata is required: {metadata_path}")
        return obj

    @classmethod
    def _metadata_path(cls, path: Path) -> Path:
        return path.with_suffix(".metadata.json")

    def _validate_metadata(self, metadata: FactualSuccessCheckpointMetadata) -> None:
        if metadata.critic_class != self.__class__.__name__:
            raise TypeError("factual checkpoint metadata class mismatch")
        if metadata.encoder_class != self.encoder.__class__.__name__:
            raise TypeError("factual checkpoint encoder mismatch")
        if metadata.method_family != "factual_success_ablation":
            raise ValueError("factual checkpoint method_family mismatch")
        if metadata.target != "share_success":
            raise ValueError("factual checkpoint target mismatch")
        if metadata.uses_pairwise_counterfactual_labels:
            raise ValueError("factual checkpoint must not use pairwise labels")
        if metadata.feature_block != getattr(self.encoder, "feature_block", None):
            raise ValueError("factual checkpoint feature_block mismatch")


def choose_threshold_for_exposure(
    *,
    probabilities: list[float],
    target_mean_exposure: float,
    invocation_count: int,
) -> float:
    """Choose a validation-only threshold matching SMTR mean exposure."""
    if invocation_count < 1:
        raise ValueError("invocation_count must be positive")
    if target_mean_exposure < 0:
        raise ValueError("target_mean_exposure must be non-negative")
    if not probabilities:
        return 1.0
    count = max(
        0,
        min(len(probabilities), round(target_mean_exposure * invocation_count)),
    )
    if count == 0:
        return 1.0
    ordered = sorted(probabilities, reverse=True)
    if count == len(ordered):
        return float(ordered[-1])
    # Midpoint avoids accidentally including a tied/adjacent lower score while
    # retaining the requested top validation predictions.
    return float((ordered[count - 1] + ordered[count]) / 2.0)
