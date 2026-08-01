"""Four-outcome transfer critic for cross-agent memory exposure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from smtr.core.types import CandidateExposureInput, TransferPrediction
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

    def fit(
        self,
        inputs: list[CandidateExposureInput],
        labels: list[str],
    ) -> None:
        """Train bootstrap ensemble on paired record features."""
        X = self.encoder.encode_batch(inputs)
        y = np.array([LABEL_TO_INDEX[lb] for lb in labels])
        rng = np.random.default_rng(self.seed)
        self.members = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(y), size=len(y), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
            clf.fit(X_boot, y_boot)
            self.members.append(clf)
        self._fitted = True

    def predict(self, item: CandidateExposureInput) -> TransferPrediction:
        """Predict four-outcome distribution for a candidate exposure."""
        if not self._fitted:
            raise RuntimeError("critic not fitted")
        X = self.encoder.encode_one(item)
        probs = np.zeros(4)
        for clf in self.members:
            p = clf.predict_proba(X)[0]
            # Align to 4 classes
            full_p = np.zeros(4)
            for i, c in enumerate(clf.classes_):
                full_p[int(c)] = p[i]
            probs += full_p
        probs /= len(self.members)
        return TransferPrediction(
            q00_neutral_failure=float(probs[0]),
            q01_negative_transfer=float(probs[1]),
            q10_positive_transfer=float(probs[2]),
            q11_neutral_success=float(probs[3]),
        )

    def predict_batch(self, items: list[CandidateExposureInput]) -> list[TransferPrediction]:
        """Predict for a batch."""
        return [self.predict(item) for item in items]

    def save(self, path: Path) -> None:
        """Save critic checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "members": self.members,
                "n_features": self.n_features,
                "n_bootstrap": self.n_bootstrap,
                "feature_block": self.feature_block,
                "seed": self.seed,
                "encoder": self.encoder,
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
        critic._fitted = True
        return critic
