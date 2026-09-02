"""Official-score transfer critic for RIMA (Phase 2).

Two-potential-outcome critic replacing the binary four-outcome critic for
the formal MultiAgentBench method (paper Section 4.2)::

    mu_1(h) = E[Y(1) | h],   mu_0(h) = E[Y(0) | h],
    tau_hat(h) = mu_1(h) - mu_0(h)

where ``Y`` is the normalized official MultiAgentBench Task Score in [0,1]
and ``h = phi_R(m, a_r, x_t)`` are receiver-conditioned features.

Training data are **historical matched interventions**; each example carries
``official_expose_score`` (Y1) and ``official_withhold_score`` (Y0).

Loss: Huber (recommended), MSE also supported — applied independently to
each potential-outcome head.

Admission rule (paper Eq. 8, Phase 2.3/28)::

    A_r^t(m) = I[tau_hat(m, r | x_t) > 0]

NO epsilon-star gate, NO eta threshold, NO validation-tuned threshold.

Fail-closed invariants:

* invalid training examples (missing score on either branch) are excluded
  and counted — never treated as zero;
* self-transfer pairs (source == receiver) are excluded from training;
* predicting with an unfitted critic raises;
* re-fitting a frozen critic raises.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge

from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)

__all__ = [
    "MatchedInterventionExample",
    "OfficialScoreTransferCritic",
    "CriticPrediction",
    "ADMISSION_RULE",
]

#: Formal admission rule (paper Eq. 8). No gates beyond sign of tau_hat.
ADMISSION_RULE = "admit iff tau_hat > 0"

_VALID_LOSSES = frozenset({"huber", "mse"})


@dataclass(frozen=True)
class MatchedInterventionExample:
    """One historical matched expose/withhold intervention record.

    Attributes:
        task_id: task on which the intervention was run.
        memory_id: memory exposed in the expose branch.
        receiver_id: receiver agent of the exposure.
        source_agent_id: agent that produced the memory (self-transfer
            pairs are excluded from training when source == receiver).
        official_expose_score: Y(1) — normalized official TS of expose run.
        official_withhold_score: Y(0) — normalized official TS of withhold run.
        features: receiver-conditioned feature input (routing card only).
    """

    task_id: str
    memory_id: str
    receiver_id: str
    source_agent_id: str
    official_expose_score: float | None
    official_withhold_score: float | None
    features: ReceiverConditionedTransferFeatures


@dataclass(frozen=True)
class CriticPrediction:
    """Prediction for one (memory, receiver, task) triple."""

    memory_id: str
    receiver_id: str
    task_id: str
    mu_expose: float | None
    mu_withhold: float | None
    tau_hat: float | None

    @property
    def is_valid(self) -> bool:
        return self.tau_hat is not None

    @property
    def admitted(self) -> bool:
        """Formal admission decision (paper Eq. 8). Invalid -> rejected."""
        return self.tau_hat is not None and self.tau_hat > 0.0


class OfficialScoreTransferCritic:
    """Continuous potential-outcome critic with official-score supervision.

    Parameters:
        encoder: receiver-conditioned feature encoder.
        loss: "huber" (recommended) or "mse".
        receiver_conditioned: whether receiver features are included.
            ``False`` yields the receiver-agnostic RIMA-Uniform critic
            (clean causal ablation, Phase 22).
    """

    def __init__(
        self,
        *,
        encoder: RimaFeatureEncoder,
        loss: str = "huber",
        receiver_conditioned: bool = True,
    ) -> None:
        if loss not in _VALID_LOSSES:
            raise ValueError(f"loss must be one of {sorted(_VALID_LOSSES)}, got {loss!r}")
        if receiver_conditioned != encoder.include_receiver:
            raise ValueError(
                "receiver_conditioned flag is inconsistent with the encoder: "
                f"flag={receiver_conditioned}, encoder.include_receiver="
                f"{encoder.include_receiver}"
            )
        self.encoder = encoder
        self.loss = loss
        self.receiver_conditioned = receiver_conditioned
        self._mu1: Any = None
        self._mu0: Any = None
        self._frozen = False
        self._training_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, examples: list[MatchedInterventionExample]) -> dict[str, Any]:
        """Train both potential-outcome heads on matched interventions.

        Invalid examples (None score on either branch) and self-transfer
        pairs are excluded (fail-closed, counted).

        Returns:
            Training statistics dict.
        """
        if self._frozen:
            raise RuntimeError("Critic is frozen; re-fitting is forbidden (Phase 17).")

        usable: list[MatchedInterventionExample] = []
        invalid_count = 0
        self_transfer_count = 0
        for ex in examples:
            if ex.source_agent_id == ex.receiver_id:
                self_transfer_count += 1
                continue
            if ex.official_expose_score is None or ex.official_withhold_score is None:
                invalid_count += 1
                continue
            usable.append(ex)

        if not usable:
            raise ValueError(
                "No usable training examples after exclusion "
                f"(invalid={invalid_count}, self_transfer={self_transfer_count})."
            )

        X = self.encoder.encode_batch([ex.features for ex in usable])
        y1 = np.array([ex.official_expose_score for ex in usable], dtype=float)
        y0 = np.array([ex.official_withhold_score for ex in usable], dtype=float)

        self._mu1 = self._make_head().fit(X, y1)
        self._mu0 = self._make_head().fit(X, y0)
        self._training_stats = {
            "n_examples_total": len(examples),
            "n_examples_used": len(usable),
            "invalid_excluded": invalid_count,
            "self_transfer_excluded": self_transfer_count,
            "loss": self.loss,
            "receiver_conditioned": self.receiver_conditioned,
        }
        return dict(self._training_stats)

    def _make_head(self) -> Any:
        if self.loss == "huber":
            return HuberRegressor(max_iter=500)
        return Ridge(alpha=1.0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_one(self, ex: MatchedInterventionExample) -> CriticPrediction:
        """Predict mu_expose, mu_withhold, tau_hat for one triple."""
        self._require_fitted()
        if ex.source_agent_id == ex.receiver_id:
            # Self-transfer excluded from critic evaluation/admission entirely.
            return CriticPrediction(
                memory_id=ex.memory_id,
                receiver_id=ex.receiver_id,
                task_id=ex.task_id,
                mu_expose=None,
                mu_withhold=None,
                tau_hat=None,
            )
        X = self.encoder.encode_one(ex.features)
        mu1 = float(self._mu1.predict(X)[0])
        mu0 = float(self._mu0.predict(X)[0])
        return CriticPrediction(
            memory_id=ex.memory_id,
            receiver_id=ex.receiver_id,
            task_id=ex.task_id,
            mu_expose=mu1,
            mu_withhold=mu0,
            tau_hat=mu1 - mu0,
        )

    def predict_batch(
        self, examples: list[MatchedInterventionExample]
    ) -> list[CriticPrediction]:
        return [self.predict_one(ex) for ex in examples]

    def _require_fitted(self) -> None:
        if self._mu1 is None or self._mu0 is None:
            raise RuntimeError("Critic is not fitted; call fit() first.")

    # ------------------------------------------------------------------
    # Freeze / checkpoint (Phase 17)
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        self._require_fitted()
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def dump_bytes(self) -> bytes:
        """Serialize critic + encoder + config to bytes."""
        self._require_fitted()
        buf = io.BytesIO()
        joblib.dump(
            {
                "mu1": self._mu1,
                "mu0": self._mu0,
                "encoder": self.encoder,
                "loss": self.loss,
                "receiver_conditioned": self.receiver_conditioned,
                "training_stats": self._training_stats,
                "frozen": self._frozen,
            },
            buf,
        )
        return buf.getvalue()

    def checkpoint_sha256(self) -> str:
        """Stable identifier of the critic checkpoint (Phase 17 audit)."""
        return hashlib.sha256(self.dump_bytes()).hexdigest()

    def save(self, path: str) -> str:
        data = self.dump_bytes()
        with open(path, "wb") as f:
            f.write(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path: str) -> "OfficialScoreTransferCritic":
        with open(path, "rb") as f:
            payload = joblib.load(io.BytesIO(f.read()))
        critic = cls(
            encoder=payload["encoder"],
            loss=payload["loss"],
            receiver_conditioned=payload["receiver_conditioned"],
        )
        critic._mu1 = payload["mu1"]
        critic._mu0 = payload["mu0"]
        critic._training_stats = payload.get("training_stats", {})
        critic._frozen = bool(payload.get("frozen", True))
        return critic
