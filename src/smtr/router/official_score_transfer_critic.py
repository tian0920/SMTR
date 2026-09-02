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
    "PotentialOutcomeMember",
    "TransferEffectDistribution",
    "BootstrapOfficialScoreTransferCritic",
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


# ---------------------------------------------------------------------------
# Bootstrap transfer distribution (RIMA-v2, §4-8)
# ---------------------------------------------------------------------------


@dataclass
class PotentialOutcomeMember:
    """One bootstrap member's trained potential-outcome models.

    Attributes:
        mu1_model: fitted model for E[Y(1) | h].
        mu0_model: fitted model for E[Y(0) | h].
    """

    mu1_model: Any
    mu0_model: Any


@dataclass(frozen=True)
class TransferEffectDistribution:
    """Bootstrap-aggregated transfer effect distribution for one triple.

    Attributes:
        memory_id: candidate memory.
        receiver_id: receiver agent.
        task_id: current task.
        mu_expose: mean of member mu1 predictions.
        mu_withhold: mean of member mu0 predictions.
        mu_tau: mean of member tau predictions (mu1 - mu0).
        sigma_tau: std of member tau predictions.
        n_members: number of bootstrap members.
    """

    memory_id: str
    receiver_id: str
    task_id: str

    mu_expose: float | None
    mu_withhold: float | None

    mu_tau: float | None
    sigma_tau: float | None

    n_members: int


class BootstrapOfficialScoreTransferCritic:
    """Bootstrap ensemble transfer critic producing (mu, sigma) predictions.

    Each bootstrap member is trained on a cluster-bootstrap sample (by
    task_id) of the training data, producing independent mu1/mu0 models.
    Predictions aggregate across members:

        mu_tau  = mean_b[ tau^(b) ]
        sigma_tau = std_b[ tau^(b) ]

    where tau^(b) = mu1^(b) - mu0^(b).

    Parameters:
        encoder: receiver-conditioned feature encoder.
        n_bootstrap: number of bootstrap members (default 31).
        seed: random seed for bootstrap sampling.
        loss: "huber" (recommended) or "mse".
        receiver_conditioned: whether receiver features are included.
    """

    _SCHEMA_VERSION = "rima_bootstrap_official_v1"

    def __init__(
        self,
        *,
        encoder: RimaFeatureEncoder,
        n_bootstrap: int = 31,
        seed: int = 0,
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
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.loss = loss
        self.receiver_conditioned = receiver_conditioned
        self.members: list[PotentialOutcomeMember] = []
        self._frozen = False
        self._training_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training (cluster bootstrap by task_id, §5)
    # ------------------------------------------------------------------
    def fit(self, examples: list[MatchedInterventionExample]) -> dict[str, Any]:
        """Train bootstrap ensemble on matched interventions.

        Cluster bootstrap: sample ``(task_id, receiver_id)`` families
        with replacement, then include all examples from each sampled
        family (§16.4 — dependency-compatible family). This ensures
        shared-control treatment edges within the same (task, receiver)
        are kept together and not treated as independent samples.

        Returns:
            Training statistics dict.
        """
        if self._frozen:
            raise RuntimeError("Critic is frozen; re-fitting is forbidden.")

        # Fail-closed filtering (same as point critic).
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

        # Group by (task_id, receiver_id) for cluster bootstrap (§16.4).
        by_family: dict[tuple[str, str], list[MatchedInterventionExample]] = {}
        for ex in usable:
            key = (ex.task_id, ex.receiver_id)
            by_family.setdefault(key, []).append(ex)
        family_keys = sorted(by_family.keys())

        rng = np.random.RandomState(self.seed)
        self.members = []

        for _b in range(self.n_bootstrap):
            # Sample (task_id, receiver_id) families with replacement.
            sampled_indices = rng.choice(
                len(family_keys), size=len(family_keys), replace=True
            )
            # Collect all examples from sampled families (with duplicates).
            sample_examples: list[MatchedInterventionExample] = []
            for idx in sampled_indices:
                sample_examples.extend(by_family[family_keys[idx]])

            X = self.encoder.encode_batch([ex.features for ex in sample_examples])
            y1 = np.array(
                [ex.official_expose_score for ex in sample_examples], dtype=float
            )
            y0 = np.array(
                [ex.official_withhold_score for ex in sample_examples], dtype=float
            )

            mu1_model = self._make_head().fit(X, y1)
            mu0_model = self._make_head().fit(X, y0)
            self.members.append(
                PotentialOutcomeMember(mu1_model=mu1_model, mu0_model=mu0_model)
            )

        self._training_stats = {
            "n_examples_total": len(examples),
            "n_examples_used": len(usable),
            "invalid_excluded": invalid_count,
            "self_transfer_excluded": self_transfer_count,
            "loss": self.loss,
            "receiver_conditioned": self.receiver_conditioned,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "bootstrap_cluster_unit": "(task_id, receiver_id)",
            "n_unique_families": len(family_keys),
        }
        return dict(self._training_stats)

    def _make_head(self) -> Any:
        if self.loss == "huber":
            return HuberRegressor(max_iter=500)
        return Ridge(alpha=1.0)

    # ------------------------------------------------------------------
    # Inference (§6)
    # ------------------------------------------------------------------
    def predict_distribution(
        self, ex: MatchedInterventionExample
    ) -> TransferEffectDistribution:
        """Predict bootstrap-aggregated transfer effect distribution."""
        self._require_fitted()
        if ex.source_agent_id == ex.receiver_id:
            return TransferEffectDistribution(
                memory_id=ex.memory_id,
                receiver_id=ex.receiver_id,
                task_id=ex.task_id,
                mu_expose=None,
                mu_withhold=None,
                mu_tau=None,
                sigma_tau=None,
                n_members=len(self.members),
            )

        X = self.encoder.encode_one(ex.features)
        mu1_preds = np.array(
            [float(m.mu1_model.predict(X)[0]) for m in self.members]
        )
        mu0_preds = np.array(
            [float(m.mu0_model.predict(X)[0]) for m in self.members]
        )
        tau_preds = mu1_preds - mu0_preds

        return TransferEffectDistribution(
            memory_id=ex.memory_id,
            receiver_id=ex.receiver_id,
            task_id=ex.task_id,
            mu_expose=float(mu1_preds.mean()),
            mu_withhold=float(mu0_preds.mean()),
            mu_tau=float(tau_preds.mean()),
            sigma_tau=float(tau_preds.std()),
            n_members=len(self.members),
        )

    def predict_one(self, ex: MatchedInterventionExample) -> CriticPrediction:
        """Compatibility API: tau_hat == distribution.mu_tau."""
        dist = self.predict_distribution(ex)
        return CriticPrediction(
            memory_id=ex.memory_id,
            receiver_id=ex.receiver_id,
            task_id=ex.task_id,
            mu_expose=dist.mu_expose,
            mu_withhold=dist.mu_withhold,
            tau_hat=dist.mu_tau,
        )

    def predict_batch(
        self, examples: list[MatchedInterventionExample]
    ) -> list[CriticPrediction]:
        return [self.predict_one(ex) for ex in examples]

    def _require_fitted(self) -> None:
        if not self.members:
            raise RuntimeError("Critic is not fitted; call fit() first.")

    # ------------------------------------------------------------------
    # Freeze / checkpoint (§8)
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
                "schema_version": self._SCHEMA_VERSION,
                "members": self.members,
                "encoder": self.encoder,
                "loss": self.loss,
                "receiver_conditioned": self.receiver_conditioned,
                "n_bootstrap": self.n_bootstrap,
                "seed": self.seed,
                "bootstrap_cluster_unit": "task_id",
                "training_stats": self._training_stats,
                "frozen": self._frozen,
            },
            buf,
        )
        return buf.getvalue()

    def checkpoint_sha256(self) -> str:
        """Stable identifier of the critic checkpoint."""
        return hashlib.sha256(self.dump_bytes()).hexdigest()

    def save(self, path: str) -> str:
        data = self.dump_bytes()
        with open(path, "wb") as f:
            f.write(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path: str) -> "BootstrapOfficialScoreTransferCritic":
        with open(path, "rb") as f:
            payload = joblib.load(io.BytesIO(f.read()))
        schema = payload.get("schema_version")
        if schema != cls._SCHEMA_VERSION:
            raise ValueError(
                f"Schema mismatch: expected {cls._SCHEMA_VERSION}, got {schema!r}"
            )
        critic = cls(
            encoder=payload["encoder"],
            n_bootstrap=payload["n_bootstrap"],
            seed=payload["seed"],
            loss=payload["loss"],
            receiver_conditioned=payload["receiver_conditioned"],
        )
        critic.members = payload["members"]
        critic._training_stats = payload.get("training_stats", {})
        critic._frozen = bool(payload.get("frozen", True))
        return critic


# ---------------------------------------------------------------------------
# Original point-estimate critic (unchanged, static RIMA baseline)
# ---------------------------------------------------------------------------


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
