"""Continual transfer learner with forward-only critic refit (§15).

Implements the ``rima_transfer_adaptive`` method that continually
improves the transfer critic by incorporating online causal evidence
collected from post-task probes.

The existing frozen critic (``rima_transfer_frozen``) remains as a
controlled-ablation baseline.

Forward-only invariant (§15.1):
    Task t's prediction may only use D_0 ∪ D^online_{<t}.
    Evidence from task t (D^online_t) must NEVER influence task t's
    own memory selection.

Refit strategy (§15.2):
    Periodic full refit — train a fresh bootstrap critic on
    ``base_training_examples + online_examples``. No incremental
    optimizer; Huber/Ridge training is lightweight.

Update interval (§15.3):
    Default ``refit_every_new_edges=5`` — only refit after accumulating
    5 new valid causal edges. Avoids per-task retraining.

Each refit (§15.4):
    1. ``training_data = base_examples + online_examples``
    2. Bootstrap cluster resample on full training data
    3. ``new_critic.fit(training_data)``
    4. ``new_critic.freeze()``
    5. ``current_critic = new_critic``
    6. ``critic_version += 1``

Leakage prevention (§15.5):
    Every prediction logs ``critic_version`` and
    ``critic_trained_through_task_position``.
    Invariant: ``critic_trained_through_task_position < current_task_position``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from smtr.rima.features import ReceiverConditionedTransferFeatures, RimaFeatureEncoder
from smtr.rima.online_transfer_evidence import OnlineTransferEvidence
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
    TransferEffectDistribution,
)

__all__ = [
    "ContinualTransferLearner",
    "CriticPredictionLog",
    "RIMA_TRANSFER_FROZEN",
    "RIMA_TRANSFER_ADAPTIVE",
]

#: Baseline method name — frozen critic, no online update.
RIMA_TRANSFER_FROZEN = "rima_transfer_frozen"

#: Adaptive method name — continual refit with online evidence.
RIMA_TRANSFER_ADAPTIVE = "rima_transfer_adaptive"

#: Default number of new causal edges before triggering a refit.
DEFAULT_REFIT_EVERY_NEW_EDGES = 5


@dataclass(frozen=True)
class CriticPredictionLog:
    """Audit metadata attached to every prediction (§15.5).

    Attributes:
        critic_version: monotonically increasing version counter.
        critic_trained_through_task_position: max task_position of
            online evidence incorporated in the current critic.
    """

    critic_version: int
    critic_trained_through_task_position: int


class ContinualTransferLearner:
    """Continual transfer learner with forward-only critic refit (§15).

    Maintains:
        - ``base_training_examples``: historical D_0 interventions.
        - ``online_examples``: accumulated D^online causal observations.
        - ``current_critic``: the active frozen bootstrap critic.
        - ``critic_version``: incremented on each refit.

    Args:
        base_examples: historical training interventions (D_0).
        encoder: feature encoder for the bootstrap critic.
        source_agent_ids: memory_id → source_agent_id mapping for
            self-transfer exclusion.
        n_bootstrap: number of bootstrap members (default 31).
        seed: random seed for bootstrap sampling.
        loss: ``"huber"`` or ``"mse"``.
        receiver_conditioned: whether to include receiver features.
        refit_every_new_edges: trigger refit after this many new
            valid causal edges (default 5).
    """

    def __init__(
        self,
        *,
        base_examples: list[MatchedInterventionExample],
        encoder: RimaFeatureEncoder,
        source_agent_ids: dict[str, str] | None = None,
        n_bootstrap: int = 31,
        seed: int = 0,
        loss: str = "huber",
        receiver_conditioned: bool = True,
        refit_every_new_edges: int = DEFAULT_REFIT_EVERY_NEW_EDGES,
    ) -> None:
        self.base_training_examples: list[MatchedInterventionExample] = list(
            base_examples
        )
        self.online_examples: list[MatchedInterventionExample] = []

        self._encoder = encoder
        self._source_agent_ids: dict[str, str] = dict(source_agent_ids or {})
        self._n_bootstrap = n_bootstrap
        self._seed = seed
        self._loss = loss
        self._receiver_conditioned = receiver_conditioned
        self._refit_every_new_edges = refit_every_new_edges

        # Initial critic: fitted on base examples only, then frozen.
        self.current_critic: BootstrapOfficialScoreTransferCritic | None = None
        self.critic_version: int = 0
        self._critic_trained_through_task_position: int = -1
        self._edges_since_last_refit: int = 0

        # Track the max task_position seen across online evidence.
        self._max_online_task_position: int = -1

        # Fit initial critic if base examples are available.
        if self.base_training_examples:
            self._fit_initial_critic()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _fit_initial_critic(self) -> None:
        """Fit and freeze the initial critic on base examples."""
        critic = self._make_critic()
        critic.fit(self.base_training_examples)
        critic.freeze()
        self.current_critic = critic
        self.critic_version = 1

    def _make_critic(self) -> BootstrapOfficialScoreTransferCritic:
        """Create a fresh (unfitted) bootstrap critic."""
        return BootstrapOfficialScoreTransferCritic(
            encoder=self._encoder,
            n_bootstrap=self._n_bootstrap,
            seed=self._seed,
            loss=self._loss,
            receiver_conditioned=self._receiver_conditioned,
        )

    # ------------------------------------------------------------------
    # Online evidence ingestion (§15.2)
    # ------------------------------------------------------------------

    def add_online_evidence(
        self,
        evidence: OnlineTransferEvidence,
        *,
        features: ReceiverConditionedTransferFeatures,
        source_agent_id: str | None = None,
    ) -> MatchedInterventionExample:
        """Convert online evidence to a training example and accumulate.

        Args:
            evidence: post-task causal probe result.
            features: receiver-conditioned features for this evidence.
            source_agent_id: agent that produced the memory. If None,
                looked up from ``source_agent_ids`` mapping.

        Returns:
            The created MatchedInterventionExample.
        """
        src_id = source_agent_id or self._source_agent_ids.get(
            evidence.memory_id, ""
        )

        # Aggregate per-seed scores into mean expose/withhold.
        expose_score = statistics.mean(evidence.expose_scores)
        withhold_score = statistics.mean(evidence.withhold_scores)

        example = MatchedInterventionExample(
            task_id=evidence.task_id,
            memory_id=evidence.memory_id,
            receiver_id=evidence.receiver_id,
            source_agent_id=src_id,
            official_expose_score=expose_score,
            official_withhold_score=withhold_score,
            features=features,
        )

        self.online_examples.append(example)
        self._edges_since_last_refit += 1
        self._max_online_task_position = max(
            self._max_online_task_position, evidence.task_position
        )

        return example

    # ------------------------------------------------------------------
    # Refit logic (§15.3 / §15.4)
    # ------------------------------------------------------------------

    def should_refit(self) -> bool:
        """Check whether enough new causal edges have accumulated."""
        return self._edges_since_last_refit >= self._refit_every_new_edges

    def maybe_refit(self) -> bool:
        """Refit if enough new causal edges have accumulated.

        Returns:
            True if a refit was performed, False otherwise.
        """
        if not self.should_refit():
            return False
        self._do_refit()
        return True

    def force_refit(self) -> None:
        """Force a refit regardless of the edge count threshold."""
        self._do_refit()

    def _do_refit(self) -> None:
        """Perform a full refit: base + online → new frozen critic (§15.4).

        Steps:
            1. Concatenate base_examples + online_examples.
            2. Create fresh bootstrap critic.
            3. Fit on full data (cluster resample by task_id).
            4. Freeze.
            5. Replace current_critic.
            6. Increment critic_version.
        """
        training_data = (
            self.base_training_examples + self.online_examples
        )

        new_critic = self._make_critic()
        new_critic.fit(training_data)
        new_critic.freeze()

        self.current_critic = new_critic
        self.critic_version += 1
        self._critic_trained_through_task_position = (
            self._max_online_task_position
        )
        self._edges_since_last_refit = 0

    # ------------------------------------------------------------------
    # Prediction with leakage guard (§15.5)
    # ------------------------------------------------------------------

    def predict_distribution(
        self,
        example: MatchedInterventionExample,
        current_task_position: int,
    ) -> tuple[TransferEffectDistribution, CriticPredictionLog]:
        """Predict with leakage-preventing assertion (§15.5).

        Raises:
            AssertionError: if the critic was trained on evidence from
                a task_position >= current_task_position.

        Returns:
            (distribution, prediction_log) tuple.
        """
        if self.current_critic is None:
            raise RuntimeError("No critic available; fit or refit first.")

        # Forward-only invariant: critic trained on <t, predicting at t.
        if self._critic_trained_through_task_position >= 0:
            assert (
                self._critic_trained_through_task_position
                < current_task_position
            ), (
                f"Leakage: critic trained through position "
                f"{self._critic_trained_through_task_position} "
                f"cannot predict at position {current_task_position}"
            )

        distribution = self.current_critic.predict_distribution(example)

        prediction_log = CriticPredictionLog(
            critic_version=self.critic_version,
            critic_trained_through_task_position=(
                self._critic_trained_through_task_position
            ),
        )

        return distribution, prediction_log

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_base_examples(self) -> int:
        return len(self.base_training_examples)

    @property
    def n_online_examples(self) -> int:
        return len(self.online_examples)

    @property
    def edges_since_last_refit(self) -> int:
        return self._edges_since_last_refit

    @property
    def critic_trained_through_task_position(self) -> int:
        return self._critic_trained_through_task_position
