"""Factory for explicitly constructed Robust-SMTR routers."""

from __future__ import annotations

from pathlib import Path

from smtr.robust.config import RobustSMTRGateConfig
from smtr.robust.robust_gate import RobustSMTRGate
from smtr.robust.uncertainty import summarize_member_predictions
from smtr.router.factory import CheckpointCompatibilityError, _validate_feature_block
from smtr.router.sequential_router import (
    ProductionSequentialRouter,
    SequentialRouterConfig,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import TransferPredictionInput


class RobustEstimateCriticAdapter:
    """Adapter that makes member predictions explicit for Robust-SMTR."""

    def __init__(
        self,
        critic: FourOutcomeTransferCritic,
        *,
        confidence_level: float,
    ) -> None:
        self.critic = critic
        self.confidence_level = confidence_level

    def predict_point(self, item: TransferPredictionInput):
        member_predictions = self.critic.predict_members(item)
        if len(member_predictions.probabilities) < 2:
            raise ValueError("Robust-SMTR requires at least two ensemble members")
        return summarize_member_predictions(
            member_predictions,
            confidence_level=self.confidence_level,
        )


def build_robust_smtr_router(
    *,
    critic_checkpoint: str | Path,
    negative_risk_budget: float,
    confidence_level: float,
    max_shares_per_invocation: int | None,
    seed: int,
) -> ProductionSequentialRouter:
    critic = FourOutcomeTransferCritic.load(Path(critic_checkpoint), require_metadata=True)
    _validate_feature_block(critic, "full")
    metadata = critic.checkpoint_metadata
    if metadata is not None and not metadata.supports_member_predictions:
        raise CheckpointCompatibilityError(
            "Robust-SMTR requires checkpoint member predictions"
        )
    if len(critic.models) < 2:
        raise CheckpointCompatibilityError(
            "Robust-SMTR requires at least two ensemble members"
        )
    gate_config = RobustSMTRGateConfig(
        negative_risk_budget=negative_risk_budget,
        confidence_level=confidence_level,
    )
    return ProductionSequentialRouter(
        critic=RobustEstimateCriticAdapter(
            critic,
            confidence_level=gate_config.confidence_level,
        ),
        gate=RobustSMTRGate(gate_config),
        config=SequentialRouterConfig(
            max_shares_per_invocation=max_shares_per_invocation
        ),
        seed=seed,
    )
