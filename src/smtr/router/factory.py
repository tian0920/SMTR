"""Unified router factory for SMTR routing modes.

Provides entry points for constructing formal SMTR routers by mode:
- "no-memory": NoMemoryRouter (B0 baseline)
- "relevance-topk": RelevanceTopKRouter (B1 baseline)
- "learned": ProductionSequentialRouter with trained critic (SMTR)
"""

from pathlib import Path
from typing import Literal

from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.baselines import RelevanceTopKRouter, RelevanceTopKRouterConfig
from smtr.router.interfaces import MemoryRouter
from smtr.router.sequential_router import (
    ProductionSequentialRouter,
    SequentialRouterConfig,
)
from smtr.router.smtr_gate import SMTRGate, SMTRGateConfig
from smtr.router.transfer_critic import FourOutcomeTransferCritic

RouterMode = Literal["no-memory", "relevance-topk", "learned"]


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be used for the requested router."""


def _validate_feature_block(
    critic: FourOutcomeTransferCritic,
    expected_feature_block: str | None,
) -> None:
    if expected_feature_block is None:
        return
    actual = getattr(critic.encoder, "feature_block", None)
    if actual != expected_feature_block:
        raise CheckpointCompatibilityError(
            "critic checkpoint feature_block mismatch: "
            f"expected {expected_feature_block!r}, got {actual!r}"
        )


def build_router(
    mode: RouterMode,
    *,
    critic_checkpoint: str | Path | None = None,
    max_shares_per_invocation: int | None = None,
    seed: int = 0,
    critic_config: SequentialRouterConfig | None = None,
    expected_feature_block: str | None = None,
    negative_risk_budget: float = 0.2,
) -> MemoryRouter:
    """Build a router by mode.

    Args:
        mode: Router mode — "no-memory", "relevance-topk", or "learned".
        critic_checkpoint: Path to critic checkpoint (required for "learned" mode).
        max_shares_per_invocation: Max memories to share per invocation.
            Applies to both B1 (RelevanceTopKRouter) and M0 (ProductionSequentialRouter).
        seed: Traversal seed for deterministic routing. Passed to routers that
            support it (e.g., ProductionSequentialRouter).
        critic_config: Optional SequentialRouterConfig for "learned" mode.
        expected_feature_block: Optional feature block name to validate. The
            checkpoint is never mutated after loading.

    Returns:
        A MemoryRouter instance.

    Raises:
        ValueError: If mode is unknown, or "learned" mode lacks a checkpoint,
            or max_shares_per_invocation is negative.
    """
    if max_shares_per_invocation is not None and max_shares_per_invocation < 0:
        raise ValueError(
            f"max_shares_per_invocation must be >= 0, got {max_shares_per_invocation}"
        )

    if mode == "no-memory":
        return NoMemoryRouter()

    if mode == "relevance-topk":
        return RelevanceTopKRouter(
            config=RelevanceTopKRouterConfig(
                max_shares_per_invocation=max_shares_per_invocation,
            )
        )

    if mode == "learned":
        if critic_checkpoint is None:
            raise ValueError(
                "learned mode requires a critic_checkpoint path; "
                "cannot fall back to no-memory silently"
            )
        critic = FourOutcomeTransferCritic.load(
            Path(critic_checkpoint),
            require_metadata=expected_feature_block is not None,
        )
        _validate_feature_block(critic, expected_feature_block)
        config = critic_config or SequentialRouterConfig()
        # Apply max_shares_per_invocation to M0 if specified
        if max_shares_per_invocation is not None:
            config = SequentialRouterConfig(
                **{
                    **config.model_dump(),
                    "max_shares_per_invocation": max_shares_per_invocation,
                }
            )
        return ProductionSequentialRouter(
            critic=critic,
            gate=SMTRGate(SMTRGateConfig(negative_risk_budget=negative_risk_budget)),
            config=config,
            seed=seed,
        )

    raise ValueError(
        f"unknown router mode: {mode!r}; "
        f"expected one of: 'no-memory', 'relevance-topk', 'learned'"
    )


def load_learned_router(
    *,
    critic_checkpoint: str | Path,
    expected_feature_block: str,
    config: SequentialRouterConfig,
    seed: int,
    negative_risk_budget: float = 0.2,
) -> ProductionSequentialRouter:
    """Load a learned router after validating checkpoint compatibility."""
    router = build_router(
        mode="learned",
        critic_checkpoint=critic_checkpoint,
        expected_feature_block=expected_feature_block,
        critic_config=config,
        seed=seed,
        negative_risk_budget=negative_risk_budget,
    )
    if not isinstance(router, ProductionSequentialRouter):
        raise TypeError("expected ProductionSequentialRouter")
    return router


def build_smtr_router(
    *,
    critic_checkpoint: str | Path,
    negative_risk_budget: float,
    max_shares_per_invocation: int | None,
    seed: int,
) -> ProductionSequentialRouter:
    """Build the formal SMTR router without importing robust extensions."""
    router = build_router(
        mode="learned",
        critic_checkpoint=critic_checkpoint,
        expected_feature_block="full",
        max_shares_per_invocation=max_shares_per_invocation,
        seed=seed,
        negative_risk_budget=negative_risk_budget,
    )
    if not isinstance(router, ProductionSequentialRouter):
        raise TypeError("expected ProductionSequentialRouter")
    return router


def smtr_router_observability(
    *,
    router: ProductionSequentialRouter,
    critic_checkpoint_digest: str,
) -> dict[str, object]:
    """Return formal SMTR router observability fields."""
    feature_block = getattr(router.critic.encoder, "feature_block", None)
    gate_config = getattr(router.gate, "config", None)
    return {
        "router_class": router.__class__.__name__,
        "gate_name": router.gate.gate_name,
        "negative_risk_budget": getattr(gate_config, "negative_risk_budget", None),
        "critic_checkpoint_digest": critic_checkpoint_digest,
        "feature_block": feature_block,
        "max_shares_per_invocation": router.config.max_shares_per_invocation,
    }
