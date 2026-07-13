"""Unified router factory for SMTR routing modes.

Provides a single entry point for constructing routers by mode:
- "no-memory": NoMemoryRouter (B0 baseline)
- "relevance-topk": RelevanceTopKRouter (B1 baseline)
- "learned": ProductionSequentialRouter with trained critic (M0)
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
from smtr.router.transfer_critic import FourOutcomeTransferCritic

RouterMode = Literal["no-memory", "relevance-topk", "learned"]


def build_router(
    mode: RouterMode,
    *,
    critic_checkpoint: str | Path | None = None,
    max_shares_per_invocation: int | None = None,
    seed: int = 0,
    critic_config: SequentialRouterConfig | None = None,
    feature_block: str | None = None,
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
        feature_block: Optional feature block name (e.g. "context_plus_candidate"
            for A1 ablation). If None, uses the checkpoint's default.

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
        critic = FourOutcomeTransferCritic.load(Path(critic_checkpoint))
        # Override feature_block if specified (for A1 ablation)
        if feature_block is not None:
            critic.encoder.feature_block = feature_block
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
            config=config,
            seed=seed,
        )

    raise ValueError(
        f"unknown router mode: {mode!r}; "
        f"expected one of: 'no-memory', 'relevance-topk', 'learned'"
    )
