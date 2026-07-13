"""Method registry for ablation comparison experiments.

Defines the canonical set of methods and their specifications.
Each method maps to a router class, feature block, share budget policy,
and gate policy. The registry is the single source of truth for method
configuration — no arbitrary string-to-method mapping is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MethodId = Literal[
    "b0_no_memory",
    "b1_top1",
    "b1_top3",
    "b1_matched",
    "a1_no_selected_set",
    "m0_full",
]

# Display labels for output
METHOD_DISPLAY: dict[str, str] = {
    "b0_no_memory": "B0",
    "b1_top1": "B1-Top1",
    "b1_top3": "B1-Top3",
    "b1_matched": "B1-Matched",
    "a1_no_selected_set": "A1-NoSet",
    "m0_full": "M0-Full",
}

# Reverse mapping: display label -> method_id
DISPLAY_TO_METHOD: dict[str, str] = {v: k for k, v in METHOD_DISPLAY.items()}

ALL_METHOD_IDS: tuple[str, ...] = (
    "b0_no_memory",
    "b1_top1",
    "b1_top3",
    "b1_matched",
    "a1_no_selected_set",
    "m0_full",
)


@dataclass(frozen=True)
class MethodSpec:
    """Complete specification for a comparison experiment method."""

    method_id: str
    display_label: str
    router_class: str
    critic_checkpoint: str | None = None
    feature_block: str | None = None
    share_budget_policy: str = "zero"
    gate_policy: str = "none"
    uses_selected_set: bool = False
    uses_pairwise_interactions: bool = False
    # For B1 variants: fixed max_shares_per_invocation
    fixed_max_shares: int | None = None
    # For B1-Matched: path to budget manifest JSON
    budget_manifest_path: str | None = None


def get_method_spec(method_id: str) -> MethodSpec:
    """Look up a method spec by ID. Raises ValueError for unknown IDs."""
    if method_id not in METHOD_REGISTRY:
        raise ValueError(
            f"unknown method_id: {method_id!r}; "
            f"expected one of: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[method_id]


def build_default_specs(
    *,
    critic_checkpoint: str | None = None,
    a1_checkpoint: str | None = None,
    budget_manifest_path: str | None = None,
    max_shares_per_invocation: int = 3,
) -> dict[str, MethodSpec]:
    """Build method specs with concrete checkpoint paths.

    Args:
        critic_checkpoint: Path to M0-Full critic checkpoint.
        a1_checkpoint: Path to A1-NoSet critic checkpoint (if different from M0).
        budget_manifest_path: Path to B1-Matched budget manifest JSON.
        max_shares_per_invocation: Default max shares for M0-Full.

    Returns:
        Dict of method_id -> MethodSpec with concrete paths.
    """
    a1_ckpt = a1_checkpoint or critic_checkpoint
    return {
        "b0_no_memory": MethodSpec(
            method_id="b0_no_memory",
            display_label="B0",
            router_class="NoMemoryRouter",
            share_budget_policy="zero",
            gate_policy="none",
        ),
        "b1_top1": MethodSpec(
            method_id="b1_top1",
            display_label="B1-Top1",
            router_class="RelevanceTopKRouter",
            share_budget_policy="fixed_1",
            gate_policy="none",
            fixed_max_shares=1,
        ),
        "b1_top3": MethodSpec(
            method_id="b1_top3",
            display_label="B1-Top3",
            router_class="RelevanceTopKRouter",
            share_budget_policy="fixed_3",
            gate_policy="none",
            fixed_max_shares=3,
        ),
        "b1_matched": MethodSpec(
            method_id="b1_matched",
            display_label="B1-Matched",
            router_class="RelevanceTopKRouter",
            share_budget_policy="validation_matched",
            gate_policy="none",
            budget_manifest_path=budget_manifest_path,
        ),
        "a1_no_selected_set": MethodSpec(
            method_id="a1_no_selected_set",
            display_label="A1-NoSet",
            router_class="ProductionSequentialRouter",
            critic_checkpoint=a1_ckpt,
            feature_block="context_plus_candidate",
            share_budget_policy=f"fixed_{max_shares_per_invocation}",
            gate_policy="strict_lcb_ucb",
            uses_selected_set=False,
            uses_pairwise_interactions=False,
            fixed_max_shares=max_shares_per_invocation,
        ),
        "m0_full": MethodSpec(
            method_id="m0_full",
            display_label="M0-Full",
            router_class="ProductionSequentialRouter",
            critic_checkpoint=critic_checkpoint,
            feature_block="full",
            share_budget_policy=f"fixed_{max_shares_per_invocation}",
            gate_policy="strict_lcb_ucb",
            uses_selected_set=True,
            uses_pairwise_interactions=True,
            fixed_max_shares=max_shares_per_invocation,
        ),
    }


# Default registry (without concrete checkpoint paths).
# Use build_default_specs() to get specs with paths filled in.
METHOD_REGISTRY: dict[str, MethodSpec] = {
    "b0_no_memory": MethodSpec(
        method_id="b0_no_memory",
        display_label="B0",
        router_class="NoMemoryRouter",
        share_budget_policy="zero",
        gate_policy="none",
    ),
    "b1_top1": MethodSpec(
        method_id="b1_top1",
        display_label="B1-Top1",
        router_class="RelevanceTopKRouter",
        share_budget_policy="fixed_1",
        gate_policy="none",
        fixed_max_shares=1,
    ),
    "b1_top3": MethodSpec(
        method_id="b1_top3",
        display_label="B1-Top3",
        router_class="RelevanceTopKRouter",
        share_budget_policy="fixed_3",
        gate_policy="none",
        fixed_max_shares=3,
    ),
    "b1_matched": MethodSpec(
        method_id="b1_matched",
        display_label="B1-Matched",
        router_class="RelevanceTopKRouter",
        share_budget_policy="validation_matched",
        gate_policy="none",
    ),
    "a1_no_selected_set": MethodSpec(
        method_id="a1_no_selected_set",
        display_label="A1-NoSet",
        router_class="ProductionSequentialRouter",
        feature_block="context_plus_candidate",
        share_budget_policy="fixed_3",
        gate_policy="strict_lcb_ucb",
        uses_selected_set=False,
        uses_pairwise_interactions=False,
    ),
    "m0_full": MethodSpec(
        method_id="m0_full",
        display_label="M0-Full",
        router_class="ProductionSequentialRouter",
        feature_block="full",
        share_budget_policy="fixed_3",
        gate_policy="strict_lcb_ucb",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
    ),
}
