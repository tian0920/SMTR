"""Method registry for formal SMTR comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MethodId = Literal[
    "b0_no_memory",
    "b1_top1",
    "b1_all_candidates",
    "b1_matched",
    "smtr",
    "effect_only_smtr",
    "static_smtr",
    "factual_success_smtr",
]

METHOD_DISPLAY: dict[str, str] = {
    "b0_no_memory": "B0",
    "b1_top1": "B1-Top1",
    "b1_all_candidates": "B1-AllCandidates",
    "b1_matched": "B1-Matched",
    "smtr": "SMTR",
    "effect_only_smtr": "EffectOnly-SMTR",
    "static_smtr": "Static-SMTR",
    "factual_success_smtr": "FactualSuccess-SMTR",
}

DISPLAY_TO_METHOD: dict[str, str] = {v: k for k, v in METHOD_DISPLAY.items()}


@dataclass(frozen=True)
class MethodSpec:
    """Complete specification for a comparison experiment method."""

    method_id: str
    display_label: str
    router_class: str
    router_type: str = "baseline"
    checkpoint_path: str | None = None
    critic_checkpoint: str | None = None
    feature_block: str | None = None
    share_budget_policy: str = "zero"
    gate_policy: str = "none"
    uses_selected_set: bool = False
    uses_pairwise_interactions: bool = False
    fixed_max_shares: int | None = None
    budget_manifest_path: str | None = None
    gate_name: str | None = None
    conditioning_policy_name: str | None = None
    traversal_policy_name: str | None = None
    negative_risk_budget: float | None = None
    factual_success_threshold: float | None = None
    uses_effect_condition: bool | None = None
    uses_risk_condition: bool | None = None
    uses_dynamic_selected_set: bool | None = None
    uses_pairwise_counterfactual_labels: bool | None = None
    target: str | None = None
    is_ablation: bool = False
    robust_extension: bool = False
    is_robust_extension: bool = False


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "b0_no_memory": MethodSpec(
        method_id="b0_no_memory",
        display_label="B0",
        router_class="NoMemoryRouter",
        router_type="no-memory",
        share_budget_policy="zero",
    ),
    "b1_top1": MethodSpec(
        method_id="b1_top1",
        display_label="B1-Top1",
        router_class="RelevanceTopKRouter",
        router_type="relevance-topk",
        share_budget_policy="fixed_1",
        fixed_max_shares=1,
    ),
    "b1_all_candidates": MethodSpec(
        method_id="b1_all_candidates",
        display_label="B1-AllCandidates",
        router_class="RelevanceTopKRouter",
        router_type="relevance-topk",
        share_budget_policy="all_candidates",
        fixed_max_shares=None,
    ),
    "b1_matched": MethodSpec(
        method_id="b1_matched",
        display_label="B1-Matched",
        router_class="RelevanceTopKRouter",
        router_type="relevance-topk",
        share_budget_policy="validation_matched",
    ),
    "smtr": MethodSpec(
        method_id="smtr",
        display_label="SMTR",
        router_class="ProductionSequentialRouter",
        router_type="learned",
        feature_block="full",
        share_budget_policy="none",
        gate_policy="smtr_mean_effect_mean_risk",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        uses_pairwise_counterfactual_labels=True,
        fixed_max_shares=None,
        gate_name="smtr_mean_effect_mean_risk",
        conditioning_policy_name="dynamic_selected_set",
        traversal_policy_name="random_order",
        uses_effect_condition=True,
        uses_risk_condition=True,
        uses_dynamic_selected_set=True,
        is_ablation=False,
        is_robust_extension=False,
    ),
}

ABLATION_METHODS: dict[str, MethodSpec] = {
    "effect_only_smtr": MethodSpec(
        method_id="effect_only_smtr",
        display_label="EffectOnly-SMTR",
        router_class="ProductionSequentialRouter",
        router_type="learned",
        feature_block="full",
        share_budget_policy="none",
        gate_policy="effect_only_smtr",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        fixed_max_shares=None,
        gate_name="effect_only_smtr",
        conditioning_policy_name="dynamic_selected_set",
        traversal_policy_name="random_order",
        uses_effect_condition=True,
        uses_risk_condition=False,
        uses_dynamic_selected_set=True,
        is_ablation=True,
    ),
    "static_smtr": MethodSpec(
        method_id="static_smtr",
        display_label="Static-SMTR",
        router_class="ProductionSequentialRouter",
        router_type="learned",
        feature_block="full",
        share_budget_policy="none",
        gate_policy="smtr_mean_effect_mean_risk",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        uses_pairwise_counterfactual_labels=True,
        fixed_max_shares=None,
        gate_name="smtr_mean_effect_mean_risk",
        conditioning_policy_name="frozen_initial_selected_set",
        traversal_policy_name="random_order",
        uses_effect_condition=True,
        uses_risk_condition=True,
        uses_dynamic_selected_set=False,
        is_ablation=True,
    ),
    "factual_success_smtr": MethodSpec(
        method_id="factual_success_smtr",
        display_label="FactualSuccess-SMTR",
        router_class="ProductionSequentialRouter",
        router_type="learned",
        feature_block="full",
        share_budget_policy="none",
        gate_policy="factual_success_smtr",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        uses_pairwise_counterfactual_labels=False,
        target="share_success",
        gate_name="factual_success_smtr",
        conditioning_policy_name="dynamic_selected_set",
        traversal_policy_name="random_order",
        uses_effect_condition=False,
        uses_risk_condition=False,
        uses_dynamic_selected_set=True,
        is_ablation=True,
    ),
}

ALL_METHOD_IDS: tuple[str, ...] = tuple(METHOD_REGISTRY.keys())
ALL_EXPERIMENT_METHOD_IDS: tuple[str, ...] = tuple(
    [*METHOD_REGISTRY.keys(), *ABLATION_METHODS.keys()]
)


def get_method_spec(method_id: str, *, include_ablations: bool = False) -> MethodSpec:
    """Look up a method spec by ID. Raises ValueError for unknown IDs."""
    registry = _registry(include_ablations=include_ablations)
    if method_id not in registry:
        raise ValueError(
            f"unknown method_id: {method_id!r}; expected one of: {list(registry)}"
        )
    return registry[method_id]


def build_default_specs(
    *,
    critic_checkpoint: str | None = None,
    factual_success_checkpoint: str | None = None,
    budget_manifest_path: str | None = None,
    max_shares_per_invocation: int | None = None,
    negative_risk_budget: float = 0.2,
    include_ablations: bool = False,
) -> dict[str, MethodSpec]:
    """Build method specs with concrete checkpoint paths."""
    return {
        key: _with_paths(
            spec,
            critic_checkpoint=critic_checkpoint,
            factual_success_checkpoint=factual_success_checkpoint,
            budget_manifest_path=budget_manifest_path,
            max_shares_per_invocation=max_shares_per_invocation,
            negative_risk_budget=negative_risk_budget,
        )
        for key, spec in _registry(include_ablations=include_ablations).items()
    }


def _registry(*, include_ablations: bool) -> dict[str, MethodSpec]:
    if not include_ablations:
        return dict(METHOD_REGISTRY)
    return {**METHOD_REGISTRY, **ABLATION_METHODS}


def _with_paths(
    spec: MethodSpec,
    *,
    critic_checkpoint: str | None,
    factual_success_checkpoint: str | None,
    budget_manifest_path: str | None,
    max_shares_per_invocation: int | None,
    negative_risk_budget: float,
) -> MethodSpec:
    values = spec.__dict__.copy()
    if spec.display_label == "B1-Matched":
        values["budget_manifest_path"] = budget_manifest_path
    if spec.router_class == "ProductionSequentialRouter":
        checkpoint = (
            factual_success_checkpoint
            if spec.display_label == "FactualSuccess-SMTR"
            else critic_checkpoint
        )
        values.update(
            {
                "critic_checkpoint": checkpoint,
                "checkpoint_path": checkpoint,
                "fixed_max_shares": None,
                "share_budget_policy": "none",
                "negative_risk_budget": negative_risk_budget,
            }
        )
    return MethodSpec(**values)
