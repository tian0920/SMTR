"""Method registry for formal SMTR comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MethodId = Literal[
    "b0_no_memory",
    "b1_top1",
    "b1_top3",
    "b1_matched",
    "smtr",
    "effect_only_smtr",
]

METHOD_DISPLAY: dict[str, str] = {
    "b0_no_memory": "B0",
    "b1_top1": "B1-Top1",
    "b1_top3": "B1-Top3",
    "b1_matched": "B1-Matched",
    "smtr": "SMTR",
    "effect_only_smtr": "EffectOnly-SMTR",
}

DISPLAY_TO_METHOD: dict[str, str] = {v: k for k, v in METHOD_DISPLAY.items()}


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
    fixed_max_shares: int | None = None
    budget_manifest_path: str | None = None
    gate_name: str | None = None
    robust_extension: bool = False


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "b0_no_memory": MethodSpec(
        method_id="b0_no_memory",
        display_label="B0",
        router_class="NoMemoryRouter",
        share_budget_policy="zero",
    ),
    "b1_top1": MethodSpec(
        method_id="b1_top1",
        display_label="B1-Top1",
        router_class="RelevanceTopKRouter",
        share_budget_policy="fixed_1",
        fixed_max_shares=1,
    ),
    "b1_top3": MethodSpec(
        method_id="b1_top3",
        display_label="B1-Top3",
        router_class="RelevanceTopKRouter",
        share_budget_policy="fixed_3",
        fixed_max_shares=3,
    ),
    "b1_matched": MethodSpec(
        method_id="b1_matched",
        display_label="B1-Matched",
        router_class="RelevanceTopKRouter",
        share_budget_policy="validation_matched",
    ),
    "smtr": MethodSpec(
        method_id="smtr",
        display_label="SMTR",
        router_class="ProductionSequentialRouter",
        feature_block="full",
        share_budget_policy="fixed_3",
        gate_policy="smtr_mean_effect_mean_risk",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        fixed_max_shares=3,
        gate_name="smtr_mean_effect_mean_risk",
    ),
}

ABLATION_METHODS: dict[str, MethodSpec] = {
    "effect_only_smtr": MethodSpec(
        method_id="effect_only_smtr",
        display_label="EffectOnly-SMTR",
        router_class="ProductionSequentialRouter",
        feature_block="full",
        share_budget_policy="fixed_3",
        gate_policy="effect_only_smtr",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        fixed_max_shares=3,
        gate_name="effect_only_smtr",
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
    a1_checkpoint: str | None = None,
    budget_manifest_path: str | None = None,
    max_shares_per_invocation: int = 3,
    include_ablations: bool = False,
) -> dict[str, MethodSpec]:
    """Build method specs with concrete checkpoint paths."""
    del a1_checkpoint
    specs = {
        key: _with_paths(
            spec,
            critic_checkpoint=critic_checkpoint,
            budget_manifest_path=budget_manifest_path,
            max_shares_per_invocation=max_shares_per_invocation,
        )
        for key, spec in _registry(include_ablations=include_ablations).items()
    }
    return specs


def _registry(*, include_ablations: bool) -> dict[str, MethodSpec]:
    if not include_ablations:
        return dict(METHOD_REGISTRY)
    return {**METHOD_REGISTRY, **ABLATION_METHODS}


def _with_paths(
    spec: MethodSpec,
    *,
    critic_checkpoint: str | None,
    budget_manifest_path: str | None,
    max_shares_per_invocation: int,
) -> MethodSpec:
    if spec.display_label == "B1-Matched":
        return MethodSpec(
            **{
                **spec.__dict__,
                "budget_manifest_path": budget_manifest_path,
            }
        )
    if spec.router_class != "ProductionSequentialRouter":
        return spec
    return MethodSpec(
        **{
            **spec.__dict__,
            "critic_checkpoint": critic_checkpoint,
            "fixed_max_shares": max_shares_per_invocation,
            "share_budget_policy": f"fixed_{max_shares_per_invocation}",
        }
    )
