"""Method registry for explicitly requested Robust-SMTR experiments."""

from __future__ import annotations

from smtr.experiment.methods import MethodSpec

ROBUST_METHODS: dict[str, MethodSpec] = {
    "robust_smtr": MethodSpec(
        method_id="robust_smtr",
        display_label="Robust-SMTR",
        router_class="ProductionSequentialRouter",
        feature_block="full",
        share_budget_policy="fixed_3",
        gate_policy="robust_smtr_lcb_ucb",
        uses_selected_set=True,
        uses_pairwise_interactions=True,
        gate_name="robust_smtr_lcb_ucb",
    )
}
