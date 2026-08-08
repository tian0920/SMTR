"""Formal Protocol §17: routing rule depends only on tau_hat, eta_hat, epsilon*.

Formal routing must remain based only on:
- ``tau_hat > 0``
- ``eta_hat <= epsilon*``
- ``argmax tau_hat`` over eligible memories.

These tests assert that:
1. The eligible selection in the policy curve does not use confidence
   intervals, bootstrap variance, or uncertainty thresholds.
2. The router decision trace only records diagnostic quantities (never
   uses them for the decision).
3. The FourOutcomeTransferCritic ``adaptive_sampling_used`` and
   ``adaptive_stopping_used`` flags are False for formal checkpoints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


# Regex patterns for forbidden routing inputs.
FORBIDDEN_ROUTING_PATTERNS = re.compile(
    r"confidence_interval|bootstrap_variance|uncertainty_threshold"
    r"|confidence_threshold|entropy_threshold|active_acquisition"
    r"|adaptive_sampling|acquisition_weight|early_stopping",
    re.IGNORECASE,
)


class TestFormalRoutingRule:
    """Routing decision must depend only on tau_hat, eta_hat, epsilon*."""

    def test_eligible_selection_uses_only_tau_and_eta(self):
        """Inspect the eligible-selection code in metrics.py."""
        metrics_path = Path("src/smtr/evaluation/metrics.py")
        content = metrics_path.read_text(encoding="utf-8")

        # Find the eligible candidates selection block.
        match = re.search(
            r"eligible.*?if eligible:.*?Select exactly one memory.*?\n",
            content,
            re.DOTALL,
        )
        assert match is not None, "Could not find eligible selection block"
        block = match.group(0)

        # Only tau_hat and eta_hat should appear in the eligibility check.
        assert "tau_hat" in block
        assert "eta_hat" in block or "eta_calibrated" in block

        # Forbidden patterns must not appear in the selection block.
        assert not FORBIDDEN_ROUTING_PATTERNS.search(block)

    def test_critic_forbids_adaptive_flags_by_default(self):
        """FourOutcomeTransferCritic defaults to no adaptive sampling."""
        from smtr.router.transfer_critic import FourOutcomeTransferCritic

        critic = FourOutcomeTransferCritic(
            n_features=64, n_bootstrap=4, seed=0
        )
        assert critic.adaptive_sampling_used is False
        assert critic.adaptive_stopping_used is False

    def test_router_decision_trace_schema(self):
        """RouterDecision trace must not contain uncertainty-based fields
        that would indicate they influence the routing decision.
        Diagnostic-only fields are permitted."""
        from smtr.router.traces import RouterDecision

        # Inspect the schema fields for forbidden decision inputs.
        fields = RouterDecision.model_fields
        for field_name in fields:
            assert not FORBIDDEN_ROUTING_PATTERNS.search(field_name), (
                f"RouterDecision field {field_name!r} suggests "
                "uncertainty-based decision input"
            )
