"""Tests for formal and optional SMTR gate policies."""

import pytest

from smtr.evaluation.ablation_gates import EffectOnlyGate, FactualSuccessGate
from smtr.robust.config import RobustSMTRGateConfig
from smtr.robust.estimates import RobustTransferEstimate
from smtr.robust.robust_gate import RobustSMTRGate
from smtr.router.gate_protocol import TransferPointEstimate
from smtr.router.smtr_gate import SMTRGate, SMTRGateConfig


def test_smtr_gate_rejects_nonpositive_tau_mean():
    gate = SMTRGate(SMTRGateConfig(negative_risk_budget=0.2))
    decision = gate.decide(TransferPointEstimate(tau_mean=0.0, negative_risk_mean=0.0))
    assert decision.share is False
    assert decision.reason == "tau_mean_nonpositive"
    assert decision.effect_condition_passed is False


def test_smtr_gate_rejects_risk_over_budget():
    gate = SMTRGate(SMTRGateConfig(negative_risk_budget=0.2))
    decision = gate.decide(TransferPointEstimate(tau_mean=0.1, negative_risk_mean=0.21))
    assert decision.share is False
    assert decision.reason == "negative_risk_mean_exceeded"
    assert decision.risk_condition_passed is False


def test_smtr_gate_shares_when_effect_and_risk_pass():
    gate = SMTRGate(SMTRGateConfig(negative_risk_budget=0.2))
    decision = gate.decide(TransferPointEstimate(tau_mean=0.1, negative_risk_mean=0.2))
    assert decision.share is True
    assert decision.reason == "shared"
    assert decision.gate_name == "smtr_mean_effect_mean_risk"


def test_smtr_gate_rejects_invalid_budget_and_has_no_confidence():
    with pytest.raises(ValueError, match="negative_risk_budget"):
        SMTRGateConfig(negative_risk_budget=1.1)
    assert "confidence" not in SMTRGateConfig.__dataclass_fields__


def test_effect_only_smtr_ignores_risk():
    decision = EffectOnlyGate().decide(
        TransferPointEstimate(tau_mean=0.1, negative_risk_mean=1.0)
    )
    assert decision.share is True
    assert decision.risk_condition_passed is None
    assert decision.risk_condition_status == "not_applicable"


def test_factual_success_gate_uses_threshold_only():
    estimate = type("Estimate", (), {"p_share_success": 0.7})()
    decision = FactualSuccessGate(threshold=0.6).decide(estimate)
    assert decision.share is True
    assert decision.reason == "shared"
    assert decision.effect_condition_status == "not_applicable"
    assert decision.risk_condition_status == "not_applicable"


def test_robust_gate_uses_lcb_ucb_and_unified_confidence_config():
    config = RobustSMTRGateConfig(negative_risk_budget=0.2, confidence_level=0.9)
    gate = RobustSMTRGate(config)
    estimate = RobustTransferEstimate(
        tau_mean=0.5,
        tau_lcb=0.1,
        tau_ucb=0.7,
        negative_risk_mean=0.1,
        negative_risk_lcb=0.0,
        negative_risk_ucb=0.2,
        confidence_level=0.9,
    )
    assert gate.decide(estimate).share is True
    assert gate.decide(estimate).reason == "shared"
    assert "tau_confidence_level" not in RobustSMTRGateConfig.__dataclass_fields__
    assert "risk_confidence_level" not in RobustSMTRGateConfig.__dataclass_fields__


def test_robust_gate_rejects_nonpositive_lcb_and_unsafe_ucb():
    gate = RobustSMTRGate(RobustSMTRGateConfig(negative_risk_budget=0.2))
    low_tau = RobustTransferEstimate(
        tau_mean=0.5,
        tau_lcb=0.0,
        tau_ucb=0.7,
        negative_risk_mean=0.1,
        negative_risk_lcb=0.0,
        negative_risk_ucb=0.1,
        confidence_level=0.9,
    )
    high_risk = RobustTransferEstimate(
        tau_mean=0.5,
        tau_lcb=0.1,
        tau_ucb=0.7,
        negative_risk_mean=0.1,
        negative_risk_lcb=0.0,
        negative_risk_ucb=0.21,
        confidence_level=0.9,
    )
    assert gate.decide(low_tau).reason == "tau_lcb_nonpositive"
    assert gate.decide(high_risk).reason == "negative_risk_ucb_exceeded"
