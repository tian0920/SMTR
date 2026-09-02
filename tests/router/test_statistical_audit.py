"""Tests for uncertainty and gamma statistical audit (§16).

Covers:
    1. BETA_LABEL constant
    2. UNCERTAINTY_UNCALIBRATED constant
    3. Coverage audit (calibrated & uncalibrated)
    4. Gamma report computation
    5. Gamma report per-scenario diagnostics
    6. Gamma report to_dict serialization
    7. Warning on uncalibrated uncertainty
    8. Edge cases (no calibration data, single point)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from smtr.rima.features import (
    ReceiverConditionedTransferFeatures,
    RimaFeatureEncoder,
)
from smtr.rima.statistical_audit import (
    BETA_LABEL,
    UNCERTAINTY_UNCALIBRATED,
    GammaReport,
    UncertaintyAuditReport,
    audit_coverage,
    compute_gamma_report,
)
from smtr.router.official_score_transfer_critic import (
    BootstrapOfficialScoreTransferCritic,
    MatchedInterventionExample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    task_id: str = "t1",
    memory_id: str = "m1",
    receiver_id: str = "r1",
) -> ReceiverConditionedTransferFeatures:
    return ReceiverConditionedTransferFeatures(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        task_repr={"scenario": "test", "task_type": "test", "text": ""},
        receiver_repr={"role": "executor", "capabilities": []},
        routing_card={
            "goal_summary": "",
            "task_tags": ["test"],
            "precondition_summary": "",
            "compatible_receiver_roles": [],
            "compatible_receiver_capabilities": [],
            "procedure_type": "experience",
        },
    )


def _make_example(
    task_id: str,
    memory_id: str,
    receiver_id: str,
    expose: float | None,
    withhold: float | None,
    source_agent_id: str = "src",
    scenario: str = "test",
) -> MatchedInterventionExample:
    return MatchedInterventionExample(
        task_id=task_id,
        memory_id=memory_id,
        receiver_id=receiver_id,
        source_agent_id=source_agent_id,
        official_expose_score=expose,
        official_withhold_score=withhold,
        features=ReceiverConditionedTransferFeatures(
            task_id=task_id,
            memory_id=memory_id,
            receiver_id=receiver_id,
            task_repr={"scenario": scenario, "task_type": "test", "text": ""},
            receiver_repr={"role": "executor", "capabilities": []},
            routing_card={
                "goal_summary": "",
                "task_tags": ["test"],
                "precondition_summary": "",
                "compatible_receiver_roles": [],
                "compatible_receiver_capabilities": [],
                "procedure_type": "experience",
            },
        ),
    )


def _fit_critic(
    examples: list[MatchedInterventionExample],
    n_bootstrap: int = 5,
) -> BootstrapOfficialScoreTransferCritic:
    encoder = RimaFeatureEncoder(n_features=64, include_receiver=True)
    critic = BootstrapOfficialScoreTransferCritic(
        encoder=encoder, n_bootstrap=n_bootstrap, seed=0
    )
    critic.fit(examples)
    return critic


def _make_training_examples(
    n_tasks: int = 6,
    n_per_task: int = 2,
) -> list[MatchedInterventionExample]:
    """Create training examples with positive tau."""
    examples = []
    for t in range(n_tasks):
        for i in range(n_per_task):
            examples.append(
                _make_example(
                    task_id=f"task_{t}",
                    memory_id=f"mem_{t}_{i}",
                    receiver_id=f"recv_{i}",
                    source_agent_id=f"src_{t}_{i}",
                    expose=0.6 + 0.05 * i + 0.02 * t,
                    withhold=0.3 + 0.02 * i,
                )
            )
    return examples


# ---------------------------------------------------------------------------
# Test: constants
# ---------------------------------------------------------------------------


def test_beta_label():
    """Beta label must be 'conservative uncertainty coefficient'."""
    assert BETA_LABEL == "conservative uncertainty coefficient"


def test_uncertainty_uncalibrated_constant():
    """UNCERTAINTY_UNCALIBRATED warning code must be defined."""
    assert UNCERTAINTY_UNCALIBRATED == "UNCERTAINTY_UNCALIBRATED"


# ---------------------------------------------------------------------------
# Test: coverage audit
# ---------------------------------------------------------------------------


def test_audit_coverage_basic():
    """audit_coverage must return valid UncertaintyAuditReport."""
    train = _make_training_examples()
    critic = _fit_critic(train, n_bootstrap=5)

    # Use different examples for calibration (held-out)
    cal = [
        _make_example(
            task_id="cal_t0",
            memory_id="cal_m0",
            receiver_id="recv_0",
            expose=0.7,
            withhold=0.4,
        ),
        _make_example(
            task_id="cal_t1",
            memory_id="cal_m1",
            receiver_id="recv_1",
            expose=0.8,
            withhold=0.3,
        ),
    ]

    report = audit_coverage(critic, cal, beta=1.64)

    assert isinstance(report, UncertaintyAuditReport)
    assert 0.0 <= report.lcb_empirical_coverage <= 1.0
    assert report.n_calibration_edges == 2
    assert report.mean_sigma >= 0
    assert report.median_sigma >= 0


def test_audit_coverage_empty_calibration():
    """Empty calibration data must return zeroed report."""
    train = _make_training_examples()
    critic = _fit_critic(train)

    report = audit_coverage(critic, [], beta=1.64)
    assert report.n_calibration_edges == 0
    assert report.lcb_empirical_coverage == 0.0
    assert report.sigma_abs_error_correlation is None
    assert report.uncertainty_calibrated is False


def test_audit_coverage_skips_self_transfer():
    """Self-transfer examples must be excluded from calibration."""
    train = _make_training_examples()
    critic = _fit_critic(train)

    cal = [
        _make_example(
            task_id="t1",
            memory_id="m1",
            receiver_id="r1",
            expose=0.8,
            withhold=0.5,
            source_agent_id="r1",  # self-transfer
        ),
    ]

    report = audit_coverage(critic, cal)
    assert report.n_calibration_edges == 0


def test_audit_coverage_skips_invalid_scores():
    """Examples with None scores must be excluded."""
    train = _make_training_examples()
    critic = _fit_critic(train)

    cal = [
        _make_example(
            task_id="t1",
            memory_id="m1",
            receiver_id="r1",
            expose=None,
            withhold=0.5,
        ),
    ]

    report = audit_coverage(critic, cal)
    assert report.n_calibration_edges == 0


def test_uncalibrated_emits_warning():
    """Uncalibrated sigma must emit UNCERTAINTY_UNCALIBRATED warning."""
    train = _make_training_examples()
    critic = _fit_critic(train, n_bootstrap=3)

    # Create calibration examples where sigma might not correlate
    cal = [
        _make_example(
            task_id=f"cal_{i}",
            memory_id=f"cal_m_{i}",
            receiver_id=f"recv_{i % 2}",
            expose=0.5 + 0.01 * i,
            withhold=0.4,
        )
        for i in range(5)
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        report = audit_coverage(critic, cal, beta=1.64)

        if not report.uncertainty_calibrated:
            uncal_warnings = [
                x for x in w if UNCERTAINTY_UNCALIBRATED in str(x.message)
            ]
            assert len(uncal_warnings) >= 1


# ---------------------------------------------------------------------------
# Test: UncertaintyAuditReport dataclass
# ---------------------------------------------------------------------------


def test_uncertainty_audit_report_frozen():
    """UncertaintyAuditReport must be frozen."""
    report = UncertaintyAuditReport(
        lcb_empirical_coverage=0.9,
        sigma_abs_error_correlation=0.5,
        mean_sigma=0.1,
        median_sigma=0.09,
        n_calibration_edges=20,
        uncertainty_calibrated=True,
    )
    with pytest.raises(AttributeError):
        report.lcb_empirical_coverage = 0.8  # type: ignore


# ---------------------------------------------------------------------------
# Test: gamma report
# ---------------------------------------------------------------------------


def test_gamma_report_basic():
    """compute_gamma_report must return valid GammaReport."""
    examples = []
    # Create 8 edges with positive tau
    for i in range(8):
        examples.append(
            _make_example(
                task_id=f"t{i}",
                memory_id=f"m{i}",
                receiver_id=f"r{i}",
                expose=0.5 + 0.1 * i,  # tau = 0.1 * i + 0.2
                withhold=0.3,
                scenario="scenario_a",
            )
        )

    report = compute_gamma_report(examples)

    assert isinstance(report, GammaReport)
    assert report.positive_tau_count == 8
    assert report.gamma > 0
    assert report.positive_tau_mean > 0
    assert report.positive_tau_q25 <= report.positive_tau_q50
    assert report.positive_tau_q50 <= report.positive_tau_q75
    assert report.positive_tau_q75 <= report.positive_tau_q90
    # gamma == q75
    assert report.gamma == pytest.approx(report.positive_tau_q75)


def test_gamma_report_ignores_negative_tau():
    """Negative tau edges must be excluded from gamma report."""
    examples = [
        _make_example("t0", "m0", "r0", expose=0.8, withhold=0.3),  # tau=0.5
        _make_example("t1", "m1", "r1", expose=0.2, withhold=0.7),  # tau=-0.5
        _make_example("t2", "m2", "r2", expose=0.6, withhold=0.3),  # tau=0.3
    ]

    report = compute_gamma_report(examples)
    assert report.positive_tau_count == 2


def test_gamma_report_per_scenario():
    """Per-scenario diagnostics must be populated."""
    examples = [
        _make_example("t0", "m0", "r0", expose=0.8, withhold=0.3, scenario="sql"),
        _make_example("t1", "m1", "r1", expose=0.7, withhold=0.3, scenario="sql"),
        _make_example("t2", "m2", "r2", expose=0.9, withhold=0.3, scenario="web"),
        _make_example("t3", "m3", "r3", expose=0.6, withhold=0.3, scenario="web"),
    ]

    report = compute_gamma_report(examples)
    assert "sql" in report.per_scenario
    assert "web" in report.per_scenario
    assert report.per_scenario["sql"]["count"] == 2
    assert report.per_scenario["web"]["count"] == 2


def test_gamma_report_to_dict():
    """to_dict must produce JSON-serializable dict."""
    examples = [
        _make_example(f"t{i}", f"m{i}", f"r{i}", expose=0.5 + 0.1 * i, withhold=0.3)
        for i in range(5)
    ]

    report = compute_gamma_report(examples)
    d = report.to_dict()

    assert isinstance(d, dict)
    assert "gamma" in d
    assert "positive_tau_count" in d
    assert "positive_tau_mean" in d
    assert "positive_tau_median" in d
    assert "positive_tau_q25" in d
    assert "positive_tau_q50" in d
    assert "positive_tau_q75" in d
    assert "positive_tau_q90" in d
    assert d["gamma"] == report.gamma


def test_gamma_report_no_positive_tau_raises():
    """All non-positive tau must raise ValueError."""
    examples = [
        _make_example("t0", "m0", "r0", expose=0.3, withhold=0.8),  # tau=-0.5
        _make_example("t1", "m1", "r1", expose=0.5, withhold=0.5),  # tau=0
    ]

    with pytest.raises(ValueError, match="No positive observed tau"):
        compute_gamma_report(examples)


def test_gamma_report_aggregates_same_edge():
    """Multiple seeds for same edge must be averaged before filtering."""
    # Same edge with tau=0.4 and tau=0.2 → mean tau=0.3 (positive)
    ex1 = _make_example("t1", "m1", "r1", expose=0.8, withhold=0.4)
    ex2 = _make_example("t1", "m1", "r1", expose=0.6, withhold=0.4)
    # Different edge with tau=0.5
    ex3 = _make_example("t2", "m2", "r2", expose=0.7, withhold=0.2)

    report = compute_gamma_report([ex1, ex2, ex3])
    assert report.positive_tau_count == 2  # 2 unique edges


def test_gamma_report_frozen():
    """GammaReport must be frozen."""
    report = GammaReport(
        gamma=0.5,
        positive_tau_count=10,
        positive_tau_mean=0.4,
        positive_tau_median=0.35,
        positive_tau_q25=0.2,
        positive_tau_q50=0.35,
        positive_tau_q75=0.5,
        positive_tau_q90=0.6,
    )
    with pytest.raises(AttributeError):
        report.gamma = 0.6  # type: ignore
