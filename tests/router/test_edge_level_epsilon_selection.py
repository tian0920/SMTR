"""清单 Test 3: edge-level epsilon selection (P0-4~7).

epsilon_star is selected over treatment edges: edge A with 10 seeds must
carry exactly the same weight as edges B/C with 5 seeds, and the selector
only ever reads validation edges.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES
from smtr.router.transfer_calibration import (
    Q01Calibrator,
    EdgeThresholdExample,
    build_edge_calibration_examples,
    build_edge_threshold_examples,
    select_epsilon_edge_level,
)


def _edge(key, *, tau, eta, share, withhold, neg, seeds):
    return EdgeThresholdExample(
        edge_key=key,
        predicted_tau=tau,
        calibrated_eta=eta,
        empirical_share_success=share,
        empirical_withhold_success=withhold,
        empirical_negative_transfer_rate=neg,
        valid_seed_count=seeds,
    )


def _abc_examples():
    """Edge A has 10 seeds; B and C have 5 seeds each."""
    return [
        _edge(("t1", "r1", "A"), tau=0.5, eta=0.05, share=0.5, withhold=0.5,
              neg=0.0, seeds=10),
        _edge(("t1", "r1", "B"), tau=0.4, eta=0.40, share=0.0, withhold=1.0,
              neg=0.5, seeds=5),
        _edge(("t1", "r1", "C"), tau=0.4, eta=0.60, share=0.0, withhold=1.0,
              neg=0.5, seeds=5),
    ]


def _rows_by_epsilon(result):
    return {row["epsilon"]: row for row in result["candidate_rows"]}


def test_edges_equally_weighted_regardless_of_seed_count():
    """Edge A's extra seeds must not give it double weight (P0-5)."""
    result = select_epsilon_edge_level(
        examples=_abc_examples(),
        candidate_epsilons=[0.1, 0.5, 0.7],
        max_negative_exposure_rate=None,
    )
    assert result["selection_unit"] == "treatment_edge"
    assert result["validation_edge_count"] == 3
    rows = _rows_by_epsilon(result)

    # epsilon=0.1 shares only A: edge-equal value (0.5+1+1)/3.
    equal_weight_value = (0.5 + 1.0 + 1.0) / 3
    seed_weight_value = (10 * 0.5 + 5 * 1.0 + 5 * 1.0) / 20
    assert rows[0.1]["policy_value"] == pytest.approx(equal_weight_value)
    assert rows[0.1]["policy_value"] != pytest.approx(seed_weight_value)
    assert rows[0.1]["shared_edge_rate"] == pytest.approx(1 / 3)
    assert rows[0.1]["negative_exposure_rate"] == pytest.approx(0.0)


def test_risk_constraint_filters_epsilons():
    """Epsilons violating the exposure cap are dropped (P0-7)."""
    result = select_epsilon_edge_level(
        examples=_abc_examples(),
        candidate_epsilons=[0.1, 0.5, 0.7],
        max_negative_exposure_rate=0.1,
    )
    rows = _rows_by_epsilon(result)
    # 0.5 exposes (0.0+0.5)/2=0.25 and 0.7 exposes 0.1667: both infeasible.
    assert rows[0.5]["negative_exposure_rate"] == pytest.approx(0.25)
    assert result["epsilon_star"] == 0.1
    assert result["validation_policy_value"] == pytest.approx((0.5 + 2.0) / 3)


def test_tie_prefers_smaller_epsilon():
    examples = [
        _edge(("t1", "r1", "A"), tau=0.5, eta=0.20, share=0.0, withhold=1.0,
              neg=0.5, seeds=3),
        _edge(("t1", "r1", "B"), tau=-0.1, eta=0.05, share=1.0, withhold=1.0,
              neg=0.0, seeds=3),
    ]
    # Neither epsilon shares anything (A's eta above both, B's tau <= 0),
    # so both tie at the withhold-only value; the smaller epsilon wins.
    result = select_epsilon_edge_level(
        examples=examples,
        candidate_epsilons=[0.05, 0.1],
        max_negative_exposure_rate=None,
    )
    assert result["epsilon_star"] == 0.05


def test_no_feasible_epsilon_raises():
    examples = [
        # Shared at every candidate epsilon, and always with a negative
        # rate far above the cap: nothing can satisfy the constraint.
        _edge(("t1", "r1", "A"), tau=0.5, eta=0.05, share=1.0, withhold=0.0,
              neg=0.8, seeds=4),
    ]
    with pytest.raises(
        ValueError, match="no epsilon satisfies the validation risk constraint"
    ):
        select_epsilon_edge_level(
            examples=examples,
            candidate_epsilons=[0.1, 0.2],
            max_negative_exposure_rate=0.05,
        )


def test_selector_only_accepts_validation_edges():
    """No test-split argument may sneak into the selector signature."""
    params = set(inspect.signature(select_epsilon_edge_level).parameters)
    assert params == {
        "examples",
        "candidate_epsilons",
        "max_negative_exposure_rate",
    }, "select_epsilon_edge_level must not accept test-split data"


def test_one_threshold_example_per_edge_despite_seed_counts():
    """10/5/5 seed records collapse to exactly three threshold examples."""
    records = []

    def _append(edge_memory, seed, label):
        y_share, y_withhold = LABEL_TO_OUTCOMES[label]
        records.append({
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "candidate_memory_id": edge_memory,
            "generation_seed": seed,
            "label": label,
            "share": {"team_success": bool(y_share)},
            "withhold": {"team_success": bool(y_withhold)},
        })

    for s in range(10):
        # One negative seed out of ten: empirical negative rate 0.1,
        # exactly meeting the exposure cap below.
        _append("A", s, "negative_transfer" if s == 0 else "positive_transfer")
    for s in range(5):
        _append("B", s, "negative_transfer")
    for s in range(5):
        _append("C", s, "positive_transfer")

    predictions = {
        ("t1", "r1", "A"): {"predicted_q01": 0.05, "predicted_tau": 0.5},
        ("t1", "r1", "B"): {"predicted_q01": 0.40, "predicted_tau": 0.4},
        ("t1", "r1", "C"): {"predicted_q01": 0.60, "predicted_tau": 0.4},
    }
    calibration_examples = build_edge_calibration_examples(
        records=records, predictions_by_edge=predictions
    )
    assert len(calibration_examples) == 3
    seed_counts = sorted(
        ex.valid_seed_count for ex in calibration_examples
    )
    assert seed_counts == [5, 5, 10]

    calibrator = Q01Calibrator().fit(
        predicted_q01=np.array(
            [ex.predicted_q01 for ex in calibration_examples]
        ),
        empirical_eta=np.array(
            [ex.empirical_eta for ex in calibration_examples]
        ),
    )
    threshold_examples = build_edge_threshold_examples(
        calibration_examples, calibrator
    )
    assert len(threshold_examples) == 3
    result = select_epsilon_edge_level(
        examples=threshold_examples,
        candidate_epsilons=[0.1, 0.2],
        max_negative_exposure_rate=0.1,
    )
    assert result["validation_edge_count"] == 3
    assert result["selection_unit"] == "treatment_edge"
