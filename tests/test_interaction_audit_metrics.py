from types import SimpleNamespace

import smtr.evaluation.interaction_audit as interaction_audit
from smtr.evaluation.interaction_audit import (
    _baseline_modified,
    _pearson,
    audit_interaction,
)


def _rec(record_id, *, mech, prefix_family, prefix_size, transfer_class, marginal, tau):
    return SimpleNamespace(
        record_id=record_id,
        prefix_size=prefix_size,
        transfer_class=transfer_class,
        marginal_effect=marginal,
        tau=tau,
        evaluation_group_metadata=SimpleNamespace(
            mechanism_group_id=mech,
            prefix_structure_family=prefix_family,
            scenario_family="s",
            surface_variant_id="v",
            target_memory_family="t",
        ),
    )


class _FakeCritic:
    def predict(self, item):
        return SimpleNamespace(tau_mean=item.tau)


def test_pearson_basic() -> None:
    assert _pearson([1, -1, 1, -1], [0.3, -0.3, 0.3, -0.3]) > 0.99
    assert _pearson([1, -1, 1, -1], [-0.3, 0.3, -0.3, 0.3]) < -0.99
    assert _pearson([1, 1, 1], [0.5, 0.5, 0.5]) is None  # zero variance
    assert _pearson([1], [1]) is None  # too few points


def test_baseline_modified_prefers_empty_then_smaller_prefix() -> None:
    empty = _rec("e", mech="m", prefix_family="empty", prefix_size=0,
                 transfer_class="positive", marginal=1, tau=0.4)
    lock = _rec("l", mech="m", prefix_family="lock-prefix", prefix_size=1,
                transfer_class="neutral_failure", marginal=0, tau=0.05)
    assert _baseline_modified(lock, empty) == (empty, lock)
    assert _baseline_modified(empty, lock) == (empty, lock)


def test_audit_reports_s3_metrics(monkeypatch) -> None:
    monkeypatch.setattr(interaction_audit, "prediction_input_from_record", lambda r: r)
    records = [
        # m1: empty positive -> lock neutral (positive_to_neutral), critic correct.
        _rec("a", mech="m1", prefix_family="empty", prefix_size=0,
             transfer_class="positive", marginal=1, tau=0.40),
        _rec("b", mech="m1", prefix_family="lock-prefix", prefix_size=1,
             transfer_class="neutral_failure", marginal=0, tau=0.05),
        # m2: empty neutral -> lock positive (neutral_to_positive), critic correct.
        _rec("c", mech="m2", prefix_family="empty", prefix_size=0,
             transfer_class="neutral_failure", marginal=0, tau=0.05),
        _rec("d", mech="m2", prefix_family="lock-prefix", prefix_size=1,
             transfer_class="positive", marginal=1, tau=0.40),
    ]
    report = audit_interaction(records, _FakeCritic(), mode="prefix")

    assert report["matched_pair_count"] == 4
    assert report["direction_accuracy"] == 1.0
    assert report["delta_correlation"] > 0.99
    assert report["delta_mae"] == report["mean_abs_delta_tau_error"]
    assert report["transfer_region_flip_accuracy"] == 1.0
    assert report["transfer_region_flip_pair_count"] == 4

    flip = report["flip_detection"]
    assert flip["positive_to_neutral"] == {"pair_count": 2, "direction_accuracy": 1.0}
    assert flip["neutral_to_positive"] == {"pair_count": 2, "direction_accuracy": 1.0}
    assert flip["positive_to_negative"]["pair_count"] == 0
    assert flip["positive_to_negative"]["direction_accuracy"] is None


def test_audit_penalizes_wrong_direction(monkeypatch) -> None:
    monkeypatch.setattr(interaction_audit, "prediction_input_from_record", lambda r: r)
    records = [
        # Critic predicts the WRONG direction: empty positive has low tau,
        # lock neutral has high tau.
        _rec("a", mech="m1", prefix_family="empty", prefix_size=0,
             transfer_class="positive", marginal=1, tau=0.05),
        _rec("b", mech="m1", prefix_family="lock-prefix", prefix_size=1,
             transfer_class="neutral_failure", marginal=0, tau=0.40),
    ]
    report = audit_interaction(records, _FakeCritic(), mode="prefix")

    assert report["matched_pair_count"] == 2
    assert report["direction_accuracy"] == 0.0
    assert report["delta_correlation"] < 0
    assert report["flip_detection"]["positive_to_neutral"]["direction_accuracy"] == 0.0
