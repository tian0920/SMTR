"""清单 Test 3: formal checkpoint calibration gate.

A formal checkpoint must carry a fitted isotonic q01 calibrator and a
validation-edge-selected epsilon_star; every degraded variant fails fast.
"""

from __future__ import annotations

import pytest

from smtr.marble.formal_protocol import require_formal_calibration
from smtr.router.transfer_calibration import Q01Calibrator
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _formal_critic() -> FourOutcomeTransferCritic:
    critic = FourOutcomeTransferCritic(feature_block="full")
    calibrator = Q01Calibrator()
    calibrator.method = "isotonic"
    calibrator.calibration_status = "fitted"
    critic.q01_calibrator = calibrator
    critic.calibration_split = "validation"
    critic.epsilon_selection_split = "validation"
    critic.risk_calibration = {"epsilon_selection_unit": "treatment_edge"}
    critic.epsilon_star = 0.2
    critic.validation_edge_count = 25
    return critic


def test_calibrator_missing_fails():
    critic = _formal_critic()
    critic.q01_calibrator = None
    with pytest.raises(ValueError, match="no q01 calibrator"):
        require_formal_calibration(critic, method="SMTR")


def test_unfitted_status_fails():
    critic = _formal_critic()
    critic.q01_calibrator.calibration_status = "insufficient_validation_edges"
    with pytest.raises(ValueError, match="fitted risk calibrator"):
        require_formal_calibration(critic, method="SMTR")


def test_identity_method_fails():
    critic = _formal_critic()
    critic.q01_calibrator.method = "identity"
    with pytest.raises(ValueError, match="isotonic"):
        require_formal_calibration(critic, method="SMTR")


def test_train_calibration_split_fails():
    critic = _formal_critic()
    critic.calibration_split = "train"
    with pytest.raises(ValueError, match="validation edges"):
        require_formal_calibration(critic, method="SMTR")


def test_test_epsilon_selection_split_fails():
    critic = _formal_critic()
    critic.epsilon_selection_split = "test"
    with pytest.raises(ValueError, match="validation edges"):
        require_formal_calibration(critic, method="SMTR")


def test_record_level_epsilon_unit_fails():
    critic = _formal_critic()
    critic.risk_calibration = {"epsilon_selection_unit": "record"}
    with pytest.raises(ValueError, match="treatment_edge"):
        require_formal_calibration(critic, method="SMTR")


def test_missing_epsilon_star_fails():
    critic = _formal_critic()
    critic.epsilon_star = None
    with pytest.raises(ValueError, match="no epsilon_star"):
        require_formal_calibration(critic, method="SMTR")


def test_insufficient_validation_edges_fails():
    critic = _formal_critic()
    critic.validation_edge_count = 5
    with pytest.raises(ValueError, match="insufficient validation edges"):
        require_formal_calibration(critic, method="SMTR")


def test_isotonic_validation_edge_level_passes():
    critic = _formal_critic()
    require_formal_calibration(critic, method="SMTR")
