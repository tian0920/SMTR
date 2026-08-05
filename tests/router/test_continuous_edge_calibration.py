"""清单 Test 1: continuous edge-level eta calibration (P0-1).

Edge-level empirical eta is continuous (0, 1/n, ..., 1). The calibrator
must fit isotonic regression on these continuous targets: it never requires
exactly two unique target values and never binarizes at 0.5.
"""

from __future__ import annotations

import numpy as np

from smtr.router.transfer_calibration import Q01Calibrator

PREDICTED_Q01 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
EMPIRICAL_ETA = np.array([0.0, 0.2, 0.4, 0.8, 1.0])


def _fitted_calibrator() -> Q01Calibrator:
    # 5 validation edges; lower the isotonic threshold to match.
    return Q01Calibrator(min_edges_for_isotonic=5).fit(
        PREDICTED_Q01, EMPIRICAL_ETA
    )


def test_continuous_eta_fits_isotonic_not_identity():
    calibrator = _fitted_calibrator()
    assert calibrator.method == "isotonic"
    assert calibrator.calibration_status == "fitted"
    assert calibrator.model is not None
    assert calibrator.n_edges == 5


def test_no_binarization_of_empirical_eta():
    """Continuous targets are used as-is; no 0.5 thresholding."""
    calibrator = _fitted_calibrator()
    # If targets had been binarized at 0.5 the fitted map would only take
    # values in {0, 1}; the continuous fit reproduces intermediate rates.
    calibrated = calibrator.transform(PREDICTED_Q01)
    assert np.any((calibrated > 0.0) & (calibrated < 1.0))
    np.testing.assert_allclose(calibrated, EMPIRICAL_ETA, atol=1e-12)


def test_single_class_targets_do_not_trigger_identity_fallback():
    """The removed both-classes check must not demote the fit to identity."""
    predicted = np.linspace(0.1, 0.9, 25)
    empirical = np.zeros(25)
    calibrator = Q01Calibrator().fit(predicted, empirical)
    assert calibrator.method == "isotonic"
    assert calibrator.calibration_status == "fitted"
    np.testing.assert_allclose(calibrator.transform(predicted), 0.0)


def test_output_bounds_and_monotonicity_on_grid():
    calibrator = _fitted_calibrator()
    grid = np.linspace(0.0, 1.0, 101)
    calibrated = calibrator.transform(grid)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert np.all(np.diff(calibrated) >= 0.0)
