from __future__ import annotations

import numpy as np
import pytest

from cfb.evaluation.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    mean_absolute_error,
)


def test_perfect_predictions_have_zero_loss():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert log_loss(y, p) == pytest.approx(0.0, abs=1e-6)
    assert brier_score(y, p) == pytest.approx(0.0)
    assert accuracy(y, p) == pytest.approx(1.0)


def test_uniform_50_50_predictions_are_well_calibrated_but_uninformative():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.full(6, 0.5)
    assert expected_calibration_error(y, p, n_bins=5) == pytest.approx(0.0, abs=1e-6)
    assert brier_score(y, p) == pytest.approx(0.25)


def test_overconfident_predictions_increase_log_loss_and_brier():
    p_calibrated = np.array([0.6, 0.6])
    p_overconfident = np.array([0.99, 0.99])
    # When right, overconfidence helps slightly...
    y_right = np.array([1, 1])
    assert log_loss(y_right, p_overconfident) < log_loss(y_right, p_calibrated)
    # ...but when even one of those calls is wrong, overconfidence is
    # punished far more harshly than a well-calibrated (60%) prediction --
    # this asymmetry is exactly why calibration, not raw confidence, is
    # what the project selects models on.
    y_one_wrong = np.array([1, 0])
    assert log_loss(y_one_wrong, p_overconfident) > log_loss(y_one_wrong, p_calibrated)
    assert brier_score(y_one_wrong, p_overconfident) > brier_score(y_one_wrong, p_calibrated)


def test_mean_absolute_error():
    assert mean_absolute_error([3, 7, 10], [1, 8, 10]) == pytest.approx(1.0)
