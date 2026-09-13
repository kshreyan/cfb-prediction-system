"""Unit tests for walk-forward isotonic calibration using synthetic,
deliberately-miscalibrated probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.calibration.isotonic_calibrator import (
    MIN_TRAIN_GAMES,
    walk_forward_isotonic_calibrate,
)
from cfb.evaluation.metrics import expected_calibration_error


def _overconfident_synthetic_df(n_per_season: int, seasons: list[int], seed: int = 0) -> pd.DataFrame:
    """Ground truth win prob is `true_p`; the "raw" prediction is
    deliberately over-confident (pushed toward 0/1), so calibration has
    something real to fix."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for _ in range(n_per_season):
            true_p = rng.uniform(0.05, 0.95)
            outcome = int(rng.uniform() < true_p)
            raw_p = np.clip(0.5 + (true_p - 0.5) * 1.8, 0.01, 0.99)  # overconfident
            rows.append({"season": season, "raw_p": raw_p, "outcome": outcome})
    return pd.DataFrame(rows)


def test_insufficient_training_data_passes_through_unchanged():
    df = _overconfident_synthetic_df(n_per_season=50, seasons=[2020])
    assert len(df) < MIN_TRAIN_GAMES
    calibrated = walk_forward_isotonic_calibrate(df, "raw_p", "outcome")
    assert np.allclose(calibrated, df["raw_p"].to_numpy())


def test_calibration_improves_ece_on_later_season():
    df = _overconfident_synthetic_df(n_per_season=500, seasons=[2020, 2021])
    calibrated = walk_forward_isotonic_calibrate(df, "raw_p", "outcome")

    season_2021 = df["season"] == 2021
    raw_ece = expected_calibration_error(
        df.loc[season_2021, "outcome"].to_numpy(), df.loc[season_2021, "raw_p"].to_numpy()
    )
    calibrated_ece = expected_calibration_error(
        df.loc[season_2021, "outcome"].to_numpy(), calibrated[season_2021.to_numpy()]
    )
    assert calibrated_ece < raw_ece


def test_calibrated_probs_stay_in_unit_interval():
    df = _overconfident_synthetic_df(n_per_season=400, seasons=[2020, 2021])
    calibrated = walk_forward_isotonic_calibrate(df, "raw_p", "outcome")
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)
