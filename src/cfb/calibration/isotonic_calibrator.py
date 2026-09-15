"""Walk-forward isotonic calibration.

Fits sklearn's IsotonicRegression on strictly-prior-season (raw
probability, outcome) pairs and applies it to the current season -- same
expanding-window-by-season discipline as the spread/total models, so a
calibrator can never see the outcomes of games it's about to calibrate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

MIN_TRAIN_GAMES = 300  # isotonic regression needs more data than a 1-D
                        # linear fit to avoid overfitting to training noise


def fit_calibrator_for_season(df: pd.DataFrame, prob_col: str, outcome_col: str,
                               target_season: int, season_col: str = "season"
                               ) -> IsotonicRegression | None:
    """Fits an isotonic calibrator for `target_season` using only
    strictly-prior-season rows. Returns None if there isn't enough prior
    data yet (caller should pass raw probabilities through unchanged in
    that case) -- shared by the backtest loop below and by the live
    weekly-prediction pipeline, same reasoning as the spread/total
    models' fit_season_params helpers."""
    train = df[df[season_col] < target_season]
    if len(train) < MIN_TRAIN_GAMES:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train[prob_col].to_numpy(dtype=float), train[outcome_col].to_numpy(dtype=float))
    return iso


def walk_forward_isotonic_apply(df: pd.DataFrame, fit_prob_col: str, outcome_col: str,
                                 apply_prob_col: str, season_col: str = "season") -> np.ndarray:
    """Fits the calibrator on (`fit_prob_col`, `outcome_col`) pairs for
    each season (strictly-prior seasons only), then applies it to
    `apply_prob_col` -- usually the same column, but can differ: e.g. fit
    the mapping from closing-line cover probability to actual cover rate,
    then apply that same mapping to opening-line cover probability for a
    CLV computation, since both come from the same underlying model
    distribution and only the threshold (the line) differs. Seasons with
    insufficient prior training data pass `apply_prob_col` through
    unchanged."""
    calibrated = df[apply_prob_col].to_numpy(dtype=float).copy()

    for season in sorted(df[season_col].unique()):
        iso = fit_calibrator_for_season(df, fit_prob_col, outcome_col, season, season_col)
        if iso is None:
            continue  # leave raw probability in place for this season
        test_idx = df.index[df[season_col] == season]
        test_probs = df.loc[test_idx, apply_prob_col].to_numpy(dtype=float)
        calibrated[df.index.get_indexer(test_idx)] = iso.predict(test_probs)

    return calibrated


def walk_forward_isotonic_calibrate(df: pd.DataFrame, prob_col: str, outcome_col: str,
                                     season_col: str = "season") -> np.ndarray:
    """Returns an array aligned to df's index: the isotonic-calibrated
    probability for each row, using only strictly-prior-season data to
    fit the calibrator for that row's season. Seasons with insufficient
    prior training data pass the raw probability through unchanged
    (calibration is a refinement, not something to fake on too little
    data)."""
    return walk_forward_isotonic_apply(df, prob_col, outcome_col, prob_col, season_col)
