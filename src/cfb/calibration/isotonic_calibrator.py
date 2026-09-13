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


def walk_forward_isotonic_calibrate(df: pd.DataFrame, prob_col: str, outcome_col: str,
                                     season_col: str = "season") -> np.ndarray:
    """Returns an array aligned to df's index: the isotonic-calibrated
    probability for each row, using only strictly-prior-season data to
    fit the calibrator for that row's season. Seasons with insufficient
    prior training data pass the raw probability through unchanged
    (calibration is a refinement, not something to fake on too little
    data)."""
    calibrated = df[prob_col].to_numpy(dtype=float).copy()

    for season in sorted(df[season_col].unique()):
        train = df[df[season_col] < season]
        test_idx = df.index[df[season_col] == season]

        if len(train) < MIN_TRAIN_GAMES:
            continue  # leave raw probability in place for this season

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(train[prob_col].to_numpy(dtype=float),
                train[outcome_col].to_numpy(dtype=float))
        test_probs = df.loc[test_idx, prob_col].to_numpy(dtype=float)
        calibrated[df.index.get_indexer(test_idx)] = iso.predict(test_probs)

    return calibrated
