"""Margin (spread) model.

Predicts the full home-margin distribution from Elo, walk-forward and
leak-free. Margins are NOT modeled as Gaussian: blowouts and garbage-time
scoring make the tails fat and (for big favorites especially) skewed, so
residuals are fit with a skew-normal rather than assumed Normal (see the
master brief's "fat tails / non-normality" requirement).

Design choice, stated plainly: the regression (Elo-diff -> expected
margin) and the residual distribution are refit once per season boundary,
on an expanding window of all STRICTLY PRIOR seasons -- never refit
mid-season and never using the current or a future season. Refitting
every single week would be marginally more adaptive but isn't needed for
a single-feature (Elo-diff only) linear model; this is a deliberate
simplification, not a leak. See docs/methodology.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_TRAIN_GAMES = 150
# Until MIN_TRAIN_GAMES prior games exist (i.e. during the first season in
# the dataset), fall back to a widely-cited Elo-to-points heuristic
# (~28 Elo points per point of expected margin) with a wide, symmetric
# residual prior, rather than fitting on too little data.
FALLBACK_SLOPE = 1.0 / 28.0
FALLBACK_RESID = (0.0, 0.0, 14.0)  # (skew a, loc, scale)


def recover_elo_diff(home_win_prob: pd.Series) -> pd.Series:
    """Recovers the exact (rating_diff + home_field_adj) that produced
    this win probability -- the exact logistic inverse, so no information
    is lost or re-approximated versus re-deriving it from raw ratings."""
    p = home_win_prob.clip(1e-6, 1 - 1e-6)
    return 400.0 * np.log10(p / (1 - p))


def _fit_ols_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form simple linear regression y = slope*x + intercept."""
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0, float(y_mean)
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def run_margin_backtest(elo_df: pd.DataFrame) -> pd.DataFrame:
    """Adds predicted_margin + fitted residual-distribution params to a
    copy of the Elo backtest dataframe, walk-forward by season."""
    df = elo_df.copy()
    df["elo_diff"] = recover_elo_diff(df["home_win_prob"])
    df = df.sort_values(["season", "start_date"]).reset_index(drop=True)

    n = len(df)
    predicted_margin = np.full(n, np.nan)
    resid_a = np.full(n, np.nan)
    resid_loc = np.full(n, np.nan)
    resid_scale = np.full(n, np.nan)

    for season in sorted(df["season"].unique()):
        train = df[df["season"] < season]
        test_idx = df.index[df["season"] == season]

        if len(train) >= MIN_TRAIN_GAMES:
            slope, intercept = _fit_ols_1d(
                train["elo_diff"].to_numpy(), train["home_margin"].to_numpy()
            )
            residuals = train["home_margin"].to_numpy() - (
                slope * train["elo_diff"].to_numpy() + intercept
            )
            a, loc, scale = stats.skewnorm.fit(residuals)
        else:
            slope, intercept = FALLBACK_SLOPE, 0.0
            a, loc, scale = FALLBACK_RESID

        elo_diff_test = df.loc[test_idx, "elo_diff"].to_numpy()
        predicted_margin[test_idx] = slope * elo_diff_test + intercept
        resid_a[test_idx] = a
        resid_loc[test_idx] = loc
        resid_scale[test_idx] = scale

    df["predicted_margin"] = predicted_margin
    df["resid_a"] = resid_a
    df["resid_loc"] = resid_loc
    df["resid_scale"] = resid_scale
    df["margin_error"] = df["home_margin"] - df["predicted_margin"]
    return df


def home_cover_probability(predicted_margin: float, resid_a: float, resid_loc: float,
                            resid_scale: float, spread_home: float) -> float:
    """P(home covers `spread_home`), where spread_home follows standard
    market convention (negative = home favored by that many points).
    Home covers iff actual_margin > -spread_home; actual_margin =
    predicted_margin + residual, residual ~ skewnorm(a, loc, scale)."""
    threshold = -spread_home - predicted_margin
    return float(1.0 - stats.skewnorm.cdf(threshold, resid_a, resid_loc, resid_scale))
