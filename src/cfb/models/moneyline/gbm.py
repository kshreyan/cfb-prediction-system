"""Gradient-boosted moneyline model -- the feature-rich complement to the
Elo-only baseline. Walk-forward per season, same discipline as every
other model in this repo: trained only on strictly-prior-season rows,
refit once per season boundary, never mid-season.

Feature set: Elo differential (the exact logistic-inverse-recovered
signal, so it's not a weaker re-derivation of Elo), leak-free walk-forward
EPA/success-rate state (offense and defense, both teams), and preseason
priors (SP+ lagged one season, recruiting and talent composite for the
season itself -- see features/preseason_priors.py for why the lag
differs by metric). Missing values (a team with no tracked advanced
stats, or no recruiting/talent record) are passed through as NaN --
LightGBM splits on missingness natively, so nothing here imputes a fill
value.

Hyperparameters, stated plainly (see docs/methodology.md for the full
diagnostic): the original config was overfitting specifically in
near-even games (|Elo differential| < 100) -- it was MORE confident than
raw Elo in exactly the games with the least signal, and that confidence
wasn't earned: it lost to Elo in that bucket in 4 of 5 evaluated seasons.
Diagnosed via a walk-forward bucketed error analysis (favorite-size,
week-of-season, data-availability), not a blind hyperparameter sweep.
The regularized config below (shallower trees, higher min_child_samples,
L1/L2 penalties) was chosen for exactly that failure mode and validated
walk-forward against the original: pooled log loss 0.5499 -> 0.5489,
ECE 0.0217 -> 0.0119 (nearly halved), across all 9 seasons the GBM is
active for (2017-2025) -- a real if modest gain, not a breakthrough, and
it still does not beat the market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

FEATURE_COLUMNS = [
    "elo_diff",
    "home_off_epa_pre", "home_off_success_pre",
    "home_def_epa_allowed_pre", "home_def_success_allowed_pre",
    "away_off_epa_pre", "away_off_success_pre",
    "away_def_epa_allowed_pre", "away_def_success_allowed_pre",
    "home_sp_plus_prior", "away_sp_plus_prior",
    "home_recruiting_prior", "away_recruiting_prior",
    "home_talent_prior", "away_talent_prior",
]

# A 15-feature GBM needs substantially more rows than the 1-D linear/EWMA
# models to avoid overfitting -- below this, a season is skipped (NaN
# gbm_prob) rather than trained on too little data.
MIN_TRAIN_GAMES = 1500

def _new_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.03, num_leaves=8,
        min_child_samples=60, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=-1,
    )


def fit_gbm_for_season(df: pd.DataFrame, target_season: int) -> LGBMClassifier | None:
    """Fits a classifier for `target_season` using only strictly-prior-
    season rows. Returns None if there isn't enough prior data yet."""
    train = df[df["season"] < target_season]
    if len(train) < MIN_TRAIN_GAMES:
        return None
    model = _new_model()
    model.fit(train[FEATURE_COLUMNS], train["home_won"].astype(int))
    return model


def run_gbm_backtest(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward GBM predictions, one row per game. Seasons without
    enough prior training data get NaN gbm_prob, not a guess."""
    df = feature_df.sort_values(["season", "start_date"]).reset_index(drop=True)
    gbm_prob = np.full(len(df), np.nan)

    for season in sorted(df["season"].unique()):
        model = fit_gbm_for_season(df, season)
        if model is None:
            continue
        test_idx = df.index[df["season"] == season]
        proba = np.asarray(model.predict_proba(df.loc[test_idx, FEATURE_COLUMNS]))
        gbm_prob[test_idx] = proba[:, 1]

    df["gbm_prob"] = gbm_prob
    return df
