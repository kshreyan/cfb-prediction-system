"""Moneyline ensemble: blends a model probability with the market
probability on the log-odds scale, with the blend weight learned only
from strictly-prior seasons (walk-forward, expanding window -- the same
discipline as every other walk-forward component in this repo). This is
the "weights learned only via nested time-series CV" requirement: no
weight is ever chosen using a season it will then be scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.evaluation.metrics import log_loss

MIN_TRAIN_GAMES = 150
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)  # weight on the model; 1-w on market


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def blend_probs(model_p: np.ndarray, market_p: np.ndarray, weight_on_model: float) -> np.ndarray:
    blended_logit = weight_on_model * _logit(model_p) + (1 - weight_on_model) * _logit(market_p)
    return _sigmoid(blended_logit)


def _best_weight(train: pd.DataFrame, model_col: str, market_col: str, outcome_col: str) -> float:
    y = train[outcome_col].to_numpy(dtype=float)
    model_p = train[model_col].to_numpy(dtype=float)
    market_p = train[market_col].to_numpy(dtype=float)
    losses = [log_loss(y, blend_probs(model_p, market_p, w)) for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmin(losses))])


def walk_forward_ensemble(df: pd.DataFrame, model_col: str, market_col: str,
                           outcome_col: str, season_col: str = "season") -> pd.DataFrame:
    """Adds `ensemble_prob` and `ensemble_weight_on_model` columns. For a
    season with fewer than MIN_TRAIN_GAMES strictly-prior rows (i.e. the
    first season this ensemble has market data for), falls back to an
    even 50/50 blend rather than fitting a weight on too little data."""
    out = df.copy()
    out["ensemble_prob"] = np.nan
    out["ensemble_weight_on_model"] = np.nan

    for season in sorted(out[season_col].unique()):
        train = out[out[season_col] < season]
        test_idx = out.index[out[season_col] == season]

        weight = _best_weight(train, model_col, market_col, outcome_col) \
            if len(train) >= MIN_TRAIN_GAMES else 0.5

        model_p = out.loc[test_idx, model_col].to_numpy(dtype=float)
        market_p = out.loc[test_idx, market_col].to_numpy(dtype=float)
        out.loc[test_idx, "ensemble_prob"] = blend_probs(model_p, market_p, weight)
        out.loc[test_idx, "ensemble_weight_on_model"] = weight

    return out
