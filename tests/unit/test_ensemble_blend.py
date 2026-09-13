"""Unit tests for the log-odds ensemble blend, using synthetic data where
the "market" is far more accurate than the "model", so the walk-forward
weight search should learn to lean heavily on the market."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.ensemble.blend import MIN_TRAIN_GAMES, blend_probs, walk_forward_ensemble
from cfb.evaluation.metrics import log_loss


def test_blend_probs_weight_one_equals_model():
    model_p = np.array([0.7, 0.3, 0.9])
    market_p = np.array([0.5, 0.5, 0.5])
    blended = blend_probs(model_p, market_p, weight_on_model=1.0)
    assert np.allclose(blended, model_p, atol=1e-6)


def test_blend_probs_weight_zero_equals_market():
    model_p = np.array([0.7, 0.3, 0.9])
    market_p = np.array([0.4, 0.6, 0.2])
    blended = blend_probs(model_p, market_p, weight_on_model=0.0)
    assert np.allclose(blended, market_p, atol=1e-6)


def _synthetic_df(n_per_season: int, seasons: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for _ in range(n_per_season):
            true_p = rng.uniform(0.05, 0.95)
            outcome = int(rng.uniform() < true_p)
            # Market: accurate. Model: noisy/uninformative around 0.5.
            market_p = np.clip(true_p + rng.normal(0, 0.02), 0.01, 0.99)
            model_p = np.clip(0.5 + rng.normal(0, 0.05), 0.01, 0.99)
            rows.append({"season": season, "model_p": model_p, "market_p": market_p,
                         "outcome": outcome})
    return pd.DataFrame(rows)


def test_walk_forward_learns_to_favor_accurate_market_over_noisy_model():
    df = _synthetic_df(n_per_season=400, seasons=[2020, 2021, 2022])
    result = walk_forward_ensemble(df, "model_p", "market_p", "outcome")

    season_2022 = result[result["season"] == 2022]
    assert len(result[result["season"] < 2022]) >= MIN_TRAIN_GAMES
    # Weight on the noisy model should be small (market dominates).
    assert season_2022["ensemble_weight_on_model"].iloc[0] <= 0.2


def test_ensemble_beats_or_matches_the_worse_input_on_held_out_season():
    df = _synthetic_df(n_per_season=400, seasons=[2020, 2021, 2022])
    result = walk_forward_ensemble(df, "model_p", "market_p", "outcome")
    season_2022 = result[result["season"] == 2022]

    y = season_2022["outcome"].to_numpy()
    ensemble_loss = log_loss(y, season_2022["ensemble_prob"].to_numpy())
    model_loss = log_loss(y, season_2022["model_p"].to_numpy())
    assert ensemble_loss < model_loss


def test_first_eligible_season_without_enough_training_data_uses_even_blend():
    df = _synthetic_df(n_per_season=50, seasons=[2020])
    result = walk_forward_ensemble(df, "model_p", "market_p", "outcome")
    assert (result["ensemble_weight_on_model"] == 0.5).all()
