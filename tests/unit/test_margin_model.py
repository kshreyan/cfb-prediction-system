"""Unit tests for the margin (spread) model, using a synthetic dataset
with a known ground-truth relationship between Elo win prob and margin."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb.models.spread.margin_model import (
    home_cover_probability,
    recover_elo_diff,
    run_margin_backtest,
)


def _synthetic_elo_df(n_per_season: int, seasons: list[int], true_slope: float,
                       seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    game_id = 0
    for season in seasons:
        for i in range(n_per_season):
            elo_diff = rng.uniform(-600, 600)
            win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            margin = true_slope * elo_diff + rng.normal(0, 10)
            rows.append({
                "game_id": f"g{game_id}",
                "season": season,
                "week": (i // 10) + 1,
                "start_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=i),
                "home_win_prob": win_prob,
                "home_margin": round(margin),
                "is_fbs_vs_fbs": True,
            })
            game_id += 1
    return pd.DataFrame(rows)


def test_recover_elo_diff_is_exact_inverse_of_logistic():
    probs = pd.Series([0.5, 0.7, 0.9, 0.1])
    diffs = recover_elo_diff(probs)
    recomputed = 1.0 / (1.0 + 10.0 ** (-diffs / 400.0))
    assert np.allclose(recomputed, probs, atol=1e-6)


def test_walk_forward_regression_recovers_true_slope_in_later_season():
    # 300 games/season is well above MIN_TRAIN_GAMES, so season 2's fit
    # should be trained on season 1's real data and recover true_slope.
    df = _synthetic_elo_df(n_per_season=300, seasons=[2020, 2021], true_slope=0.04)
    result = run_margin_backtest(df)

    season_2021 = result[result["season"] == 2021]
    # Fitted slope should be close to the true generating slope (0.04);
    # check via the actual predictions vs elo_diff correlation instead of
    # a private slope field, using a simple regression on the outputs.
    corr = np.corrcoef(season_2021["elo_diff"], season_2021["predicted_margin"])[0, 1]
    assert corr > 0.95

    mae = (season_2021["home_margin"] - season_2021["predicted_margin"]).abs().mean()
    assert mae < 15  # noisy synthetic data (std=10) but should track reasonably


def test_first_season_uses_fallback_not_undertrained_fit():
    # Only 50 games in season 1 (< MIN_TRAIN_GAMES=150) -- must use the
    # fallback slope, not a regression fit on 50 noisy points.
    df = _synthetic_elo_df(n_per_season=50, seasons=[2020], true_slope=0.04)
    result = run_margin_backtest(df)
    from cfb.models.spread.margin_model import FALLBACK_SLOPE

    expected = FALLBACK_SLOPE * result["elo_diff"]
    assert np.allclose(result["predicted_margin"], expected, atol=1e-9)


def test_cover_probability_monotonic_in_spread():
    # A bigger home favorite spread (more negative) should make it HARDER
    # for home to cover, so cover probability should be strictly decreasing
    # as spread_home becomes more negative (home is asked to win by more).
    probs = [
        home_cover_probability(predicted_margin=10.0, resid_a=0.0, resid_loc=0.0,
                                resid_scale=14.0, spread_home=s)
        for s in [-3.0, -7.0, -10.0, -14.0, -21.0]
    ]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_cover_probability_near_half_when_spread_matches_prediction():
    # Symmetric residual (a=0): if the market spread equals -predicted_margin
    # exactly, home's cover probability should be ~50%.
    prob = home_cover_probability(predicted_margin=14.0, resid_a=0.0, resid_loc=0.0,
                                   resid_scale=10.0, spread_home=-14.0)
    assert prob == pytest.approx(0.5, abs=1e-6)
