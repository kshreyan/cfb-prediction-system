"""Unit tests for the walk-forward GBM model using a synthetic dataset
with a known generative relationship between features and outcome."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.evaluation.metrics import accuracy
from cfb.models.moneyline.gbm import FEATURE_COLUMNS, MIN_TRAIN_GAMES, run_gbm_backtest


def _synthetic_df(n_per_season: int, seasons: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    game_id = 0
    for season in seasons:
        for i in range(n_per_season):
            row = {"game_id": f"g{game_id}", "season": season,
                   "start_date": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=i)}
            for col in FEATURE_COLUMNS:
                row[col] = rng.normal(0, 1)
            # True signal lives entirely in elo_diff and home_sp_plus_prior;
            # everything else is pure noise -- a working GBM should still
            # separate classes using just those two real features.
            logit = 1.5 * row["elo_diff"] + 1.2 * row["home_sp_plus_prior"]
            prob = 1 / (1 + np.exp(-logit))
            row["home_won"] = int(rng.uniform() < prob)
            rows.append(row)
            game_id += 1
    return pd.DataFrame(rows)


def test_seasons_below_min_train_games_get_nan():
    n_small = MIN_TRAIN_GAMES // 3
    df = _synthetic_df(n_per_season=n_small, seasons=[2020])
    result = run_gbm_backtest(df)
    assert result["gbm_prob"].isna().all()


def test_gbm_learns_real_signal_on_held_out_season():
    per_season = MIN_TRAIN_GAMES + 50
    df = _synthetic_df(n_per_season=per_season, seasons=[2020, 2021])
    result = run_gbm_backtest(df)

    season_2021 = result[result["season"] == 2021]
    assert season_2021["gbm_prob"].notna().all()  # had >= MIN_TRAIN_GAMES prior rows

    acc = accuracy(season_2021["home_won"].to_numpy(), season_2021["gbm_prob"].to_numpy())
    assert acc > 0.65  # well above chance given the strong synthetic signal
