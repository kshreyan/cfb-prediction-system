"""Assembles the full leak-free feature matrix the GBM model trains on:
Elo backtest output + EPA engine output + preseason priors, joined on
game_id. Each input is independently leak-free (see their own modules);
this module only joins them, adding no new information.
"""
from __future__ import annotations

import pandas as pd

from cfb.features.epa_engine import build_epa_features
from cfb.features.preseason_priors import attach_preseason_priors
from cfb.models.spread.margin_model import recover_elo_diff


def build_feature_matrix(elo_df: pd.DataFrame, games, raw_advanced_stats: list[dict],
                          sp_lookup: dict, recruiting_lookup: dict,
                          talent_lookup: dict) -> pd.DataFrame:
    df = elo_df.copy()
    df["elo_diff"] = recover_elo_diff(df["home_win_prob"])

    epa_df = build_epa_features(games, raw_advanced_stats)
    df = df.merge(epa_df, on="game_id", how="left")

    df = attach_preseason_priors(df, sp_lookup, recruiting_lookup, talent_lookup)
    return df
