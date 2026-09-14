"""Unit tests for preseason prior lagging -- the core thing to verify is
that SP+ is lagged by one season (leak-free) while recruiting/talent are
not (correctly, since those are inherently preseason signals)."""
from __future__ import annotations

import pandas as pd

from cfb.features.preseason_priors import (
    attach_preseason_priors,
    build_recruiting_lookup,
    build_sp_plus_lookup,
    build_talent_lookup,
)

RAW_SP = [
    {"year": 2023, "team": "Georgia", "rating": 30.0},
    {"year": 2024, "team": "Georgia", "rating": 25.0},
]
RAW_RECRUITING = [{"year": 2024, "team": "Georgia", "points": 300.0}]
RAW_TALENT = [{"year": 2024, "team": "Georgia", "talent": 1000.0}]

GAMES = pd.DataFrame([
    {"game_id": "g1", "season": 2024, "home_team": "Georgia", "away_team": "Unknown U"},
])


def test_sp_plus_is_lagged_by_one_season():
    sp_lookup = build_sp_plus_lookup(RAW_SP)
    recruiting_lookup = build_recruiting_lookup(RAW_RECRUITING)
    talent_lookup = build_talent_lookup(RAW_TALENT)
    result = attach_preseason_priors(GAMES, sp_lookup, recruiting_lookup, talent_lookup)

    # 2024 game must use the 2023 (prior-season) SP+ rating, NOT 2024's.
    assert result.loc[0, "home_sp_plus_prior"] == 30.0


def test_recruiting_and_talent_use_current_season_not_lagged():
    sp_lookup = build_sp_plus_lookup(RAW_SP)
    recruiting_lookup = build_recruiting_lookup(RAW_RECRUITING)
    talent_lookup = build_talent_lookup(RAW_TALENT)
    result = attach_preseason_priors(GAMES, sp_lookup, recruiting_lookup, talent_lookup)

    assert result.loc[0, "home_recruiting_prior"] == 300.0
    assert result.loc[0, "home_talent_prior"] == 1000.0


def test_missing_data_is_nan_not_imputed():
    sp_lookup = build_sp_plus_lookup(RAW_SP)
    recruiting_lookup = build_recruiting_lookup(RAW_RECRUITING)
    talent_lookup = build_talent_lookup(RAW_TALENT)
    result = attach_preseason_priors(GAMES, sp_lookup, recruiting_lookup, talent_lookup)

    # "Unknown U" has no data anywhere -- must be NaN, not 0 or a league average.
    assert pd.isna(result.loc[0, "away_sp_plus_prior"])
    assert pd.isna(result.loc[0, "away_recruiting_prior"])
    assert pd.isna(result.loc[0, "away_talent_prior"])
