"""Preseason priors: SP+, recruiting, and talent-composite features.

Leakage discipline differs by metric, and that difference matters:

- **SP+** is an outcome-based rating that CFBD's free tier only exposes
  as one number per team per season, updated *as the season progresses*
  -- there is no week-by-week history available. Using a season's own
  SP+ rating for that season's games would leak future game results into
  early-season predictions (exactly the trap the master brief calls out
  by name). So SP+ is lagged by exactly one full season: a game in
  season S uses season S-1's final SP+ rating as a "how good was this
  program last year" prior.
- **Recruiting rankings** and the **talent composite** are inherently
  preseason signals -- a recruiting class is signed and a roster's talent
  composite is set before the season kicks off -- so season S's own
  recruiting/talent numbers are safe to use directly for season S's games
  without lagging.

A team missing a given season's data (new to FBS, or CFBD gap) gets NaN,
not an imputed league-average value -- the GBM model handles missing
features natively rather than this module fabricating a fill value.
"""
from __future__ import annotations

import pandas as pd


def build_sp_plus_lookup(raw_sp_by_season: list[dict]) -> dict[tuple[int, str], float]:
    """{(season, team): sp_plus_rating} straight from CFBD, NOT yet lagged
    -- lagging happens in attach_preseason_priors below, where it's
    explicit which season's games are consuming which year's rating."""
    return {(row["year"], row["team"]): row["rating"] for row in raw_sp_by_season
            if row.get("rating") is not None}


def build_recruiting_lookup(raw_recruiting_by_season: list[dict]) -> dict[tuple[int, str], float]:
    return {(row["year"], row["team"]): row["points"] for row in raw_recruiting_by_season
            if row.get("points") is not None}


def build_talent_lookup(raw_talent_by_season: list[dict]) -> dict[tuple[int, str], float]:
    return {(row["year"], row["team"]): row["talent"] for row in raw_talent_by_season
            if row.get("talent") is not None}


def attach_preseason_priors(games_df: pd.DataFrame, sp_lookup: dict, recruiting_lookup: dict,
                             talent_lookup: dict) -> pd.DataFrame:
    """`games_df` must have game_id, season, home_team, away_team. Adds:
    home/away sp_plus_prior_lag1 (season-1 SP+), home/away
    recruiting_prior (this season's recruiting points), home/away
    talent_prior (this season's talent composite). NaN where unavailable.
    """
    df = games_df.copy()

    def _sp(season: int, team: str) -> float | None:
        return sp_lookup.get((season - 1, team))

    def _recruit(season: int, team: str) -> float | None:
        return recruiting_lookup.get((season, team))

    def _talent(season: int, team: str) -> float | None:
        return talent_lookup.get((season, team))

    df["home_sp_plus_prior"] = df.apply(lambda r: _sp(r["season"], r["home_team"]), axis=1)
    df["away_sp_plus_prior"] = df.apply(lambda r: _sp(r["season"], r["away_team"]), axis=1)
    df["home_recruiting_prior"] = df.apply(lambda r: _recruit(r["season"], r["home_team"]), axis=1)
    df["away_recruiting_prior"] = df.apply(lambda r: _recruit(r["season"], r["away_team"]), axis=1)
    df["home_talent_prior"] = df.apply(lambda r: _talent(r["season"], r["home_team"]), axis=1)
    df["away_talent_prior"] = df.apply(lambda r: _talent(r["season"], r["away_team"]), axis=1)
    return df
