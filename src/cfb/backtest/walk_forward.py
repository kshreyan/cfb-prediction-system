"""Walk-forward backtest harness.

The Elo engine is *inherently* walk-forward: process_game() predicts from
current ratings and only afterwards folds in the result, so simply
feeding games through in chronological order (engine.process_all) already
satisfies "train on prior games only, predict upcoming, reveal after."
This module turns that stream of (prediction, actual) pairs into the
immutable prediction log and the per-season/week evaluation tables used
by every downstream report.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cfb.elo.engine import EloEngine
from cfb.elo.types import GameResult
from cfb.evaluation.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
)


@dataclass
class BacktestRow:
    game_id: str
    season: int
    week: int
    start_date: object
    home_team: str
    away_team: str
    home_rating_pre: float
    away_rating_pre: float
    home_win_prob: float
    home_won: bool
    home_points: int
    away_points: int
    home_margin: int
    is_fbs_vs_fbs: bool


def run_elo_backtest(games: list[GameResult], engine: EloEngine) -> pd.DataFrame:
    """Runs the leak-free Elo walk-forward backtest end to end.

    Returns one immutable row per game with the pre-game prediction and
    the (only-attached-after-the-fact) actual result -- predictions and
    results are computed independently and joined here for evaluation,
    mirroring how a real deployment stores them in separate tables.
    """
    games_by_id = {g.game_id: g for g in games}
    predictions = engine.process_all(games)

    rows = []
    for pred in predictions:
        game = games_by_id[pred.game_id]
        rows.append(
            BacktestRow(
                game_id=pred.game_id,
                season=pred.season,
                week=pred.week,
                start_date=pred.start_date,
                home_team=pred.home_team,
                away_team=pred.away_team,
                home_rating_pre=pred.home_rating_pre,
                away_rating_pre=pred.away_rating_pre,
                home_win_prob=pred.home_win_prob,
                home_won=game.home_won,
                home_points=game.home_points,
                away_points=game.away_points,
                home_margin=game.margin,
                is_fbs_vs_fbs=game.home_is_fbs and game.away_is_fbs,
            )
        )
    return pd.DataFrame(rows)


def summarize_by_season(df: pd.DataFrame, fbs_only: bool = True) -> pd.DataFrame:
    """Per-season accuracy / log loss / Brier / ECE, matching the metrics
    the README commits to reporting. `fbs_only` excludes FBS-vs-FCS
    mismatches, since those inflate straight-up accuracy without
    representing a real market edge.
    """
    data = df[df["is_fbs_vs_fbs"]] if fbs_only else df
    out = []
    for season, group in data.groupby("season"):
        y_true = group["home_won"].astype(int).to_numpy()
        p_pred = group["home_win_prob"].to_numpy()
        out.append(
            {
                "season": season,
                "n_games": len(group),
                "accuracy": accuracy(y_true, p_pred),
                "log_loss": log_loss(y_true, p_pred),
                "brier": brier_score(y_true, p_pred),
                "ece": expected_calibration_error(y_true, p_pred),
            }
        )
    return pd.DataFrame(out).sort_values("season").reset_index(drop=True)
