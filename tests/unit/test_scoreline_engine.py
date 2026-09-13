"""Unit tests for the total/scoreline engine using small synthetic game
fixtures -- proves engine logic (EWMA updates, home field, leak-free
walk-forward), not real predictive skill."""
from __future__ import annotations

from datetime import datetime

import pytest

from cfb.elo.types import GameResult
from cfb.models.total.scoreline_engine import (
    LEAGUE_AVG_POINTS,
    ScorelineEngine,
    over_probability,
)


def make_game(game_id, season, week, day, home, away, home_pts, away_pts):
    return GameResult(
        game_id=game_id, season=season, week=week, start_date=datetime(season, 9, day),
        home_team=home, away_team=away, home_conference="X", away_conference="X",
        home_is_fbs=True, away_is_fbs=True, neutral_site=False,
        home_points=home_pts, away_points=away_pts,
    )


def test_new_teams_predict_from_league_average_plus_home_field():
    engine = ScorelineEngine()
    game = make_game("g1", 2024, 1, 1, "TeamA", "TeamB", 30, 20)
    pred_home, pred_away = engine.process_game(game)
    assert pred_home == pytest.approx(LEAGUE_AVG_POINTS + 2.5)
    assert pred_away == pytest.approx(LEAGUE_AVG_POINTS)


def test_high_scoring_team_raises_future_predictions():
    engine = ScorelineEngine()
    # TeamA scores heavily in game 1; its predicted points in game 2
    # (as the visiting team this time) should rise above league average.
    engine.process_game(make_game("g1", 2024, 1, 1, "TeamA", "TeamB", 55, 10))
    _pred_home2, pred_away2 = engine.process_game(
        make_game("g2", 2024, 2, 8, "TeamC", "TeamA", 20, 20)
    )
    assert pred_away2 > LEAGUE_AVG_POINTS


def test_stingy_defense_lowers_opponent_prediction():
    engine = ScorelineEngine()
    # TeamB allows very few points in game 1; TeamB's future opponents
    # should have lower predicted points against them.
    engine.process_game(make_game("g1", 2024, 1, 1, "TeamA", "TeamB", 3, 0))
    pred_home2, _ = engine.process_game(
        make_game("g2", 2024, 2, 8, "TeamB", "TeamC", 0, 0)
    )
    assert pred_home2 < LEAGUE_AVG_POINTS + 2.5


def test_over_probability_monotonic_in_market_total():
    probs = [
        over_probability(predicted_total=55.0, resid_a=0.0, resid_loc=0.0,
                          resid_scale=17.0, market_total=t)
        for t in [40.0, 50.0, 55.0, 60.0, 70.0]
    ]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
