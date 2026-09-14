"""Unit tests for the EPA/success-rate engine using synthetic fixtures."""
from __future__ import annotations

from datetime import datetime

import pytest

from cfb.elo.types import GameResult
from cfb.features.epa_engine import OFF_EPA_DEFAULT, SUCCESS_RATE_DEFAULT, EpaEngine


def make_game(game_id, season, week, day, home, away, home_pts=20, away_pts=10):
    return GameResult(
        game_id=game_id, season=season, week=week, start_date=datetime(season, 9, day),
        home_team=home, away_team=away, home_conference="X", away_conference="X",
        home_is_fbs=True, away_is_fbs=True, neutral_site=False,
        home_points=home_pts, away_points=away_pts,
    )


def test_new_teams_predict_from_league_defaults():
    engine = EpaEngine()
    features = engine.process_game(make_game("g1", 2024, 1, 1, "A", "B"), None, None)
    assert features["home_off_epa_pre"] == OFF_EPA_DEFAULT
    assert features["home_off_success_pre"] == SUCCESS_RATE_DEFAULT
    assert features["home_stats_available"] is False


def test_efficient_offense_raises_future_predictions():
    engine = EpaEngine()
    hot_stats = {"off_ppa": 0.6, "off_success": 0.7, "def_ppa": 0.05, "def_success": 0.4}
    engine.process_game(make_game("g1", 2024, 1, 1, "A", "B"), hot_stats, None)
    features = engine.process_game(make_game("g2", 2024, 2, 8, "C", "A"), None, None)
    assert features["away_off_epa_pre"] > OFF_EPA_DEFAULT
    assert features["away_off_success_pre"] > SUCCESS_RATE_DEFAULT


def test_missing_stats_leaves_state_unchanged():
    engine = EpaEngine()
    engine.process_game(make_game("g1", 2024, 1, 1, "A", "B"), None, None)
    features = engine.process_game(make_game("g2", 2024, 2, 8, "A", "C"), None, None)
    assert features["home_off_epa_pre"] == pytest.approx(OFF_EPA_DEFAULT)


def test_offseason_regression_pulls_toward_default():
    engine = EpaEngine()
    hot_stats = {"off_ppa": 0.6, "off_success": 0.7, "def_ppa": 0.05, "def_success": 0.4}
    for i in range(5):
        engine.process_game(make_game(f"g{i}", 2024, i + 1, i + 1, "A", "B"), hot_stats, None)
    pre_regression = engine.teams["A"].off_epa_ewma
    assert pre_regression > OFF_EPA_DEFAULT + 0.1

    features = engine.process_game(make_game("g2025", 2025, 1, 1, "A", "C"), None, None)
    assert OFF_EPA_DEFAULT < features["home_off_epa_pre"] < pre_regression
