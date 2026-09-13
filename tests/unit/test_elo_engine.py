"""Unit tests for the Elo engine using small, hand-built synthetic game
fixtures. These verify the ENGINE'S LOGIC is correct (rating updates,
home field, offseason regression). They say nothing about real-world
predictive skill -- that can only be measured on real CFBD data, which
requires an API key (see README)."""
from __future__ import annotations

from datetime import datetime

import pytest

from cfb.elo.config import EloConfig
from cfb.elo.engine import EloEngine, LeakageError
from cfb.elo.types import GameResult

CONFIG = EloConfig(
    base_rating=1500.0,
    tier_priors={"power": 1620.0, "group_of_five": 1440.0, "fcs": 1250.0},
    conference_tiers={"power": ["SEC", "Big Ten"], "group_of_five": ["Sun Belt"]},
)


def make_game(game_id, season, week, day, home, away, home_conf, away_conf,
              home_pts, away_pts, neutral=False, home_fbs=True, away_fbs=True):
    return GameResult(
        game_id=game_id,
        season=season,
        week=week,
        start_date=datetime(season, 9, day),
        home_team=home,
        away_team=away,
        home_conference=home_conf,
        away_conference=away_conf,
        home_is_fbs=home_fbs,
        away_is_fbs=away_fbs,
        neutral_site=neutral,
        home_points=home_pts,
        away_points=away_pts,
    )


def test_new_team_starts_at_tier_prior():
    engine = EloEngine(config=CONFIG)
    game = make_game("g1", 2024, 1, 1, "Georgia", "Clemson", "SEC", "ACC", 30, 10)
    pred = engine.process_game(game)
    # Georgia (SEC power) should start above Clemson if Clemson isn't in a
    # configured power conference (here ACC falls to group_of_five prior).
    assert pred.home_rating_pre == pytest.approx(1620.0)
    assert pred.away_rating_pre == pytest.approx(1440.0)


def test_winner_rating_increases_loser_decreases():
    engine = EloEngine(config=CONFIG)
    game = make_game("g1", 2024, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 30, 10)
    engine.process_game(game)
    assert engine.rating_of("TeamA") > 1440.0
    assert engine.rating_of("TeamB") < 1440.0
    # Zero-sum-ish: total rating change for the two teams should roughly cancel.
    assert (engine.rating_of("TeamA") - 1440.0) == pytest.approx(
        -(engine.rating_of("TeamB") - 1440.0)
    )


def test_home_field_advantage_shifts_win_probability():
    engine_neutral = EloEngine(config=CONFIG)
    engine_home = EloEngine(config=CONFIG)
    g_neutral = make_game("g1", 2024, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt",
                           20, 20, neutral=True)
    g_home = make_game("g2", 2024, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt",
                        20, 20, neutral=False)
    pred_neutral = engine_neutral.process_game(g_neutral)
    pred_home = engine_home.process_game(g_home)
    assert pred_home.home_win_prob > pred_neutral.home_win_prob


def test_bigger_margin_causes_bigger_rating_swing():
    engine_blowout = EloEngine(config=CONFIG)
    engine_close = EloEngine(config=CONFIG)
    g_blowout = make_game("g1", 2024, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 55, 3)
    g_close = make_game("g2", 2024, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 24, 21)
    engine_blowout.process_game(g_blowout)
    engine_close.process_game(g_close)
    blowout_gain = engine_blowout.rating_of("TeamA") - 1440.0
    close_gain = engine_close.rating_of("TeamA") - 1440.0
    assert blowout_gain > close_gain > 0


def test_out_of_order_games_raise_leakage_error():
    engine = EloEngine(config=CONFIG)
    g_later = make_game("g1", 2024, 2, 10, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 20, 10)
    g_earlier = make_game("g2", 2024, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 20, 10)
    engine.process_game(g_later)
    with pytest.raises(LeakageError):
        engine.process_game(g_earlier)


def test_offseason_regression_pulls_rating_toward_tier_prior():
    engine = EloEngine(config=CONFIG)
    # TeamA wins big repeatedly in 2024, rating rises well above its prior.
    for i in range(5):
        g = make_game(f"g{i}", 2024, i + 1, i + 1, "TeamA", "TeamB",
                       "Sun Belt", "Sun Belt", 45, 3)
        engine.process_game(g)
    rating_end_2024 = engine.rating_of("TeamA")
    assert rating_end_2024 > 1440.0 + 50

    # First game of 2025 triggers offseason regression before prediction.
    g_2025 = make_game("g2025", 2025, 1, 1, "TeamA", "TeamC", "Sun Belt", "Sun Belt", 0, 0)
    pred = engine.process_game(g_2025)
    assert 1440.0 < pred.home_rating_pre < rating_end_2024


def test_season_cannot_go_backwards():
    engine = EloEngine(config=CONFIG)
    engine.process_game(
        make_game("g1", 2025, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 20, 10)
    )
    with pytest.raises(LeakageError):
        engine.process_game(
            make_game("g2", 2024, 1, 1, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 20, 10)
        )


def test_process_all_sorts_before_processing():
    engine = EloEngine(config=CONFIG)
    g_late = make_game("g1", 2024, 2, 10, "TeamA", "TeamB", "Sun Belt", "Sun Belt", 20, 10)
    g_early = make_game("g2", 2024, 1, 1, "TeamB", "TeamA", "Sun Belt", "Sun Belt", 10, 20)
    preds = engine.process_all([g_late, g_early])
    assert [p.game_id for p in preds] == ["g2", "g1"]
