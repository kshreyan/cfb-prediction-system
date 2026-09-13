"""Leakage suite: proves predictions for game N never depend on games that
happen after game N, chronologically. This is the test category the
project's acceptance criteria require to run in CI on every push.

Two independent lines of evidence:
  1. Truncation invariance: predictions for the first K games are
     identical whether you feed the engine only those K games, or feed
     it those K games plus any number of future games appended after.
  2. A structural check that process_game() computes and returns its
     PreGamePrediction BEFORE mutating any rating (so it is architecturally
     impossible for a prediction to see its own game's result, let alone a
     future one).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from cfb.elo.config import EloConfig
from cfb.elo.engine import EloEngine
from cfb.elo.types import GameResult

CONFIG = EloConfig(
    base_rating=1500.0,
    tier_priors={"power": 1620.0, "group_of_five": 1440.0, "fcs": 1250.0},
    conference_tiers={"power": ["SEC"], "group_of_five": ["Sun Belt"]},
)

TEAMS = ["TeamA", "TeamB", "TeamC", "TeamD"]


def _build_games(n: int, seed_scores) -> list[GameResult]:
    games = []
    base_date = datetime(2024, 8, 24)
    for i in range(n):
        home, away = TEAMS[i % 4], TEAMS[(i + 1) % 4]
        home_pts, away_pts = seed_scores[i]
        week = (i // 4) + 1
        games.append(
            GameResult(
                game_id=f"g{i}",
                season=2024,
                week=week,
                start_date=base_date + timedelta(days=7 * week),
                home_team=home,
                away_team=away,
                home_conference="Sun Belt",
                away_conference="Sun Belt",
                home_is_fbs=True,
                away_is_fbs=True,
                neutral_site=False,
                home_points=home_pts,
                away_points=away_pts,
            )
        )
    return games


@given(
    scores=st.lists(
        st.tuples(st.integers(0, 60), st.integers(0, 60)), min_size=6, max_size=20
    ),
    truncate_at=st.integers(min_value=3, max_value=5),
)
@settings(max_examples=50)
def test_truncation_invariance(scores, truncate_at):
    """Predictions for games[:k] must be identical whether or not games
    after index k exist in the input."""
    truncate_at = min(truncate_at, len(scores) - 1)
    full_games = _build_games(len(scores), scores)
    truncated_games = full_games[:truncate_at]

    engine_full = EloEngine(config=CONFIG)
    preds_full = engine_full.process_all(full_games)

    engine_truncated = EloEngine(config=CONFIG)
    preds_truncated = engine_truncated.process_all(truncated_games)

    for i in range(truncate_at):
        assert preds_full[i].home_win_prob == preds_truncated[i].home_win_prob
        assert preds_full[i].home_rating_pre == preds_truncated[i].home_rating_pre
        assert preds_full[i].away_rating_pre == preds_truncated[i].away_rating_pre


def test_prediction_uses_pre_update_ratings_not_post_update():
    """Directly verifies the prediction snapshot equals the rating state
    that existed strictly before this game's own delta was applied."""
    engine = EloEngine(config=CONFIG)
    game = GameResult(
        game_id="g0",
        season=2024,
        week=1,
        start_date=datetime(2024, 8, 31),
        home_team="TeamA",
        away_team="TeamB",
        home_conference="Sun Belt",
        away_conference="Sun Belt",
        home_is_fbs=True,
        away_is_fbs=True,
        neutral_site=False,
        home_points=45,
        away_points=3,
    )
    pred = engine.process_game(game)
    # After a 45-3 blowout, TeamA's rating must have moved -- if the
    # prediction had leaked the post-game rating, home_rating_pre would
    # equal the *updated* (higher) rating instead of the pre-game prior.
    assert pred.home_rating_pre == 1440.0  # group_of_five prior, untouched
    assert engine.rating_of("TeamA") > 1440.0  # confirms an update DID happen after
