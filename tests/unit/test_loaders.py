"""Unit tests for CFBD raw-dict -> GameResult parsing, using a small
hand-built synthetic dict shaped like a real CFBD response (verified
against an actual API call during development) -- not real data."""
from __future__ import annotations

from cfb.data.loaders import games_from_cfbd_dicts

RAW_COMPLETED = {
    "id": 401693678,
    "season": 2024,
    "week": 1,
    "startDate": "2024-08-24 20:00:00+00:00",
    "completed": True,
    "neutralSite": False,
    "homeTeam": "New Mexico",
    "homeConference": "Mountain West",
    "homeClassification": "fbs",
    "homePoints": 31,
    "awayTeam": "Montana State",
    "awayConference": "Big Sky",
    "awayClassification": "fcs",
    "awayPoints": 35,
}

RAW_UNCOMPLETED = {**RAW_COMPLETED, "id": 999, "completed": False,
                    "homePoints": None, "awayPoints": None}


def test_parses_completed_game():
    games = games_from_cfbd_dicts([RAW_COMPLETED])
    assert len(games) == 1
    g = games[0]
    assert g.game_id == "401693678"
    assert g.home_team == "New Mexico"
    assert g.away_is_fbs is False
    assert g.home_is_fbs is True
    assert g.home_points == 31 and g.away_points == 35
    assert g.neutral_site is False


def test_drops_uncompleted_games():
    games = games_from_cfbd_dicts([RAW_COMPLETED, RAW_UNCOMPLETED])
    assert len(games) == 1
    assert games[0].game_id == "401693678"
