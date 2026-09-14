"""Unit tests for Odds-API consensus building, using a small synthetic
fixture shaped like a real live-odds response (verified against a real
pull during development)."""
from __future__ import annotations

from cfb.data.odds_api_loader import odds_api_consensus

CFBD_GAMES = [
    {"id": 12345, "homeTeam": "Pittsburgh", "awayTeam": "Syracuse"},
]

ODDS_API_GAMES = [
    {
        "home_team": "Pittsburgh Panthers",
        "away_team": "Syracuse Orange",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Pittsburgh Panthers", "price": -440},
                        {"name": "Syracuse Orange", "price": 340},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "Pittsburgh Panthers", "price": -112, "point": -10.5},
                        {"name": "Syracuse Orange", "price": -108, "point": 10.5},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": -102, "point": 51.5},
                        {"name": "Under", "price": -118, "point": 51.5},
                    ]},
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "Pittsburgh Panthers", "price": -115, "point": -11.0},
                        {"name": "Syracuse Orange", "price": -105, "point": 11.0},
                    ]},
                ],
            },
        ],
    },
    {
        "home_team": "Unmatchable University",
        "away_team": "Nowhere State",
        "bookmakers": [],
    },
]


def test_consensus_extracts_spread_total_and_moneyline():
    result = odds_api_consensus(ODDS_API_GAMES, CFBD_GAMES, season=2026, week=3)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["game_id"] == "12345"
    assert row["market_spread_home"] == -10.75  # median of -10.5, -11.0
    assert row["market_total"] == 51.5
    assert row["n_books_spread"] == 2
    assert row["n_books_ml"] == 1
    assert 0.0 < row["market_home_win_prob"] < 1.0


def test_unmatched_game_is_dropped_not_guessed():
    result = odds_api_consensus(ODDS_API_GAMES, CFBD_GAMES, season=2026, week=3)
    # Only the matched Pittsburgh/Syracuse game should appear.
    assert len(result) == 1


def test_flipped_home_away_is_matched_and_sign_corrected():
    # CFBD says Kansas is home, Arizona State is away (neutral-site game);
    # Odds API disagrees and calls Arizona State home -- the loader must
    # still find the game AND express the spread/moneyline relative to
    # CFBD's home team (Kansas), not Odds API's.
    cfbd_games = [{"id": 999, "homeTeam": "Kansas", "awayTeam": "Arizona State"}]
    odds_games = [{
        "home_team": "Arizona State Sun Devils",
        "away_team": "Kansas Jayhawks",
        "bookmakers": [{
            "key": "draftkings",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Arizona State Sun Devils", "price": -150},
                    {"name": "Kansas Jayhawks", "price": 130},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": "Arizona State Sun Devils", "price": -110, "point": -3.0},
                    {"name": "Kansas Jayhawks", "price": -110, "point": 3.0},
                ]},
            ],
        }],
    }]
    result = odds_api_consensus(odds_games, cfbd_games, season=2026, week=3)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["game_id"] == "999"
    # Arizona State (Odds API's "home") was favored by 3 -- from CFBD's
    # home team (Kansas)'s perspective, Kansas is a +3 underdog.
    assert row["market_spread_home"] == 3.0
    # Kansas's (CFBD home) moneyline is the underdog price (+130), not -150.
    assert row["market_home_win_prob"] < 0.5
