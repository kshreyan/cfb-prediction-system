"""Unit tests for betting-line parsing and consensus computation, using a
small synthetic fixture shaped like a real CFBD lines response."""
from __future__ import annotations

from cfb.data.lines_loader import consensus_closing_lines, parse_lines

RAW_GAME_TWO_BOOKS = {
    "id": 1,
    "season": 2024,
    "week": 1,
    "lines": [
        {
            "provider": "DraftKings",
            "spread": -21,
            "spreadOpen": -17.5,
            "overUnder": 48,
            "overUnderOpen": 50,
            "homeMoneyline": -1800,
            "awayMoneyline": 1000,
        },
        {
            "provider": "Bovada",
            "spread": -20,
            "spreadOpen": -17,
            "overUnder": 47,
            "overUnderOpen": 49.5,
            "homeMoneyline": -1600,
            "awayMoneyline": 900,
        },
    ],
}

RAW_GAME_NO_MONEYLINE = {
    "id": 2,
    "season": 2024,
    "week": 1,
    "lines": [
        {
            "provider": "DraftKings",
            "spread": -3,
            "spreadOpen": -3,
            "overUnder": 55,
            "overUnderOpen": 55,
            "homeMoneyline": None,
            "awayMoneyline": None,
        }
    ],
}

RAW_GAME_NO_LINES = {"id": 3, "season": 2024, "week": 1, "lines": []}


def test_parse_lines_produces_closing_and_opening_rows():
    rows = parse_lines([RAW_GAME_TWO_BOOKS])
    closing = [r for r in rows if r.is_closing]
    opening = [r for r in rows if not r.is_closing]
    assert len(closing) == 2
    assert len(opening) == 2
    assert {r.book for r in closing} == {"DraftKings", "Bovada"}


def test_consensus_closing_lines_medians_across_books():
    rows = parse_lines([RAW_GAME_TWO_BOOKS])
    consensus = consensus_closing_lines(rows)
    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["market_spread_home"] == -20.5  # median of -21, -20
    assert row["market_total"] == 47.5
    assert row["n_books_spread"] == 2
    assert row["n_books_ml"] == 2
    assert 0.0 < row["market_home_win_prob"] < 1.0


def test_game_with_no_moneyline_has_zero_ml_books():
    rows = parse_lines([RAW_GAME_NO_MONEYLINE])
    consensus = consensus_closing_lines(rows)
    row = consensus.iloc[0]
    assert row["n_books_ml"] == 0
    assert row["market_spread_home"] == -3


def test_game_with_no_lines_at_all_produces_no_rows():
    rows = parse_lines([RAW_GAME_NO_LINES])
    assert rows == []
