"""Converts raw CFBD game dicts (as returned by CfbdClient.fetch_games) into
the engine's GameResult type. Kept separate from the client so the parsing
logic is unit-testable without hitting the network."""
from __future__ import annotations

from datetime import datetime

from cfb.elo.types import GameResult


def _parse_date(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    # CFBD's to_dict() renders timestamps like "2024-08-24 20:00:00+00:00"
    return datetime.fromisoformat(value)


def games_from_cfbd_dicts(raw_games: list[dict]) -> list[GameResult]:
    """Only completed games are usable for backtesting -- future/unplayed
    games have no result to leak or to score against, so they're dropped
    here rather than passed through with fabricated scores."""
    out = []
    for g in raw_games:
        if not g.get("completed"):
            continue
        if g.get("homePoints") is None or g.get("awayPoints") is None:
            continue
        out.append(
            GameResult(
                game_id=str(g["id"]),
                season=g["season"],
                week=g["week"],
                start_date=_parse_date(g["startDate"]),
                home_team=g["homeTeam"],
                away_team=g["awayTeam"],
                home_conference=g.get("homeConference"),
                away_conference=g.get("awayConference"),
                home_is_fbs=g.get("homeClassification") == "fbs",
                away_is_fbs=g.get("awayClassification") == "fbs",
                neutral_site=bool(g.get("neutralSite")),
                home_points=g["homePoints"],
                away_points=g["awayPoints"],
            )
        )
    return out
