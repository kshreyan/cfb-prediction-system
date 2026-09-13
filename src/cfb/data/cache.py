"""Local JSON cache for raw CFBD pulls, keyed by season.

This is a cache, not a data lake: it exists so re-running a backtest
doesn't re-hit the API for seasons whose games are already final. It is
not committed to git (see .gitignore) -- on a fresh clone it's rebuilt
from CFBD directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from cfb.data.cfbd_client import CfbdClient

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def games_cache_path(season: int) -> Path:
    return RAW_DIR / f"games_{season}.json"


def lines_cache_path(season: int) -> Path:
    return RAW_DIR / f"lines_{season}.json"


def fetch_season_games(client: CfbdClient, season: int, force_refresh: bool = False) -> list[dict]:
    """Returns raw CFBD game dicts for one season (FBS games, both regular
    + postseason), using the local cache unless force_refresh=True or the
    cached season isn't fully complete yet (so an in-progress season is
    always re-fetched to pick up newly completed games)."""
    path = games_cache_path(season)
    if path.exists() and not force_refresh:
        cached = json.loads(path.read_text())
        if all(g.get("completed") for g in cached) and cached:
            return cached

    games = client.fetch_games(season=season, season_type="both", classification="fbs")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(games, indent=2, default=str))
    return games


def fetch_seasons(client: CfbdClient, seasons: list[int], force_refresh: bool = False) -> list[dict]:
    all_games: list[dict] = []
    for season in seasons:
        all_games.extend(fetch_season_games(client, season, force_refresh=force_refresh))
    return all_games


def fetch_season_lines(client: CfbdClient, season: int, force_refresh: bool = False) -> list[dict]:
    path = lines_cache_path(season)
    if path.exists() and not force_refresh:
        return json.loads(path.read_text())

    lines = client.fetch_lines(season=season, season_type="both")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lines, indent=2, default=str))
    return lines


def fetch_lines_for_seasons(client: CfbdClient, seasons: list[int],
                             force_refresh: bool = False) -> list[dict]:
    all_lines: list[dict] = []
    for season in seasons:
        all_lines.extend(fetch_season_lines(client, season, force_refresh=force_refresh))
    return all_lines
