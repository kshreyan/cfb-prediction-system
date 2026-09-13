"""Thin, typed wrapper around the CFBD API.

Requires a free API key from https://collegefootballdata.com/key, supplied
via the CFBD_API_KEY environment variable -- never hardcode it. Every
record fetched is stamped with provenance (source, fetched_at, season,
week) per the project's "no fabricated data" rule: if CFBD doesn't have a
field for a given game (e.g. no line was ever posted), we store null and
mark is_real_data=False downstream -- we do not impute or invent it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime

import cfbd
import structlog

logger = structlog.get_logger(__name__)

CFBD_API_KEY_ENV = "CFBD_API_KEY"


class MissingApiKeyError(RuntimeError):
    pass


@dataclass
class CfbdClient:
    api_key: str

    @classmethod
    def from_env(cls) -> CfbdClient:
        key = os.environ.get(CFBD_API_KEY_ENV)
        if not key:
            raise MissingApiKeyError(
                f"Set {CFBD_API_KEY_ENV} (get a free key at "
                "https://collegefootballdata.com/key). Never hardcode it in source."
            )
        return cls(api_key=key)

    def _configuration(self) -> cfbd.Configuration:
        config = cfbd.Configuration(access_token=self.api_key)
        return config

    def fetch_games(self, season: int, week: int | None = None,
                     season_type: str = "regular") -> list[dict]:
        """Fetch games with full provenance stamping. Raises on API error
        rather than silently returning partial/fabricated data."""
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.GamesApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            # cfbd's generated stubs type season_type as its SeasonType enum;
            # the runtime client accepts the plain string values fine.
            games = api.get_games(
                year=season, week=week, season_type=season_type  # type: ignore[arg-type]
            )
            logger.info("fetched_games", season=season, week=week, count=len(games))
            return [
                {
                    **g.to_dict(),
                    "_source": "cfbd.games",
                    "_fetched_at": fetched_at,
                    "_season": season,
                    "_week": week,
                }
                for g in games
            ]

    def fetch_lines(self, season: int, week: int | None = None,
                     season_type: str = "regular") -> list[dict]:
        """Fetch betting lines. A game with no posted line comes back with
        no lines entry at all -- callers must treat that as unavailable,
        not as a missing-value-to-impute."""
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.BettingApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            lines = api.get_lines(
                year=season, week=week, season_type=season_type  # type: ignore[arg-type]
            )
            logger.info("fetched_lines", season=season, week=week, count=len(lines))
            return [
                {
                    **l.to_dict(),
                    "_source": "cfbd.lines",
                    "_fetched_at": fetched_at,
                    "_season": season,
                    "_week": week,
                }
                for l in lines
            ]

    def fetch_team_talent(self, season: int) -> list[dict]:
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.TeamsApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            talent = api.get_talent(year=season)
            return [
                {**t.to_dict(), "_source": "cfbd.talent", "_fetched_at": fetched_at,
                 "_season": season}
                for t in talent
            ]

    def fetch_advanced_stats(self, season: int, week: int | None = None) -> list[dict]:
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.StatsApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            stats = api.get_advanced_game_stats(year=season, week=week)
            return [
                {**s.to_dict(), "_source": "cfbd.advanced_stats",
                 "_fetched_at": fetched_at, "_season": season, "_week": week}
                for s in stats
            ]
