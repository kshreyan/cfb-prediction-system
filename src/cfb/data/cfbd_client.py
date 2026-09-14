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
from dotenv import load_dotenv

# Loads CFBD_API_KEY (and any other project secrets) from a local .env file
# if present. .env is gitignored -- the key never enters source control or
# any committed file. Safe to call repeatedly; does not override a var
# already set in the real environment.
load_dotenv()

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
                     season_type: str = "both", classification: str = "fbs") -> list[dict]:
        """Fetch games with full provenance stamping. Raises on API error
        rather than silently returning partial/fabricated data.

        `classification="fbs"` returns every game with an FBS home team,
        including FBS-vs-FCS games (needed to flag those separately) --
        it does NOT return FCS-vs-FCS games, which are out of scope.
        `season_type="both"` includes postseason (bowls/CFP).
        """
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.GamesApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            # cfbd's generated stubs type season_type/classification as enums;
            # the runtime client accepts the plain string values fine.
            games = api.get_games(
                year=season, week=week, season_type=season_type,  # type: ignore[arg-type]
                classification=classification,  # type: ignore[arg-type]
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
                     season_type: str = "both") -> list[dict]:
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
        """Per-game, per-team advanced stats (PPA/EPA, success rate,
        explosiveness) -- the raw material for a leak-free, walk-forward
        EPA feature engine (never a season-aggregated stat, which would
        leak future games' plays into an early-season prediction)."""
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.StatsApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            stats = api.get_advanced_game_stats(
                year=season, week=week, season_type="both"  # type: ignore[arg-type]
            )
            logger.info("fetched_advanced_stats", season=season, week=week, count=len(stats))
            return [
                {**s.to_dict(), "_source": "cfbd.advanced_stats",
                 "_fetched_at": fetched_at, "_season": season, "_week": week}
                for s in stats
            ]

    def fetch_sp_ratings(self, season: int) -> list[dict]:
        """One SP+ rating per team per season -- CFBD's free tier does not
        expose a week-by-week SP+ history, so this is only safe to use as
        a PRIOR-season preseason prior (see docs/methodology.md), never as
        an in-season feature for games in the same season it was computed."""
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.RatingsApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            ratings = api.get_sp(year=season)
            logger.info("fetched_sp_ratings", season=season, count=len(ratings))
            return [
                {**r.to_dict(), "_source": "cfbd.sp_ratings", "_fetched_at": fetched_at,
                 "_season": season}
                for r in ratings
            ]

    def fetch_recruiting_rankings(self, season: int) -> list[dict]:
        """One recruiting class ranking per team per season -- inherently a
        preseason signal (the class was signed before the season), so safe
        to use directly as that season's feature (not just prior-season)."""
        with cfbd.ApiClient(self._configuration()) as api_client:
            api = cfbd.RecruitingApi(api_client)
            fetched_at = datetime.now(UTC).isoformat()
            rankings = api.get_team_recruiting_rankings(year=season)
            logger.info("fetched_recruiting_rankings", season=season, count=len(rankings))
            return [
                {**r.to_dict(), "_source": "cfbd.recruiting", "_fetched_at": fetched_at,
                 "_season": season}
                for r in rankings
            ]
