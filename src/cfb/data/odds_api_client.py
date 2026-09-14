"""Thin, typed wrapper around The Odds API (the-odds-api.com) -- a paid,
real-time, multi-book odds source used to supplement CFBD's free-tier
lines (which have only one open/close snapshot per book, not a real
intraday time series).

Requires ODDS_API_KEY, loaded the same way as CFBD_API_KEY (local .env,
gitignored, never hardcoded). Every call is provenance-stamped and, since
this is a metered paid API, every call logs its exact credit cost from
the `x-requests-*` response headers so spend is always visible, not
silently accumulated.

Cost model (as observed against the live API, subject to the provider's
own pricing): the *live* odds endpoint charges a small, roughly fixed
number of credits per (markets x regions) combination regardless of how
many games are returned -- cheap enough for routine weekly use. The
*historical* endpoint charges per (market x region x timestamp)
snapshot requested, and does NOT scale with games either, but you need
one timestamp per distinct kickoff-time cluster to get accurate
closing lines for a full week -- meaningfully more expensive. Never
call the historical endpoint in a loop without the caller (and the
human on the other end) knowing roughly how many credits that will
cost; this module logs the cost of every single call so that's always
visible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests
import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)

ODDS_API_KEY_ENV = "ODDS_API_KEY"
BASE_URL = "https://api.the-odds-api.com/v4"
NCAAF_SPORT_KEY = "americanfootball_ncaaf"


class MissingOddsApiKeyError(RuntimeError):
    pass


@dataclass
class OddsApiClient:
    api_key: str

    @classmethod
    def from_env(cls) -> OddsApiClient:
        key = os.environ.get(ODDS_API_KEY_ENV)
        if not key:
            raise MissingOddsApiKeyError(
                f"Set {ODDS_API_KEY_ENV} in .env. This is a paid API -- never hardcode it."
            )
        return cls(api_key=key)

    def _get(self, path: str, params: dict) -> tuple[Any, dict]:
        resp = requests.get(
            f"{BASE_URL}{path}", params={**params, "apiKey": self.api_key}, timeout=30
        )
        resp.raise_for_status()
        cost_info = {
            "requests_used_this_call": resp.headers.get("x-requests-last"),
            "requests_remaining": resp.headers.get("x-requests-remaining"),
        }
        logger.info("odds_api_call", path=path, **cost_info)
        return resp.json(), cost_info

    def fetch_live_odds(self, sport: str = NCAAF_SPORT_KEY, markets: str = "h2h,spreads,totals",
                         regions: str = "us") -> list[dict]:
        """Current live odds for every upcoming game in `sport`. Cheap --
        cost is per (markets x regions) combination, not per game."""
        data, cost = self._get(f"/sports/{sport}/odds/",
                                {"regions": regions, "markets": markets, "oddsFormat": "american"})
        fetched_at = datetime.now(UTC).isoformat()
        return [
            {**game, "_source": "the-odds-api.live", "_fetched_at": fetched_at,
             "_credit_cost": cost}
            for game in data
        ]

    def fetch_historical_odds(self, timestamp_iso: str, sport: str = NCAAF_SPORT_KEY,
                               markets: str = "h2h,spreads,totals",
                               regions: str = "us") -> dict:
        """One point-in-time snapshot of every game's odds as of
        `timestamp_iso`. Meaningfully more expensive than the live
        endpoint -- see module docstring. Returns the raw envelope
        (timestamp/previous_timestamp/next_timestamp/data) since callers
        need those to know how stale/fresh this snapshot actually is."""
        data, cost = self._get(f"/historical/sports/{sport}/odds/",
                                {"regions": regions, "markets": markets,
                                 "oddsFormat": "american", "date": timestamp_iso})
        fetched_at = datetime.now(UTC).isoformat()
        return {**data, "_source": "the-odds-api.historical", "_fetched_at": fetched_at,
                "_credit_cost": cost}
