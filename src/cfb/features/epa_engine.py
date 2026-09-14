"""Leak-free, walk-forward EPA/success-rate feature engine.

Per the master brief, EPA/play (offense & defense) and success rate are
"the single most predictive public metric" for CFB -- but CFBD's
advanced-stats endpoint only gives a *season-aggregate* view unless you
pull it per-game, and a season aggregate for a game in week 3 would
include that game's own plays (and week 4-15's) -- a direct leak. So
this engine treats each team's per-game advanced stats (PPA = predicted
points added, i.e. EPA; success rate) exactly like the Elo and scoreline
engines: predict from an EWMA of PRIOR games only, then update.

Not every game has advanced stats (FCS opponents in particular usually
don't have tracked play-by-play) -- when a team's stats for a game are
missing, that team's EWMA simply isn't updated for that game, rather
than updating it with a fabricated value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cfb.elo.types import GameResult

# Rough FBS long-run averages (PPA is points-added-per-play, success rate
# is the fraction of "successful" plays under Bill Connelly's down/distance
# definition) -- only used as the untrained-team starting prior.
OFF_EPA_DEFAULT = 0.07
SUCCESS_RATE_DEFAULT = 0.42
EWMA_ALPHA = 0.2
OFFSEASON_REGRESSION_WEIGHT = 0.5


@dataclass
class TeamEpaState:
    off_epa_ewma: float = OFF_EPA_DEFAULT
    off_success_ewma: float = SUCCESS_RATE_DEFAULT
    def_epa_allowed_ewma: float = OFF_EPA_DEFAULT
    def_success_allowed_ewma: float = SUCCESS_RATE_DEFAULT


@dataclass
class EpaEngine:
    teams: dict[str, TeamEpaState] = field(default_factory=dict)
    _current_season: int | None = field(default=None, repr=False)

    def _get_or_init(self, team: str) -> TeamEpaState:
        if team not in self.teams:
            self.teams[team] = TeamEpaState()
        return self.teams[team]

    def regress_offseason(self) -> None:
        w = OFFSEASON_REGRESSION_WEIGHT
        for s in self.teams.values():
            s.off_epa_ewma = OFF_EPA_DEFAULT + (s.off_epa_ewma - OFF_EPA_DEFAULT) * (1 - w)
            s.off_success_ewma = SUCCESS_RATE_DEFAULT + (s.off_success_ewma - SUCCESS_RATE_DEFAULT) * (1 - w)
            s.def_epa_allowed_ewma = OFF_EPA_DEFAULT + (s.def_epa_allowed_ewma - OFF_EPA_DEFAULT) * (1 - w)
            s.def_success_allowed_ewma = SUCCESS_RATE_DEFAULT + (s.def_success_allowed_ewma - SUCCESS_RATE_DEFAULT) * (1 - w)

    def process_game(self, game: GameResult, home_stats: dict | None,
                      away_stats: dict | None) -> dict:
        """Returns the PRE-game EWMA features for both teams, then updates
        state using this game's actual stats (skipped for a team whose
        stats are missing for this game)."""
        if self._current_season is None:
            self._current_season = game.season
        elif game.season != self._current_season:
            self.regress_offseason()
            self._current_season = game.season

        home = self._get_or_init(game.home_team)
        away = self._get_or_init(game.away_team)

        features = {
            "home_off_epa_pre": home.off_epa_ewma,
            "home_off_success_pre": home.off_success_ewma,
            "home_def_epa_allowed_pre": home.def_epa_allowed_ewma,
            "home_def_success_allowed_pre": home.def_success_allowed_ewma,
            "away_off_epa_pre": away.off_epa_ewma,
            "away_off_success_pre": away.off_success_ewma,
            "away_def_epa_allowed_pre": away.def_epa_allowed_ewma,
            "away_def_success_allowed_pre": away.def_success_allowed_ewma,
            "home_stats_available": home_stats is not None,
            "away_stats_available": away_stats is not None,
        }

        alpha = EWMA_ALPHA
        if home_stats is not None:
            home.off_epa_ewma = alpha * home_stats["off_ppa"] + (1 - alpha) * home.off_epa_ewma
            home.off_success_ewma = alpha * home_stats["off_success"] + (1 - alpha) * home.off_success_ewma
            home.def_epa_allowed_ewma = alpha * home_stats["def_ppa"] + (1 - alpha) * home.def_epa_allowed_ewma
            home.def_success_allowed_ewma = alpha * home_stats["def_success"] + (1 - alpha) * home.def_success_allowed_ewma
        if away_stats is not None:
            away.off_epa_ewma = alpha * away_stats["off_ppa"] + (1 - alpha) * away.off_epa_ewma
            away.off_success_ewma = alpha * away_stats["off_success"] + (1 - alpha) * away.off_success_ewma
            away.def_epa_allowed_ewma = alpha * away_stats["def_ppa"] + (1 - alpha) * away.def_epa_allowed_ewma
            away.def_success_allowed_ewma = alpha * away_stats["def_success"] + (1 - alpha) * away.def_success_allowed_ewma

        return features


def _stats_lookup(raw_advanced_stats: list[dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for row in raw_advanced_stats:
        game_id = str(row.get("gameId"))
        team = row.get("team")
        offense, defense = row.get("offense") or {}, row.get("defense") or {}
        if team is None or offense.get("ppa") is None or defense.get("ppa") is None:
            continue
        lookup[(game_id, team)] = {
            "off_ppa": offense["ppa"],
            "off_success": offense.get("successRate"),
            "def_ppa": defense["ppa"],
            "def_success": defense.get("successRate"),
        }
    return lookup


def build_epa_features(games: list[GameResult], raw_advanced_stats: list[dict]) -> pd.DataFrame:
    """Leak-free, walk-forward: one row per game with each team's PRE-game
    EPA/success-rate EWMA state."""
    lookup = _stats_lookup(raw_advanced_stats)
    ordered = sorted(games, key=lambda g: (g.start_date, g.season, g.week, g.game_id))
    engine = EpaEngine()

    rows = []
    for g in ordered:
        home_stats = lookup.get((g.game_id, g.home_team))
        away_stats = lookup.get((g.game_id, g.away_team))
        features = engine.process_game(g, home_stats, away_stats)
        rows.append({"game_id": g.game_id, **features})
    return pd.DataFrame(rows)
