"""Chronological, leak-free Elo engine for FBS college football.

Design goals (see docs/methodology.md):
  * Strict chronological processing -- a game's prediction is generated
    from ratings as they stood immediately before that game, and only
    afterwards is the result folded in. Calling process_game() twice or
    out of date order raises, so leakage is a structural impossibility
    rather than a convention.
  * Conference-tier priors -- a new/unrated team starts at its
    conference tier's prior rating rather than a single global mean, so
    an unrated Sun Belt team and an unrated SEC team are not treated as
    equal.
  * Offseason regression to the mean -- ratings partially reset toward
    the team's current tier prior between seasons, controlled by
    `offseason_regression_weight`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from cfb.elo.config import EloConfig
from cfb.elo.types import GameResult, PreGamePrediction


@dataclass
class TeamState:
    rating: float
    conference: str | None
    is_fbs: bool


class LeakageError(RuntimeError):
    """Raised when games are fed to the engine out of chronological order."""


@dataclass
class EloEngine:
    config: EloConfig
    teams: dict[str, TeamState] = field(default_factory=dict)
    _current_season: int | None = field(default=None, repr=False)
    _last_key: tuple | None = field(default=None, repr=False)

    def _get_or_init(self, team: str, conference: str | None, is_fbs: bool) -> TeamState:
        state = self.teams.get(team)
        if state is None:
            prior = self.config.prior_for_conference(conference, is_fbs)
            state = TeamState(rating=prior, conference=conference, is_fbs=is_fbs)
            self.teams[team] = state
        else:
            # Keep conference/FBS status current for future offseason regression.
            state.conference = conference
            state.is_fbs = is_fbs
        return state

    @staticmethod
    def win_probability(rating_diff: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))

    def _mov_multiplier(self, margin: int, elo_diff_winner: float) -> float:
        base = self.config.mov_multiplier_base
        scale = self.config.mov_multiplier_scale
        return math.log(abs(margin) + 1) * (base / (abs(elo_diff_winner) * scale + base))

    def regress_offseason(self) -> None:
        """Pull every rated team partway back toward its tier prior.

        Must be called exactly once between seasons, before the first
        game of the new season is processed.
        """
        weight = self.config.offseason_regression_weight
        for state in self.teams.values():
            prior = self.config.prior_for_conference(state.conference, state.is_fbs)
            state.rating = prior + (state.rating - prior) * (1.0 - weight)

    def process_game(self, game: GameResult) -> PreGamePrediction:
        key = (game.start_date, game.season, game.week, game.game_id)
        if self._last_key is not None and key < self._last_key:
            raise LeakageError(
                f"Game {game.game_id} (start={game.start_date}) processed out of "
                f"chronological order after {self._last_key}. Feeding games "
                "out of order risks using future information to predict the past."
            )
        self._last_key = key

        if self._current_season is None:
            self._current_season = game.season
        elif game.season != self._current_season:
            if game.season < self._current_season:
                raise LeakageError(
                    f"Season went backwards: {game.season} after {self._current_season}"
                )
            self.regress_offseason()
            self._current_season = game.season

        home = self._get_or_init(game.home_team, game.home_conference, game.home_is_fbs)
        away = self._get_or_init(game.away_team, game.away_conference, game.away_is_fbs)

        home_field_adj = 0.0 if game.neutral_site else self.config.home_field_advantage
        rating_diff = (home.rating + home_field_adj) - away.rating
        home_win_prob = self.win_probability(rating_diff)

        prediction = PreGamePrediction(
            game_id=game.game_id,
            season=game.season,
            week=game.week,
            start_date=game.start_date,
            home_team=game.home_team,
            away_team=game.away_team,
            home_rating_pre=home.rating,
            away_rating_pre=away.rating,
            home_win_prob=home_win_prob,
        )

        # --- Update ratings using the now-known result. Everything above
        # this line used only pre-game information; everything below
        # only affects *future* predictions. ---
        home_won = game.home_won
        actual_home_score = 1.0 if home_won else (0.5 if game.margin == 0 else 0.0)
        winner_elo_diff = rating_diff if home_won else -rating_diff
        mov_mult = self._mov_multiplier(game.margin, winner_elo_diff)
        delta = self.config.k_factor * mov_mult * (actual_home_score - home_win_prob)

        home.rating += delta
        away.rating -= delta

        return prediction

    def process_all(self, games: list[GameResult]) -> list[PreGamePrediction]:
        ordered = sorted(games, key=lambda g: (g.start_date, g.season, g.week, g.game_id))
        return [self.process_game(g) for g in ordered]

    def rating_of(self, team: str) -> float | None:
        state = self.teams.get(team)
        return state.rating if state else None
