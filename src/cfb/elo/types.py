"""Data types shared by the Elo engine and its backtest harness."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class GameResult:
    """A single completed FBS/FCS game, chronologically orderable.

    All fields must be knowable strictly *before* kickoff except
    home_points/away_points, which are only used to update ratings
    *after* the pre-game prediction has been produced.
    """

    game_id: str
    season: int
    week: int
    start_date: datetime
    home_team: str
    away_team: str
    home_conference: str | None
    away_conference: str | None
    home_is_fbs: bool
    away_is_fbs: bool
    neutral_site: bool
    home_points: int
    away_points: int

    def __post_init__(self) -> None:
        if self.home_points < 0 or self.away_points < 0:
            raise ValueError(f"Negative score in game {self.game_id}")

    @property
    def home_won(self) -> bool:
        return self.home_points > self.away_points

    @property
    def margin(self) -> int:
        """Home margin: positive means home won by this many points."""
        return self.home_points - self.away_points


@dataclass(frozen=True, slots=True)
class PreGamePrediction:
    """Elo's prediction, captured BEFORE the game's result is applied.

    This is the artifact that must be persisted immutably in the real
    backtest -- it is generated strictly from ratings as they stood
    before this game's own result was known.
    """

    game_id: str
    season: int
    week: int
    start_date: datetime
    home_team: str
    away_team: str
    home_rating_pre: float
    away_rating_pre: float
    home_win_prob: float
