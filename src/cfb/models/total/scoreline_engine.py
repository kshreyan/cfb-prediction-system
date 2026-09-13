"""Total (game points) model -- the "scoreline engine".

Design note, stated plainly rather than overclaimed: this is NOT a full
bivariate-Poisson goal-scoring simulation ported from a soccer project.
CFB point totals are sums of discrete scoring plays worth 3/6/7/8 points
each, not unit-increment events like soccer goals, and are meaningfully
overdispersed relative to a Poisson process (garbage time, onside-kick
chaos, and overtime periods all fatten the right tail). Modeling each
team's own points as Poisson would understate that variance.

Instead: each team's expected points are a leak-free, walk-forward EWMA
of its own scoring rate blended with its opponents' allowed rate (the
same attack/defense-strength idea bivariate-Poisson models use, without
assuming a Poisson *shape*). The total's residual distribution is then
fit walk-forward with a skew-normal (same approach and same honest
fat-tail justification as the spread model) rather than assumed Normal.
A true discrete scoreline simulation (e.g. Negative-Binomial per team,
convolved for the total) is a documented future enhancement -- see
docs/methodology.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from cfb.elo.types import GameResult

LEAGUE_AVG_POINTS = 27.5  # rough FBS long-run per-team average; only the
                          # starting prior for teams with no history yet.
EWMA_ALPHA = 0.12
HOME_FIELD_POINTS = 2.5
OFFSEASON_REGRESSION_WEIGHT = 0.5  # scoring rates regress faster than Elo
                                   # ratings -- year-to-year roster turnover
                                   # swings scoring more than "true strength".

MIN_TRAIN_GAMES = 150
FALLBACK_RESID = (0.0, 0.0, 17.0)  # (skew a, loc, scale) -- wide symmetric prior


@dataclass
class TeamScoreState:
    points_scored_ewma: float = LEAGUE_AVG_POINTS
    points_allowed_ewma: float = LEAGUE_AVG_POINTS


@dataclass
class ScorelineEngine:
    """Chronological, leak-free, like the Elo engine: predict from current
    EWMAs, then update them with the result."""
    teams: dict[str, TeamScoreState] = field(default_factory=dict)
    _current_season: int | None = field(default=None, repr=False)

    def _get_or_init(self, team: str) -> TeamScoreState:
        if team not in self.teams:
            self.teams[team] = TeamScoreState()
        return self.teams[team]

    def regress_offseason(self) -> None:
        for state in self.teams.values():
            state.points_scored_ewma = (
                LEAGUE_AVG_POINTS
                + (state.points_scored_ewma - LEAGUE_AVG_POINTS) * (1 - OFFSEASON_REGRESSION_WEIGHT)
            )
            state.points_allowed_ewma = (
                LEAGUE_AVG_POINTS
                + (state.points_allowed_ewma - LEAGUE_AVG_POINTS) * (1 - OFFSEASON_REGRESSION_WEIGHT)
            )

    def process_game(self, game: GameResult) -> tuple[float, float]:
        """Returns (predicted_home_points, predicted_away_points) computed
        from PRE-game state, then updates state with the actual result."""
        if self._current_season is None:
            self._current_season = game.season
        elif game.season != self._current_season:
            self.regress_offseason()
            self._current_season = game.season

        home = self._get_or_init(game.home_team)
        away = self._get_or_init(game.away_team)

        predicted_home = 0.5 * (home.points_scored_ewma + away.points_allowed_ewma) + HOME_FIELD_POINTS
        predicted_away = 0.5 * (away.points_scored_ewma + home.points_allowed_ewma)

        alpha = EWMA_ALPHA
        home.points_scored_ewma = alpha * game.home_points + (1 - alpha) * home.points_scored_ewma
        home.points_allowed_ewma = alpha * game.away_points + (1 - alpha) * home.points_allowed_ewma
        away.points_scored_ewma = alpha * game.away_points + (1 - alpha) * away.points_scored_ewma
        away.points_allowed_ewma = alpha * game.home_points + (1 - alpha) * away.points_allowed_ewma

        return predicted_home, predicted_away


def run_total_backtest(games: list[GameResult]) -> pd.DataFrame:
    """Runs the scoreline engine walk-forward over chronologically-sorted
    games and returns one row per game with the pre-game point predictions
    and the actual result."""
    ordered = sorted(games, key=lambda g: (g.start_date, g.season, g.week, g.game_id))
    engine = ScorelineEngine()
    rows = []
    for g in ordered:
        pred_home, pred_away = engine.process_game(g)
        rows.append({
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "start_date": g.start_date,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "predicted_home_points": pred_home,
            "predicted_away_points": pred_away,
            "predicted_total": pred_home + pred_away,
            "actual_total": g.home_points + g.away_points,
            "is_fbs_vs_fbs": g.home_is_fbs and g.away_is_fbs,
        })
    return pd.DataFrame(rows)


def fit_total_residuals(total_df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward skew-normal fit of (actual_total - predicted_total)
    residuals, refit once per season boundary on strictly-prior seasons
    -- same expanding-window discipline as the margin model."""
    df = total_df.sort_values(["season", "start_date"]).reset_index(drop=True)
    n = len(df)
    resid_a = np.full(n, np.nan)
    resid_loc = np.full(n, np.nan)
    resid_scale = np.full(n, np.nan)

    for season in sorted(df["season"].unique()):
        train = df[df["season"] < season]
        test_idx = df.index[df["season"] == season]

        if len(train) >= MIN_TRAIN_GAMES:
            residuals = (train["actual_total"] - train["predicted_total"]).to_numpy()
            a, loc, scale = stats.skewnorm.fit(residuals)
        else:
            a, loc, scale = FALLBACK_RESID

        resid_a[test_idx] = a
        resid_loc[test_idx] = loc
        resid_scale[test_idx] = scale

    df["resid_a"] = resid_a
    df["resid_loc"] = resid_loc
    df["resid_scale"] = resid_scale
    return df


def over_probability(predicted_total: float, resid_a: float, resid_loc: float,
                      resid_scale: float, market_total: float) -> float:
    """P(actual_total > market_total)."""
    threshold = market_total - predicted_total
    return float(1.0 - stats.skewnorm.cdf(threshold, resid_a, resid_loc, resid_scale))
