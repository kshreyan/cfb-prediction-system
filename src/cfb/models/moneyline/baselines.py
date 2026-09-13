"""Baseline moneyline models.

These exist for two reasons: (1) they are legitimate baselines the full
ensemble must beat out-of-sample, and (2) an Elo-only baseline gives us
something end-to-end and honest to run before any market/stats data is
wired in.
"""
from __future__ import annotations

from dataclasses import dataclass


def american_to_implied_prob(odds: float) -> float:
    """Convert American moneyline odds to a *vig-included* implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove the vig from a two-way market by normalizing implied probs to sum to 1."""
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Implied probabilities must be positive")
    return prob_a / total, prob_b / total


@dataclass
class EloOnlyMoneylineBaseline:
    """Wraps raw Elo win probability as a moneyline prediction. No calibration
    layer is applied here on purpose -- calibration is measured and applied
    downstream (see cfb.calibration) so we can report how far off the raw
    Elo probabilities are before and after calibration.
    """

    def predict_home_win_prob(self, home_rating: float, away_rating: float,
                               home_field_adj: float) -> float:
        diff = (home_rating + home_field_adj) - away_rating
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


@dataclass
class HomeTeamAlwaysBaseline:
    """Predicts the home team always wins. A sanity-check baseline the real
    model must beat -- if it doesn't, something is badly wrong."""

    home_field_win_rate_prior: float = 0.58

    def predict_home_win_prob(self, *_args, **_kwargs) -> float:
        return self.home_field_win_rate_prior


@dataclass
class MarketOnlyBaseline:
    """De-vigged market-implied probability. This is the bar the ensemble
    must clear to claim any edge at all."""

    def predict_home_win_prob(self, home_ml_odds: float, away_ml_odds: float) -> float:
        home_implied = american_to_implied_prob(home_ml_odds)
        away_implied = american_to_implied_prob(away_ml_odds)
        home_fair, _ = devig_two_way(home_implied, away_implied)
        return home_fair
