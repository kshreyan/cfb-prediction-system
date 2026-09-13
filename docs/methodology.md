# Methodology notes

## Elo engine (`src/cfb/elo/engine.py`)

- **Update rule**: standard logistic Elo, `P(home win) = 1 / (1 + 10^(-diff/400))`,
  where `diff = (home_rating + home_field_adj) - away_rating`. `home_field_adj`
  is a single global constant (`EloConfig.home_field_advantage`, default 65
  points) applied only to non-neutral-site games. This is a simplification —
  per-venue home-field effects are a planned feature (see README roadmap) but
  are not yet implemented; don't read per-venue precision into the current
  numbers.
- **Margin-of-victory multiplier**: FiveThirtyEight-style,
  `ln(|margin| + 1) * (2.2 / (|winner_elo_diff| * 0.001 + 2.2))`, which
  dampens blowout rating swings against already-favored teams (avoids a
  55-3 win over an FCS team inflating a top team's rating as much as a
  55-3 win over a peer).
- **Conference-tier priors**: an unrated team is seeded at its conference
  tier's prior rather than a single global mean (`configs/conference_tiers.yaml`).
  Tiers: power conferences (1620), FBS independents (1560), Group of Five
  (1440), FCS (1250). These are modeling choices, not measured constants —
  they should be recalibrated against realized outcomes once real
  historical data is loaded (see README, "blocked on CFBD key").
- **Offseason regression**: at the first game of a new season, every
  team's rating is pulled 35% of the way back toward its (current) tier
  prior before that game is predicted. This reflects real turnover
  (roster churn, coaching changes) and prevents a rating built up over
  several strong seasons from persisting indefinitely.
- **Leak-free by construction**: `process_game()` computes and returns the
  `PreGamePrediction` from ratings as they stood *before* this game, then
  updates ratings afterward. Feeding games out of chronological order
  raises `LeakageError` rather than silently mispredicting. See
  `tests/leakage/test_no_future_leakage.py` for the property-based proof
  (truncation invariance under `hypothesis`).

## What is not yet built

Per the acceptance criteria in the README, the following are designed for
but not yet implemented, because they require real CFBD data (blocked on
an API key) and/or come after the Elo/moneyline baseline milestone:

- SP+/FPI-derived features, EPA/success-rate features, recruiting/returning
  production, transfer portal, weather/altitude/travel features.
- The spread model (margin distribution) and total model (bivariate-Poisson
  scoreline engine).
- The stacked moneyline ensemble (logistic + GBM + Elo + market on
  log-odds) — currently only the Elo-only, home-team-always, and
  market-only baselines exist (`src/cfb/models/moneyline/baselines.py`).
- Isotonic/Platt calibration layer (reliability diagrams and ECE are
  already implemented in `src/cfb/evaluation/metrics.py` and used by the
  Elo backtest).
- CLV computation, GitHub Pages site generation, and the weekly refresh
  workflow.
