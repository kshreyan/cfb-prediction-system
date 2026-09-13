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

## Spread model (`src/cfb/models/spread/margin_model.py`)

- **Predicted margin**: `home_margin = slope * elo_diff + intercept`, a
  simple 1-D OLS fit walk-forward once per season boundary on an
  expanding window of all *strictly prior* seasons (never mid-season
  refit, never using the current/future season). `elo_diff` is recovered
  exactly from the Elo win probability via the logistic's inverse
  (`400 * log10(p/(1-p))`), so it's the same information the moneyline
  model used, not a re-approximation.
- **Residuals**: fit with `scipy.stats.skewnorm` (not Normal) on the
  training window's residuals, to capture the fat right tail and skew
  real CFB margins have (blowouts, garbage time, occasional overtime).
- **First season fallback**: with fewer than `MIN_TRAIN_GAMES` (150)
  prior games (i.e. the first season in the dataset), falls back to a
  fixed heuristic slope (~28 Elo points per point of margin) with a wide
  symmetric residual prior, rather than fitting on too little data.
- **Cover probability**: `P(home covers) = 1 - skewnorm.cdf(-spread_home
  - predicted_margin; a, loc, scale)`.

## Total model (`src/cfb/models/total/scoreline_engine.py`)

Explicitly **not** a full bivariate-Poisson goal-scoring simulation —
see the module's own docstring for why (CFB points are sums of 3/6/7/8
discrete scoring plays, not unit Poisson events, and are overdispersed
relative to Poisson). Instead: each team's expected points are a
leak-free EWMA (α=0.12) blend of its own scoring rate and its opponents'
allowed rate — the same attack/defense-strength idea bivariate-Poisson
models use, without assuming a Poisson *shape*. EWMAs regress toward the
league average (27.5) at each offseason boundary, faster than Elo's own
regression (roster turnover swings scoring more than "true strength").
The total's residual distribution is fit walk-forward with a skew-normal,
same discipline as the spread model. A genuine discrete scoreline
simulation (e.g. Negative-Binomial per team, convolved for the total) is
a real future enhancement, not implemented here.

## What is not yet built

The Elo engine, spread model, and total model have all been run
end-to-end against real CFBD data (2015–2025 — see README for results).
Still not implemented:

- CLV vs a true closing line — CFBD's free tier gives one line snapshot
  per book per game, not full line-movement history (see README's CLV
  caveat); a proper CLV computation needs a source with real closing
  timestamps.
- SP+/FPI-derived features, EPA/success-rate features, recruiting/returning
  production, transfer portal, weather/altitude/travel features — none of
  the three models above use anything but Elo/scoring-rate state yet.
- The stacked moneyline ensemble (logistic + GBM + Elo + market on
  log-odds) — currently only the Elo-only, home-team-always, and
  market-only baselines exist (`src/cfb/models/moneyline/baselines.py`),
  now validated against real odds (README: market beats raw Elo 5/5
  seasons on log loss, as expected pre-ensemble).
- Isotonic/Platt calibration layer (reliability diagrams and ECE are
  already implemented in `src/cfb/evaluation/metrics.py` and used
  throughout the moneyline/spread/total backtests).
- GitHub Pages site generation and the weekly refresh workflow.
