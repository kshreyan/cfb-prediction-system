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

## Calibration (`src/cfb/calibration/isotonic_calibrator.py`)

Walk-forward isotonic regression (`sklearn.isotonic.IsotonicRegression`),
refit per season on an expanding window of strictly-prior seasons only.
Below `MIN_TRAIN_GAMES` (300) prior rows, the raw probability passes
through uncalibrated rather than fitting isotonic regression on too
little data (isotonic regression, being non-parametric, overfits small
samples more readily than the spread/total models' 1-D linear/EWMA fits).

## Ensemble (`src/cfb/ensemble/blend.py`)

Blends two probabilities on the **log-odds** scale:
`blended = sigmoid(w * logit(model_p) + (1-w) * logit(market_p))`. The
weight `w` is chosen by grid search (0.0 to 1.0, step 0.05) minimizing
log loss on an expanding window of strictly-prior seasons — this is the
project's "weights learned only via nested time-series CV" requirement.
Below `MIN_TRAIN_GAMES` (150) prior rows (the first season a model has
market data to blend against), falls back to an even 50/50 weight.

Real result (README): the weight walk-forward-learned on Elo shrank from
0.50 to 0.05 across 2021→2025 as it correctly detected Elo wasn't adding
information beyond the market — the ensemble ended up statistically tied
with market-only on log loss (beat it in 2/5 seasons), not ahead of it.

## CLV (`src/cfb/evaluation/clv.py`)

`spread_clv_points` / `total_clv_points` compute points of closing-line
value for a pick made at the opening line and held to close, using
CFBD's real opening+closing spread/total snapshots (`consensus_opening_lines`
/ `consensus_closing_lines` in `src/cfb/data/lines_loader.py`). Moneyline
CLV is not computed: CFBD's free tier has no opening-moneyline field, so
there's nothing to measure movement against without fabricating one.
Real result (README): pooled mean CLV is ≈0 for both spread (−0.002 pts)
and total (−0.033 pts) — no demonstrated edge, consistent with the
ATS/O-U records.

## EPA/success-rate feature engine (`src/cfb/features/epa_engine.py`)

Same discipline as the Elo and scoreline engines: predicts from an EWMA
(α=0.2) of PRIOR games' per-team PPA (predicted points added, i.e. EPA)
and success rate -- offense and defense separately -- then updates. Built
from CFBD's per-game advanced-stats endpoint specifically because the
season-aggregate version of that endpoint would leak an early-season
game's own (and later weeks') plays into its own prediction. A team's
state isn't updated for a game with no tracked advanced stats (common
for FCS opponents) rather than updating with a fabricated value.
Regresses toward the league-average prior at each offseason boundary,
same as Elo and the scoreline engine.

## Preseason priors (`src/cfb/features/preseason_priors.py`)

SP+, recruiting-class ranking, and talent composite, with deliberately
different leakage handling per metric: **SP+ is lagged a full season**
(a game in season S uses season S-1's final SP+ -- CFBD's free tier
exposes no week-by-week SP+ history, so using season S's own rating for
season S games would be the exact "end-of-season SP+ on early-season
games" trap the master brief calls out by name). **Recruiting and talent
are used directly for the season they describe** -- both are set before
the season kicks off, so no lag is needed or correct. Missing data is
NaN, not imputed.

## GBM moneyline model (`src/cfb/models/moneyline/gbm.py`)

LightGBM classifier, walk-forward per season (train = strictly prior
seasons only, refit at each boundary, `MIN_TRAIN_GAMES=1500` before it
activates -- roughly seasons 2017+ given ~800 FBS games/season). 15
features: Elo differential + the EPA engine's 8 offense/defense state
values (both teams) + the 6 preseason-prior values (both teams). NaN
features pass through untouched; LightGBM splits on missingness
natively rather than this pipeline imputing a fill value.

**Hyperparameters were changed once, for a diagnosed reason, not tuned
by grid search.** A walk-forward bucketed error analysis (favorite size,
week-of-season, EPA/SP+ data availability) found the original config was
overconfident specifically in near-even games (|Elo diff| < 100, ~43% of
all games): mean |prob-0.5| of 0.116 vs. Elo's own 0.088 there, despite
there being less real signal to be confident about -- and losing to Elo
in that bucket in 4/5 seasons individually. Classic overfitting symptom.
Fixed with shallower trees (max_depth 4->3, num_leaves 15->8), higher
min_child_samples (30->60), and L1/L2 regularization -- validated
walk-forward across all 9 active seasons (2017-2025), not just cherry-
picked on the bucket that motivated the change: pooled log loss
0.5499->0.5489, ECE 0.0217->0.0119 (nearly halved), pickem-bucket log
loss 0.6810->0.6765. Real, modest, not a breakthrough.

Real result (README, `gbm-eval`, current config): GBM beat Elo-only on
log loss in 5/5 seasons (2021-2025) -- up from 4/5 pre-regularization --
but did not beat market-only in any season (0/5), and folding it into a
3-way ensemble (calibrated Elo ↔ GBM ↔ market) didn't improve on the
2-way Elo+market ensemble's already-modest 2/5 record. Live weekly
predictions (`cfb predict`) do not yet use the GBM -- only Elo -- since
wiring the full feature pipeline into the not-yet-played-game path is
separate, not-yet-done work.

## Real-time odds (`src/cfb/data/odds_api_client.py`, `odds_api_loader.py`, `odds_api_matching.py`)

A paid, real-time, multi-book source (The Odds API), used as the
preferred market feed for `cfb predict` with CFBD as fallback. Design
choices worth remembering:

- **Live endpoint only, not a historical backfill.** The live endpoint
  is cheap (fixed credits per call, not per game); the historical
  endpoint is priced per timestamp snapshot, and backfilling 2015-2025
  at the granularity needed for real tick-level CLV would cost far more
  than a 20,000-credit quota. Every call logs its exact cost
  (`x-requests-*` response headers) rather than spending silently.
- **Team-name matching bug, found and fixed**: naive startswith prefix
  matching let an Odds API name like "Iowa State Cyclones" ambiguously
  resolve to either "Iowa" or "Iowa State" depending on Python's
  per-process hash-randomized set/dict iteration order -- the exact same
  input could match differently across runs. Fixed by always preferring
  the longest (most specific) candidate name; regression-tested.
- **Home/away disagreement**: the two sources occasionally label
  different teams "home" for the same neutral-site game. The loader
  tries the flipped team pairing before giving up, and when it uses a
  flipped match, negates the spread and swaps the moneylines so the
  result is always expressed relative to CFBD's home team.
- Sources are never averaged together -- one wins per game/field
  (Odds API preferred), recorded in a `market_source` field, not blended.

## What is not yet built

The Elo engine, spread model, total model, calibration layer, ensemble,
CLV computation, EPA/preseason-prior features, and GBM model have all
been run end-to-end against real CFBD data (2015–2025 — see README for
results). Still not implemented:

- Transfer portal, weather, travel/rest, injuries/QB status, and
  per-venue home field — none of these are in any model's feature set
  yet, and are the most likely remaining lever to close the gap to the
  market (weather/injuries especially are exactly the kind of
  game-specific information a market prices that a season-level or
  EWMA-based feature structurally can't capture).
- GBM/richer features for the spread and total models — currently only
  the moneyline stack has them; spread/total are still Elo-diff-only and
  scoring-rate-EWMA-only respectively.
- Wiring the GBM into live weekly predictions (`cfb predict` still uses
  Elo only).
- Walk-forward hyperparameter tuning — Elo's K-factor/home-field-adj,
  the EWMA α values, and the GBM's tree hyperparameters are all fixed
  heuristics, never tuned via nested time-series CV against alternatives.
- GitHub Pages is built but not enabled in repo settings; the weekly
  refresh workflow is written but inert without a repo secret — both
  deliberately left for the user to opt into (see README, Credentials).
