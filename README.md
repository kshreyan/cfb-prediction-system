# CFB Prediction System (moneyline · spread · total)

A reproducible, honestly-backtested college football (FBS) prediction
system. This is a research/analytics project, **not betting advice**.

## The honesty standard (read this first)

**Do not target raw accuracy. Target calibration and honest out-of-sample
performance vs the closing line.**

- Straight-up moneyline accuracy will likely land around 75–80%. **This is
  expected and near-meaningless** — most FBS games are mismatches the
  market already prices correctly, so high win-rate mostly reflects talent
  gaps, not model skill.
- The real tests are: **ATS record (~52–54% is a realistic ceiling)**,
  **CLV vs the closing line**, and **calibration** (reliability, Brier,
  log loss, ECE) on all three markets.
- Any ATS backtest materially above ~54% is **presumed leakage** until
  proven otherwise — it gets investigated, not celebrated.
- Straight-up accuracy is always reported alongside ATS-excluding-FCS and
  CLV, so a reader can't mistake mismatch-driven accuracy for skill.
- Predictions are immutable once made; results are stored separately and
  joined in afterward. Missing data is reported as unavailable — **never
  fabricated, imputed, or guessed**. No "beats Vegas" claim is made
  without multi-season CLV evidence.

## Current status

This repo is under active build-out. Honest status as of the last commit:

| Component | Status |
|---|---|
| Repo scaffold, CI, leakage test suite | ✅ built |
| Conference-tier-aware, offseason-regressed Elo engine | ✅ built, unit + property tested |
| Moneyline baselines (Elo-only, market-only, home-always) | ✅ built |
| Calibration metrics (log loss, Brier, ECE, reliability curve) | ✅ built |
| CFBD data client, real-data loader, season cache | ✅ built and run against live CFBD data |
| Walk-forward Elo backtest, 2015–2025 (9,505 real FBS games) | ✅ run — see results below |
| Betting lines (spreads/totals/ML) pulled + market baseline | ✅ run — see results below |
| Spread model (margin distribution, skew-normal residuals) | ✅ built and backtested vs real market spreads |
| Total model (scoreline engine, skew-normal residuals) | ✅ built and backtested vs real market totals |
| CLV vs closing line (spread, total) | ✅ computed from real opening/closing snapshots — see results below |
| Isotonic calibration layer, log-odds ensemble (weight learned walk-forward) | ✅ built and backtested vs real market moneylines |
| SP+/FPI, EPA, recruiting, portal, weather/travel features | 🚧 not started — every model above uses only Elo/scoring-rate state |
| GitHub Pages site, weekly workflow | 🚧 not started |

### Real backtest results (raw Elo, no calibration layer, no market blend)

Walk-forward, leak-free, 2015–2025 regular + postseason, offseason
regression applied at every season boundary. This is **Elo alone** — no
SP+/EPA features, no market data, no calibration layer yet, so treat
these as a first honest baseline, not a finished model.

| Season | Games (FBS-only) | Accuracy | Log loss | Brier | ECE |
|---|---|---|---|---|---|
| 2015 | 765 | 69.0% | 0.579 | 0.198 | 0.095 |
| 2016 | 760 | 70.4% | 0.565 | 0.193 | 0.054 |
| 2017 | 776 | 69.7% | 0.558 | 0.190 | 0.053 |
| 2018 | 772 | 71.8% | 0.535 | 0.180 | 0.047 |
| 2019 | 774 | 72.6% | 0.538 | 0.181 | 0.037 |
| 2020 | 534 | 69.3% | 0.572 | 0.195 | 0.070 |
| 2021 | 770 | 71.7% | 0.553 | 0.187 | 0.047 |
| 2022 | 776 | 67.9% | 0.589 | 0.202 | 0.043 |
| 2023 | 792 | 69.9% | 0.557 | 0.189 | 0.045 |
| 2024 | 798 | 67.3% | 0.577 | 0.198 | 0.059 |
| 2025 | 808 | 72.0% | 0.553 | 0.187 | 0.047 |

**Read this correctly, per the honesty standard above:** 67–73% straight-up
on FBS-vs-FBS games is Elo working as intended, not a headline result —
raw Elo has no market or efficiency data yet, so this is a floor, not a
ceiling. Including FBS-vs-FCS mismatches raises every season's accuracy by
2–6 points (e.g. 2024: 67.3% → 70.9%) with zero added skill — exactly the
inflation effect this project's honesty standard warns about, which is why
FBS-only is the headline number and mismatches are reported separately.
ECE (0.04–0.10) shows raw logistic Elo is reasonably but not perfectly
calibrated; an isotonic/Platt layer (planned) should tighten this further.

Reproduce with `make backtest` (or `python -m cfb.cli backtest
--start-season 2015 --end-season 2025`); raw CFBD pulls cache to
`data/raw/` (gitignored) and predictions/summaries write to
`data/processed/` (also gitignored — regenerate, don't expect them in
git).

### Elo vs the de-vigged market (moneyline)

CFBD's moneyline coverage in this feed only starts in **2021** — earlier
seasons have spreads/totals but essentially no posted moneyline, so this
comparison is 2021–2025 only, on the 3,771 FBS-vs-FBS games with a real
posted line (`python -m cfb.cli market-eval`):

| Season | Games | Elo log loss | Market log loss | Elo wins? |
|---|---|---|---|---|
| 2021 | 721 | 0.572 | 0.550 | No |
| 2022 | 708 | 0.619 | 0.589 | No |
| 2023 | 772 | 0.568 | 0.523 | No |
| 2024 | 787 | 0.584 | 0.541 | No |
| 2025 | 783 | 0.568 | 0.536 | No |

**The market beat raw Elo on log loss in 5/5 seasons.** This is exactly
what the honesty standard predicts: a single-signal, uncalibrated Elo
model has no business beating a market that prices in injuries, weather,
motivation, and everything else Elo doesn't see. This is the expected,
reported-without-spin result at this stage — the target for the finished
ensemble is to close this gap by blending with the market, not to beat it
on Elo alone.

### Ensemble: isotonic-calibrated Elo blended with the market

Isotonic calibration is fit walk-forward on Elo's raw probabilities
(strictly-prior seasons only), then blended with the market on the
log-odds scale using a weight chosen by grid search on strictly-prior
seasons' log loss only — never the season it's then scored on
(`python -m cfb.cli ensemble-eval`):

| Season | n | Weight on Elo | Elo raw LL | Elo calib. LL | Market LL | Ensemble LL |
|---|---|---|---|---|---|---|
| 2021 | 721 | 0.50 | 0.572 | 0.565 | 0.550 | **0.545** |
| 2022 | 708 | 0.30 | 0.619 | 0.675 | 0.589 | 0.596 |
| 2023 | 772 | 0.15 | 0.568 | 0.607 | 0.523 | 0.526 |
| 2024 | 787 | 0.05 | 0.584 | 0.623 | 0.541 | 0.542 |
| 2025 | 783 | 0.05 | 0.568 | 0.564 | 0.536 | **0.535** |

**Ensemble beat Elo-only on log loss 5/5 seasons (expected — it contains
the market). Ensemble beat market-only in only 2/5 seasons (2021,
2025) — essentially statistically tied with the market, not a
demonstrated edge.** The walk-forward weight search correctly recognized
this in real time: the learned weight on Elo shrinks from an even 0.50 in
its first eligible season down to 0.05 by 2024–2025, i.e. the ensemble
learned to mostly defer to the market because Elo wasn't adding
information — which is the walk-forward weight-learning working exactly
as intended, not a failure of it. **Per the honesty standard, this is not
a "beats Vegas" result and is not reported as one.**

One genuine, worth-stating nuance: **ECE often improves under the
ensemble even in seasons where log loss doesn't** (e.g. 2021: ECE 0.024
vs. Elo's 0.046 and the market's 0.040; 2022: 0.026 vs. 0.046/0.048).
Averaging two differently-biased predictors can improve calibration
without improving discrimination — a known property of ensembling, not
an edge claim; log loss (which rewards discrimination, not just
calibration) is still the metric this project selects on.

### Spread model: real ATS backtest (the actual skill test)

Margin predicted from Elo rating differential via a walk-forward linear
regression (refit each season on all strictly-prior seasons' games,
skew-normal residuals for fat tails), compared against the real,
posted consensus market spread (`python -m cfb.cli spread-eval`):

**Overall ATS record, FBS-only, 2015–2025: 4,023–4,151–151 (49.2%).**

| Season | Record | Win % | Margin MAE |
|---|---|---|---|
| 2015 | 352–403–10 | 46.6% | 15.4 |
| 2016 | 359–385–16 | 48.3% | 14.0 |
| 2017 | 367–390–19 | 48.5% | 13.7 |
| 2018 | 383–376–13 | 50.5% | 13.6 |
| 2019 | 385–378–11 | 50.5% | 13.3 |
| 2020 | 256–268–10 | 48.9% | 13.7 |
| 2021 | 385–374–11 | 50.7% | 13.5 |
| 2022 | 376–387–13 | 49.3% | 13.1 |
| 2023 | 380–397–15 | 48.9% | 13.3 |
| 2024 | 372–409–17 | 47.6% | 13.6 |
| 2025 | 408–384–16 | 51.5% | 12.8 |

By favorite size: `|spread| < 14` → 49.0% (2,736–2,843–106); `|spread| ≥
14` → 49.6% (1,287–1,308–45). Cover-probability ECE: 0.125 (pooled).

**No season is materially above ~54% — the honesty standard's leakage
trip-wire never fires here.** A single-feature (Elo-diff-only) margin
model with no efficiency, injury, weather, or market-derived features
shows **no ATS edge**, landing at or slightly below breakeven every year.
That is the expected, legitimate result for this stage of the build, not
a bug — see the honesty standard at the top of this README.

### Total model: real O/U backtest

Team scoring-rate EWMA (attack/defense blend, leak-free, offseason-
regressed) vs. the real posted consensus total, skew-normal residuals
(`python -m cfb.cli total-eval`):

**Overall O/U record, FBS-only, 2015–2025: 4,222–4,016–87 (51.3%).**

Per-season win rates range 47.6%–54.8%; two seasons (2020: 54.8%, n=529;
2023: 54.7%, n=783) individually cross the honesty standard's 54%
trip-wire. Investigated, not celebrated: with 11 independent seasons
tested, 1–2 crossing p<0.05 by chance is expected under the null, both
values are only marginally significant (binomial p≈0.01–0.03, not
overwhelming), and there is no plausible leakage channel — the total
prediction is generated entirely from each team's own EWMA scoring state
and a residual distribution fit only on strictly prior seasons, with
market data entering nowhere upstream of the final O/U comparison. The
pooled 51.3% across all 9,132 decided games is the number that matters,
and it shows no real edge. Over-probability ECE: 0.079 (pooled).

### CLV: the decisive benchmark

CFBD's free-tier feed carries one **opening** and one **closing**
snapshot per book per game for spreads and totals (not a full intraday
time series — this is a real limitation, not a tick-by-tick closing line,
so treat these as directionally meaningful rather than precise to the
tenth of a point). That's enough for a genuine CLV computation: decide
the pick using the model's cover/over probability against the **opening**
line, then measure whether the line moved in the model's favor by close
— independent of whether the individual bet won (`python -m cfb.cli
spread-eval` / `total-eval`, 2021–2025, the seasons with real
opening+closing coverage).

**No moneyline CLV is reported** — CFBD's free tier has no opening
moneyline field, only opening spread/total, so a moneyline CLV number
would have to be fabricated to fill that gap. It isn't.

| Market | Games | Mean CLV (points) | % picks with positive CLV |
|---|---|---|---|
| Spread | 3,937 | **−0.002** | 41.2% |
| Total | 3,942 | **−0.033** | 44.3% |

**Both are indistinguishable from zero — no demonstrated CLV edge on
either market.** One nuance worth flagging rather than hiding: the
"% positive" figures look worse than the near-zero means suggest, because
~13% of spread lines (and a comparable share of totals) don't move at all
between open and close, and a strict `> 0` threshold counts every
zero-movement game as "not positive." The pooled **mean** CLV — which
does credit zero-movement games as exactly neutral rather than as a loss
— is the more honest single number, and it says the same thing both
markets' ATS/O-U records already said: **this system currently shows no
measurable edge over the market, on any metric, on any of the three
targets.** That is the correct, current, and complete answer to this
project's own acceptance criteria — not a hedge.

## Credentials

- **CFBD API key**: stored locally in `.env` (gitignored, never committed,
  loaded automatically via `python-dotenv`). Not stored in any synced or
  cross-session memory — `.env` on this machine is the single source of
  truth for it.
- **GitHub**: repo created at https://github.com/kshreyan/cfb-prediction-system
  (private) and connected as `origin`, using the already-authenticated
  local `gh` CLI session.

## Repo layout

```
src/cfb/
  data/         CFBD API client (provenance-stamped: source, fetched_at, season, week)
  elo/          Chronological, leak-free, conference-tier-aware Elo engine
  features/     (planned) leak-free, as-of-kickoff feature builders
  ratings/      (planned) SP+/FPI ingestion and blending
  models/
    moneyline/  Baselines today; stacked ensemble planned
    spread/     (planned) margin distribution model
    total/      (planned) bivariate-Poisson scoreline engine
  ensemble/     (planned) model+market blending on log-odds
  calibration/  (planned) isotonic/Platt calibration layer
  backtest/     Walk-forward harness
  evaluation/   Accuracy, log loss, Brier, ECE, reliability curves
  reporting/    (planned) report + site generation
configs/        conference_tiers.yaml — Elo tier priors, not ground truth
tests/
  unit/         Engine + metrics logic, synthetic fixtures
  property/     (hypothesis-based invariants beyond leakage)
  leakage/      Structural proof no future info reaches a prediction
docs/           methodology notes
```

## Setup

```bash
make setup        # creates .venv, installs the package + dev deps
make test         # runs the full test suite (43 tests, all passing)
make lint         # ruff + mypy, both clean
export CFBD_API_KEY=...  # or rely on the local .env (already configured)
make backtest      # walk-forward Elo backtest, 2015-2025 by default
make market-eval   # Elo vs de-vigged market moneyline, real odds
make ensemble-eval # calibrated Elo + market ensemble, real odds
make spread-eval   # real ATS backtest vs the market spread
make total-eval    # real O/U backtest vs the market total
```

Dependencies are pinned as ranges in `pyproject.toml` and fully resolved
in `requirements-lock.txt` (generated via `pip freeze` against the exact
environment the tests above were run in — Python 3.13, macOS). Note:
`pydantic` is pinned to `<2` because the `cfbd` SDK (5.x) requires it;
config models in this repo intentionally avoid pydantic-v2-only features.

## Backtesting philosophy

Walk-forward, week by week, expanding window across seasons, with
offseason Elo regression applied between seasons — never a random
train/test split. Baselines the ensemble must beat out-of-sample:
Elo-only, SP+/FPI-only, market-only, home-team-always,
higher-ranked-always. If it doesn't beat the market, that's what gets
reported — a legitimate, respectable result in this project's own terms.

See `docs/methodology.md` for the Elo engine's exact update rule, prior
values, and what is/isn't implemented yet.
