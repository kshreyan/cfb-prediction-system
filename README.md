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
| Walk-forward backtest harness | ✅ built, **not yet run on real games** |
| CFBD data client | ✅ built, **not yet run — needs an API key** |
| Real historical backtest + calibration report | ⛔ blocked on CFBD API key |
| Spread model (margin distribution), total model (scoreline engine) | 🚧 not started |
| Stacked ensemble, calibration layer (isotonic/Platt) | 🚧 not started |
| CLV computation, GitHub Pages site, weekly workflow | 🚧 not started |

**Nothing in this repo currently claims a real backtest result.** The 14
tests that pass (`make test`) run against small hand-built synthetic game
fixtures to prove the *engine logic* is correct (rating updates, home
field, offseason regression, chronological/leakage safety) — they are not
a demonstration of predictive skill. That demonstration requires real
CFBD game/line data, which requires the API key described below.

## What I need from you to proceed

1. **A CFBD API key** — free, instant signup at
   https://collegefootballdata.com/key. Once you have it:
   ```
   export CFBD_API_KEY=your_key_here
   make check-cfbd
   ```
   Do not paste the key into chat or commit it to the repo — set it as an
   environment variable (or a local `.env` that's already gitignored).
2. **A GitHub repo URL** to push this to (I can help create one via `gh
   repo create` if you'd rather I do that — just confirm).
3. **A GitHub token** (or confirm you're authenticated via `gh auth
   login` already) if pushing requires auth beyond what's configured
   locally.

Once (1) arrives I'll run the real walk-forward Elo backtest across every
available CFBD season, report real calibration numbers, and continue
through the spread model, total model, ensemble, and deployment. (2) and
(3) are needed for the GitHub storage and Pages deployment steps.

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
make setup     # creates .venv, installs the package + dev deps
make test      # runs the full test suite (14 tests, all passing)
make lint      # ruff + mypy, both clean
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
