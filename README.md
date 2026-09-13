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
| Betting lines (spreads/totals/ML), market baseline, CLV | 🚧 not pulled yet — no live line source connected |
| Spread model (margin distribution), total model (scoreline engine) | 🚧 not started |
| Stacked ensemble, calibration layer (isotonic/Platt) | 🚧 not started |
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
There is no ATS number yet because there's no spread model or market data
connected — that's next.

Reproduce with `make backtest` (or `python -m cfb.cli backtest
--start-season 2015 --end-season 2025`); raw CFBD pulls cache to
`data/raw/` (gitignored) and predictions/summaries write to
`data/processed/` (also gitignored — regenerate, don't expect them in
git).

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
make setup     # creates .venv, installs the package + dev deps
make test      # runs the full test suite (16 tests, all passing)
make lint      # ruff + mypy, both clean
export CFBD_API_KEY=...  # or rely on the local .env (already configured)
make backtest  # real walk-forward Elo backtest, 2015-2025 by default
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
