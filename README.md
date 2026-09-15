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
| Live weekly predictions (immutable JSON snapshot, real upcoming games) | ✅ built and run — see "This week" below |
| Static site (GitHub Pages, `docs/`) — predictions, calibration & CLV charts, honest ATS record | ✅ built; **Pages not yet enabled in repo settings — ask before I flip that on** |
| EPA/success-rate (leak-free walk-forward) + SP+/recruiting/talent priors + GBM moneyline model | ✅ built and backtested — see "GBM vs Elo vs market" below |
| Transfer portal, weather, travel/rest, injuries/QB status, per-venue home field | 🚧 not started — still not in any model's feature set |
| Scheduled weekly-refresh GitHub Actions workflow | ✅ workflow written (`.github/workflows/weekly-refresh.yml`), **inert until `CFBD_API_KEY` is added as a repo secret — won't add without asking first** (see Credentials) |
| Real-time multi-book odds (The Odds API) for live weekly predictions | ✅ integrated as the preferred market source, CFBD as fallback — see below |

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

### GBM vs Elo vs market: does feature-rich modeling actually help?

The single biggest acknowledged gap after the first pass of this project
was that every model used only Elo/scoring-rate state — no EPA, no SP+,
no recruiting, no talent. This closes that gap for moneyline: a
LightGBM classifier trained walk-forward per season (strictly-prior
seasons only, refit at each boundary) on 15 features — Elo differential,
leak-free walk-forward EPA/success-rate state (offense **and** defense,
both teams, built the same way as the Elo/scoreline engines: predict
from an EWMA of prior games, then update — see
`src/cfb/features/epa_engine.py`), and three preseason priors (SP+
**lagged one full season** to avoid leaking in-season results, recruiting
class ranking and talent composite used directly since those are signed
before the season starts — see `src/cfb/features/preseason_priors.py`
for why the lag differs by metric). Missing values (no tracked advanced
stats for a game, no recruiting/talent record for a team) are passed
through as NaN, not imputed — LightGBM splits on missingness natively
(`python -m cfb.cli gbm-eval`, 2021–2025, the games with real posted
moneylines):

| Season | n | Elo-raw LL | **GBM LL** | Market LL | 3-way ensemble LL |
|---|---|---|---|---|---|
| 2021 | 721 | 0.572 | **0.563** | 0.550 | 0.545 |
| 2022 | 708 | 0.619 | **0.616** | 0.589 | 0.592 |
| 2023 | 772 | 0.568 | **0.558** | 0.523 | 0.527 |
| 2024 | 787 | 0.584 | **0.574** | 0.541 | 0.542 |
| 2025 | 783 | 0.568 | **0.556** | 0.536 | 0.535 |

**GBM beat Elo-only on log loss in 5/5 seasons** (up from 4/5 — see the
diagnostic + fix below). **GBM did not beat market-only in any season
(0/5)**, and the 3-way ensemble still only beats market-only in 2/5
seasons. **Read plainly: richer features made the model consistently
better than its own bare-Elo predecessor, but still not competitive with
the market.** That's a real result, not a disappointing one to talk
around — beating a market that prices in injuries, weather, and
information this system still doesn't have was never the honest
expectation at this stage.

#### Diagnostic: finding and fixing a real overfitting pattern

Per a direct ask to find what's working, what isn't, and actually fix
it — not just report the headline numbers again — the 3,771 backtested
GBM predictions were bucketed by favorite size, week-of-season, and
data availability to look for a genuine, non-noise error pattern (not a
blind hyperparameter sweep). One held up cleanly across seasons:

**In truly close games (|Elo differential| < 100 — the "pickem" bucket,
~43% of all games), the GBM was *more confident* than raw Elo
(mean |prob − 0.5| of 0.116 vs. Elo's 0.088) despite there being
structurally less signal to be confident about — and that extra
confidence wasn't earned: GBM lost to Elo in this bucket in 4 of 5
seasons, individually.** Classic GBM overfitting symptom — fitting noise
in the EPA/recruiting/talent features specifically where the true signal
is weakest.

Fix: retrained with a more conservative config targeted at exactly that
failure mode — shallower trees (`max_depth` 4→3, `num_leaves` 15→8),
higher `min_child_samples` (30→60), and L1/L2 regularization added
(`reg_alpha`/`reg_lambda` = 1.0). Re-validated walk-forward across all 9
seasons the GBM is active for (2017–2025, not just the 5 with posted
moneylines) — a real, if modest, improvement, not a lucky overfit to one
metric:

| Metric | Original | Regularized |
|---|---|---|
| Pooled log loss | 0.5499 | **0.5489** |
| Pooled Brier | 0.1863 | **0.1859** |
| Pooled ECE | 0.0217 | **0.0119** (nearly halved) |
| Pooled accuracy | 0.7131 | 0.7115 (negligible dip, expected — not the target metric) |
| Pickem-bucket log loss | 0.6810 | **0.6765** |

This is the config now shipped in `src/cfb/models/moneyline/gbm.py`
(see its docstring for the same writeup) — it's what produced the 5/5
table above. The calibration gain (ECE nearly halved) is the more
meaningful part of this result; the log-loss gain is real but small, and
none of this closes the gap to the market — reported as exactly what it
is, not oversold.

Known limitation: live weekly predictions (`cfb predict`) still use Elo
only, not the GBM — wiring the full EPA/SP+/recruiting/talent feature
pipeline into the live (not-yet-played-game) path is real remaining
work, not yet done.

### Spread and total models: diagnosed, fixed, and re-validated

Per a direct ask to find what's working, what isn't, and actually fix
it, the same bucketed-diagnostic treatment applied to the GBM (see
above) was applied to the spread and total models. It found something
more fundamental than the GBM's overfitting: **the raw cover/over
probability's reliability curve was badly non-monotonic** — e.g. for
spread, predicted cover probability of 5% carried an observed cover rate
of 40%, and predicted 94% carried an observed rate of only 58%. That's
not a subtle miscalibration, it's close to uninformative in the middle
of the range. Root cause: the point predictions (predicted margin/total)
do carry *some* real signal, but disagreements with the market are
mostly estimation noise rather than insight (consistent with the ATS/O-U
records already being near 50%) — running that noise through a
skew-normal CDF manufactures false precision.

**Fix**: the same walk-forward isotonic calibration already proven for
the moneyline model (`src/cfb/calibration/isotonic_calibrator.py`),
applied here to cover/over probability, fit only on decided (non-push)
games. The ATS/O-U pick now uses the calibrated probability, not the
raw one:

| Metric | Spread — raw | Spread — calibrated | Total — raw | Total — calibrated |
|---|---|---|---|---|
| ECE (pooled) | 0.1251 | **0.0351** | 0.0792 | **0.0154** |
| ATS/O-U record | 4,023–4,151–151 (49.2%) | **4,082–4,092–151 (49.9%)** | 4,222–4,016–87 (51.3%) | **4,196–4,042–87 (50.9%)** |
| Mean CLV (points) | −0.002 | **+0.081** | −0.033 | **+0.255** |

**Read this precisely, because it's a genuinely two-sided result.**
Calibration is a real, large, honest fix (ECE cut by 3–5×) — but it is
*not* a newly discovered edge: ATS/O-U win rates barely moved (both
still within a point of 50%, no season materially above the 54%
honesty-standard trip-wire), because **isotonic regression, given a
genuinely near-flat/non-monotonic input, correctly outputs something
close to flat** — about 71% of calibrated cover/over probabilities now
land within 2 points of 50%, with real differentiation concentrated in
extreme mismatches. That's not a display bug (the live site and CLI now
show one decimal place specifically so this doesn't look like one) — 
it's the honest picture: **a single-feature margin model and an EWMA
scoring-rate model mostly don't know enough to differentiate closer
games, and now they honestly say so instead of pretending otherwise.**

The mean-CLV improvement (both markets moved from slightly negative to
modestly positive) is the one number here worth a second look rather
than filing under "no change" — it's small (well under half a point per
bet) and this is a single-cut result, not multi-season-validated proof
of an edge, but it's a genuinely different sign than before, driven by
the calibrated probability changing which side gets picked on a large
fraction of games (roughly half, across both markets). Worth continued
monitoring, not yet a claim.

By favorite size (spread): `|spread| < 14` → 49.4% (2,754–2,825–106);
`|spread| ≥ 14` → 51.2% (1,328–1,267–45).

Reproduce with `python -m cfb.cli spread-eval` / `total-eval` — both now
print raw-vs-calibrated ECE explicitly.

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

| Market | Games | Mean CLV (points) — pre-calibration | Mean CLV (points) — post-calibration |
|---|---|---|---|
| Spread | 3,937 | −0.002 | **+0.081** |
| Total | 3,942 | −0.033 | **+0.255** |

**Both moved from ~zero to modestly positive after the isotonic
calibration fix documented above** (the CLV pick now uses the calibrated
probability too) — real, but small (well under a point per bet, on a
single historical cut, not yet re-validated on a fresh season) and not
yet a claim of a demonstrated edge. The pooled **mean** CLV — which
credits zero-movement games as exactly neutral rather than as a loss —
remains the more honest single number than "% positive," since roughly
13% of lines don't move between open and close at all. Read together
with the ATS/O-U records above (both still within a point of 50%):
**this system shows no large, reliably demonstrated edge over the
market on any of the three targets, but the calibration fix measurably
improved probability quality and CLV sign on both spread and total.**
That's the correct, current, complete answer — better than the last
version of this section, not hedged into sounding bigger than it is.

## Credentials

- **CFBD API key**: stored locally in `.env` (gitignored, never committed,
  loaded automatically via `python-dotenv`). Not stored in any synced or
  cross-session memory — `.env` on this machine is the single source of
  truth for it.
- **The Odds API key (paid)**: same treatment — local `.env` only
  (`ODDS_API_KEY`), auto-loaded, never in memory or git. This is a
  metered paid API; every call logs its real credit cost from the
  response headers rather than spending silently (see
  `src/cfb/data/odds_api_client.py`). Used for live current-week odds
  in `cfb predict` (cheap — ~3 credits/week regardless of game count);
  **not** used for a full historical backfill — see "Real-time odds"
  below for why and what a scoped alternative would cost.
- **GitHub**: repo created at https://github.com/kshreyan/cfb-prediction-system
  (private) and connected as `origin`, using the already-authenticated
  local `gh` CLI session.
- **GitHub Pages / Actions secret — deliberately not yet done**: the site
  in `docs/` is built and committed, but I haven't enabled Pages in repo
  settings, because on a private repo that makes this content reachable
  at a public URL — a visibility change, not just a local file write, so
  it's your call rather than mine to make unilaterally. Similarly, a
  scheduled weekly-refresh GitHub Action would need `CFBD_API_KEY` added
  as a repo secret (`.env` alone doesn't reach CI) — say the word for
  either and I'll do it.

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
make gbm-eval      # feature-rich GBM (EPA/SP+/recruiting/talent) vs Elo vs market
make spread-eval   # real ATS backtest vs the market spread
make total-eval    # real O/U backtest vs the market total
make predict       # immutable prediction snapshot for the next upcoming week
make report        # builds docs/ (site) from whatever's in data/processed/
```

### Real-time odds (The Odds API)

Added a paid, real-time, multi-book odds source (`src/cfb/data/odds_api_client.py`)
as the **preferred** market feed for live weekly predictions, with CFBD
kept as the fallback for games it doesn't cover — never averaged
together, since blending two different snapshot times would produce a
meaningless number (`_merge_market_sources` in
`weekly_predictions.py` picks one source per game and records which in
a new `market_source` field, shown as a column on the site).

- **Cost-aware by design**: the live-odds endpoint costs a small, fixed
  number of credits per call regardless of game count (~3 credits for
  the *entire* week's NCAAF slate across moneyline/spread/total) — cheap
  enough for routine weekly use out of a 20,000-credit quota. The
  historical endpoint is priced per timestamp snapshot instead and would
  cost far more than the remaining quota to backfill 2015–2025 at the
  granularity needed for real CLV — so that backfill was **not**
  attempted; every API call logs its exact credit cost so spend is
  never silent.
- **Team-name matching** (`odds_api_matching.py`) handles the "City
  Mascot" vs. CFBD's plain-school-name conventions (e.g. "Pittsburgh
  Panthers" → "Pittsburgh"), including a real bug caught and fixed
  during this integration: naive prefix matching let "Iowa State
  Cyclones" ambiguously match either "Iowa" or "Iowa State" depending on
  Python's per-process hash-randomized set/dict iteration order —
  non-deterministic across runs. Fixed by always preferring the longest
  (most specific) matching name, with a regression test locking that in.
- **Home/away disagreement handling**: the two sources occasionally
  disagree on which team is "home" for a neutral-site game (observed
  live: CFBD called Kansas home vs. Arizona State; Odds API called it
  the other way). The loader tries the flipped pairing before giving up,
  and correctly negates the spread / swaps the moneylines so everything
  ends up expressed relative to *CFBD's* home team, not whichever team
  Odds API happened to label home.
- A game unmatched by name, or with no market posted anywhere, is
  reported as unavailable — never guessed or imputed.

### This week (live, generated 2026-09-14)

`make predict` was run against the real, live 2026 season (currently week
3, 75 upcoming FBS games) — not a demo, and now backed by real-time
multi-book odds where available. Sample rows (see `docs/` after
`make report`, or `data/processed/predictions/*.json` for the full,
immutable snapshot):

| Game | Home win % | Pred. margin | Market spread (cover %) | Market ML | Source |
|---|---|---|---|---|---|
| Georgia @ Arkansas | 12% | −18.1 | +24.5 (64%) | 7% | both |
| Florida State @ Alabama | 91% | +20.9 | −20.5 (51%) | 90% | both |
| Portland State @ Oregon `[FBS–FCS]` | 100% | +46.6 | −57.5 (26%) | no line | the-odds-api |

That last row is the real-time feed's contribution: CFBD had no line at
all for that mismatch, but The Odds API did — more coverage, not just
more precision.

The full site (`docs/index.html`) renders all 75 games plus the
calibration reliability charts, ATS/O-U record charts, CLV charts, and
the GBM-vs-Elo-vs-market comparison shown above — it's built and
committed, but **GitHub Pages is not yet enabled** in the repo settings
(see Credentials below for why that's a separate ask, not something I
turned on unilaterally).

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
