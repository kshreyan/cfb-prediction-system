"""Builds the static GitHub Pages site (docs/) from the latest
prediction snapshot and the backtest CSVs already on disk.

Static means static: every number and chart here is pre-computed by the
CLI commands that write data/processed/*.csv and
data/processed/predictions/*.json; this module only reads those files
and renders HTML/PNG. Nothing here calls CFBD or does any live
computation, per the project's "no live browser compute" rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from cfb.evaluation.metrics import accuracy, expected_calibration_error, log_loss
from cfb.reporting.charts import (
    render_ats_record_chart,
    render_clv_chart,
    render_model_comparison_chart,
    render_reliability_chart,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _read_csv_if_exists(name: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / name
    return pd.read_csv(path) if path.exists() else None


def latest_prediction_snapshot() -> dict | None:
    if not PREDICTIONS_DIR.exists():
        return None
    files = sorted(PREDICTIONS_DIR.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def build_site(output_dir: str = "docs") -> None:
    out = REPO_ROOT / output_dir
    assets = out / "assets"
    data_dir = out / "data"
    assets.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    snapshot = latest_prediction_snapshot()
    if snapshot is not None:
        (data_dir / "latest_predictions.json").write_text(json.dumps(snapshot, indent=2))

    elo_df = _read_csv_if_exists("elo_backtest_predictions.csv")
    ensemble_df = _read_csv_if_exists("ensemble_predictions.csv")
    spread_by_season = _read_csv_if_exists("spread_backtest_by_season.csv")
    total_by_season = _read_csv_if_exists("total_backtest_by_season.csv")
    spread_clv = _read_csv_if_exists("spread_clv_by_season.csv")
    total_clv = _read_csv_if_exists("total_clv_by_season.csv")
    gbm_summary = _read_csv_if_exists("gbm_summary.csv")

    charts: dict[str, str] = {}
    headline: dict[str, object] = {}

    if elo_df is not None:
        fbs = elo_df[elo_df["is_fbs_vs_fbs"]]
        y, p = fbs["home_won"].astype(int).to_numpy(), fbs["home_win_prob"].to_numpy()
        render_reliability_chart(y, p, assets / "reliability_elo.png",
                                  "Moneyline calibration -- raw Elo (FBS-only, 2015-2025, pooled)")
        charts["reliability_elo"] = "assets/reliability_elo.png"
        headline["elo_accuracy"] = accuracy(y, p)
        headline["elo_logloss"] = log_loss(y, p)
        headline["elo_ece"] = expected_calibration_error(y, p)
        headline["elo_n_games"] = len(fbs)

    if ensemble_df is not None:
        y = ensemble_df["home_won_int"].to_numpy()
        p_ens = ensemble_df["ensemble_prob"].to_numpy()
        p_mkt = ensemble_df["market_home_win_prob"].to_numpy()
        render_reliability_chart(y, p_ens, assets / "reliability_ensemble.png",
                                  "Moneyline calibration -- calibrated Elo + market ensemble "
                                  "(2021-2025, pooled)")
        charts["reliability_ensemble"] = "assets/reliability_ensemble.png"
        headline["ensemble_logloss"] = log_loss(y, p_ens)
        headline["market_logloss"] = log_loss(y, p_mkt)
        headline["ensemble_n_games"] = len(ensemble_df)

    if spread_by_season is not None:
        render_ats_record_chart(spread_by_season, assets / "ats_record.png",
                                 "ATS win % by season (FBS-only) -- the real skill test")
        charts["ats_record"] = "assets/ats_record.png"
        decided_wins = spread_by_season["wins"].sum()
        decided_losses = spread_by_season["losses"].sum()
        decided_pushes = spread_by_season["pushes"].sum()
        headline["ats_record"] = f"{decided_wins}-{decided_losses}-{decided_pushes}"
        headline["ats_win_pct"] = decided_wins / (decided_wins + decided_losses)

    if total_by_season is not None:
        render_ats_record_chart(total_by_season, assets / "ou_record.png",
                                 "O/U win % by season (FBS-only)")
        charts["ou_record"] = "assets/ou_record.png"
        wins = total_by_season["wins"].sum()
        losses = total_by_season["losses"].sum()
        pushes = total_by_season["pushes"].sum()
        headline["ou_record"] = f"{wins}-{losses}-{pushes}"
        headline["ou_win_pct"] = wins / (wins + losses)

    if spread_clv is not None:
        render_clv_chart(spread_clv, assets / "clv_spread.png", "Spread CLV by season (points)")
        charts["clv_spread"] = "assets/clv_spread.png"
        headline["spread_clv_mean"] = (
            (spread_clv["mean_clv_points"] * spread_clv["n_games"]).sum() / spread_clv["n_games"].sum()
        )

    if total_clv is not None:
        render_clv_chart(total_clv, assets / "clv_total.png", "Total CLV by season (points)")
        charts["clv_total"] = "assets/clv_total.png"
        headline["total_clv_mean"] = (
            (total_clv["mean_clv_points"] * total_clv["n_games"]).sum() / total_clv["n_games"].sum()
        )

    if gbm_summary is not None:
        render_model_comparison_chart(
            gbm_summary, assets / "gbm_comparison.png",
            "Log loss by season: Elo-raw vs. feature-rich GBM vs. market"
        )
        charts["gbm_comparison"] = "assets/gbm_comparison.png"
        gbm_beats_elo = int((gbm_summary["gbm_logloss"] < gbm_summary["elo_raw_logloss"]).sum())
        headline["gbm_beats_elo_seasons"] = f"{gbm_beats_elo}/{len(gbm_summary)}"
        headline["gbm_n_games"] = int(gbm_summary["n_games"].sum())

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("index.html.jinja")
    html = template.render(snapshot=snapshot, charts=charts, headline=headline)
    (out / "index.html").write_text(html)

    (out / ".nojekyll").write_text("")
