"""Generates immutable prediction snapshots for a week's not-yet-played
FBS games.

Uses exactly the same walk-forward-fit parameters (Elo ratings, margin
regression, scoreline EWMA state, residual distributions, isotonic
calibration) that a backtested game in the same season would have used
-- see fit_season_params / fit_total_residual_params / fit_calibrator_for_season
docstrings for why the live and backtested code paths deliberately share
that fitting logic instead of duplicating it.

Each call writes a NEW timestamped file; nothing already written is ever
edited or overwritten, per the project's "predictions are immutable"
rule. Results (once games are played) are joined in separately by the
backtest/evaluation modules, never back into this file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.calibration.isotonic_calibrator import fit_calibrator_for_season
from cfb.data.cache import fetch_seasons
from cfb.data.cfbd_client import CfbdClient
from cfb.data.lines_loader import consensus_closing_lines, parse_lines
from cfb.data.loaders import games_from_cfbd_dicts
from cfb.data.odds_api_client import MissingOddsApiKeyError, OddsApiClient
from cfb.data.odds_api_loader import odds_api_consensus
from cfb.elo.config import EloConfig
from cfb.elo.engine import EloEngine
from cfb.models.spread.margin_model import (
    fit_season_params as fit_margin_params,
)
from cfb.models.spread.margin_model import (
    home_cover_probability,
    recover_elo_diff,
    run_margin_backtest,
)
from cfb.models.total.scoreline_engine import (
    build_scoreline_engine,
    fit_total_residual_params,
    over_probability,
    run_total_backtest,
)

PREDICTIONS_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "predictions"


@dataclass
class GamePrediction:
    game_id: str
    season: int
    week: int
    start_date: str
    home_team: str
    away_team: str
    home_conference: str | None
    away_conference: str | None
    is_fbs_vs_fbs: bool
    neutral_site: bool
    home_win_prob_raw: float
    home_win_prob_calibrated: float | None
    calibration_available: bool
    predicted_margin: float
    predicted_total: float
    market_spread_home: float | None
    market_total: float | None
    market_home_win_prob: float | None
    home_cover_prob: float | None
    over_prob: float | None
    spread_edge_points: float | None  # predicted_margin - (-market_spread_home)
    total_edge_points: float | None   # predicted_total - market_total
    moneyline_edge_prob: float | None  # model - market win prob
    n_books_spread: int
    n_books_total: int
    n_books_ml: int
    market_source: str  # "the-odds-api", "cfbd", "both" (different fields from each), or "none"


def _merge_market_sources(cfbd_consensus: pd.DataFrame, odds_consensus: pd.DataFrame) -> pd.DataFrame:
    """Coalesces two consensus tables by game_id, field by field, always
    preferring The Odds API's real-time multi-book data (more books,
    real timestamps) and falling back to CFBD's for any game/field it
    doesn't cover -- never averaging the two sources together, since
    that would blur two different snapshot times into a meaningless
    number. Adds `market_source` so it's visible which source won."""
    merged = cfbd_consensus.merge(
        odds_consensus, on="game_id", how="outer", suffixes=("_cfbd", "_odds")
    )
    out = pd.DataFrame({"game_id": merged["game_id"]})
    for field, count_field in [("market_spread_home", "n_books_spread"),
                                ("market_total", "n_books_total"),
                                ("market_home_win_prob", "n_books_ml")]:
        odds_col, cfbd_col = f"{field}_odds", f"{field}_cfbd"
        odds_count = merged.get(f"{count_field}_odds", pd.Series(0, index=merged.index)).fillna(0)
        out[field] = merged[odds_col].where(odds_count > 0, merged[cfbd_col])
        out[count_field] = merged.get(f"{count_field}_odds", 0).where(
            odds_count > 0, merged.get(f"{count_field}_cfbd", 0)
        ).fillna(0).astype(int)

    used_odds = (merged.get("n_books_spread_odds", 0).fillna(0) > 0) | \
                (merged.get("n_books_total_odds", 0).fillna(0) > 0) | \
                (merged.get("n_books_ml_odds", 0).fillna(0) > 0)
    used_cfbd = (merged.get("n_books_spread_cfbd", 0).fillna(0) > 0) | \
                (merged.get("n_books_total_cfbd", 0).fillna(0) > 0) | \
                (merged.get("n_books_ml_cfbd", 0).fillna(0) > 0)
    out["market_source"] = np.select(
        [used_odds & used_cfbd, used_odds, used_cfbd], ["both", "the-odds-api", "cfbd"], "none"
    )
    return out


def _resolve_target_week(client: CfbdClient, season: int, week: int | None) -> int:
    if week is not None:
        return week
    raw = client.fetch_games(season=season, season_type="regular", classification="fbs")
    upcoming_weeks = sorted({g["week"] for g in raw if not g["completed"]})
    if not upcoming_weeks:
        raise RuntimeError(f"No upcoming FBS games found for season {season}.")
    return upcoming_weeks[0]


def generate_weekly_predictions(season: int, week: int | None = None,
                                 history_start_season: int = 2015,
                                 refresh: bool = False) -> dict:
    client = CfbdClient.from_env()
    target_week = _resolve_target_week(client, season, week)

    history_seasons = list(range(history_start_season, season + 1))
    raw_history = fetch_seasons(client, history_seasons, force_refresh=refresh)
    completed_games = games_from_cfbd_dicts(raw_history)

    elo_engine = EloEngine(config=EloConfig.from_yaml("configs/conference_tiers.yaml"))
    elo_predictions = elo_engine.process_all(completed_games)

    elo_rows = []
    games_by_id = {g.game_id: g for g in completed_games}
    for pred in elo_predictions:
        g = games_by_id[pred.game_id]
        elo_rows.append({
            "game_id": pred.game_id, "season": pred.season, "week": pred.week,
            "start_date": pred.start_date, "home_team": pred.home_team,
            "away_team": pred.away_team, "home_win_prob": pred.home_win_prob,
            "home_won": g.home_won, "home_margin": g.margin,
            "is_fbs_vs_fbs": g.home_is_fbs and g.away_is_fbs,
        })
    elo_df = pd.DataFrame(elo_rows)

    margin_df = run_margin_backtest(elo_df)
    margin_params = fit_margin_params(margin_df, season)

    total_engine = build_scoreline_engine(completed_games)
    total_df = run_total_backtest(completed_games)
    total_params = fit_total_residual_params(total_df, season)

    fbs_elo_df = elo_df[elo_df["is_fbs_vs_fbs"]].copy()
    fbs_elo_df["home_won_int"] = fbs_elo_df["home_won"].astype(int)
    calibrator = fit_calibrator_for_season(fbs_elo_df, "home_win_prob", "home_won_int", season)

    raw_upcoming = client.fetch_games(season=season, week=target_week,
                                       season_type="regular", classification="fbs")
    upcoming = [rg for rg in raw_upcoming if not rg["completed"]]

    raw_lines = client.fetch_lines(season=season, week=target_week, season_type="regular")
    cfbd_consensus = consensus_closing_lines(parse_lines(raw_lines))

    try:
        odds_client = OddsApiClient.from_env()
        odds_games = odds_client.fetch_live_odds()
        odds_consensus = odds_api_consensus(odds_games, raw_upcoming, season, target_week)
    except MissingOddsApiKeyError:
        odds_consensus = odds_api_consensus([], raw_upcoming, season, target_week)  # empty

    consensus = _merge_market_sources(cfbd_consensus, odds_consensus)
    consensus_by_game = {row["game_id"]: row for row in consensus.to_dict("records")}

    predictions: list[GamePrediction] = []
    for rg in upcoming:
        game_id = str(rg["id"])
        home_team, away_team = rg["homeTeam"], rg["awayTeam"]
        home_is_fbs = rg.get("homeClassification") == "fbs"
        away_is_fbs = rg.get("awayClassification") == "fbs"
        neutral = bool(rg.get("neutralSite"))

        home_win_prob_raw = elo_engine.predict_win_prob(
            home_team, rg.get("homeConference"), home_is_fbs,
            away_team, rg.get("awayConference"), away_is_fbs, neutral,
        )
        calibration_available = calibrator is not None
        home_win_prob_calibrated = (
            float(calibrator.predict([home_win_prob_raw])[0]) if calibrator is not None else None
        )

        elo_diff = float(recover_elo_diff(pd.Series([home_win_prob_raw])).iloc[0])
        predicted_margin = margin_params["slope"] * elo_diff + margin_params["intercept"]

        pred_home_pts, pred_away_pts = total_engine.predict_only(home_team, away_team)
        predicted_total = pred_home_pts + pred_away_pts

        market_row = consensus_by_game.get(game_id)
        market_spread_home = market_total = market_home_win_prob = None
        n_books_spread = n_books_total = n_books_ml = 0
        home_cover_prob = over_prob = spread_edge = total_edge = ml_edge = None
        market_source = market_row.get("market_source", "none") if market_row else "none"

        if market_row is not None:
            n_books_spread = int(market_row.get("n_books_spread") or 0)
            n_books_total = int(market_row.get("n_books_total") or 0)
            n_books_ml = int(market_row.get("n_books_ml") or 0)

            if n_books_spread > 0:
                market_spread_home = float(market_row["market_spread_home"])
                home_cover_prob = home_cover_probability(
                    predicted_margin, margin_params["resid_a"], margin_params["resid_loc"],
                    margin_params["resid_scale"], market_spread_home,
                )
                spread_edge = predicted_margin - (-market_spread_home)

            if n_books_total > 0:
                market_total = float(market_row["market_total"])
                over_prob = over_probability(
                    predicted_total, total_params["resid_a"], total_params["resid_loc"],
                    total_params["resid_scale"], market_total,
                )
                total_edge = predicted_total - market_total

            if n_books_ml > 0:
                market_home_win_prob = float(market_row["market_home_win_prob"])
                model_prob_for_edge = home_win_prob_calibrated or home_win_prob_raw
                ml_edge = model_prob_for_edge - market_home_win_prob

        predictions.append(GamePrediction(
            game_id=game_id, season=season, week=target_week,
            start_date=str(rg["startDate"]), home_team=home_team, away_team=away_team,
            home_conference=rg.get("homeConference"), away_conference=rg.get("awayConference"),
            is_fbs_vs_fbs=home_is_fbs and away_is_fbs, neutral_site=neutral,
            home_win_prob_raw=home_win_prob_raw, home_win_prob_calibrated=home_win_prob_calibrated,
            calibration_available=calibration_available,
            predicted_margin=predicted_margin, predicted_total=predicted_total,
            market_spread_home=market_spread_home, market_total=market_total,
            market_home_win_prob=market_home_win_prob,
            home_cover_prob=home_cover_prob, over_prob=over_prob,
            spread_edge_points=spread_edge, total_edge_points=total_edge,
            moneyline_edge_prob=ml_edge,
            n_books_spread=n_books_spread, n_books_total=n_books_total, n_books_ml=n_books_ml,
            market_source=market_source,
        ))

    generated_at = datetime.now(UTC).isoformat()
    return {
        "generated_at": generated_at,
        "season": season,
        "week": target_week,
        "n_games": len(predictions),
        "margin_model_params": margin_params,
        "total_model_params": total_params,
        "calibration_available": calibrator is not None,
        "games": [asdict(p) for p in predictions],
    }


def save_prediction_snapshot(payload: dict) -> Path:
    """Writes a new, timestamped, never-overwritten JSON file."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = payload["generated_at"].replace(":", "").replace("+00:00", "Z")
    path = PREDICTIONS_DIR / f"{payload['season']}_week{payload['week']}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
