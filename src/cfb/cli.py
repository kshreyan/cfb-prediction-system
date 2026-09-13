"""Command-line entry points (`cfb ...`). Thin wiring only -- logic lives
in the cfb package modules so it stays testable without a CLI harness."""
from __future__ import annotations

import sys

import click
import structlog

logger = structlog.get_logger(__name__)


@click.group()
def main() -> None:
    """College football prediction system CLI."""


@main.command("check-cfbd")
def check_cfbd() -> None:
    """Verify CFBD_API_KEY is set and the API is reachable."""
    from cfb.data.cfbd_client import CfbdClient, MissingApiKeyError

    try:
        client = CfbdClient.from_env()
    except MissingApiKeyError as e:
        click.echo(f"BLOCKED: {e}", err=True)
        sys.exit(1)

    try:
        games = client.fetch_games(season=2024, week=1, season_type="regular")
        click.echo(f"OK: fetched {len(games)} games for 2024 week 1.")
    except Exception as e:  # noqa: BLE001 - surface any API error plainly
        click.echo(f"CFBD API call failed: {e}", err=True)
        sys.exit(1)


def _get_client():
    from cfb.data.cfbd_client import CfbdClient, MissingApiKeyError

    try:
        return CfbdClient.from_env()
    except MissingApiKeyError as e:
        click.echo(f"BLOCKED: {e}", err=True)
        sys.exit(1)


def _run_elo_backtest(client, seasons: list[int], refresh: bool):
    from cfb.backtest.walk_forward import run_elo_backtest
    from cfb.data.cache import fetch_seasons
    from cfb.data.loaders import games_from_cfbd_dicts
    from cfb.elo.config import EloConfig
    from cfb.elo.engine import EloEngine

    click.echo(f"Fetching FBS games for seasons {seasons[0]}-{seasons[-1]} from CFBD...")
    raw_games = fetch_seasons(client, seasons, force_refresh=refresh)
    games = games_from_cfbd_dicts(raw_games)
    click.echo(f"{len(games)} completed FBS games loaded.")

    config = EloConfig.from_yaml("configs/conference_tiers.yaml")
    engine = EloEngine(config=config)
    return run_elo_backtest(games, engine)


@main.command("backtest")
@click.option("--start-season", default=2015, show_default=True)
@click.option("--end-season", default=2025, show_default=True)
@click.option("--refresh/--no-refresh", default=False, help="Force re-fetch from CFBD.")
def backtest(start_season: int, end_season: int, refresh: bool) -> None:
    """Run the leak-free walk-forward Elo backtest over real CFBD seasons
    and print per-season calibration metrics."""
    from cfb.backtest.walk_forward import summarize_by_season

    client = _get_client()
    seasons = list(range(start_season, end_season + 1))
    df = _run_elo_backtest(client, seasons, refresh)
    df.to_csv("data/processed/elo_backtest_predictions.csv", index=False)

    click.echo("\n=== FBS-vs-FBS only (excludes mismatches) ===")
    summary_fbs = summarize_by_season(df, fbs_only=True)
    click.echo(summary_fbs.to_string(index=False))

    click.echo("\n=== All games (including FBS-vs-FCS) ===")
    summary_all = summarize_by_season(df, fbs_only=False)
    click.echo(summary_all.to_string(index=False))

    summary_fbs.to_csv("data/processed/elo_backtest_summary_fbs_only.csv", index=False)
    summary_all.to_csv("data/processed/elo_backtest_summary_all_games.csv", index=False)
    click.echo("\nSaved: data/processed/elo_backtest_predictions.csv, "
               "elo_backtest_summary_{fbs_only,all_games}.csv")


@main.command("market-eval")
@click.option("--start-season", default=2015, show_default=True)
@click.option("--end-season", default=2025, show_default=True)
@click.option("--refresh/--no-refresh", default=False, help="Force re-fetch from CFBD.")
def market_eval(start_season: int, end_season: int, refresh: bool) -> None:
    """Compare raw Elo against the de-vigged market-implied moneyline
    probability, on exactly the games where a real moneyline was posted
    (never imputed for games without one)."""
    import pandas as pd

    from cfb.data.cache import fetch_lines_for_seasons
    from cfb.data.lines_loader import consensus_closing_lines, parse_lines
    from cfb.evaluation.metrics import (
        accuracy,
        brier_score,
        expected_calibration_error,
        log_loss,
    )

    client = _get_client()
    seasons = list(range(start_season, end_season + 1))

    df = _run_elo_backtest(client, seasons, refresh)

    click.echo(f"Fetching betting lines for seasons {seasons[0]}-{seasons[-1]} from CFBD...")
    raw_lines = fetch_lines_for_seasons(client, seasons, force_refresh=refresh)
    line_rows = parse_lines(raw_lines)
    consensus = consensus_closing_lines(line_rows)
    click.echo(f"{len(consensus)} games have at least one posted line; "
               f"{(consensus['n_books_ml'] > 0).sum()} have a posted moneyline.")

    merged = df.merge(
        consensus[["game_id", "market_home_win_prob", "n_books_ml"]],
        on="game_id", how="inner",
    )
    has_ml = merged[merged["n_books_ml"] > 0].copy()
    has_ml = has_ml[has_ml["is_fbs_vs_fbs"]]  # apples-to-apples with the FBS-only Elo numbers
    click.echo(f"\n{len(has_ml)} FBS-vs-FBS games have both an Elo prediction and a "
               "real posted moneyline -- comparison below is on exactly this subset.")

    rows = []
    for season, group in has_ml.groupby("season"):
        y = group["home_won"].astype(int).to_numpy()
        elo_p = group["home_win_prob"].to_numpy()
        mkt_p = group["market_home_win_prob"].to_numpy()
        rows.append({
            "season": season,
            "n_games": len(group),
            "elo_acc": accuracy(y, elo_p),
            "elo_logloss": log_loss(y, elo_p),
            "elo_brier": brier_score(y, elo_p),
            "elo_ece": expected_calibration_error(y, elo_p),
            "mkt_acc": accuracy(y, mkt_p),
            "mkt_logloss": log_loss(y, mkt_p),
            "mkt_brier": brier_score(y, mkt_p),
            "mkt_ece": expected_calibration_error(y, mkt_p),
        })
    summary = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    click.echo("\n=== Elo vs de-vigged market, same games, FBS-vs-FBS only ===")
    click.echo(summary.to_string(index=False))

    beats = (summary["elo_logloss"] < summary["mkt_logloss"]).sum()
    click.echo(f"\nElo beat the market on log loss in {beats}/{len(summary)} seasons.")
    if beats < len(summary):
        click.echo("Per the honesty standard: where the market wins, that's reported "
                   "plainly, not explained away.")

    has_ml.to_csv("data/processed/elo_vs_market_predictions.csv", index=False)
    summary.to_csv("data/processed/elo_vs_market_summary.csv", index=False)


@main.command("ensemble-eval")
@click.option("--start-season", default=2015, show_default=True)
@click.option("--end-season", default=2025, show_default=True)
@click.option("--refresh/--no-refresh", default=False, help="Force re-fetch from CFBD.")
def ensemble_eval(start_season: int, end_season: int, refresh: bool) -> None:
    """The decisive check: does isotonic-calibrated Elo, blended with the
    market via a walk-forward-learned log-odds weight, beat Elo-only and
    market-only out-of-sample? Reports all three side by side."""
    import pandas as pd

    from cfb.calibration.isotonic_calibrator import walk_forward_isotonic_calibrate
    from cfb.data.cache import fetch_lines_for_seasons
    from cfb.data.lines_loader import consensus_closing_lines, parse_lines
    from cfb.ensemble.blend import walk_forward_ensemble
    from cfb.evaluation.metrics import (
        accuracy,
        brier_score,
        expected_calibration_error,
        log_loss,
    )

    client = _get_client()
    seasons = list(range(start_season, end_season + 1))

    elo_df = _run_elo_backtest(client, seasons, refresh)
    fbs_df = elo_df[elo_df["is_fbs_vs_fbs"]].copy()

    click.echo("Fitting walk-forward isotonic calibration on raw Elo (FBS-only)...")
    fbs_df["home_won_int"] = fbs_df["home_won"].astype(int)
    fbs_df["elo_calibrated_prob"] = walk_forward_isotonic_calibrate(
        fbs_df, "home_win_prob", "home_won_int"
    )

    click.echo(f"Fetching betting lines for seasons {seasons[0]}-{seasons[-1]} from CFBD...")
    raw_lines = fetch_lines_for_seasons(client, seasons, force_refresh=refresh)
    consensus = consensus_closing_lines(parse_lines(raw_lines))

    merged = fbs_df.merge(
        consensus[["game_id", "market_home_win_prob", "n_books_ml"]], on="game_id", how="inner"
    )
    has_ml = merged[merged["n_books_ml"] > 0].copy()
    click.echo(f"\n{len(has_ml)} FBS-vs-FBS games have Elo + a real posted moneyline.")

    click.echo("Learning walk-forward ensemble weight (calibrated Elo <-> market)...")
    has_ml = walk_forward_ensemble(
        has_ml, "elo_calibrated_prob", "market_home_win_prob", "home_won_int"
    )

    rows = []
    for season, group in has_ml.groupby("season"):
        y = group["home_won_int"].to_numpy()
        preds = {
            "elo_raw": group["home_win_prob"].to_numpy(),
            "elo_calibrated": group["elo_calibrated_prob"].to_numpy(),
            "market": group["market_home_win_prob"].to_numpy(),
            "ensemble": group["ensemble_prob"].to_numpy(),
        }
        row = {"season": season, "n_games": len(group),
               "ensemble_weight_on_elo": group["ensemble_weight_on_model"].iloc[0]}
        for name, p in preds.items():
            row[f"{name}_acc"] = accuracy(y, p)
            row[f"{name}_logloss"] = log_loss(y, p)
            row[f"{name}_brier"] = brier_score(y, p)
            row[f"{name}_ece"] = expected_calibration_error(y, p)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)

    click.echo("\n=== Log loss: elo_raw vs elo_calibrated vs market vs ensemble ===")
    click.echo(summary[["season", "n_games", "ensemble_weight_on_elo",
                         "elo_raw_logloss", "elo_calibrated_logloss",
                         "market_logloss", "ensemble_logloss"]].to_string(index=False))

    click.echo("\n=== ECE: elo_raw vs elo_calibrated vs market vs ensemble ===")
    click.echo(summary[["season", "elo_raw_ece", "elo_calibrated_ece",
                         "market_ece", "ensemble_ece"]].to_string(index=False))

    beats_market = (summary["ensemble_logloss"] < summary["market_logloss"]).sum()
    beats_elo = (summary["ensemble_logloss"] < summary["elo_raw_logloss"]).sum()
    click.echo(f"\nEnsemble beat market-only on log loss in {beats_market}/{len(summary)} seasons.")
    click.echo(f"Ensemble beat Elo-only on log loss in {beats_elo}/{len(summary)} seasons.")
    click.echo("Per the honesty standard: no 'beats the market' claim stands on this "
               "alone -- multi-season CLV evidence is still required (not yet built).")

    has_ml.to_csv("data/processed/ensemble_predictions.csv", index=False)
    summary.to_csv("data/processed/ensemble_summary.csv", index=False)
    click.echo("\nSaved: data/processed/ensemble_predictions.csv, ensemble_summary.csv")


@main.command("spread-eval")
@click.option("--start-season", default=2015, show_default=True)
@click.option("--end-season", default=2025, show_default=True)
@click.option("--refresh/--no-refresh", default=False, help="Force re-fetch from CFBD.")
def spread_eval(start_season: int, end_season: int, refresh: bool) -> None:
    """Real ATS backtest: walk-forward margin model vs the market spread,
    on every FBS-vs-FBS game with a posted spread. This is the project's
    real skill test, not straight-up accuracy -- see README."""
    import numpy as np
    import pandas as pd

    from cfb.data.cache import fetch_lines_for_seasons
    from cfb.data.lines_loader import consensus_closing_lines, parse_lines
    from cfb.evaluation.metrics import expected_calibration_error, mean_absolute_error
    from cfb.models.spread.margin_model import home_cover_probability, run_margin_backtest

    client = _get_client()
    seasons = list(range(start_season, end_season + 1))

    elo_df = _run_elo_backtest(client, seasons, refresh)
    click.echo("Fitting walk-forward margin model (per-season expanding window)...")
    margin_df = run_margin_backtest(elo_df)

    click.echo(f"Fetching betting lines for seasons {seasons[0]}-{seasons[-1]} from CFBD...")
    raw_lines = fetch_lines_for_seasons(client, seasons, force_refresh=refresh)
    consensus = consensus_closing_lines(parse_lines(raw_lines))

    df = margin_df.merge(
        consensus[["game_id", "market_spread_home", "n_books_spread"]],
        on="game_id", how="inner",
    )
    df = df[df["n_books_spread"] > 0].copy()

    df["home_cover_prob"] = df.apply(
        lambda r: home_cover_probability(
            r["predicted_margin"], r["resid_a"], r["resid_loc"], r["resid_scale"],
            r["market_spread_home"],
        ),
        axis=1,
    )
    # ATS outcome: home covers iff actual margin beats the spread; exact
    # equality is a push and is excluded from the win-rate denominator,
    # matching standard ATS bookkeeping.
    home_ats_margin = df["home_margin"] + df["market_spread_home"]
    df["push"] = np.isclose(home_ats_margin, 0.0)
    df["home_covered"] = home_ats_margin > 0
    df["model_picked_home"] = df["home_cover_prob"] > 0.5
    df["pick_correct"] = df["model_picked_home"] == df["home_covered"]

    def _ats_record(data: pd.DataFrame) -> tuple[int, int, int]:
        decided = data[~data["push"]]
        wins = int(decided["pick_correct"].sum())
        losses = len(decided) - wins
        pushes = int(data["push"].sum())
        return wins, losses, pushes

    click.echo(f"\n{len(df)} FBS-vs-FBS games have a posted spread and a model prediction.")

    fbs_only = df[df["is_fbs_vs_fbs"]]
    wins, losses, pushes = _ats_record(fbs_only)
    win_pct = wins / (wins + losses) if (wins + losses) else float("nan")
    click.echo("\n=== Overall ATS record (FBS-only, 2015-2025) ===")
    click.echo(f"{wins}-{losses}-{pushes}  ({win_pct:.1%} of decided picks)")
    click.echo("Honesty check: >54% here is presumed leakage until investigated, per README.")

    click.echo("\n=== ATS record by season ===")
    rows = []
    for season, group in fbs_only.groupby("season"):
        w, l, p = _ats_record(group)
        wp = w / (w + l) if (w + l) else float("nan")
        rows.append({"season": season, "wins": w, "losses": l, "pushes": p,
                      "win_pct": wp, "margin_mae": mean_absolute_error(
                          group["home_margin"], group["predicted_margin"])})
    season_summary = pd.DataFrame(rows)
    click.echo(season_summary.to_string(index=False))

    click.echo("\n=== ATS record split by favorite size (|spread| threshold 14) ===")
    small_fav = fbs_only[fbs_only["market_spread_home"].abs() < 14]
    big_fav = fbs_only[fbs_only["market_spread_home"].abs() >= 14]
    for label, subset in [("|spread| < 14", small_fav), ("|spread| >= 14", big_fav)]:
        w, l, p = _ats_record(subset)
        wp = w / (w + l) if (w + l) else float("nan")
        click.echo(f"{label}: {w}-{l}-{p} ({wp:.1%})")

    cover_ece = expected_calibration_error(
        fbs_only["home_covered"].astype(int).to_numpy(), fbs_only["home_cover_prob"].to_numpy()
    )
    click.echo(f"\nCover-probability ECE (FBS-only, all seasons pooled): {cover_ece:.4f}")

    df.to_csv("data/processed/spread_backtest_predictions.csv", index=False)
    season_summary.to_csv("data/processed/spread_backtest_by_season.csv", index=False)
    click.echo("\nSaved: data/processed/spread_backtest_predictions.csv, "
               "spread_backtest_by_season.csv")


@main.command("total-eval")
@click.option("--start-season", default=2015, show_default=True)
@click.option("--end-season", default=2025, show_default=True)
@click.option("--refresh/--no-refresh", default=False, help="Force re-fetch from CFBD.")
def total_eval(start_season: int, end_season: int, refresh: bool) -> None:
    """Real O/U backtest: walk-forward scoreline (total points) model vs
    the market total, on every FBS-vs-FBS game with a posted total."""
    import numpy as np
    import pandas as pd

    from cfb.data.cache import fetch_lines_for_seasons, fetch_seasons
    from cfb.data.lines_loader import consensus_closing_lines, parse_lines
    from cfb.data.loaders import games_from_cfbd_dicts
    from cfb.evaluation.metrics import expected_calibration_error, mean_absolute_error
    from cfb.models.total.scoreline_engine import (
        fit_total_residuals,
        over_probability,
        run_total_backtest,
    )

    client = _get_client()
    seasons = list(range(start_season, end_season + 1))

    click.echo(f"Fetching FBS games for seasons {seasons[0]}-{seasons[-1]} from CFBD...")
    raw_games = fetch_seasons(client, seasons, force_refresh=refresh)
    games = games_from_cfbd_dicts(raw_games)
    click.echo(f"{len(games)} completed FBS games loaded.")

    click.echo("Running walk-forward scoreline engine + skew-normal residual fit...")
    total_df = run_total_backtest(games)
    total_df = fit_total_residuals(total_df)

    click.echo(f"Fetching betting lines for seasons {seasons[0]}-{seasons[-1]} from CFBD...")
    raw_lines = fetch_lines_for_seasons(client, seasons, force_refresh=refresh)
    consensus = consensus_closing_lines(parse_lines(raw_lines))

    df = total_df.merge(
        consensus[["game_id", "market_total", "n_books_total"]], on="game_id", how="inner"
    )
    df = df[df["n_books_total"] > 0].copy()

    df["over_prob"] = df.apply(
        lambda r: over_probability(r["predicted_total"], r["resid_a"], r["resid_loc"],
                                    r["resid_scale"], r["market_total"]),
        axis=1,
    )
    df["push"] = np.isclose(df["actual_total"], df["market_total"])
    df["went_over"] = df["actual_total"] > df["market_total"]
    df["model_picked_over"] = df["over_prob"] > 0.5
    df["pick_correct"] = df["model_picked_over"] == df["went_over"]

    def _ou_record(data: pd.DataFrame) -> tuple[int, int, int]:
        decided = data[~data["push"]]
        wins = int(decided["pick_correct"].sum())
        losses = len(decided) - wins
        pushes = int(data["push"].sum())
        return wins, losses, pushes

    click.echo(f"\n{len(df)} FBS-vs-FBS games have a posted total and a model prediction.")

    fbs_only = df[df["is_fbs_vs_fbs"]]
    wins, losses, pushes = _ou_record(fbs_only)
    win_pct = wins / (wins + losses) if (wins + losses) else float("nan")
    click.echo("\n=== Overall O/U record (FBS-only, 2015-2025) ===")
    click.echo(f"{wins}-{losses}-{pushes}  ({win_pct:.1%} of decided picks)")
    click.echo("Honesty check: >54% here is presumed leakage until investigated, per README.")

    click.echo("\n=== O/U record by season ===")
    rows = []
    for season, group in fbs_only.groupby("season"):
        w, l, p = _ou_record(group)
        wp = w / (w + l) if (w + l) else float("nan")
        rows.append({"season": season, "wins": w, "losses": l, "pushes": p,
                      "win_pct": wp, "total_mae": mean_absolute_error(
                          group["actual_total"], group["predicted_total"])})
    season_summary = pd.DataFrame(rows)
    click.echo(season_summary.to_string(index=False))

    ou_ece = expected_calibration_error(
        fbs_only["went_over"].astype(int).to_numpy(), fbs_only["over_prob"].to_numpy()
    )
    click.echo(f"\nOver-probability ECE (FBS-only, all seasons pooled): {ou_ece:.4f}")

    df.to_csv("data/processed/total_backtest_predictions.csv", index=False)
    season_summary.to_csv("data/processed/total_backtest_by_season.csv", index=False)
    click.echo("\nSaved: data/processed/total_backtest_predictions.csv, "
               "total_backtest_by_season.csv")


if __name__ == "__main__":
    main()
