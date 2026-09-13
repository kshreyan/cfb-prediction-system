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


if __name__ == "__main__":
    main()
