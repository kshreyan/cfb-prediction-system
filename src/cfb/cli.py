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


@main.command("backtest")
@click.option("--start-season", default=2015, show_default=True)
@click.option("--end-season", default=2025, show_default=True)
@click.option("--refresh/--no-refresh", default=False, help="Force re-fetch from CFBD.")
def backtest(start_season: int, end_season: int, refresh: bool) -> None:
    """Run the leak-free walk-forward Elo backtest over real CFBD seasons
    and print per-season calibration metrics."""
    from cfb.backtest.walk_forward import run_elo_backtest, summarize_by_season
    from cfb.data.cache import fetch_seasons
    from cfb.data.cfbd_client import CfbdClient, MissingApiKeyError
    from cfb.data.loaders import games_from_cfbd_dicts
    from cfb.elo.config import EloConfig
    from cfb.elo.engine import EloEngine

    try:
        client = CfbdClient.from_env()
    except MissingApiKeyError as e:
        click.echo(f"BLOCKED: {e}", err=True)
        sys.exit(1)

    seasons = list(range(start_season, end_season + 1))
    click.echo(f"Fetching FBS games for seasons {seasons[0]}-{seasons[-1]} from CFBD...")
    raw_games = fetch_seasons(client, seasons, force_refresh=refresh)
    games = games_from_cfbd_dicts(raw_games)
    click.echo(f"{len(games)} completed FBS games loaded.")

    config = EloConfig.from_yaml("configs/conference_tiers.yaml")
    engine = EloEngine(config=config)

    df = run_elo_backtest(games, engine)
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


if __name__ == "__main__":
    main()
