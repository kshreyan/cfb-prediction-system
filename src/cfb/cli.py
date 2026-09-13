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
        games = client.fetch_games(season=2024, week=1)
        click.echo(f"OK: fetched {len(games)} games for 2024 week 1.")
    except Exception as e:  # noqa: BLE001 - surface any API error plainly
        click.echo(f"CFBD API call failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
