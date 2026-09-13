"""Parses raw CFBD betting-line dicts into a flat, provenance-stamped table
and derives a consensus closing line per game.

Real-world caveat (state this plainly, don't hide it): CFBD's free-tier
historical endpoint returns one line snapshot per book per game, not a
full time series of line movement. For completed games we treat that
snapshot as the closing line (`spread`/`overUnder` fields, as opposed to
`spreadOpen`/`overUnderOpen`) -- it is the last line CFBD captured, which
is the best available proxy for closing without a paid odds-history feed.
This is documented, not fabricated: `is_closing` is set accordingly and
CLV computed against it should be read with that caveat in mind.

A game with no posted line for a market is simply absent from that
market's consensus table -- never imputed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from cfb.models.moneyline.baselines import american_to_implied_prob, devig_two_way


@dataclass(frozen=True, slots=True)
class LineRow:
    game_id: str
    season: int
    week: int
    book: str
    is_closing: bool
    spread_home: float | None  # negative = home favored
    over_under: float | None
    home_moneyline: int | None
    away_moneyline: int | None


def parse_lines(raw_games: list[dict]) -> list[LineRow]:
    rows: list[LineRow] = []
    for g in raw_games:
        game_id = str(g["id"])
        for book_line in g.get("lines") or []:
            book = book_line.get("provider")
            if not book:
                continue
            rows.append(
                LineRow(
                    game_id=game_id,
                    season=g["season"],
                    week=g["week"],
                    book=book,
                    is_closing=True,
                    spread_home=book_line.get("spread"),
                    over_under=book_line.get("overUnder"),
                    home_moneyline=book_line.get("homeMoneyline"),
                    away_moneyline=book_line.get("awayMoneyline"),
                )
            )
            if book_line.get("spreadOpen") is not None or book_line.get("overUnderOpen") is not None:
                rows.append(
                    LineRow(
                        game_id=game_id,
                        season=g["season"],
                        week=g["week"],
                        book=book,
                        is_closing=False,
                        spread_home=book_line.get("spreadOpen"),
                        over_under=book_line.get("overUnderOpen"),
                        home_moneyline=None,
                        away_moneyline=None,
                    )
                )
    return rows


def consensus_closing_lines(rows: list[LineRow]) -> pd.DataFrame:
    """One row per game_id: median closing spread/total across books that
    posted one, and a devigged average home-win probability across books
    that posted a moneyline. Any field is NaN (not zero, not the other
    field's value) if no book posted that market for that game.
    """
    df = pd.DataFrame([asdict(r) for r in rows if r.is_closing])
    if df.empty:
        return pd.DataFrame(
            columns=["game_id", "season", "week", "market_spread_home",
                     "market_total", "market_home_win_prob", "n_books_spread",
                     "n_books_total", "n_books_ml"]
        )

    def _devig_row(row: pd.Series) -> float | None:
        if pd.isna(row["home_moneyline"]) or pd.isna(row["away_moneyline"]):
            return None
        home_implied = american_to_implied_prob(row["home_moneyline"])
        away_implied = american_to_implied_prob(row["away_moneyline"])
        home_fair, _ = devig_two_way(home_implied, away_implied)
        return home_fair

    df = df.copy()
    df["devigged_home_prob"] = df.apply(_devig_row, axis=1)

    grouped = df.groupby(["game_id", "season", "week"], as_index=False).agg(
        market_spread_home=("spread_home", "median"),
        n_books_spread=("spread_home", "count"),
        market_total=("over_under", "median"),
        n_books_total=("over_under", "count"),
        market_home_win_prob=("devigged_home_prob", "mean"),
        n_books_ml=("devigged_home_prob", "count"),
    )
    # median()/count() on all-NaN groups yields NaN/0 already -- explicit
    # about "unavailable" rather than silently coercing to 0.
    return grouped
