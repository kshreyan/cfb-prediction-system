"""Parses The Odds API's live-odds response into the same consensus
shape `lines_loader.consensus_closing_lines` produces (market_spread_home,
market_total, market_home_win_prob, n_books_*), so downstream code
(cover_probability, over_probability, the weekly-predictions pipeline)
doesn't need to know which source a line came from.

Games are matched to CFBD's game_id via team-name matching
(odds_api_matching.match_team_name) plus the (season, week) the caller
already knows it's asking about -- a game unmatched on team names is
dropped with a logged warning, never guessed into the wrong game_id.
"""
from __future__ import annotations

import pandas as pd
import structlog

from cfb.data.odds_api_matching import match_team_name
from cfb.models.moneyline.baselines import american_to_implied_prob, devig_two_way

logger = structlog.get_logger(__name__)


def build_game_id_lookup(cfbd_raw_games: list[dict]) -> tuple[dict[tuple[str, str], str], set[str]]:
    """{(home_team, away_team): game_id} and the full set of CFBD team
    names, both needed to match Odds API games to CFBD's game_id."""
    lookup = {}
    names = set()
    for g in cfbd_raw_games:
        lookup[(g["homeTeam"], g["awayTeam"])] = str(g["id"])
        names.add(g["homeTeam"])
        names.add(g["awayTeam"])
    return lookup, names


def odds_api_consensus(odds_api_games: list[dict], cfbd_raw_games: list[dict],
                        season: int, week: int) -> pd.DataFrame:
    """Matches on team names first, but the two sources occasionally
    disagree on which team is "home" -- observed in practice for
    neutral-site games, where the designation is a bookkeeping
    convention rather than a real home-field fact. If the direct
    (home, away) pair isn't found, this tries the flipped pair before
    giving up; a flipped match negates the spread and swaps the
    moneylines so everything downstream still ends up expressed
    relative to *CFBD's* home team, never silently relative to
    whichever team Odds API happened to call home.
    """
    game_id_lookup, cfbd_names = build_game_id_lookup(cfbd_raw_games)

    rows = []
    unmatched = []
    for game in odds_api_games:
        home_cfbd = match_team_name(game["home_team"], cfbd_names)
        away_cfbd = match_team_name(game["away_team"], cfbd_names)
        if home_cfbd is None or away_cfbd is None:
            unmatched.append((game["home_team"], game["away_team"]))
            continue

        game_id = game_id_lookup.get((home_cfbd, away_cfbd))
        flipped = False
        if game_id is None:
            game_id = game_id_lookup.get((away_cfbd, home_cfbd))
            flipped = game_id is not None
        if game_id is None:
            unmatched.append((game["home_team"], game["away_team"]))
            continue

        for book in game.get("bookmakers", []):
            spread_home = total = home_ml = away_ml = None
            for market in book.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == game["home_team"]:
                            spread_home = outcome.get("point")
                elif market["key"] == "totals":
                    over = next((o for o in market["outcomes"] if o["name"] == "Over"), None)
                    total = over.get("point") if over else None
                elif market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == game["home_team"]:
                            home_ml = outcome.get("price")
                        elif outcome["name"] == game["away_team"]:
                            away_ml = outcome.get("price")
            if flipped:
                spread_home = -spread_home if spread_home is not None else None
                home_ml, away_ml = away_ml, home_ml
            rows.append({
                "game_id": game_id, "book": book["key"], "spread_home": spread_home,
                "over_under": total, "home_moneyline": home_ml, "away_moneyline": away_ml,
            })

    if unmatched:
        logger.warning("odds_api_unmatched_games", count=len(unmatched), games=unmatched)

    if not rows:
        return pd.DataFrame(columns=["game_id", "season", "week", "market_spread_home",
                                      "market_total", "market_home_win_prob",
                                      "n_books_spread", "n_books_total", "n_books_ml"])

    df = pd.DataFrame(rows)

    def _devig(row: pd.Series) -> float | None:
        if pd.isna(row["home_moneyline"]) or pd.isna(row["away_moneyline"]):
            return None
        home_p = american_to_implied_prob(row["home_moneyline"])
        away_p = american_to_implied_prob(row["away_moneyline"])
        fair, _ = devig_two_way(home_p, away_p)
        return fair

    df["devigged_home_prob"] = df.apply(_devig, axis=1)
    grouped = df.groupby("game_id", as_index=False).agg(
        market_spread_home=("spread_home", "median"),
        n_books_spread=("spread_home", "count"),
        market_total=("over_under", "median"),
        n_books_total=("over_under", "count"),
        market_home_win_prob=("devigged_home_prob", "mean"),
        n_books_ml=("devigged_home_prob", "count"),
    )
    grouped["season"] = season
    grouped["week"] = week
    return grouped
