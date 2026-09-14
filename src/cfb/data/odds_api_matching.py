"""Matches The Odds API's "City Mascot" team names (e.g. "Pittsburgh
Panthers") to CFBD's plain school names (e.g. "Pittsburgh").

Most pairs match via a simple startswith check once both sides are
accent-normalized. A short, explicit alias table covers the real
exceptions observed in practice (verified against a live pull, not
guessed) -- anything that still can't be matched is reported as
unmatched rather than silently paired with the wrong team, per the
project's "never fabricate" rule.
"""
from __future__ import annotations

import unicodedata

# Odds-API name (or its normalized prefix) -> CFBD name, for pairs a plain
# startswith can't resolve.
ALIASES: dict[str, str] = {
    "southern mississippi": "southern miss",
    "umass": "massachusetts",
    "hawaii": "hawai'i",
    "appalachian state": "app state",
    "ul monroe": "louisiana monroe",
    "louisiana-monroe": "louisiana monroe",
}


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return stripped.lower().strip()


def match_team_name(odds_api_name: str, cfbd_team_names: set[str]) -> str | None:
    """Returns the matching CFBD team name, or None if no confident match
    exists (caller must treat that game as unmatched, not guess).

    Picks the LONGEST matching CFBD name among all candidates, not the
    first one encountered: "Iowa State Cyclones" is a startswith-match
    for both "Iowa" and "Iowa State", and iterating a set/dict in
    Python has no guaranteed order (PYTHONHASHSEED is randomized per
    process by default) -- taking the first hit would silently pick
    "Iowa" on some runs and "Iowa State" on others. Longest-prefix-wins
    make this deterministic and correct.
    """
    normalized_lookup = {_normalize(t): t for t in cfbd_team_names}
    norm_odds_name = _normalize(odds_api_name)

    for alias_prefix, cfbd_alias in ALIASES.items():
        is_alias_match = norm_odds_name == alias_prefix or norm_odds_name.startswith(alias_prefix + " ")
        if is_alias_match and cfbd_alias in normalized_lookup:
            return normalized_lookup[cfbd_alias]

    candidates = [
        (norm_cfbd, original_cfbd)
        for norm_cfbd, original_cfbd in normalized_lookup.items()
        if norm_odds_name == norm_cfbd or norm_odds_name.startswith(norm_cfbd + " ")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: len(pair[0]), reverse=True)
    return candidates[0][1]
