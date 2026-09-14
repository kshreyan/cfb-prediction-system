"""Unit tests for Odds-API <-> CFBD team-name matching, using the exact
alias cases verified against a live pull (see commit message / session
notes) plus the plain-prefix common case."""
from __future__ import annotations

from cfb.data.odds_api_matching import match_team_name

CFBD_NAMES = {"Pittsburgh", "Syracuse", "App State", "Massachusetts",
              "Southern Miss", "San José State", "Ole Miss"}


def test_plain_prefix_match():
    assert match_team_name("Pittsburgh Panthers", CFBD_NAMES) == "Pittsburgh"
    assert match_team_name("Syracuse Orange", CFBD_NAMES) == "Syracuse"
    assert match_team_name("Ole Miss Rebels", CFBD_NAMES) == "Ole Miss"


def test_alias_matches():
    assert match_team_name("Appalachian State Mountaineers", CFBD_NAMES) == "App State"
    assert match_team_name("UMass Minutemen", CFBD_NAMES) == "Massachusetts"
    assert match_team_name("Southern Mississippi Golden Eagles", CFBD_NAMES) == "Southern Miss"


def test_accented_name_matches():
    assert match_team_name("San Jose State Spartans", CFBD_NAMES) == "San José State"


def test_unmatched_team_returns_none_not_a_guess():
    assert match_team_name("Some Totally Unknown Team", CFBD_NAMES) is None


def test_ambiguous_prefix_picks_the_longer_more_specific_name():
    # "Iowa State Cyclones" is a startswith-match for BOTH "Iowa" and
    # "Iowa State" -- must deterministically pick "Iowa State", not
    # whichever one a set/dict happens to iterate first.
    names = {"Iowa", "Iowa State", "Kansas", "Kansas State"}
    assert match_team_name("Iowa State Cyclones", names) == "Iowa State"
    assert match_team_name("Iowa Hawkeyes", names) == "Iowa"
    assert match_team_name("Kansas State Wildcats", names) == "Kansas State"
    assert match_team_name("Kansas Jayhawks", names) == "Kansas"
