"""Unit tests for CLV sign conventions, checked against hand-worked examples."""
from __future__ import annotations

import pytest

from cfb.evaluation.clv import spread_clv_points, total_clv_points


def test_home_dog_bet_gets_positive_clv_when_line_moves_toward_home():
    # Bet home +7 (home getting 7 points) at open; by close, home only
    # getting +3 -- home became relatively stronger in the market's eyes,
    # so the home bettor got a better number than closing bettors.
    clv = spread_clv_points(picked_home=True, spread_home_open=7.0, spread_home_close=3.0)
    assert clv == pytest.approx(4.0)


def test_home_favorite_bet_gets_positive_clv_when_line_moves_further_negative():
    # Bet home -7 at open; by close home is -10 (bigger favorite) --
    # home bettor needed to win by only 7, closing bettors need 10.
    clv = spread_clv_points(picked_home=True, spread_home_open=-7.0, spread_home_close=-10.0)
    assert clv == pytest.approx(3.0)


def test_away_bet_clv_is_mirror_of_home_bet_clv():
    open_, close = -7.0, -10.0
    home_clv = spread_clv_points(picked_home=True, spread_home_open=open_, spread_home_close=close)
    away_clv = spread_clv_points(picked_home=False, spread_home_open=open_, spread_home_close=close)
    assert away_clv == pytest.approx(-home_clv)


def test_over_bet_gets_positive_clv_when_total_rises():
    # Bet Over 55 at open; by close the total moved to 58 -- the over
    # bettor got a cheaper number (needed fewer points) than closing.
    clv = total_clv_points(picked_over=True, total_open=55.0, total_close=58.0)
    assert clv == pytest.approx(3.0)


def test_under_bet_clv_is_mirror_of_over_bet_clv():
    open_, close = 55.0, 58.0
    over_clv = total_clv_points(picked_over=True, total_open=open_, total_close=close)
    under_clv = total_clv_points(picked_over=False, total_open=open_, total_close=close)
    assert under_clv == pytest.approx(-over_clv)


def test_no_line_movement_means_zero_clv():
    assert spread_clv_points(True, -7.0, -7.0) == 0.0
    assert total_clv_points(True, 55.0, 55.0) == 0.0
