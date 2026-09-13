"""Closing-line value (CLV): the decisive benchmark this project commits
to reporting (see README's honesty standard). CLV measures whether the
line moved in the direction of your bet after you placed it -- it's
computed from real opening/closing line snapshots, is independent of
whether any individual bet won, and is the standard way skilled bettors
are actually identified (a positive long-run CLV predicts long-run
profitability far better than any single season's win rate).

Sign convention throughout: `spread_home` follows the standard market
convention (negative = home favored by that many points); a "total"
increasing means the market grew more bullish on scoring.

Scope, stated plainly: CFBD's free-tier feed gives an opening AND closing
snapshot for spreads and totals, so CLV is computed for those two markets
below. It does NOT give an opening moneyline, so moneyline CLV is not
computed here -- that's a real data-availability gap, not an oversight,
and no moneyline CLV number is fabricated to fill it.
"""
from __future__ import annotations


def spread_clv_points(picked_home: bool, spread_home_open: float, spread_home_close: float) -> float:
    """Points of CLV for a spread bet placed at the opening line and held
    to close. Positive = the closing line moved in the bettor's favor
    (they got a better number than someone betting the same side at
    close would have)."""
    diff = spread_home_open - spread_home_close
    return diff if picked_home else -diff


def total_clv_points(picked_over: bool, total_open: float, total_close: float) -> float:
    """Points of CLV for a total (O/U) bet placed at the opening line and
    held to close."""
    diff = total_close - total_open
    return diff if picked_over else -diff
