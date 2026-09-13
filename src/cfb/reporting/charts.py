"""Static chart rendering for the published site. Deliberately simple
matplotlib PNGs (not an interactive JS library) -- GitHub Pages is
static, and these charts are pre-generated at publish time, never
computed in the visitor's browser.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfb.evaluation.metrics import reliability_curve

_BG = "#ffffff"
_INK = "#1a1a2e"
_MUTED = "#6b7280"
_ACCENT = "#2563eb"
_GRID = "#e5e7eb"


def _style_axes(ax) -> None:
    ax.set_facecolor(_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def render_reliability_chart(y_true: np.ndarray, p_pred: np.ndarray, path: Path,
                              title: str, n_bins: int = 10) -> None:
    centers, observed, counts = reliability_curve(y_true, p_pred, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    fig.patch.set_facecolor(_BG)
    _style_axes(ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color=_MUTED, linewidth=1.2,
            label="perfect calibration")
    valid = ~np.isnan(observed)
    sizes = 30 + 300 * (counts[valid] / max(counts[valid].max(), 1))
    ax.scatter(centers[valid], observed[valid], s=sizes, color=_ACCENT, alpha=0.85,
               edgecolor="white", linewidth=0.8, zorder=3, label="observed (bin size = point size)")
    ax.plot(centers[valid], observed[valid], color=_ACCENT, alpha=0.4, linewidth=1.2, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", color=_INK, fontsize=10)
    ax.set_ylabel("Observed frequency", color=_INK, fontsize=10)
    ax.set_title(title, color=_INK, fontsize=11, pad=12)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)


def render_clv_chart(clv_by_season: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    fig.patch.set_facecolor(_BG)
    _style_axes(ax)

    seasons = clv_by_season["season"].astype(str)
    values = clv_by_season["mean_clv_points"]
    colors = [_ACCENT if v >= 0 else "#ef4444" for v in values]
    ax.bar(seasons, values, color=colors, width=0.6, zorder=3)
    ax.axhline(0, color=_INK, linewidth=1)
    ax.set_ylabel("Mean CLV (points)", color=_INK, fontsize=10)
    ax.set_title(title, color=_INK, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)


def render_ats_record_chart(by_season: pd.DataFrame, path: Path, title: str,
                             win_pct_col: str = "win_pct") -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    fig.patch.set_facecolor(_BG)
    _style_axes(ax)

    seasons = by_season["season"].astype(str)
    pct = by_season[win_pct_col] * 100
    ax.bar(seasons, pct, color=_ACCENT, width=0.6, zorder=3)
    ax.axhline(50, color=_MUTED, linewidth=1, linestyle="--", label="50% (breakeven, pre-vig)")
    ax.axhline(54, color="#ef4444", linewidth=1, linestyle=":", label="54% (honesty-standard trip-wire)")
    ax.set_ylim(35, 60)
    ax.set_ylabel("Win % (excl. pushes)", color=_INK, fontsize=10)
    ax.set_title(title, color=_INK, fontsize=11, pad=12)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=_BG)
    plt.close(fig)
