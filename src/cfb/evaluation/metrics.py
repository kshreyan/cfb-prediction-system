"""Calibration and accuracy metrics shared across all three markets.

These are the metrics the whole project is judged on -- see README.md's
"honesty standard". Raw accuracy is reported too, but log loss, Brier
score, and calibration/ECE are what select and validate models.
"""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    return float(np.mean((p_pred >= threshold).astype(int) == y_true))


def log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred)))


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    return float(np.mean((p_pred - y_true) ** 2))


def reliability_curve(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10):
    """Returns (bin_centers, observed_freq, bin_counts) for a reliability diagram."""
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_pred, edges[1:-1]), 0, n_bins - 1)

    centers, observed, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        counts.append(count)
        if count > 0:
            centers.append(float(p_pred[mask].mean()))
            observed.append(float(y_true[mask].mean()))
        else:
            centers.append(float((edges[b] + edges[b + 1]) / 2))
            observed.append(float("nan"))
    return np.array(centers), np.array(observed), np.array(counts)


def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_pred, edges[1:-1]), 0, n_bins - 1)
    n = len(p_pred)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(p_pred[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (count / n) * abs(acc - conf)
    return ece


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
