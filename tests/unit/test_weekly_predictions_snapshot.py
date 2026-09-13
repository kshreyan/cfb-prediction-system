"""Unit test for the immutable-snapshot guarantee: saving twice must
produce two distinct files, never overwrite the first."""
from __future__ import annotations

import cfb.reporting.weekly_predictions as wp


def test_save_prediction_snapshot_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(wp, "PREDICTIONS_DIR", tmp_path)

    payload_1 = {"generated_at": "2026-09-13T12:00:00+00:00", "season": 2026, "week": 3,
                 "n_games": 1, "games": []}
    payload_2 = {"generated_at": "2026-09-13T13:00:00+00:00", "season": 2026, "week": 3,
                 "n_games": 1, "games": []}

    path_1 = wp.save_prediction_snapshot(payload_1)
    path_2 = wp.save_prediction_snapshot(payload_2)

    assert path_1 != path_2
    assert path_1.exists() and path_2.exists()
    assert len(list(tmp_path.glob("*.json"))) == 2
