"""Unit tests for site generation: proves it degrades gracefully when no
backtest data exists yet, and renders real content when it does."""
from __future__ import annotations

import cfb.reporting.site as site_mod


def test_build_site_with_no_data_still_produces_valid_html(tmp_path, monkeypatch):
    monkeypatch.setattr(site_mod, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(site_mod, "PREDICTIONS_DIR", tmp_path / "processed" / "predictions")
    monkeypatch.setattr(site_mod, "REPO_ROOT", tmp_path)

    site_mod.build_site(output_dir="site_out")

    index = tmp_path / "site_out" / "index.html"
    assert index.exists()
    html = index.read_text()
    assert "<html" in html
    assert "No prediction snapshot found yet" in html


def test_build_site_renders_real_prediction_snapshot(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    predictions_dir = processed / "predictions"
    predictions_dir.mkdir(parents=True)
    monkeypatch.setattr(site_mod, "PROCESSED_DIR", processed)
    monkeypatch.setattr(site_mod, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(site_mod, "REPO_ROOT", tmp_path)

    import json
    snapshot = {
        "generated_at": "2026-09-13T12:00:00+00:00", "season": 2026, "week": 3, "n_games": 1,
        "games": [{
            "game_id": "1", "home_team": "Alpha", "away_team": "Beta", "is_fbs_vs_fbs": True,
            "home_win_prob_raw": 0.6, "home_win_prob_calibrated": 0.58,
            "calibration_available": True, "predicted_margin": 3.5, "predicted_total": 52.0,
            "market_spread_home": -3.0, "home_cover_prob": 0.55, "market_total": 50.0,
            "over_prob": 0.52, "market_home_win_prob": 0.6, "moneyline_edge_prob": -0.02,
        }],
    }
    (predictions_dir / "snap.json").write_text(json.dumps(snapshot))

    site_mod.build_site(output_dir="site_out")
    html = (tmp_path / "site_out" / "index.html").read_text()
    assert "Alpha" in html and "Beta" in html
    assert (tmp_path / "site_out" / "data" / "latest_predictions.json").exists()
