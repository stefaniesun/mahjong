"""Video-metric tests — every number verified against a hand-built example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.metrics import (
    check_acceptance,
    compute_flicker,
    compute_latency,
    estimate_track_continuity,
    evaluate_checkpoint,
    iou_xywh,
    size_bucket,
)


def box(x, y, w, h, label="w1"):
    return {"bbox": [x, y, w, h], "label": label}


def test_iou_basics():
    assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xywh([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0
    # Half overlap: intersection 50, union 150.
    assert abs(iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) - 50 / 150) < 1e-6


def test_size_buckets():
    assert size_bucket(15, 40) == "lt20"
    assert size_bucket(30, 30) == "20to40"
    assert size_bucket(40, 41) == "20to40"
    assert size_bucket(80, 90) == "gt40"


def test_checkpoint_all_correct():
    gt = [box(0, 0, 50, 50, "w1"), box(100, 0, 50, 50, "t2")]
    result = evaluate_checkpoint(gt, [box(0, 0, 50, 50, "w1"), box(100, 0, 50, 50, "t2")])
    stats = result.as_dict()
    assert stats["recall"] == 1.0
    assert stats["class_accuracy"] == 1.0
    assert stats["spurious"] == 0


def test_checkpoint_found_but_misclassified():
    # Located correctly, named wrongly: counts as recalled but not class-correct.
    result = evaluate_checkpoint([box(0, 0, 50, 50, "w1")], [box(0, 0, 50, 50, "w9")])
    stats = result.as_dict()
    assert stats["recall"] == 1.0
    assert stats["class_accuracy"] == 0.0


def test_checkpoint_missed_and_spurious():
    gt = [box(0, 0, 50, 50, "w1"), box(200, 200, 50, 50, "t3")]
    confirmed = [box(0, 0, 50, 50, "w1"), box(500, 500, 50, 50, "b4")]
    stats = evaluate_checkpoint(gt, confirmed).as_dict()
    assert stats["matched"] == 1 and stats["missed"] == 1 and stats["spurious"] == 1
    assert stats["recall"] == 0.5


def test_checkpoint_bucket_split():
    gt = [box(0, 0, 15, 15, "w1"), box(100, 0, 80, 80, "t2")]
    confirmed = [box(0, 0, 15, 15, "w1"), box(100, 0, 80, 80, "b9")]
    stats = evaluate_checkpoint(gt, confirmed).as_dict()
    assert stats["by_bucket"]["lt20"]["class_accuracy"] == 1.0
    assert stats["by_bucket"]["gt40"]["class_accuracy"] == 0.0


def test_latency_matches_forward_only():
    truth = [{"tile": "w5", "t": 10.0}]
    events = [
        {"label": "w5", "ts": 9.0},  # before the tile was placed — a different w5
        {"label": "w5", "ts": 10.4},
    ]
    result = compute_latency(truth, events)
    assert result["n_matched"] == 1
    assert abs(result["p50"] - 0.4) < 1e-6


def test_latency_reports_unmatched():
    result = compute_latency([{"tile": "t7", "t": 3.0}], [{"label": "b2", "ts": 3.2}])
    assert result["n_matched"] == 0 and result["n_missed"] == 1


def test_latency_ignores_late_confirmation():
    result = compute_latency([{"tile": "w1", "t": 1.0}], [{"label": "w1", "ts": 30.0}], window=5.0)
    assert result["n_missed"] == 1


def test_latency_percentiles():
    truth = [{"tile": "w1", "t": 0.0}, {"tile": "w2", "t": 0.0}, {"tile": "w3", "t": 0.0}]
    events = [{"label": "w1", "ts": 0.1}, {"label": "w2", "ts": 0.2}, {"label": "w3", "ts": 0.3}]
    result = compute_latency(truth, events)
    assert abs(result["p50"] - 0.2) < 1e-6
    assert abs(result["max"] - 0.3) < 1e-6


def test_flicker_counts_updates_per_tile():
    events = [
        {"type": "tile_confirmed", "track_id": 1},
        {"type": "tile_confirmed", "track_id": 2},
        {"type": "tile_updated", "track_id": 1, "previous_label": "w1", "label": "w9", "ts": 1.0},
    ]
    result = compute_flicker(events)
    assert result["confirmed_tracks"] == 2
    assert result["total_updates"] == 1
    assert abs(result["updates_per_tile"] - 0.5) < 1e-9


def test_flicker_zero_when_stable():
    events = [{"type": "tile_confirmed", "track_id": i} for i in range(10)]
    assert compute_flicker(events)["updates_per_tile"] == 0.0


def test_track_continuity_detects_id_change():
    checkpoints = [
        {"confirmed": [{"bbox": [0, 0, 50, 50], "track_id": 1}]},
        {"confirmed": [{"bbox": [0, 0, 50, 50], "track_id": 7}]},  # same place, new id
    ]
    result = estimate_track_continuity(checkpoints)
    assert result["comparisons"] == 1 and result["id_switches"] == 1


def test_track_continuity_stable():
    checkpoints = [
        {"confirmed": [{"bbox": [0, 0, 50, 50], "track_id": 1}]},
        {"confirmed": [{"bbox": [1, 1, 50, 50], "track_id": 1}]},
    ]
    assert estimate_track_continuity(checkpoints)["id_switches"] == 0


def test_acceptance_red_lines():
    good = {
        "checkpoint": {"class_accuracy": 0.996},
        "latency": {"p95": 0.4},
        "flicker": {"updates_per_tile": 0.01},
        "performance": {"fps": 31.0},
    }
    assert all(item["pass"] for item in check_acceptance(good))
    bad = {
        "checkpoint": {"class_accuracy": 0.98},
        "latency": {"p95": 0.9},
        "flicker": {"updates_per_tile": 0.2},
        "performance": {"fps": 5.0},
    }
    assert not any(item["pass"] for item in check_acceptance(bad))
