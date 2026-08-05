from __future__ import annotations

import numpy as np

from mahjong_rt.game_events import GameEvent
from mahjong_rt.raw_event_backtrack import (
    BacktrackConfig,
    LandingPoint,
    refine_events_with_raw_tracks,
    trace_landing_backwards,
)
from mahjong_rt.recording import FrameRecord, Recording


CLASSES = ["b1", "t5"]


def _frame(ts: float, detections: list[tuple[float, float, float, float, float, float]]) -> FrameRecord:
    """Detection tuple: x, y, target probability, other probability, detector score, size."""
    boxes = []
    scores = []
    probs = []
    for x, y, target_prob, other_prob, score, size in detections:
        boxes.append([x - size / 2, y - size / 2, x + size / 2, y + size / 2])
        scores.append(score)
        probs.append([other_prob, target_prob])
    probability_array = np.asarray(probs, dtype=np.float32).reshape(-1, 2)
    labels = probability_array.argmax(axis=1).astype(np.int16)
    return FrameRecord(
        frame_index=int(round(ts * 10)),
        timestamp=ts,
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(scores, dtype=np.float32),
        labels=labels,
        confidences=probability_array.max(axis=1),
        probs=probability_array,
        homography=np.eye(3, dtype=np.float32),
    )


def _recording(frames: list[FrameRecord]) -> Recording:
    return Recording("synthetic", CLASSES, 640, 480, 10.0, frames=frames)


def _event(ts: float = 1.4) -> GameEvent:
    return GameEvent(1, "discard", "me", "t5", ts, int(ts * 10), 0.6)


def _landing() -> LandingPoint:
    return LandingPoint(220.0, 100.0, 40.0)


def _config(**kwargs) -> BacktrackConfig:
    defaults = {
        "lookback_s": 1.2,
        "min_class_prob": 0.2,
        "max_gap_s": 0.45,
        "max_speed_sizes_per_s": 12.0,
        "landing_distance_ratio": 1.0,
        "min_path_nodes": 3,
        "min_displacement_ratio": 1.5,
    }
    defaults.update(kwargs)
    return BacktrackConfig(**defaults)


def test_links_non_top1_target_probability_across_short_occlusion() -> None:
    recording = _recording([
        _frame(0.4, [(80, 100, 0.35, 0.65, 0.8, 40)]),
        _frame(0.6, [(130, 100, 0.40, 0.60, 0.8, 40)]),
        _frame(0.8, []),
        _frame(1.0, [(185, 100, 0.45, 0.55, 0.8, 40)]),
        _frame(1.2, [(220, 100, 0.80, 0.20, 0.9, 40)]),
        _frame(1.4, [(220, 100, 0.90, 0.10, 0.9, 40)]),
    ])
    evidence = trace_landing_backwards(recording, _event(), _landing(), _config())
    assert evidence.has_motion
    assert evidence.path_nodes >= 4
    assert evidence.displacement_ratio >= 3.0
    assert 1.1 <= evidence.arrival_ts <= 1.3


def test_does_not_use_detections_after_stable_event_time() -> None:
    recording = _recording([
        _frame(1.2, [(220, 100, 0.9, 0.1, 0.9, 40)]),
        _frame(1.6, [(100, 100, 0.9, 0.1, 0.9, 40)]),
    ])
    evidence = trace_landing_backwards(recording, _event(1.4), _landing(), _config(min_path_nodes=1))
    assert evidence.arrival_ts <= 1.4


def test_static_detection_at_landing_is_not_motion_evidence() -> None:
    recording = _recording([
        _frame(ts, [(220, 100, 0.9, 0.1, 0.9, 40)])
        for ts in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
    ])
    evidence = trace_landing_backwards(recording, _event(), _landing(), _config())
    assert not evidence.has_motion
    assert evidence.displacement_ratio < 0.2


def test_refinement_uses_arrival_time_and_can_filter_events_without_motion() -> None:
    moving = _recording([
        _frame(0.6, [(100, 100, 0.5, 0.5, 0.8, 40)]),
        _frame(0.8, [(150, 100, 0.5, 0.5, 0.8, 40)]),
        _frame(1.0, [(190, 100, 0.6, 0.4, 0.8, 40)]),
        _frame(1.2, [(220, 100, 0.9, 0.1, 0.9, 40)]),
        _frame(1.4, [(220, 100, 0.9, 0.1, 0.9, 40)]),
    ])
    refined = refine_events_with_raw_tracks(
        moving,
        [_event()],
        [_landing()],
        _config(require_motion=True),
    )
    assert len(refined) == 1
    assert refined[0].ts == 1.2
    assert refined[0].confidence == 0.6

    static = _recording([_frame(ts, [(220, 100, 0.9, 0.1, 0.9, 40)]) for ts in (0.6, 0.8, 1.0, 1.2, 1.4)])
    assert refine_events_with_raw_tracks(static, [_event()], [_landing()], _config(require_motion=True)) == []
