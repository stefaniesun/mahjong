"""Game event extraction.

These pin the mechanics — turn order, meld anchoring, warm-up — on synthetic input.
They do not pin accuracy, because on real video it is currently poor; see the module
docstring and docs/EVENT_LAYER_STATUS.md for the measurements and why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.game_events import GameEventConfig, GameEventExtractor


def summary(tiles, ts=0.0, frame=0):
    return {"frame_idx": frame, "ts": ts, "tiles": tiles, "counts_by_zone": {}, "type": "frame_summary"}


def river(label, x, y=300.0):
    return {"track_id": hash((label, x)) % 10000, "label": label, "bbox": [x, y, 40.0, 40.0],
            "zone": "river", "state": "CONFIRMED", "visible": True}


def meld(label, zone, x):
    return [{"track_id": hash((label, zone, x, i)) % 10000, "label": label,
             "bbox": [x + i * 42.0, 200.0, 40.0, 40.0], "zone": zone,
             "state": "CONFIRMED", "visible": True} for i in range(3)]


def test_warmup_does_not_report_the_pool_already_on_the_table():
    """Tiles present before the clip started are not discards."""
    ex = GameEventExtractor(GameEventConfig(warmup_frames=5, settle_frames=2, start_player="me"))
    existing = [river("w1", 100), river("b4", 200), river("t9", 300)]
    for i in range(8):
        ex.add_frame(summary(existing, ts=i * 0.1, frame=i))
    assert ex.events == []


def test_a_new_tile_after_warmup_is_a_discard():
    ex = GameEventExtractor(GameEventConfig(warmup_frames=3, settle_frames=2, start_player="me"))
    existing = [river("w1", 100)]
    for i in range(5):
        ex.add_frame(summary(existing, ts=i * 0.1, frame=i))
    for i in range(5, 10):
        ex.add_frame(summary(existing + [river("b4", 400)], ts=i * 0.5, frame=i))
    assert [(e.event_type, e.tile, e.player) for e in ex.events] == [("discard", "b4", "me")]


def test_turn_advances_counter_clockwise():
    ex = GameEventExtractor(GameEventConfig(warmup_frames=1, settle_frames=1, min_gap_s=0.0, start_player="me"))
    ex.add_frame(summary([], ts=0.0, frame=0))
    tiles = []
    for i, (label, x) in enumerate([("w1", 100), ("b4", 300), ("t9", 500), ("w5", 700)]):
        tiles = tiles + [river(label, x)]
        ex.add_frame(summary(tiles, ts=1.0 + i, frame=i + 1))
    assert [e.player for e in ex.events] == ["me", "right", "across", "left"]


def test_a_pong_reanchors_the_turn():
    """The claimer discards next, so a pong pulls a drifting pointer back into line."""
    ex = GameEventExtractor(GameEventConfig(warmup_frames=1, settle_frames=1, min_gap_s=0.0, start_player="me"))
    ex.add_frame(summary([], ts=0.0, frame=0))
    ex.add_frame(summary(meld("t3", "seat_across", 400), ts=1.0, frame=1))
    assert [(e.event_type, e.player, e.tile) for e in ex.events] == [("pong", "across", "t3")]
    ex.add_frame(summary(meld("t3", "seat_across", 400) + [river("w9", 100)], ts=2.0, frame=2))
    assert ex.events[-1].event_type == "discard"
    assert ex.events[-1].player == "across", "碰了之后该由碰的那家出牌"


def test_a_settled_cell_is_not_reported_twice():
    ex = GameEventExtractor(GameEventConfig(warmup_frames=1, settle_frames=2, min_gap_s=0.0, start_player="me"))
    ex.add_frame(summary([], ts=0.0, frame=0))
    for i in range(10):
        ex.add_frame(summary([river("b4", 400)], ts=1.0 + i, frame=i + 1))
    assert len(ex.events) == 1


def test_flicker_never_settles_into_an_event():
    """One frame of a hand passing through is not a discard."""
    ex = GameEventExtractor(GameEventConfig(warmup_frames=1, settle_frames=3, start_player="me"))
    ex.add_frame(summary([], ts=0.0, frame=0))
    for i in range(6):
        tiles = [river("b4", 400)] if i % 2 == 0 else []
        ex.add_frame(summary(tiles, ts=1.0 + i, frame=i + 1))
    assert ex.events == []


def test_detections_mark_a_cell_as_already_occupied():
    """A tile the detector saw all along is not new, however late it gets confirmed."""
    ex = GameEventExtractor(GameEventConfig(warmup_frames=1, settle_frames=2, start_player="me"))
    boxes = np.array([[400.0, 300.0, 440.0, 340.0]], dtype=np.float32)
    ex.add_frame(summary([], ts=0.0, frame=0), detections=boxes)
    for i in range(5):
        ex.add_frame(summary([river("b4", 400)], ts=1.0 + i, frame=i + 1), detections=boxes)
    assert ex.events == []
