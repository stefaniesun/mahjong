"""State machine behaviour: zone voting and occlusion retention.

Both rules here were written after watching real output go wrong, so they are pinned
rather than left to the accuracy suite: a regression in either is invisible in the
aggregate numbers but obvious to anyone watching the overlay.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.events import TileState, Zone
from mahjong_rt.state_machine import StateMachine
from mahjong_rt.voter import Observation


def feed(machine, zones, *, label="w3", track_id=1):
    """Run one track through the given per-frame zones."""
    for i, zone in enumerate(zones):
        machine.update(
            [(track_id, [10.0, 10.0, 30.0, 30.0], Observation(label=label, confidence=0.9, short_side=30.0), zone)],
            frame_idx=i,
            ts=i / 10.0,
        )
    return machine.tiles[track_id].zone


def test_zone_follows_the_majority_not_the_last_frame():
    machine = StateMachine(emit_frame_summary=False, zone_evidence_weight=1.0)
    zones = [Zone.RIVER.value] * 5 + [Zone.SEAT_ACROSS.value]
    assert feed(machine, zones) == Zone.RIVER.value


def test_seat_evidence_outweighs_the_river_default():
    """`river` is also what a frame reports when it simply failed to see the evidence.

    A meld needs all three of its tiles detected in one frame to be recognised; miss one
    and that frame falls back to river. Counting both the same lets the frames that saw
    nothing outvote the frames that saw something.
    """
    machine = StateMachine(emit_frame_summary=False, zone_evidence_weight=2.0)
    zones = [Zone.RIVER.value] * 5 + [Zone.SEAT_ACROSS.value] * 3
    assert feed(machine, zones) == Zone.SEAT_ACROSS.value

    # Not unconditional: a clear river majority must still win.
    plain = StateMachine(emit_frame_summary=False, zone_evidence_weight=2.0)
    assert feed(plain, [Zone.RIVER.value] * 9 + [Zone.SEAT_ACROSS.value] * 3) == Zone.RIVER.value


def test_weighting_can_be_switched_off():
    machine = StateMachine(emit_frame_summary=False, zone_evidence_weight=1.0)
    zones = [Zone.RIVER.value] * 5 + [Zone.SEAT_ACROSS.value] * 3
    assert feed(machine, zones) == Zone.RIVER.value


def test_two_seats_do_not_get_a_free_pass_over_each_other():
    """The weight is river-vs-evidence, not a thumb on one seat over another."""
    machine = StateMachine(emit_frame_summary=False, zone_evidence_weight=2.0)
    zones = [Zone.SEAT_LEFT.value] * 5 + [Zone.SEAT_ACROSS.value] * 3
    assert feed(machine, zones) == Zone.SEAT_LEFT.value


def test_reacquired_tile_does_not_double_count():
    """Fragmentation is the norm; the old entry has to go when the tile comes back.

    Holding tiles through occlusion is right, but a broken track returns under a new id,
    and keeping both counted one physical tile twice. Over 30 seconds a 14-tile hand was
    reporting 62 tiles.
    """
    machine = StateMachine(emit_frame_summary=False, lost_after=300)
    feed(machine, [Zone.RIVER.value] * 5, label="w3", track_id=1)
    for i in range(5):                       # track 1 disappears
        machine.update([], frame_idx=50 + i, ts=5.0 + i)
    assert len(machine.tiles) == 1
    feed(machine, [Zone.RIVER.value] * 5, label="w3", track_id=2)   # same tile, new id
    assert len(machine.tiles) == 1, "同一张牌不该同时以两个 ID 存在"
    assert 2 in machine.tiles


def test_a_slot_holds_one_tile_even_if_it_reads_differently():
    """The label test alone misses the hand position whose tile has been replaced."""
    machine = StateMachine(emit_frame_summary=False, lost_after=300)
    feed(machine, [Zone.MY_HAND.value] * 5, label="w3", track_id=1)
    for i in range(5):
        machine.update([], frame_idx=50 + i, ts=5.0 + i)
    feed(machine, [Zone.MY_HAND.value] * 5, label="b8", track_id=2)  # same box, other tile
    assert len(machine.tiles) == 1
    assert machine.tiles[2].label == "b8"


def test_distinct_tiles_side_by_side_both_survive():
    """Merging must not eat the neighbour: hand tiles sit shoulder to shoulder."""
    machine = StateMachine(emit_frame_summary=False, lost_after=300)
    for i in range(5):
        machine.update(
            [(1, [10.0, 10.0, 30.0, 30.0], Observation(label="w3", confidence=0.9, short_side=30.0), Zone.MY_HAND.value)],
            frame_idx=i, ts=i / 10.0,
        )
    for i in range(5):
        machine.update([], frame_idx=50 + i, ts=5.0 + i)
    feed(machine, [Zone.MY_HAND.value] * 5, label="b8", track_id=2)   # bbox is the same box
    machine2 = StateMachine(emit_frame_summary=False, lost_after=300, merge_dist=0.3, merge_iou=0.5)
    for i in range(5):
        machine2.update(
            [(1, [10.0, 10.0, 30.0, 30.0], Observation(label="w3", confidence=0.9, short_side=30.0), Zone.MY_HAND.value),
             (2, [45.0, 10.0, 30.0, 30.0], Observation(label="b8", confidence=0.9, short_side=30.0), Zone.MY_HAND.value)],
            frame_idx=i, ts=i / 10.0,
        )
    assert len(machine2.tiles) == 2, "并排的两张不同的牌不能被合并掉"


def test_hand_expires_faster_than_the_table():
    """A placed tile stays; a tile in one's own hand gets played and must not linger."""
    machine = StateMachine(emit_frame_summary=False, lost_after=300, hand_lost_after=10)
    for i in range(5):
        machine.update(
            [(1, [10.0, 10.0, 30.0, 30.0], Observation(label="w3", confidence=0.9, short_side=30.0), Zone.MY_HAND.value),
             (2, [400.0, 200.0, 30.0, 30.0], Observation(label="b8", confidence=0.9, short_side=30.0), Zone.RIVER.value)],
            frame_idx=i, ts=i / 10.0,
        )
    for i in range(15):
        machine.update([], frame_idx=100 + i, ts=10.0 + i)
    assert 1 not in machine.tiles, "手牌该过期了"
    assert 2 in machine.tiles, "桌面上的牌不该过期"


def test_occluded_tile_stays_in_the_world():
    """A placed tile does not move again; a hand over it is not a removal."""
    machine = StateMachine(emit_frame_summary=False, lost_after=300)
    feed(machine, [Zone.MY_HAND.value] * 5)
    for i in range(10):
        machine.update([], frame_idx=100 + i, ts=10.0 + i)
    tile = machine.tiles[1]
    assert tile.state == TileState.OCCLUDED
    assert tile in machine.confirmed_tiles()
    assert tile not in machine.confirmed_tiles(include_occluded=False)
