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
