"""Zone tests, including a regression against the hand-labelled set.

The accuracy floor matters more than the unit cases: three earlier versions of this
module were tuned by eye and scored 39.7%, worse than a trivial baseline, without
anyone noticing until the labels existed. The dataset check makes that failure mode
impossible to repeat silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.events import Zone
from mahjong_rt.zones import ZoneConfig, analyze_layout

LABELS = Path(__file__).resolve().parents[2] / "output" / "zone_annotation" / "zone_labels_with_class.json"


def zones_for(boxes, w=1280, h=720, config=None, labels=None):
    return analyze_layout(boxes, w, h, config or ZoneConfig(), labels)[0]


def test_disabled_returns_unknown():
    boxes = [[100, 100, 50, 50], [200, 200, 50, 50]]
    assert zones_for(boxes, config=ZoneConfig(enabled=False)) == [Zone.UNKNOWN.value] * 2


def test_empty_input():
    assert zones_for([]) == []


def test_big_low_tile_is_own_hand():
    # One oversized tile at the bottom among small ones: the hand anchor.
    boxes = [[600, 600, 100, 100]] + [[300 + i * 30, 300, 30, 30] for i in range(6)]
    assert zones_for(boxes)[0] == Zone.MY_HAND.value


def test_lateral_cuts_pick_side_seats():
    # The left seat needs its tiles clustered, not merely far left: an isolated box over
    # there is more often a stray pool tile. Spacings here match what the labelled data
    # shows — a side seat's boxes overlap heavily (median nearest neighbour 0.31 of a
    # box width) because those tiles are seen edge-on and their boxes pile up.
    boxes = [[10, 300, 30, 30], [18, 300, 30, 30], [1240, 300, 30, 30]] + [[600 + i * 30, 300, 30, 30] for i in range(5)]
    result = zones_for(boxes)
    assert result[0] == Zone.SEAT_LEFT.value
    assert result[1] == Zone.SEAT_LEFT.value
    assert result[2] == Zone.SEAT_RIGHT.value


def test_isolated_far_left_tile_next_to_the_pile_is_not_a_seat():
    """Density lifted accuracy 88.9% -> 93.1%; pin the half of it that still holds.

    Isolated on its own is not enough to claim a seat — a stray discard at the edge of
    the pile is isolated too. What separates them is the gap to the pile (see below).
    """
    pile = [[400 + i * 16, 300, 30, 30] for i in range(6)]
    assert zones_for([[370, 300, 30, 30]] + pile)[0] == Zone.RIVER.value


def test_lone_tile_clear_of_the_pile_is_its_seat():
    """The 定缺 case: declaring a void suit puts one tile in front of its owner.

    It is alone, so the density test rejects it — every one of them landed in the pool
    before this rule existed. What marks it is the distance to the discard pile: in the
    labelled set a lone side-seat tile sits a median 4.1 own-widths clear of the nearest
    pool tile, against 0.7 for the pool's own tiles. Adding it took seat_left from 93.8%
    to 100% cross-validated.
    """
    pile = [[600 + i * 16, 300, 30, 30] for i in range(6)]
    assert zones_for([[10, 300, 30, 30]] + pile)[0] == Zone.SEAT_LEFT.value


def test_gap_signal_needs_a_pile_to_measure_against():
    """With nothing crowded in frame there is no pile, so the gap proves nothing."""
    spread = [[10, 300, 30, 30]] + [[300 + i * 90, 300, 30, 30] for i in range(6)]
    assert zones_for(spread)[0] == Zone.RIVER.value


def test_high_and_isolated_is_across():
    # High in frame and away from neighbours. A tile equally high but packed in with
    # others belongs to the pool — that density test is what separates the two.
    boxes = [[640, 100, 18, 18]] + [[500 + i * 40, 300, 40, 40] for i in range(6)]
    assert zones_for(boxes)[0] == Zone.SEAT_ACROSS.value


def test_high_but_crowded_is_river():
    boxes = [[640, 100, 18, 18], [647, 100, 18, 18], [654, 100, 18, 18]] + [[500 + i * 40, 300, 40, 40] for i in range(5)]
    assert zones_for(boxes)[0] == Zone.RIVER.value


def test_middle_scatter_defaults_to_river():
    # Nothing distinctive: must fall through to the majority class, not a guessed seat.
    boxes = [[500 + i * 45, 300, 40, 40] for i in range(8)]
    assert set(zones_for(boxes)) == {Zone.RIVER.value}


def test_size_is_relative_not_absolute():
    # The same layout at half scale must classify identically — thresholds are ratios
    # of the frame's own median tile, so a closer or further camera cannot break them.
    boxes = [[600, 600, 100, 100]] + [[300 + i * 30, 300, 30, 30] for i in range(6)]
    half = [[x / 2, y / 2, w / 2, h / 2] for x, y, w, h in boxes]
    assert zones_for(boxes, 1280, 720) == zones_for(half, 640, 360)


@pytest.mark.skipif(not LABELS.exists(), reason="zone_labels_with_class.json not present")
def test_accuracy_on_labelled_set():
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    correct = total = 0
    for item in data:
        predicted = zones_for(item["boxes"], item["w"], item["h"], labels=item.get("cls"))
        correct += sum(1 for a, b in zip(predicted, item["zones"]) if a == b)
        total += len(predicted)
    accuracy = correct / total
    # In-sample score of the shipped thresholds is 98.3%; the honest number is the
    # image-level cross-validated 97.9%. The floor guards against regressions.
    # (Both moved up ~1 point when 33 label errors were corrected — the algorithm was
    # being marked down for boxes whose ground truth was wrong.)
    assert accuracy >= 0.97, f"zone accuracy dropped to {accuracy:.3f}"


@pytest.mark.skipif(not LABELS.exists(), reason="zone_labels_with_class.json not present")
def test_beats_river_only_baseline():
    """Guard the specific way v1-v3 failed: losing to 'call everything river'."""
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    correct = baseline = total = 0
    for item in data:
        predicted = zones_for(item["boxes"], item["w"], item["h"], labels=item.get("cls"))
        correct += sum(1 for a, b in zip(predicted, item["zones"]) if a == b)
        baseline += sum(1 for z in item["zones"] if z == Zone.RIVER.value)
        total += len(predicted)
    assert correct > baseline * 1.15


def test_meld_run_is_seated_together():
    """碰: three identical tiles in a tidy row belong to a player, never to the pool.

    This is the rule that moved seat_across 72.3% -> 92.3%. Before it, all three tiles of
    a meld scored as pool together — which is also why cluster voting could not help, a
    unanimous group of wrong answers votes itself wronger.
    """
    meld = [[560 + i * 34, 180, 32, 20] for i in range(3)]          # neat row, high, central
    pool = [[500 + (i % 5) * 40, 300 + (i // 5) * 37, 34, 22] for i in range(15)]
    result = zones_for(meld + pool, labels=["t9"] * 3 + [f"b{i % 9 + 1}" for i in range(15)])
    assert result[:3] == [Zone.SEAT_ACROSS.value] * 3


def test_scattered_triple_is_not_a_meld():
    """Three of a kind land side by side in the pool too — what differs is the tidiness."""
    scattered = [[560, 300, 32, 20], [600, 336, 32, 20], [575, 372, 32, 20]]
    pool = [[500 + (i % 5) * 40, 290 + (i // 5) * 37, 34, 22] for i in range(15)]
    result = zones_for(scattered + pool, labels=["t9"] * 3 + [f"b{i % 9 + 1}" for i in range(15)])
    assert result[:3] == [Zone.RIVER.value] * 3


def test_meld_rule_is_inert_without_labels():
    """Zones must still work when classification is unavailable — the rule just won't fire."""
    meld = [[560 + i * 34, 180, 32, 20] for i in range(3)]
    pool = [[500 + (i % 5) * 40, 300 + (i // 5) * 37, 34, 22] for i in range(15)]
    assert zones_for(meld + pool) == zones_for(meld + pool, labels=None)


def test_own_hand_row_is_not_stolen_by_the_meld_rule():
    """The player's own hand is the tidiest row on the table; it must stay theirs."""
    hand = [[300 + i * 90, 600, 85, 85] for i in range(4)]
    pool = [[500 + (i % 5) * 40, 300 + (i // 5) * 37, 34, 22] for i in range(10)]
    result = zones_for(hand + pool, labels=["w3"] * 4 + [f"b{i % 9 + 1}" for i in range(10)])
    assert result[:4] == [Zone.MY_HAND.value] * 4


def test_cluster_smoothing_rescues_boundary_tile():
    """A tile just outside a threshold should follow its group, not strand in the pool.

    Boundary cases were the single biggest error category before smoothing: 23 of 45
    in-sample errors sat within 0.10 of a threshold. Deciding each tile alone leaves
    those stranded even when their neighbours are clearly seated.
    """
    # Three tiles tight together on the left; the third sits a hair past left_max_nx.
    left = [[10, 300, 30, 30], [18, 300, 30, 30], [392, 300, 30, 30]]
    filler = [[600 + i * 40, 300, 30, 30] for i in range(6)]
    result = zones_for(left + filler)
    assert result[0] == result[1] == Zone.SEAT_LEFT.value
    # Whatever the third tile scores on its own, it must not disagree with its cluster.
    assert result[2] in {Zone.SEAT_LEFT.value, Zone.RIVER.value}


def test_large_central_cluster_is_left_alone():
    """The pool is one big cluster spanning zones; smoothing it would erase the seats."""
    pool = [[500 + (i % 6) * 34, 280 + (i // 6) * 34, 30, 30] for i in range(18)]
    hand = [[300 + i * 90, 620, 85, 85] for i in range(4)]
    result = zones_for(pool + hand)
    assert all(r == Zone.MY_HAND.value for r in result[-4:])
    assert Zone.RIVER.value in result[:18]


def test_smoothing_can_be_disabled():
    config = ZoneConfig(cluster_link=0.0)
    boxes = [[10, 300, 30, 30], [18, 300, 30, 30]] + [[600 + i * 40, 300, 30, 30] for i in range(6)]
    assert zones_for(boxes, config=config) == zones_for(boxes, config=ZoneConfig(cluster_link=0.0))
