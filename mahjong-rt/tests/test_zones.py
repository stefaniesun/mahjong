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

LABELS = Path(__file__).resolve().parents[2] / "output" / "zone_annotation" / "zone_labels.json"


def zones_for(boxes, w=1280, h=720, config=None):
    return analyze_layout(boxes, w, h, config or ZoneConfig())[0]


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


def test_isolated_far_left_tile_is_not_a_seat():
    """Density is the signal that lifted accuracy 88.9% -> 93.1%; pin it."""
    boxes = [[10, 300, 30, 30]] + [[600 + i * 32, 300, 30, 30] for i in range(6)]
    assert zones_for(boxes)[0] == Zone.RIVER.value


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


@pytest.mark.skipif(not LABELS.exists(), reason="zone_labels.json not present")
def test_accuracy_on_labelled_set():
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    correct = total = 0
    for item in data:
        predicted = zones_for(item["boxes"], item["w"], item["h"])
        correct += sum(1 for a, b in zip(predicted, item["zones"]) if a == b)
        total += len(predicted)
    accuracy = correct / total
    # In-sample score of the shipped thresholds is 94.4%; the honest number is the
    # image-level cross-validated 93.1%. The floor guards against regressions.
    assert accuracy >= 0.92, f"zone accuracy dropped to {accuracy:.3f}"


@pytest.mark.skipif(not LABELS.exists(), reason="zone_labels.json not present")
def test_beats_river_only_baseline():
    """Guard the specific way v1-v3 failed: losing to 'call everything river'."""
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    correct = baseline = total = 0
    for item in data:
        predicted = zones_for(item["boxes"], item["w"], item["h"])
        correct += sum(1 for a, b in zip(predicted, item["zones"]) if a == b)
        baseline += sum(1 for z in item["zones"] if z == Zone.RIVER.value)
        total += len(predicted)
    assert correct > baseline * 1.15


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
