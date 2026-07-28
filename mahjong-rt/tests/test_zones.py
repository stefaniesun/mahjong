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
    boxes = [[10, 300, 30, 30], [1240, 300, 30, 30]] + [[600 + i * 30, 300, 30, 30] for i in range(5)]
    result = zones_for(boxes)
    assert result[0] == Zone.SEAT_LEFT.value
    assert result[1] == Zone.SEAT_RIGHT.value


def test_high_and_small_is_across():
    boxes = [[640, 100, 18, 18]] + [[500 + i * 40, 300, 40, 40] for i in range(6)]
    assert zones_for(boxes)[0] == Zone.SEAT_ACROSS.value


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
    # Measured 92.5%; the floor guards against regressions, not against improvement.
    assert accuracy >= 0.90, f"zone accuracy dropped to {accuracy:.3f}"


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
