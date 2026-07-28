"""Table region assignment (Phase 4 task 6).

Pure geometry, no model — but the thresholds come from 899 hand-labelled boxes rather
than from intuition. Zones are what turn a bag of recognised tiles into game meaning:
the engine's VisionSnapshot keys rivers and melds by player, so "whose tile is this"
has to be answered before anything downstream can use the output.

Three earlier versions guessed at thresholds and scored 39.7% against those labels —
worse than a baseline that calls everything except the hand "river" (75.8%). The
mistake was structural, not numeric: they tried hard to carve out three seats, and in
doing so shredded the pool, which is *the majority class* at 52.5% of all tiles. The
seats together are only 17.5%.

So the rule now defaults to river and only claims a seat on clear evidence. Measured
per-zone recall on the labelled set: hand 97.0%, river 95.3%, right 97.8%, left 73.5%,
across 71.0% — 92.5% overall.

What the labels showed (median, and 10-90% spread of normalised x):

    my_hand      nx 0.47 [0.18,0.77]  ny 0.78  size 2.97x frame median
    river        nx 0.52 [0.37,0.68]  ny 0.42  size 0.93x
    seat_left    nx 0.23 [0.14,0.29]  ny 0.39  size 0.71x
    seat_across  nx 0.60 [0.46,0.68]  ny 0.27  size 0.71x
    seat_right   nx 0.86 [0.79,0.92]  ny 0.43  size 0.78x

Left and right sit almost entirely outside the pool's horizontal spread, so a lateral
cut separates them cleanly. Across overlaps the pool horizontally and is told apart by
being higher and smaller — which is why it is the weakest of the five.

Size is expressed relative to the frame's own median tile, never in pixels: tile size
tracks depth (short side runs 15.9px to 81.7px as ny runs 0.39 to 0.81), so a ratio
survives a change of seat, resolution, or how close the player sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .events import Zone


@dataclass
class ZoneConfig:
    enabled: bool = True
    # Own hand: much larger than everything else, and low in frame.
    hand_size_ratio: float = 1.4
    hand_min_ny: float = 0.60
    # Side seats: outside the pool's lateral spread.
    left_max_nx: float = 0.28
    right_min_nx: float = 0.78
    # Seat across: overlaps the pool horizontally, so it needs both "high" and "small".
    across_max_ny: float = 0.28
    across_max_size_ratio: float = 0.9


def analyze_layout(
    boxes: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
    config: ZoneConfig,
) -> tuple[list[str], dict]:
    """boxes: xywh in pixels. Returns (zone per box, debug info)."""
    n = len(boxes)
    if not config.enabled or n == 0:
        return [Zone.UNKNOWN.value] * n, {}

    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    short = np.minimum(arr[:, 2], arr[:, 3])
    median = float(np.median(short))
    size_ratio = short / max(median, 1e-6)
    nx = (arr[:, 0] + arr[:, 2] / 2.0) / max(frame_w, 1)
    ny = (arr[:, 1] + arr[:, 3] / 2.0) / max(frame_h, 1)

    zones: list[str] = []
    for i in range(n):
        if size_ratio[i] >= config.hand_size_ratio and ny[i] >= config.hand_min_ny:
            zones.append(Zone.MY_HAND.value)
        elif nx[i] <= config.left_max_nx:
            zones.append(Zone.SEAT_LEFT.value)
        elif nx[i] >= config.right_min_nx:
            zones.append(Zone.SEAT_RIGHT.value)
        elif ny[i] <= config.across_max_ny and size_ratio[i] <= config.across_max_size_ratio:
            zones.append(Zone.SEAT_ACROSS.value)
        else:
            # Everything unclaimed is the shared middle scatter. Defaulting here is what
            # took this module from 39.7% to 92.5%: the pool is the majority class, and
            # a rule that guesses seats aggressively loses far more than it gains.
            zones.append(Zone.RIVER.value)

    return zones, {"median_short": round(median, 1), "counts": {z: zones.count(z) for z in set(zones)}}


def assign_zones(boxes: Sequence[Sequence[float]], frame_w: int, frame_h: int, config: ZoneConfig) -> list[str]:
    """Thin wrapper kept for the pipeline's call site."""
    zones, _ = analyze_layout(boxes, frame_w, frame_h, config)
    return zones
