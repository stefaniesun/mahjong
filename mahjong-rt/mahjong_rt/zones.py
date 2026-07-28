"""Table region assignment from layout geometry (Phase 4 task 6).

No model here — the table's own structure carries the signal, and a heuristic can be
inspected and corrected when it is wrong. Zones are what turn a bag of recognised tiles
into game meaning: a face-down tile in a seat's meld run is a concealed kong, the same
tile in the middle scatter is nothing.

Three measured facts drive the design, checked against 4446 annotated boxes:

* **Size is depth.** Short side runs 15.9px → 81.7px as vertical position runs 0.39 →
  0.81 of frame height, monotonically. So tile size estimates distance from the viewer
  and, unlike a pixel threshold, survives a change of seat or resolution.
* **Horizontal position alone separates nothing.** nx sits at ~0.5 with std ~0.17 in
  every depth band, so "left means 上家" is false on its own — direction only means
  something once measured against the table centre.
* **Seats are regular, the pool is not.** A seat's tiles form aligned runs (even when
  stacked); the middle scatter does not. Regularity, not position, is what tells the
  river apart from a seat's discards.

The module stays optional (`zones.enabled`); with it off every tile reports
`unknown_zone` and the main pipeline is unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .events import Zone


@dataclass
class ZoneConfig:
    enabled: bool = True
    hand_size_quantile: float = 0.72  # tiles above this size quantile can be own hand
    hand_min_ny: float = 0.60  # ...and must sit at least this low in frame
    hand_row_tol: float = 0.9  # row grouping tolerance, in units of tile height
    run_gap_ratio: float = 0.8  # max horizontal gap inside a run, in tile widths
    min_run_len: int = 3  # a run shorter than this is not evidence of structure
    river_radius: float = 0.30  # distance from table centre (frame diagonals) still "middle"
    wall_size_quantile: float = 0.22  # smallest tiles, far side, likely wall
    seat_span_deg: float = 80.0  # angular width of each seat sector


def _rows(centres: np.ndarray, heights: np.ndarray, tol: float) -> list[list[int]]:
    """Group boxes into horizontal bands — the first step of finding aligned runs.

    Membership is tested against the band's running mean rather than its last member.
    Comparing to the last member lets a band chain-drift: each tile sits within
    tolerance of its predecessor while the band as a whole walks across the frame,
    swallowing unrelated rows into one giant "run".
    """
    order = np.argsort(centres[:, 1])
    bands: list[list[int]] = []
    means: list[float] = []
    for idx in order:
        placed = False
        for b, band in enumerate(bands):
            if abs(centres[idx, 1] - means[b]) <= heights[idx] * tol:
                band.append(int(idx))
                means[b] = float(np.mean(centres[band, 1]))
                placed = True
                break
        if not placed:
            bands.append([int(idx)])
            means.append(float(centres[idx, 1]))
    return bands


def _runs_in_row(band: list[int], boxes: np.ndarray, gap_ratio: float, min_len: int) -> list[list[int]]:
    """Split a horizontal band into tightly-spaced runs of tiles."""
    band = sorted(band, key=lambda i: boxes[i, 0])
    runs: list[list[int]] = []
    current = [band[0]]
    for prev, idx in zip(band, band[1:]):
        gap = boxes[idx, 0] - (boxes[prev, 0] + boxes[prev, 2])
        if gap <= boxes[prev, 2] * gap_ratio:
            current.append(idx)
        else:
            if len(current) >= min_len:
                runs.append(current)
            current = [idx]
    if len(current) >= min_len:
        runs.append(current)
    return runs


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
    centres = np.stack([arr[:, 0] + arr[:, 2] / 2.0, arr[:, 1] + arr[:, 3] / 2.0], axis=1)
    ny = centres[:, 1] / max(frame_h, 1)
    zones = [Zone.UNKNOWN.value] * n

    # --- 1. Own hand: the anchor everything else is measured from. It is the one group
    # that is simultaneously the largest, the lowest, and a single tight run.
    size_cut = float(np.quantile(short, config.hand_size_quantile))
    hand_candidates = [i for i in range(n) if short[i] >= size_cut and ny[i] >= config.hand_min_ny]
    hand: list[int] = []
    if len(hand_candidates) >= config.min_run_len:
        sub = np.array(hand_candidates)
        bands = _rows(centres[sub], arr[sub, 3], config.hand_row_tol)
        best: list[int] = []
        for band in bands:
            for run in _runs_in_row([int(sub[i]) for i in band], arr, config.run_gap_ratio, config.min_run_len):
                if len(run) > len(best):
                    best = run
        hand = best
    for i in hand:
        zones[i] = Zone.MY_HAND.value

    # --- 2. Table centre. With a hand found, the centre sits ahead of it; without one,
    # fall back to the tile centroid so the rest of the logic still has an origin.
    if hand:
        hand_centre = centres[hand].mean(axis=0)
        others = [i for i in range(n) if i not in set(hand)]
        forward = centres[others].mean(axis=0) if others else hand_centre
        table_centre = np.array([hand_centre[0] * 0.35 + forward[0] * 0.65, forward[1]], dtype=np.float32)
    else:
        table_centre = centres.mean(axis=0)

    # --- 3. Aligned runs mark a seat's tiles; whatever is left near the middle is the pool.
    remaining = [i for i in range(n) if zones[i] == Zone.UNKNOWN.value]
    structured: set[int] = set()
    runs_found: list[list[int]] = []
    if remaining:
        sub = np.array(remaining)
        for band in _rows(centres[sub], arr[sub, 3], config.hand_row_tol):
            for run in _runs_in_row([int(sub[i]) for i in band], arr, config.run_gap_ratio, config.min_run_len):
                runs_found.append(run)
                structured.update(run)

    diag = math.hypot(frame_w, frame_h)
    wall_cut = float(np.quantile(short, config.wall_size_quantile))

    def sector_of(point: np.ndarray) -> str:
        dx = float(point[0] - table_centre[0])
        dy = float(point[1] - table_centre[1])
        # Screen y grows downward; negate so 90° means "away from me, across the table".
        angle = math.degrees(math.atan2(-dy, dx))
        half = config.seat_span_deg / 2.0
        if 90 - half <= angle <= 90 + half:
            return Zone.SEAT_ACROSS.value
        if -half <= angle <= half:
            return Zone.SEAT_RIGHT.value
        if angle >= 180 - half or angle <= -180 + half:
            return Zone.SEAT_LEFT.value
        return Zone.UNKNOWN.value

    for run in runs_found:
        centre = centres[run].mean(axis=0)
        # A run of the very smallest tiles on the far side is the wall, not a discard row.
        if float(np.median(short[run])) <= wall_cut and centre[1] / max(frame_h, 1) <= 0.45:
            label = Zone.OPPONENT_WALL.value
        else:
            label = sector_of(centre)
        for i in run:
            zones[i] = label

    for i in range(n):
        if zones[i] != Zone.UNKNOWN.value:
            continue
        # Unstructured and close to the middle == the scattered pool.
        dist = float(np.linalg.norm(centres[i] - table_centre)) / diag
        zones[i] = Zone.RIVER.value if dist <= config.river_radius else sector_of(centres[i])

    debug = {
        "table_centre": [round(float(v), 1) for v in table_centre],
        "hand_len": len(hand),
        "runs": len(runs_found),
        "structured": len(structured),
    }
    return zones, debug


def assign_zones(boxes: Sequence[Sequence[float]], frame_w: int, frame_h: int, config: ZoneConfig) -> list[str]:
    """Thin wrapper kept for the pipeline's call site."""
    zones, _ = analyze_layout(boxes, frame_w, frame_h, config)
    return zones
