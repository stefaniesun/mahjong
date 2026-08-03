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

So the rule defaults to river and only claims a seat on clear evidence.

**Density is the fourth signal, and the one that broke the plateau.** Position and size
alone had stalled at 88.9% under image-level cross-validation (an earlier 92.5% figure
was an in-sample score — the thresholds had been grid-searched on the very data they
were then scored against). Adding nearest-neighbour distance took it to **93.1%
cross-validated**. The pool is the *dense* region — discards pile into the middle —
while a seat's tiles sit further apart relative to their own size. A side seat is the
inverse: its boxes overlap heavily (median nearest neighbour 0.31 of a box width),
because those tiles are seen edge-on and their boxes pile up.

**Clusters are the fifth signal.** A seat's tiles sit together as one compact group,
which the pool does not. Deciding each tile alone strands boundary cases — 23 of 45
in-sample errors sat within 0.10 of a threshold, in the pool while their neighbours were
correctly seated. Voting inside a cluster lets those follow their group.

Order matters here. Clustering *instead of* per-tile rules is worse (91.9% vs 93.6%):
single-linkage chains the seat across into the pool through the tiles between them, and
that zone collapses to 46.8%. Per-tile first, then smoothing within small clusters only,
reaches **94.5% cross-validated**.

Per-zone recall: hand 100%, river 97.2%, right 97.8%, left 85.7%, across 71.0%.

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
    hand_min_ny: float = 0.55
    # Side seats: outside the pool's lateral spread.
    left_max_nx: float = 0.30
    right_min_nx: float = 0.78
    # Seat across: overlaps the pool horizontally, so position alone cannot separate it.
    across_max_ny: float = 0.28
    # How isolated a tile is, as nearest-neighbour distance in units of its own width.
    # The pool is the *dense* region — discards pile into the middle — while a seat's
    # tiles sit further apart. Adding this one feature took cross-validated accuracy
    # from 88.9% to 93.1%; position and size alone had plateaued.
    across_min_nn: float = 0.55
    left_max_nn: float = 0.45
    # Tiles belonging to one seat sit together as a compact group. Deciding each tile on
    # its own leaves boundary cases stranded — a tile a hundredth outside a threshold
    # lands in the pool while its neighbours are correctly seated. Smoothing within a
    # cluster lets those tiles follow their group.
    cluster_link: float = 1.2      # same cluster if centres are within this many mean widths
    cluster_max_frac: float = 0.35  # clusters larger than this share of the frame are the
                                    # pool itself, which spans zones — leave them alone


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
    cx = arr[:, 0] + arr[:, 2] / 2.0
    cy = arr[:, 1] + arr[:, 3] / 2.0
    nx = cx / max(frame_w, 1)
    ny = cy / max(frame_h, 1)

    # Distance to the closest other tile, in units of this tile's own width. Normalising
    # by own width keeps it comparable across depth: a far tile is smaller, and so are
    # the gaps around it.
    nn = np.empty(n, dtype=np.float32)
    for i in range(n):
        d = np.hypot(cx - cx[i], cy - cy[i])
        d[i] = np.inf
        nn[i] = float(d.min()) / max(float(arr[i, 2]), 1.0) if n > 1 else 9.9

    zones: list[str] = []
    for i in range(n):
        if size_ratio[i] >= config.hand_size_ratio and ny[i] >= config.hand_min_ny:
            zones.append(Zone.MY_HAND.value)
        elif nx[i] <= config.left_max_nx and nn[i] <= config.left_max_nn:
            # Left seat's tiles cluster together; something equally far left but isolated
            # is more often a stray pool tile.
            zones.append(Zone.SEAT_LEFT.value)
        elif nx[i] >= config.right_min_nx:
            zones.append(Zone.SEAT_RIGHT.value)
        elif ny[i] <= config.across_max_ny and nn[i] >= config.across_min_nn:
            # Across overlaps the pool horizontally. What separates them is density: the
            # pool is packed, a seat's tiles are not.
            zones.append(Zone.SEAT_ACROSS.value)
        else:
            # Everything unclaimed is the shared middle scatter. Defaulting here is what
            # took this module from 39.7% to 92.5%: the pool is the majority class, and
            # a rule that guesses seats aggressively loses far more than it gains.
            zones.append(Zone.RIVER.value)

    # --- Cluster smoothing. Assigning per tile then voting inside each group beats both
    # alternatives measured on the labelled set: per-tile alone 93.6%, clustering alone
    # 91.9% (single-linkage chains the seat across into the pool and it collapses to
    # 46.8%), this hybrid 94.5% cross-validated.
    if n > 1 and config.cluster_link > 0:
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        widths = arr[:, 2]
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.hypot(cx[i] - cx[j], cy[i] - cy[j])) < config.cluster_link * (widths[i] + widths[j]) / 2:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            if len(members) < 2 or len(members) > config.cluster_max_frac * n:
                continue
            counts: dict[str, int] = {}
            for i in members:
                counts[zones[i]] = counts.get(zones[i], 0) + 1
            winner = max(counts.items(), key=lambda kv: kv[1])[0]
            for i in members:
                zones[i] = winner

    return zones, {"median_short": round(median, 1), "counts": {z: zones.count(z) for z in set(zones)}}


def assign_zones(boxes: Sequence[Sequence[float]], frame_w: int, frame_h: int, config: ZoneConfig) -> list[str]:
    """Thin wrapper kept for the pipeline's call site."""
    zones, _ = analyze_layout(boxes, frame_w, frame_h, config)
    return zones
