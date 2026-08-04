"""Table region assignment (Phase 4 task 6).

Geometry plus one class-based rule, no model — and the thresholds come from 923
hand-labelled boxes rather than from intuition. Zones are what turn a bag of recognised tiles into game meaning:
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
reaches 94.5% cross-validated.

**Distance to the dense mass is the sixth, and it came from the game rather than the
data.** 四川麻将 has 定缺: a player declares a void suit and lays that one tile in front
of themselves. It is a single tile, so the density test rejects it, and every one of them
was landing in the pool. What marks it is not density but separation — a lone side-seat
tile sits a median 4.1 own-widths clear of the nearest pool tile, against 0.7 for the
pool's own tiles. The feature needs no labels: the pool is by definition whatever is
crowded. seat_left went 93.8% -> 100%, total to **96.3% cross-validated**.

**Melds are the seventh, and the only one that looks at what a tile *is*.** 碰/杠 lays
three or four identical tiles in a tidy row in front of their owner. The pool throws up
three of a kind side by side too, but scattered: measured across the labelled set, an
across meld's row alignment (std of y over mean height) is 0.043 against the pool's 1.28,
and the two do not overlap. seat_across had survived six versions at 71-72% because a
meld's three tiles all fail the position test together — which is also why cluster voting
could not help, a unanimous group of wrong answers only votes itself wronger. Overriding
the whole run at once took it to **92.3%**, and cost the pool nothing: **97.9% total,
cross-validated**.

Per-zone recall: hand 100%, left 100%, river 97.6%, right 96.0%, across 92.3%.

A caution on all of these numbers: they moved about a point when 33 label errors were
found and fixed (`scripts/audit_zone_labels.py`). Earlier figures in this file's history
are against the uncorrected set and are not comparable.

What the labels showed (median, and 10-90% spread of normalised x):

    my_hand      nx 0.47 [0.18,0.77]  ny 0.78  size 2.97x frame median
    river        nx 0.52 [0.37,0.68]  ny 0.42  size 0.93x
    seat_left    nx 0.23 [0.14,0.29]  ny 0.39  size 0.71x
    seat_across  nx 0.60 [0.46,0.68]  ny 0.27  size 0.71x
    seat_right   nx 0.86 [0.79,0.92]  ny 0.43  size 0.78x

Left and right sit almost entirely outside the pool's horizontal spread, so a lateral
cut separates them cleanly. Across overlaps the pool horizontally and is told apart by
being higher and smaller — which is why position alone was never enough for it, and why
the meld rule above is what finally moved it.

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
    # A 定缺 tile breaks the density test above. Declaring a void suit puts one tile in
    # front of its owner, alone — so it is isolated, and `left_max_nn` rejects it, which
    # is how every one of them ended up in the pool. What actually marks it as a seat's
    # tile is the gap: it sits well clear of the discard pile. Distance to the dense mass
    # needs no labels — the pool is by definition whatever is crowded.
    # Adding this took seat_left from 93.8% to 100% cross-validated at a cost of 0.2% on
    # the pool. 4 of 5 folds independently chose 2.0.
    dense_max_nn: float = 0.55   # a tile this packed counts as part of the dense mass
    seat_min_gap: float = 2.0    # own widths clear of that mass to claim a seat alone
    # Tiles belonging to one seat sit together as a compact group. Deciding each tile on
    # its own leaves boundary cases stranded — a tile a hundredth outside a threshold
    # lands in the pool while its neighbours are correctly seated. Smoothing within a
    # cluster lets those tiles follow their group.
    cluster_link: float = 1.2      # same cluster if centres are within this many mean widths
    cluster_max_frac: float = 0.35  # clusters larger than this share of the frame are the
                                    # pool itself, which spans zones — leave them alone
    # 碰/杠: three or four identical tiles laid in a neat row in front of their owner.
    # This is the only signal here that uses what the tile *is* rather than where it sits,
    # and it is what finally moved seat_across. The pool also throws up three of a kind
    # side by side, but scattered: an across meld's row alignment (std of y over mean
    # height) is 0.043 against the pool's 1.28, and the two do not overlap at all in the
    # labelled set. Requires labels; without them the rule simply does not fire.
    meld_link: float = 1.6         # same run if same class and centres within this many widths
    meld_max_align: float = 0.12   # std of y over mean height — above this it is scatter
    meld_max_ny: float = 0.35      # a run lower than this is not somebody's meld across the table


def analyze_layout(
    boxes: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
    config: ZoneConfig,
    labels: Sequence[str | None] | None = None,
) -> tuple[list[str], dict]:
    """boxes: xywh in pixels. labels: tile class per box, or None where unknown.

    Returns (zone per box, debug info).
    """
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
    dist = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(dist, np.inf)
    nn = np.full(n, 9.9, dtype=np.float32)
    if n > 1:
        nn = (dist.min(axis=1) / np.maximum(arr[:, 2], 1.0)).astype(np.float32)

    # Distance to the nearest *crowded* tile. The dense mass is the discard pile, so a
    # tile far from it is somebody's, not the pool's — this is what rescues the lone 定缺
    # tile that the density test above throws away.
    # Zero when nothing is crowded: with no pile in view there is nothing to be clear of,
    # so the signal carries no information and must not license a seat. Claiming seats
    # without evidence is exactly how v1-v3 scored 39.7%.
    gap = np.zeros(n, dtype=np.float32)
    dense = np.where(nn <= config.dense_max_nn)[0]
    if len(dense):
        for i in range(n):
            others = dense[dense != i]
            if len(others):
                gap[i] = float(dist[i, others].min()) / max(float(arr[i, 2]), 1.0)

    zones: list[str] = []
    for i in range(n):
        clear = gap[i] >= config.seat_min_gap
        if size_ratio[i] >= config.hand_size_ratio and ny[i] >= config.hand_min_ny:
            zones.append(Zone.MY_HAND.value)
        elif nx[i] <= config.left_max_nx and (nn[i] <= config.left_max_nn or clear):
            # Left seat's tiles cluster together — or, if there is only the one 定缺 tile,
            # sit well clear of the pile. Isolated *and* close to the pile is a stray
            # discard.
            zones.append(Zone.SEAT_LEFT.value)
        elif nx[i] >= config.right_min_nx:
            zones.append(Zone.SEAT_RIGHT.value)
        elif ny[i] <= config.across_max_ny and (nn[i] >= config.across_min_nn or clear):
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
                if float(dist[i, j]) < config.cluster_link * (widths[i] + widths[j]) / 2:
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

    # --- Melds. A run of identical tiles in a tidy row is a 碰 or 杠, which is always in
    # front of a player and never in the pool. Run last and as an override, not a vote:
    # every tile of a misplaced meld is wrong together, so voting among them only makes
    # the error unanimous. This is what moved seat_across 72.3% -> 92.3%, and it costs
    # the pool nothing.
    melds = 0
    if labels is not None and n >= 3 and config.meld_max_align > 0:
        by_class: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            if label:
                by_class.setdefault(label, []).append(i)
        for members in by_class.values():
            if len(members) < 3:
                continue
            parent = {i: i for i in members}

            def root(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for a_pos, i in enumerate(members):
                for j in members[a_pos + 1:]:
                    if float(dist[i, j]) < config.meld_link * (arr[i, 2] + arr[j, 2]) / 2:
                        ri, rj = root(i), root(j)
                        if ri != rj:
                            parent[ri] = rj
            runs: dict[int, list[int]] = {}
            for i in members:
                runs.setdefault(root(i), []).append(i)
            for run in runs.values():
                if len(run) < 3:
                    continue
                heights = arr[run, 3]
                align = float(np.std(cy[run]) / max(float(np.mean(heights)), 1e-6))
                if align > config.meld_max_align:
                    continue  # scattered — three of a kind that happen to land together
                if sum(zones[i] == Zone.MY_HAND.value for i in run) * 2 > len(run):
                    continue  # the player's own hand is already a tidy row of its own
                mean_ny = float(np.mean(cy[run])) / max(frame_h, 1)
                if mean_ny > config.meld_max_ny:
                    continue
                mean_nx = float(np.mean(cx[run])) / max(frame_w, 1)
                if mean_nx <= config.left_max_nx:
                    seat = Zone.SEAT_LEFT.value
                elif mean_nx >= config.right_min_nx:
                    seat = Zone.SEAT_RIGHT.value
                else:
                    seat = Zone.SEAT_ACROSS.value
                for i in run:
                    zones[i] = seat
                melds += 1

    debug = {"median_short": round(median, 1), "counts": {z: zones.count(z) for z in set(zones)}}
    if labels is not None:
        debug["melds"] = melds
    return zones, debug


def assign_zones(
    boxes: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
    config: ZoneConfig,
    labels: Sequence[str | None] | None = None,
) -> list[str]:
    """Thin wrapper kept for the pipeline's call site."""
    zones, _ = analyze_layout(boxes, frame_w, frame_h, config, labels)
    return zones
