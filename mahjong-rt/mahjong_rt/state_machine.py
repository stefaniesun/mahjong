"""Tile lifecycle and event emission (Phase 4 task 5).

The state machine owns the class shown to the outside world. The voter proposes; the
state machine decides when that proposal becomes public. Nothing downstream — neither
the overlay nor the event stream — ever sees a raw per-frame classification, which is
what makes the "zero flicker" acceptance criterion reachable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .events import Event, FrameSummary, TileConfirmed, TileLost, TileState, TileUpdated, Zone
from .tracker import warp_boxes
from .voter import Observation, TrackVoter


def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two xywh boxes."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TileTrack:
    track_id: int
    state: TileState = TileState.TENTATIVE
    label: str | None = None
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    zone: str = Zone.UNKNOWN.value
    frames_tracked: int = 0
    frames_missing: int = 0
    zone_votes: dict[str, float] = field(default_factory=dict)
    voter: TrackVoter | None = None


class StateMachine:
    def __init__(
        self,
        *,
        voter_kwargs: dict[str, Any] | None = None,
        occluded_after: int = 2,
        lost_after: int = 30,
        emit_frame_summary: bool = True,
        zone_evidence_weight: float = 2.0,
        merge_dist: float = 0.6,
        merge_iou: float = 0.6,
        hand_lost_after: int | None = 40,
    ) -> None:
        self.voter_kwargs = voter_kwargs or {}
        self.occluded_after = occluded_after
        self.lost_after = lost_after
        self.zone_evidence_weight = zone_evidence_weight
        self.merge_dist = merge_dist
        self.merge_iou = merge_iou
        # "A placed tile does not move again" is what justifies holding tiles through a
        # long occlusion — but it is false for the player's own hand, which is picked up
        # and discarded from constantly. Held as long as the table, the hand accumulated
        # every tile it had ever contained. None disables the distinction.
        self.hand_lost_after = hand_lost_after
        self.emit_frame_summary = emit_frame_summary
        self.tiles: dict[int, TileTrack] = {}

    def _ensure(self, track_id: int) -> TileTrack:
        tile = self.tiles.get(track_id)
        if tile is None:
            tile = TileTrack(track_id=track_id, voter=TrackVoter(**self.voter_kwargs))
            self.tiles[track_id] = tile
        return tile

    def update(
        self,
        observations: Iterable[tuple[int, list[float], Observation | None, str]],
        *,
        frame_idx: int,
        ts: float,
        stats: dict[str, Any] | None = None,
        homography: Any = None,
    ) -> list[Event]:
        """observations: (track_id, bbox_xywh, classification or None, zone).

        `homography` maps the previous frame to this one, same convention the tracker
        uses. Without it an occluded tile's box stays where the camera last saw it while
        the camera moves on, which both misplaces it and stops it being recognised when
        its tile is picked up again under a new track id.
        """
        events: list[Event] = []
        seen: set[int] = set()
        self._carry_occluded(homography)

        for track_id, bbox, observation, zone in observations:
            seen.add(track_id)
            tile = self._ensure(track_id)
            tile.bbox = list(bbox)
            tile.frames_tracked += 1
            tile.frames_missing = 0
            if tile.state in (TileState.OCCLUDED,):
                tile.state = TileState.CONFIRMED if tile.label else TileState.TENTATIVE

            # Zone is decided by a weighted majority over the track's life, not per frame:
            # a tile near a boundary would otherwise flicker between zones every frame.
            #
            # The weighting is asymmetric because the evidence is. `river` is what the
            # zone rule returns when nothing else was proven — including the frames where
            # a meld went unrecognised because one of its three tiles was missed, or where
            # the gap to the pile could not be measured. A seat call, by contrast, needed
            # positive evidence to happen at all. Counting the two the same lets frames
            # that saw nothing outvote frames that saw something.
            #
            # Simulated over the labelled set (15 views per still, 15% of tiles missed,
            # boxes jittered, 2% of classes flipped): 96.7% per frame, 97.4% with a plain
            # majority, 98.4% at weight 2. The advantage inverts past ~45% missed tiles,
            # where seat calls become noise themselves — hence 2 rather than the 3 that
            # scores best at the current miss rate.
            weight = 1.0 if zone in (Zone.RIVER.value, Zone.UNKNOWN.value) else self.zone_evidence_weight
            tile.zone_votes[zone] = tile.zone_votes.get(zone, 0.0) + weight
            tile.zone = max(tile.zone_votes.items(), key=lambda kv: kv[1])[0]

            if observation is None:
                continue
            previous = tile.label
            label, changed = tile.voter.add(observation)
            tile.label = label
            tile.confidence = tile.voter.confidence
            if label is None or not changed:
                continue
            if previous is None:
                tile.state = TileState.CONFIRMED
                events.append(
                    TileConfirmed(
                        track_id=track_id,
                        label=label,
                        confidence=round(tile.confidence, 4),
                        bbox=[round(v, 1) for v in tile.bbox],
                        zone=tile.zone,
                        frame_idx=frame_idx,
                        ts=round(ts, 3),
                    )
                )
            else:
                events.append(
                    TileUpdated(
                        track_id=track_id,
                        label=label,
                        previous_label=previous,
                        confidence=round(tile.confidence, 4),
                        bbox=[round(v, 1) for v in tile.bbox],
                        zone=tile.zone,
                        frame_idx=frame_idx,
                        ts=round(ts, 3),
                    )
                )

        for track_id, tile in list(self.tiles.items()):
            if track_id in seen:
                continue
            tile.frames_missing += 1
            limit = self.lost_after
            if self.hand_lost_after is not None and tile.zone == Zone.MY_HAND.value:
                limit = min(limit, self.hand_lost_after)
            if tile.frames_missing >= limit:
                if tile.state == TileState.CONFIRMED:
                    events.append(
                        TileLost(
                            track_id=track_id,
                            label=tile.label or "",
                            last_bbox=[round(v, 1) for v in tile.bbox],
                            frames_tracked=tile.frames_tracked,
                            frame_idx=frame_idx,
                            ts=round(ts, 3),
                        )
                    )
                tile.state = TileState.LOST
                del self.tiles[track_id]
            elif tile.frames_missing >= self.occluded_after and tile.state != TileState.LOST:
                tile.state = TileState.OCCLUDED

        self._merge_reacquired(seen)

        if self.emit_frame_summary:
            confirmed = self.confirmed_tiles()
            by_zone: dict[str, int] = {}
            for tile in confirmed:
                by_zone[tile.zone] = by_zone.get(tile.zone, 0) + 1
            events.append(
                FrameSummary(
                    frame_idx=frame_idx,
                    ts=round(ts, 3),
                    tiles=[
                        {
                            "track_id": t.track_id,
                            "label": t.label,
                            "confidence": round(t.confidence, 4),
                            "bbox": [round(v, 1) for v in t.bbox],
                            "zone": t.zone,
                            "state": t.state.value,
                            # False means "still on the table, just not visible in this
                            # frame" — the consumer should keep it, not drop it.
                            "visible": t.state == TileState.CONFIRMED,
                        }
                        for t in confirmed
                    ],
                    counts_by_zone=by_zone,
                    stats=stats or {},
                )
            )
        return events

    def _carry_occluded(self, homography: Any) -> None:
        """Move tiles the camera cannot currently see along with the camera.

        Tiles are static in the world; the whole apparent motion is the head turning.
        A tile held through an occlusion must therefore be carried by the same global
        motion the tracker applies to its own predictions, or it drifts out of place and
        eventually into the wrong zone.
        """
        if homography is None:
            return
        stale = [t for t in self.tiles.values() if t.frames_missing > 0]
        if not stale:
            return
        boxes = np.array([[t.bbox[0], t.bbox[1], t.bbox[0] + t.bbox[2], t.bbox[1] + t.bbox[3]] for t in stale], np.float32)
        warped = warp_boxes(boxes, np.asarray(homography, np.float32))
        for tile, box in zip(stale, warped):
            tile.bbox = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]

    def _merge_reacquired(self, seen: set[int]) -> None:
        """Retire an occluded tile whose physical tile is back under a new track id.

        Fragmentation is the norm, not the exception: a hand passing over the table
        breaks tracks, and the tile returns as a new id. Keeping the old entry as well
        double-counts one physical tile, and it compounds — over 30 seconds a 14-tile
        hand was reporting 62 tiles, of which only the 14 visible ones were real.

        Two criteria, because two things fragment a track:

        * **Same label in the same place** is the same tile that came back. Distance is
          in units of the box's own width, so it holds at any depth.
        * **The same place at all**, whatever the label. Two tiles cannot occupy one
          slot. This catches the case the label test misses — a hand position whose tile
          has since been discarded and replaced, or simply re-read as something else.
          Without it the player's own hand kept every tile it had ever held: 31 entries
          for 14 real tiles, all the extras sitting exactly on top of live ones.

        Overlap alone would be too eager between *live* tiles — a side seat's boxes pile
        up heavily when seen edge-on — but this only ever retires a tile the camera
        cannot currently see in favour of one it can.
        """
        if self.merge_dist <= 0 and self.merge_iou <= 0:
            return
        live = [t for t in self.tiles.values() if t.track_id in seen and t.label]
        if not live:
            return
        for tile in list(self.tiles.values()):
            if tile.track_id in seen or not tile.label:
                continue
            cx, cy = tile.bbox[0] + tile.bbox[2] / 2, tile.bbox[1] + tile.bbox[3] / 2
            for other in live:
                ox, oy = other.bbox[0] + other.bbox[2] / 2, other.bbox[1] + other.bbox[3] / 2
                reach = self.merge_dist * max(other.bbox[2], tile.bbox[2], 1.0)
                same_tile = other.label == tile.label and self.merge_dist > 0 and (cx - ox) ** 2 + (cy - oy) ** 2 <= reach ** 2
                if same_tile or (self.merge_iou > 0 and _iou(tile.bbox, other.bbox) >= self.merge_iou):
                    # History carries over only when the labels agree, i.e. when this is
                    # the same tile coming back — then the survivor should not look newly
                    # seen. A slot that now reads as something else has been *replaced*,
                    # and inheriting the old tile's zone votes would drag the new one into
                    # the old one's zone.
                    if same_tile:
                        for zone_name, votes in tile.zone_votes.items():
                            other.zone_votes[zone_name] = other.zone_votes.get(zone_name, 0.0) + votes
                        other.zone = max(other.zone_votes.items(), key=lambda kv: kv[1])[0]
                        other.frames_tracked += tile.frames_tracked
                    del self.tiles[tile.track_id]
                    break

    def confirmed_tiles(self, include_occluded: bool = True) -> list[TileTrack]:
        """Tiles believed to be on the table right now.

        Occluded tiles are included by default. A mahjong tile that has been placed does
        not move again: a hand passing over it hides it for a moment, it does not remove
        it. Dropping occluded tiles from the world state made them vanish from the output
        after two missed frames — visible as tiles blinking out whenever someone reached
        across the table, even though nothing had actually left.
        """
        states = {TileState.CONFIRMED, TileState.OCCLUDED} if include_occluded else {TileState.CONFIRMED}
        return [t for t in self.tiles.values() if t.state in states and t.label]
