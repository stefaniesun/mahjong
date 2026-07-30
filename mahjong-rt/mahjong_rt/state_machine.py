"""Tile lifecycle and event emission (Phase 4 task 5).

The state machine owns the class shown to the outside world. The voter proposes; the
state machine decides when that proposal becomes public. Nothing downstream — neither
the overlay nor the event stream — ever sees a raw per-frame classification, which is
what makes the "zero flicker" acceptance criterion reachable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .events import Event, FrameSummary, TileConfirmed, TileLost, TileState, TileUpdated, Zone
from .voter import Observation, TrackVoter


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
    zone_votes: dict[str, int] = field(default_factory=dict)
    voter: TrackVoter | None = None


class StateMachine:
    def __init__(
        self,
        *,
        voter_kwargs: dict[str, Any] | None = None,
        occluded_after: int = 2,
        lost_after: int = 30,
        emit_frame_summary: bool = True,
    ) -> None:
        self.voter_kwargs = voter_kwargs or {}
        self.occluded_after = occluded_after
        self.lost_after = lost_after
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
    ) -> list[Event]:
        """observations: (track_id, bbox_xywh, classification or None, zone)."""
        events: list[Event] = []
        seen: set[int] = set()

        for track_id, bbox, observation, zone in observations:
            seen.add(track_id)
            tile = self._ensure(track_id)
            tile.bbox = list(bbox)
            tile.frames_tracked += 1
            tile.frames_missing = 0
            if tile.state in (TileState.OCCLUDED,):
                tile.state = TileState.CONFIRMED if tile.label else TileState.TENTATIVE

            # Zone is decided by majority over the track's life, not per frame — a tile
            # near a boundary would otherwise flicker between zones every frame.
            tile.zone_votes[zone] = tile.zone_votes.get(zone, 0) + 1
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
            if tile.frames_missing >= self.lost_after:
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
