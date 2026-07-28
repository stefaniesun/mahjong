"""Output event protocol (Phase 4 task 5).

This is the seam between the recognition pipeline and everything downstream — the
Phase 5 edge app, and any hand-advice logic built on top. Field names and semantics
are a published contract: add fields, never repurpose or drop them.

All events are JSON-serialisable via `asdict`. Coordinates are pixels in the source
frame; `ts` is seconds since pipeline start; `frame_idx` is the 0-based frame number
so an event can always be tied back to a recording.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

PROTOCOL_VERSION = "1.0"


class TileState(str, Enum):
    """Lifecycle of one tracked tile."""

    TENTATIVE = "TENTATIVE"  # seen, not yet voted into a stable class
    CONFIRMED = "CONFIRMED"  # voter reached a decision; safe to show and act on
    OCCLUDED = "OCCLUDED"  # track temporarily lost, still held in case it returns
    LOST = "LOST"  # gone past the buffer, removed


class Zone(str, Enum):
    """Table regions.

    Seats follow mahjong turn order as seen from the player: 下家 sits to the right,
    对家 across, 上家 to the left. `river` is the shared scatter in the middle — it is
    told apart from a seat's tiles by being irregular rather than by where it sits.
    """

    MY_HAND = "my_hand"
    RIVER = "river"  # 牌池：中间散乱的一堆
    SEAT_RIGHT = "seat_right"  # 下家
    SEAT_ACROSS = "seat_across"  # 对家
    SEAT_LEFT = "seat_left"  # 上家
    MELD_AREA = "meld_area"
    OPPONENT_WALL = "opponent_wall"
    UNKNOWN = "unknown_zone"


@dataclass
class TileConfirmed:
    """A track has settled on a class for the first time."""

    track_id: int
    label: str
    confidence: float
    bbox: list[float]  # [x, y, w, h]
    zone: str
    frame_idx: int
    ts: float
    type: str = "tile_confirmed"


@dataclass
class TileUpdated:
    """A confirmed track changed its class — only ever emitted after hysteresis."""

    track_id: int
    label: str
    previous_label: str
    confidence: float
    bbox: list[float]
    zone: str
    frame_idx: int
    ts: float
    type: str = "tile_updated"


@dataclass
class TileLost:
    track_id: int
    label: str
    last_bbox: list[float]
    frames_tracked: int
    frame_idx: int
    ts: float
    type: str = "tile_lost"


@dataclass
class FrameSummary:
    """Full snapshot of the confirmed world state for one frame. Can be disabled."""

    frame_idx: int
    ts: float
    tiles: list[dict[str, Any]] = field(default_factory=list)
    counts_by_zone: dict[str, int] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    type: str = "frame_summary"


Event = TileConfirmed | TileUpdated | TileLost | FrameSummary


def to_json(event: Event) -> str:
    """One event, one line — the on-disk jsonl format and the stdout format."""
    return json.dumps(asdict(event), ensure_ascii=False, sort_keys=False)


def dump_jsonl(events: Sequence[Event], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(to_json(event) + "\n")
