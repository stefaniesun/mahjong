"""Game events from the tile stream: who discarded what, who claimed it.

The tile stream says what is on the table; this says what just happened. The engine
consumes the latter — `VisionEvent(event_type, player, tile)` with types discard, pong,
kong, hu — so this is the last stage before the vision side hands over.

Two ideas carry most of the work.

**A discard is a tile appearing where the table was already visible and empty.** The
naive reading — "the pool gained a tile" — does not survive contact with video: over
clip01 it fires 122 times against 14 real discards, because the pool is *discovered*
progressively as tracks confirm and the camera moves, not created. Anchoring occupancy
to world coordinates rather than to track ids removes most of that (122 -> 32), and
skipping the warm-up, where the pool that was already on the table is being confirmed
for the first time, removes the rest.

**Attribution is turn order, not vision.** Which player threw a tile into a shared pool
is not visible at all. Mahjong turns run counter-clockwise, so a pointer advanced on
each discard answers it — and a pong, the one action whose owner *is* directly visible
and the one action that breaks the order, re-anchors the pointer whenever it drifts.

World coordinates come from chaining the per-frame homography the tracker already
estimates. Over 30 seconds of held-camera footage the accumulated drift measured about
20 px, well inside a tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

SEATS = ["me", "right", "across", "left"]   # counter-clockwise


@dataclass
class GameEvent:
    """One thing that happened. Mirrors the engine's VisionEvent."""

    seq: int
    event_type: str          # discard / pong / kong
    player: str              # one of SEATS
    tile: str | None
    ts: float
    frame_idx: int
    confidence: float = 1.0
    from_player: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_type": self.event_type,
            "player": self.player,
            "tile": self.tile,
            "ts": round(self.ts, 3),
            "frame_idx": self.frame_idx,
            "confidence": round(self.confidence, 4),
            "from": self.from_player,
        }


@dataclass
class GameEventConfig:
    cell_px: float = 40.0
    # Frames to ignore at the start. The pool already on the table has to be confirmed
    # before anything can be called new, and the voter needs several observations per
    # tile — so every pre-existing tile would otherwise read as a fresh discard.
    warmup_frames: int = 90
    # A cell must hold a tile for this many frames before it counts. One frame of a
    # hand passing through, or one flickering detection, is not a discard.
    settle_frames: int = 3
    # Two discards cannot land in the same instant; anything faster is one event seen
    # twice, usually a tile straddling a cell boundary.
    min_gap_s: float = 0.6
    # Melds are read straight from the zone stage, which already groups them.
    # Off by default. On clip01 — which contains no pong at all — it produced six, every
    # one of them a single-frame zone error promoted to an event. It needs cross-frame
    # confirmation before it can be trusted; until then it is a pure false-positive source.
    detect_melds: bool = False
    # Occupancy from raw detections. Correct in principle — a tile the detector saw all
    # along is not new — but a discard lands *on* the pile, so its cell has usually been
    # seen already. Measured on clip01 this removes every discard, true ones included.
    use_detection_occupancy: bool = False
    # Frames to keep watching a cell before naming what landed in it. The moment a cell
    # fills is the moment the tile arrives, but the voter has not settled on what it is
    # yet — read then, the label is whatever the first noisy frames said, or a
    # neighbour's. The event keeps the arrival time and takes the majority label from
    # this window.
    label_window: int = 30
    start_player: str | None = None   # None = unknown until a meld anchors it


class GameEventExtractor:
    """Feed it frame summaries in order; read `events` when done."""

    def __init__(self, config: GameEventConfig | None = None) -> None:
        self.config = config or GameEventConfig()
        self.events: list[GameEvent] = []
        self._world = np.eye(3, dtype=np.float64)
        self._occupied: dict[tuple[int, int], dict[str, Any]] = {}
        self._melds: set[tuple[str, str]] = set()
        self._frames = 0
        self._turn: str | None = self.config.start_player
        self._last_ts = -99.0
        self._pending: dict[tuple[int, int], dict[str, Any]] = {}
        self._ever_detected: set[tuple[int, int]] = set()
        self._detected_now: set[tuple[int, int]] = set()
        self._naming: list[dict[str, Any]] = []

    # --- geometry -------------------------------------------------------------

    def _advance(self, homography: Any) -> None:
        """Chain this frame onto the world frame. Homography maps previous -> current."""
        if homography is None:
            return
        matrix = np.asarray(homography, dtype=np.float64)
        if matrix.shape != (3, 3):
            return
        try:
            self._world = self._world @ np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            pass   # a degenerate estimate: keep the previous frame's mapping

    def _cell(self, bbox: Sequence[float]) -> tuple[int, int]:
        x, y, w, h = bbox
        point = self._world @ np.array([x + w / 2.0, y + h / 2.0, 1.0])
        point = point[:2] / point[2]
        size = max(self.config.cell_px, 1.0)
        return int(np.floor(point[0] / size)), int(np.floor(point[1] / size))

    # --- turn order -----------------------------------------------------------

    def _next_turn(self) -> str | None:
        if self._turn is None:
            return None
        return SEATS[(SEATS.index(self._turn) + 1) % 4]

    # --- ingestion ------------------------------------------------------------

    def add_frame(self, summary: dict[str, Any], homography: Any = None,
                  detections: Any = None) -> list[GameEvent]:
        """summary: one `frame_summary` event. Returns the events this frame produced.

        `detections` is the raw detector output for the frame as xyxy boxes. Occupancy is
        keyed off it rather than off confirmed tiles, and that distinction is the whole
        ball game: a tile sitting in the pool since before the clip is *detected* from
        the first frame but only *confirmed* once the voter has seen it enough times.
        Keyed off confirmations, every late-confirming tile reads as a fresh discard —
        which is most of them, spread over the whole clip rather than just the start.
        """
        self._advance(homography)
        self._frames += 1
        produced: list[GameEvent] = []
        ts = float(summary.get("ts", 0.0))
        frame_idx = int(summary.get("frame_idx", self._frames))
        tiles = summary.get("tiles", [])

        if self.config.detect_melds:
            produced += self._scan_melds(tiles, ts, frame_idx)

        warm = self._frames <= self.config.warmup_frames

        # Anything the detector sees occupies its cell, named or not.
        if detections is not None and len(detections):
            for box in np.asarray(detections, dtype=np.float64).reshape(-1, 4):
                cell = self._cell([box[0], box[1], box[2] - box[0], box[3] - box[1]])
                self._detected_now.add(cell)

        live = {}
        for tile in tiles:
            if tile.get("zone") != "river" or not tile.get("visible", True) or not tile.get("label"):
                continue
            live[self._cell(tile["bbox"])] = tile

        for cell, tile in live.items():
            if cell in self._occupied:
                continue
            if self.config.use_detection_occupancy and cell in self._ever_detected:
                continue
            entry = self._pending.setdefault(cell, {"label": tile["label"], "count": 0, "ts": ts, "frame": frame_idx})
            entry["count"] += 1
            entry["label"] = tile["label"]          # latest reading wins; the voter settles it
            if entry["count"] < self.config.settle_frames:
                continue
            self._occupied[cell] = entry
            del self._pending[cell]
            if warm:
                continue                            # the pool that was already there
            if entry["ts"] - self._last_ts < self.config.min_gap_s:
                continue                            # one landing seen twice
            self._last_ts = entry["ts"]
            event = GameEvent(seq=0, event_type="discard", player=self._turn or "unknown",
                              tile=entry["label"], ts=entry["ts"], frame_idx=entry["frame"])
            # The turn advances now — attribution depends on the order things landed in,
            # not on how long it takes to read them.
            self._turn = self._next_turn()
            self._naming.append({"event": event, "cell": cell, "votes": {}, "left": self.config.label_window})

        for cell in list(self._pending):
            if cell not in live:
                del self._pending[cell]             # never settled: noise

        # Recorded after the comparison, so a cell the detector first sees *this* frame
        # can still be a discard — the tile has to arrive somewhere.
        self._ever_detected |= self._detected_now
        self._detected_now = set()

        produced += self._name_pending(live)

        if warm:
            # Warm-up still establishes occupancy, so the pool present at the start is
            # never mistaken for something thrown during the clip.
            for cell, tile in live.items():
                self._occupied.setdefault(cell, {"label": tile["label"], "count": 99, "ts": ts, "frame": frame_idx})

        self.events += produced
        return produced

    def flush(self) -> list[GameEvent]:
        """Close every open naming window. Call once the stream ends.

        An arrival still inside its window has been detected but not yet named; without
        this the last second or two of a clip is silently dropped.
        """
        done = self._name_pending({}, force=True)
        self.events += done
        return done

    def _name_pending(self, live: dict[tuple[int, int], dict[str, Any]], force: bool = False) -> list[GameEvent]:
        """Hold each detected arrival open for a while, then name it by majority.

        Reading the label at the instant a cell fills gets it wrong most of the time —
        2 of 13 on clip01. The tile has only just landed, and the voter needs several
        clean views before it settles. Waiting costs nothing that matters: the event
        keeps the timestamp of the arrival, and downstream the engine cares about the
        order events happened in, not the order they were reported.
        """
        done: list[GameEvent] = []
        for slot in list(self._naming):
            tile = live.get(slot["cell"])
            if tile and tile.get("label"):
                slot["votes"][tile["label"]] = slot["votes"].get(tile["label"], 0) + 1
            slot["left"] -= 1
            if slot["left"] > 0 and not force:
                continue
            event = slot["event"]
            if slot["votes"]:
                best = max(slot["votes"].items(), key=lambda kv: kv[1])
                event.tile = best[0]
                event.confidence = best[1] / sum(slot["votes"].values())
            event.seq = len(self.events) + len(done)
            done.append(event)
            self._naming.remove(slot)
        return done

    def _scan_melds(self, tiles: Iterable[dict[str, Any]], ts: float, frame_idx: int) -> list[GameEvent]:
        """A seat's zone gaining a group of three or four identical tiles is a claim.

        The zone stage already identifies melds — it has to, to place them — so this
        only has to notice a group that was not there before. A pong also re-anchors the
        turn: the claimer discards next, and unlike a discard their identity is visible.
        """
        produced: list[GameEvent] = []
        groups: dict[tuple[str, str], int] = {}
        for tile in tiles:
            zone, label = tile.get("zone"), tile.get("label")
            if zone in {"seat_left", "seat_across", "seat_right", "my_hand"} and label:
                groups[(zone, label)] = groups.get((zone, label), 0) + 1
        for (zone, label), count in groups.items():
            if count < 3 or zone == "my_hand" or (zone, label) in self._melds:
                continue
            self._melds.add((zone, label))
            if self._frames <= self.config.warmup_frames:
                continue
            seat = {"seat_right": "right", "seat_across": "across", "seat_left": "left"}[zone]
            produced.append(
                GameEvent(seq=len(self.events) + len(produced),
                          event_type="kong" if count >= 4 else "pong",
                          player=seat, tile=label, ts=ts, frame_idx=frame_idx,
                          from_player=self._turn_owner_of_last_discard())
            )
            self._turn = seat            # the claimer discards next — re-anchors the pointer
        return produced

    def _turn_owner_of_last_discard(self) -> str | None:
        for event in reversed(self.events):
            if event.event_type == "discard":
                return event.player
        return None


def extract(frames: Iterable[tuple[dict[str, Any], Any]], config: GameEventConfig | None = None) -> list[GameEvent]:
    """Convenience: run the extractor over (frame_summary, homography) pairs."""
    extractor = GameEventExtractor(config)
    for summary, homography in frames:
        extractor.add_frame(summary, homography)
    extractor.flush()
    return extractor.events
