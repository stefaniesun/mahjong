"""Re-run the post-model stages from a recording (Phase 4 task 9).

Everything downstream of detection and classification — association, voting, lifecycle,
zones — is replayed here from stored model output. Same code paths as the live pipeline,
so a configuration that scores well in a sweep behaves the same way on video.

The recording holds classifications per *detection*; a track picks up whichever
detection it matched this frame. That indirection is what lets tracker parameters change
during a sweep without invalidating the stored classifications.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .events import Event
from .recording import Recording
from .state_machine import StateMachine
from .tracker import ByteTrackGMC
from .voter import Observation
from .zones import ZoneConfig, assign_zones


def replay(
    recording: Recording,
    *,
    tracker_cfg: dict[str, Any] | None = None,
    voter_cfg: dict[str, Any] | None = None,
    state_cfg: dict[str, Any] | None = None,
    zones_cfg: dict[str, Any] | None = None,
    checkpoints: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Returns events plus a snapshot of confirmed tiles at each requested frame."""
    tracker = ByteTrackGMC(**(tracker_cfg or {}))
    machine = StateMachine(voter_kwargs=voter_cfg or {}, **(state_cfg or {}))
    zone_config = ZoneConfig(**(zones_cfg or {}))

    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    wanted = checkpoints or {}

    for record in recording.frames:
        tracks = tracker.update(record.boxes, record.scores, None, homography=record.homography, descriptors=record.probs)

        xywh: list[list[float]] = []
        observations: list[Observation | None] = []
        for track in tracks:
            box = track.bbox
            xywh.append([float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])])
            index = track.det_index
            if index is None or index >= len(record.labels):
                # Coasting track: no detection this frame, so no new observation. The
                # voter simply does not get a vote, which is the correct behaviour.
                observations.append(None)
                continue
            width = float(record.boxes[index][2] - record.boxes[index][0])
            height = float(record.boxes[index][3] - record.boxes[index][1])
            observations.append(
                Observation(
                    label=recording.classes[int(record.labels[index])],
                    confidence=float(record.confidences[index]),
                    short_side=min(width, height),
                )
            )

        zones = assign_zones(xywh, recording.frame_width, recording.frame_height, zone_config)
        frame_events = machine.update(
            [(t.track_id, xywh[i], observations[i], zones[i]) for i, t in enumerate(tracks)],
            frame_idx=record.frame_index,
            ts=record.timestamp,
            stats={"detections": int(len(record.boxes)), "tracks": len(tracks)},
        )
        events.extend(asdict(e) for e in frame_events)

        if record.frame_index in wanted:
            snapshots.append(
                {
                    "clip_frame": record.frame_index,
                    "file": wanted[record.frame_index],
                    "confirmed": [
                        {"track_id": t.track_id, "label": t.label, "bbox": list(t.bbox)}
                        for t in machine.confirmed_tiles()
                    ],
                }
            )

    return {"clip": recording.clip, "events": events, "snapshots": snapshots, "tracker_stats": dict(tracker.stats)}
