"""Per-frame model output recording, so parameter sweeps skip inference (Phase 4 task 9).

A single evaluation pass over the pilot clips costs about an hour on CPU, almost all of
it in the detector and classifier. But the parameters worth tuning — voting window,
hysteresis, track buffer, association gates — live entirely *downstream* of the models.
Recording what the models produced turns each additional configuration from an hour into
a couple of seconds.

Two things make that possible:

* **Classification is recorded per detection, not per track.** Track boxes depend on
  tracker parameters, so a classification tied to a track could not be reused once those
  parameters change. Detections do not move.
* **The motion estimate is recorded too.** Global motion compensation needs optical flow
  over real pixels, which replay does not have; feeding back the matrix from the
  recording pass keeps tracking behaviour identical.

Stored as one .npz per clip: flat arrays plus an index, which keeps a few hundred frames
of detections in a couple of megabytes and loads instantly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameRecord:
    frame_index: int  # index in the source clip, before any stride
    timestamp: float
    boxes: np.ndarray  # (N,4) xyxy in clip pixels
    scores: np.ndarray  # (N,) detector confidence
    labels: np.ndarray  # (N,) class index into `classes`
    confidences: np.ndarray  # (N,) classifier confidence
    # (N,C) full class distribution, used as an appearance descriptor for association.
    # The spec suggests penultimate-layer embeddings; the class posterior is 1280 -> 27
    # dimensions for the same job and targets exactly the switches that hurt. Two
    # adjacent tiles of the same kind have near-identical posteriors, but swapping those
    # costs nothing — both read the same. Tiles of different kinds are far apart, and
    # those are the swaps that corrupt the output.
    probs: np.ndarray
    homography: np.ndarray  # (3,3) motion from the previous recorded frame


@dataclass
class Recording:
    clip: str
    classes: list[str]
    frame_width: int
    frame_height: int
    fps: float
    stride: int = 1
    frames: list[FrameRecord] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.array([len(f.boxes) for f in self.frames], dtype=np.int32)
        payload: dict[str, Any] = {
            "counts": counts,
            "frame_index": np.array([f.frame_index for f in self.frames], dtype=np.int32),
            "timestamp": np.array([f.timestamp for f in self.frames], dtype=np.float32),
            "homography": np.stack([f.homography for f in self.frames]).astype(np.float32) if self.frames else np.zeros((0, 3, 3), np.float32),
            "boxes": np.concatenate([f.boxes for f in self.frames]).astype(np.float32) if any(counts) else np.zeros((0, 4), np.float32),
            "scores": np.concatenate([f.scores for f in self.frames]).astype(np.float32) if any(counts) else np.zeros((0,), np.float32),
            "labels": np.concatenate([f.labels for f in self.frames]).astype(np.int16) if any(counts) else np.zeros((0,), np.int16),
            "confidences": np.concatenate([f.confidences for f in self.frames]).astype(np.float32) if any(counts) else np.zeros((0,), np.float32),
            "probs": np.concatenate([f.probs for f in self.frames]).astype(np.float16) if any(counts) else np.zeros((0, len(self.classes)), np.float16),
            "meta": np.array(
                [self.clip, "|".join(self.classes), str(self.frame_width), str(self.frame_height), str(self.fps), str(self.stride)],
                dtype=object,
            ),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "Recording":
        data = np.load(path, allow_pickle=True)
        meta = data["meta"]
        recording = cls(
            clip=str(meta[0]),
            classes=str(meta[1]).split("|"),
            frame_width=int(meta[2]),
            frame_height=int(meta[3]),
            fps=float(meta[4]),
            stride=int(meta[5]),
        )
        counts = data["counts"]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        for i, count in enumerate(counts):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            recording.frames.append(
                FrameRecord(
                    frame_index=int(data["frame_index"][i]),
                    timestamp=float(data["timestamp"][i]),
                    boxes=data["boxes"][lo:hi],
                    scores=data["scores"][lo:hi],
                    labels=data["labels"][lo:hi],
                    confidences=data["confidences"][lo:hi],
                    probs=data["probs"][lo:hi].astype(np.float32) if "probs" in data else np.zeros((hi - lo, len(recording.classes)), np.float32),
                    homography=data["homography"][i],
                )
            )
        return recording
