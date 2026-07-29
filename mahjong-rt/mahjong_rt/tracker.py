"""ByteTrack adapted for "tiles stay put, the camera keeps moving" (Phase 4 task 3).

Stock ByteTrack assumes a fixed camera and moving targets. First-person mahjong is the
exact inverse: the tiles are nailed to the table and the head turns constantly, so every
box translates at once. Without compensation a single head turn breaks every track.

The fix is global motion compensation: estimate the frame-to-frame homography from
sparse optical flow, warp each track's predicted box by it, and only then do IoU
association. Feature points are sampled *away from* detected tiles — tile corners move
with the tiles and would bias the estimate toward the wrong motion.

Kalman is tuned for static targets: low process noise, so observations dominate and the
filter does not invent velocity for a tile that never moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes. Shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows of `a` against rows of `b`, mapped to [0,1]. Empty inputs give an empty grid."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return np.clip(an @ bn.T, 0.0, 1.0).astype(np.float32)


def greedy_match(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy highest-IoU-first matching.

    Hungarian would be optimal, but with 40 densely packed near-identical boxes the
    optimal assignment is rarely different and greedy keeps the per-frame budget honest.
    """
    matches: list[tuple[int, int]] = []
    if cost.size:
        pairs = np.dstack(np.unravel_index(np.argsort(-cost, axis=None), cost.shape))[0]
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for row, col in pairs:
            if cost[row, col] < threshold:
                break
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(int(row))
            used_cols.add(int(col))
            matches.append((int(row), int(col)))
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return matches, [r for r in range(cost.shape[0]) if r not in matched_rows], [c for c in range(cost.shape[1]) if c not in matched_cols]


class GlobalMotionEstimator:
    """Frame-to-frame homography from sparse optical flow, ignoring tile regions."""

    def __init__(self, max_corners: int = 300, quality: float = 0.01, min_distance: int = 8, min_points: int = 12) -> None:
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.min_points = min_points
        self._prev_gray: np.ndarray | None = None
        self.last_ok = False

    def estimate(self, frame_gray: np.ndarray, exclude_boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """Return a 3x3 homography mapping previous-frame points to this frame."""
        import cv2

        identity = np.eye(3, dtype=np.float32)
        prev = self._prev_gray
        self._prev_gray = frame_gray
        if prev is None:
            self.last_ok = False
            return identity

        # Mask out tiles: their corners move with the table content we are tracking,
        # so including them pulls the "global" estimate toward the tiles themselves.
        mask = np.full(prev.shape, 255, dtype=np.uint8)
        for box in exclude_boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 0

        corners = cv2.goodFeaturesToTrack(prev, self.max_corners, self.quality, self.min_distance, mask=mask)
        if corners is None or len(corners) < self.min_points:
            self.last_ok = False
            return identity

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, frame_gray, corners, None)
        if nxt is None or status is None:
            self.last_ok = False
            return identity
        good_prev = corners[status.ravel() == 1].reshape(-1, 2)
        good_next = nxt[status.ravel() == 1].reshape(-1, 2)
        if len(good_prev) < self.min_points:
            self.last_ok = False
            return identity

        matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if matrix is None:
            self.last_ok = False
            return identity
        self.last_ok = True
        homography = np.eye(3, dtype=np.float32)
        homography[:2] = matrix.astype(np.float32)
        return homography


def warp_boxes(boxes: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply a homography to xyxy boxes by transforming their corners."""
    if len(boxes) == 0:
        return boxes
    corners = np.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        axis=1,
    )
    ones = np.ones((*corners.shape[:2], 1), dtype=np.float32)
    homo = np.concatenate([corners.astype(np.float32), ones], axis=2)
    warped = homo @ homography.T
    warped = warped[..., :2] / np.maximum(warped[..., 2:3], 1e-9)
    x1 = warped[..., 0].min(axis=1)
    y1 = warped[..., 1].min(axis=1)
    x2 = warped[..., 0].max(axis=1)
    y2 = warped[..., 1].max(axis=1)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # xyxy
    det_conf: float
    age: int = 0  # frames since created
    hits: int = 1  # frames successfully matched
    time_since_update: int = 0
    # Which detection this track matched this frame, or None if it is coasting. Replay
    # uses it to look up the recorded classification without re-running the model.
    det_index: int | None = None
    descriptor: np.ndarray | None = None  # appearance, EMA over matched detections
    history: list[np.ndarray] = field(default_factory=list)

    def predict(self, homography: np.ndarray) -> np.ndarray:
        """Tiles do not move; the whole prediction is the camera's motion."""
        return warp_boxes(self.bbox[None, :], homography)[0]

    def update(self, bbox: np.ndarray, det_conf: float, alpha: float) -> None:
        # Observation-dominated smoothing: a static tile's box should follow the
        # detector, with just enough smoothing to take the jitter off.
        self.bbox = alpha * bbox + (1.0 - alpha) * self.bbox
        self.det_conf = det_conf
        self.hits += 1
        self.time_since_update = 0


class ByteTrackGMC:
    """Two-stage ByteTrack association with global motion compensation."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.7,
        match_thresh_low: float = 0.4,
        track_buffer: int = 30,
        min_hits: int = 2,
        smooth_alpha: float = 0.8,
        gmc_enabled: bool = True,
        fallback_match_thresh: float = 0.3,
        appearance_weight: float = 0.0,
        appearance_momentum: float = 0.7,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.match_thresh_low = match_thresh_low
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.smooth_alpha = smooth_alpha
        self.gmc_enabled = gmc_enabled
        self.fallback_match_thresh = fallback_match_thresh
        # Densely packed tiles look alike to an IoU-only association: neighbouring boxes
        # overlap heavily, so the gate cannot tell which is which and ids swap. Blending
        # in appearance similarity supplies the missing information. Off by default so
        # existing behaviour is unchanged until a sweep shows it earns its place.
        self.appearance_weight = appearance_weight
        self.appearance_momentum = appearance_momentum
        self.tracks: list[Track] = []
        self._next_id = 1
        self._gmc = GlobalMotionEstimator()
        self.last_homography = np.eye(3, dtype=np.float32)
        self.stats: dict[str, Any] = {"gmc_ok": 0, "gmc_fail": 0, "new_tracks": 0, "removed": 0}

    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        frame_gray: np.ndarray | None = None,
        homography: np.ndarray | None = None,
        descriptors: np.ndarray | None = None,
    ) -> list[Track]:
        """detections: (N,4) xyxy. scores: (N,). Returns tracks visible this frame.

        `homography` lets a caller supply a motion estimate instead of computing one.
        Replay has no pixels to run optical flow on, so it feeds back the matrix recorded
        during the original pass — tracking parameters can then be swept without the
        models or the video.
        """
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if homography is not None:
            homography = np.asarray(homography, dtype=np.float32).reshape(3, 3)
            degraded = False
        elif self.gmc_enabled and frame_gray is not None:
            homography = self._gmc.estimate(frame_gray, detections)
            self.stats["gmc_ok" if self._gmc.last_ok else "gmc_fail"] += 1
            # When the motion estimate fails, association gets a looser gate rather than
            # a wrong warp — better a sloppy match than hard-breaking every track.
            degraded = not self._gmc.last_ok
        else:
            homography = np.eye(3, dtype=np.float32)
            degraded = False
        self.last_homography = homography
        high_gate = self.fallback_match_thresh if degraded else self.match_thresh
        low_gate = self.fallback_match_thresh if degraded else self.match_thresh_low

        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            track.det_index = None
            track.bbox = track.predict(homography)

        high_idx = np.where(scores >= self.high_thresh)[0]
        low_idx = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]

        # Stage 1: confident detections against every track.
        def blend(track_idx: list[int], det_idx: np.ndarray) -> np.ndarray:
            cost = iou_matrix(np.stack([self.tracks[i].bbox for i in track_idx]), detections[det_idx])
            if self.appearance_weight <= 0 or descriptors is None or not len(det_idx):
                return cost
            have = [i for i in track_idx if self.tracks[i].descriptor is not None]
            if len(have) != len(track_idx):
                return cost
            sim = cosine_similarity(np.stack([self.tracks[i].descriptor for i in track_idx]), descriptors[det_idx])
            w = self.appearance_weight
            return ((1.0 - w) * cost + w * cost * sim).astype(np.float32)

        pool = list(range(len(self.tracks)))
        matched_pairs: list[tuple[int, int]] = []
        if pool and len(high_idx):
            cost = blend(pool, high_idx)
            matches, un_tracks, un_dets = greedy_match(cost, high_gate)
            for ti, di in matches:
                matched_pairs.append((pool[ti], int(high_idx[di])))
            remaining_tracks = [pool[i] for i in un_tracks]
            remaining_high = [int(high_idx[i]) for i in un_dets]
        else:
            remaining_tracks = pool
            remaining_high = [int(i) for i in high_idx]

        # Stage 2: leftovers get a shot at the low-confidence detections. This is what
        # keeps a briefly blurred tile alive instead of spawning a fresh id later.
        if remaining_tracks and len(low_idx):
            cost = blend(remaining_tracks, low_idx)
            matches, un_tracks, _ = greedy_match(cost, low_gate)
            for ti, di in matches:
                matched_pairs.append((remaining_tracks[ti], int(low_idx[di])))
            remaining_tracks = [remaining_tracks[i] for i in un_tracks]

        for track_i, det_i in matched_pairs:
            track = self.tracks[track_i]
            track.update(detections[det_i], float(scores[det_i]), self.smooth_alpha)
            track.det_index = det_i
            if descriptors is not None and det_i < len(descriptors):
                d = descriptors[det_i].astype(np.float32)
                m = self.appearance_momentum
                track.descriptor = d if track.descriptor is None else (m * track.descriptor + (1.0 - m) * d)

        for det_i in remaining_high:
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox=detections[det_i].copy(),
                    det_conf=float(scores[det_i]),
                    det_index=det_i,
                    descriptor=None if descriptors is None or det_i >= len(descriptors) else descriptors[det_i].astype(np.float32).copy(),
                )
            )
            self._next_id += 1
            self.stats["new_tracks"] += 1

        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
        self.stats["removed"] += before - len(self.tracks)

        return [t for t in self.tracks if t.time_since_update == 0]

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.time_since_update == 0 and t.hits >= self.min_hits]
