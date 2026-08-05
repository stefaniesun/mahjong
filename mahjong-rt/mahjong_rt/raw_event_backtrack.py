"""Link stable landing events backwards through raw detector observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .game_events import GameEvent
from .recording import Recording


@dataclass(frozen=True)
class LandingPoint:
    x: float
    y: float
    size: float


@dataclass
class BacktrackConfig:
    lookback_s: float = 1.2
    min_class_prob: float = 0.2
    min_detector_score: float = 0.1
    max_gap_s: float = 0.35
    max_speed_sizes_per_s: float = 12.0
    landing_distance_ratio: float = 1.0
    min_path_nodes: int = 3
    min_displacement_ratio: float = 1.5
    require_motion: bool = False


@dataclass(frozen=True)
class BacktrackEvidence:
    has_motion: bool
    path_nodes: int
    displacement_ratio: float
    arrival_ts: float
    score: float


@dataclass(frozen=True)
class _Node:
    frame_index: int
    ts: float
    x: float
    y: float
    size: float
    probability: float
    detector_score: float


def _world_matrices(recording: Recording) -> list[np.ndarray]:
    world = np.eye(3, dtype=np.float64)
    matrices: list[np.ndarray] = []
    for frame in recording.frames:
        matrix = np.asarray(frame.homography, dtype=np.float64)
        if matrix.shape == (3, 3):
            try:
                world = world @ np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                pass
        matrices.append(world.copy())
    return matrices


def _project(matrix: np.ndarray, x: float, y: float) -> np.ndarray:
    point = matrix @ np.array([x, y, 1.0])
    if abs(point[2]) > 1e-9:
        point /= point[2]
    return point[:2]


def _nodes(recording: Recording, event: GameEvent, config: BacktrackConfig) -> list[_Node]:
    if event.tile not in recording.classes:
        return []
    class_index = recording.classes.index(event.tile)
    worlds = _world_matrices(recording)
    start = event.ts - config.lookback_s
    result: list[_Node] = []
    for frame_index, (frame, world) in enumerate(zip(recording.frames, worlds)):
        if frame.timestamp < start or frame.timestamp > event.ts:
            continue
        for detection_index, box in enumerate(frame.boxes):
            if detection_index >= len(frame.probs) or class_index >= frame.probs.shape[1]:
                continue
            probability = float(frame.probs[detection_index, class_index])
            detector_score = float(frame.scores[detection_index])
            if probability < config.min_class_prob or detector_score < config.min_detector_score:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            centre = _project(world, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
            horizontal = _project(world, x2, (y1 + y2) / 2.0)
            vertical = _project(world, (x1 + x2) / 2.0, y2)
            size = max(min(np.linalg.norm(horizontal - centre), np.linalg.norm(vertical - centre)) * 2.0, 1.0)
            result.append(
                _Node(frame_index, float(frame.timestamp), float(centre[0]), float(centre[1]), float(size), probability, detector_score)
            )
    return sorted(result, key=lambda node: (node.ts, node.frame_index))


def _distance(left: _Node, right: _Node | LandingPoint) -> float:
    return float(np.hypot(left.x - right.x, left.y - right.y))


def trace_landing_backwards(
    recording: Recording,
    event: GameEvent,
    landing: LandingPoint,
    config: BacktrackConfig | None = None,
) -> BacktrackEvidence:
    config = config or BacktrackConfig()
    nodes = _nodes(recording, event, config)
    if not nodes:
        return BacktrackEvidence(False, 0, 0.0, event.ts, 0.0)

    scores = np.array([node.probability * node.detector_score for node in nodes], dtype=np.float64)
    previous = np.full(len(nodes), -1, dtype=np.int32)
    for current_index, current in enumerate(nodes):
        for prior_index in range(current_index):
            prior = nodes[prior_index]
            gap = current.ts - prior.ts
            if gap <= 0 or gap > config.max_gap_s:
                continue
            scale = max(current.size, prior.size, 1.0)
            speed = _distance(prior, current) / scale / gap
            if speed > config.max_speed_sizes_per_s:
                continue
            candidate = scores[prior_index] + current.probability * current.detector_score
            if candidate > scores[current_index]:
                scores[current_index] = candidate
                previous[current_index] = prior_index

    endpoints = [
        index
        for index, node in enumerate(nodes)
        if _distance(node, landing) <= max(node.size, landing.size) * config.landing_distance_ratio
    ]
    if not endpoints:
        return BacktrackEvidence(False, 0, 0.0, event.ts, 0.0)
    end_index = max(endpoints, key=lambda index: (scores[index], nodes[index].ts))
    path_indices: list[int] = []
    cursor = end_index
    while cursor >= 0:
        path_indices.append(cursor)
        cursor = int(previous[cursor])
    path = [nodes[index] for index in reversed(path_indices)]

    displacement = _distance(path[0], path[-1]) / max(landing.size, 1.0)
    arrival_threshold = max(landing.size * min(config.landing_distance_ratio, 0.35), 1.0)
    arrival = path[-1]
    for node in path:
        if _distance(node, landing) <= arrival_threshold:
            arrival = node
            break
    has_motion = len(path) >= config.min_path_nodes and displacement >= config.min_displacement_ratio
    return BacktrackEvidence(has_motion, len(path), displacement, arrival.ts, float(scores[end_index]))


def refine_events_with_raw_tracks(
    recording: Recording,
    events: Sequence[GameEvent],
    landings: Sequence[LandingPoint],
    config: BacktrackConfig | None = None,
) -> list[GameEvent]:
    config = config or BacktrackConfig()
    refined: list[GameEvent] = []
    for event, landing in zip(events, landings):
        evidence = trace_landing_backwards(recording, event, landing, config)
        if config.require_motion and not evidence.has_motion:
            continue
        confidence = event.confidence
        timestamp = event.ts
        frame_index = event.frame_idx
        if evidence.has_motion:
            timestamp = evidence.arrival_ts
            frame_index = min(
                range(len(recording.frames)),
                key=lambda index: abs(recording.frames[index].timestamp - timestamp),
            )
            frame_index = recording.frames[frame_index].frame_index
        refined.append(
            GameEvent(
                seq=len(refined) + 1,
                event_type=event.event_type,
                player=event.player,
                tile=event.tile,
                ts=timestamp,
                frame_idx=frame_index,
                confidence=confidence,
            )
        )
    return refined
