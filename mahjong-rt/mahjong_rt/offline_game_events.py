"""Offline game-event reconstruction from stable before/after table states."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .game_events import GameEvent, SEATS
from .raw_event_backtrack import LandingPoint


@dataclass
class OfflineEventConfig:
    initial_window_s: float = 0.5
    stable_window_s: float = 0.7
    min_presence_ratio: float = 0.55
    sample_step_s: float = 0.15
    match_distance_ratio: float = 0.8
    min_gap_s: float = 0.6
    start_player: str | None = None


@dataclass
class _StableTile:
    label: str
    x: float
    y: float
    size: float
    confidence: float
    support: float
    frame_idx: int
    ts: float


@dataclass
class _TrackObservations:
    labels: Counter[str]
    points: list[tuple[float, float]]
    sizes: list[float]
    confidences: list[float]
    frame_indices: list[int]
    timestamps: list[float]
    frames_seen: set[int]


def _world_matrices(homographies: Sequence[Any] | None, count: int) -> list[np.ndarray]:
    world = np.eye(3, dtype=np.float64)
    result: list[np.ndarray] = []
    for index in range(count):
        if homographies is not None and index < len(homographies):
            matrix = np.asarray(homographies[index], dtype=np.float64)
            if matrix.shape == (3, 3):
                try:
                    world = world @ np.linalg.inv(matrix)
                except np.linalg.LinAlgError:
                    pass
        result.append(world.copy())
    return result


def _project(world: np.ndarray, x: float, y: float) -> np.ndarray:
    projected = world @ np.array([x, y, 1.0])
    if abs(projected[2]) > 1e-9:
        projected = projected / projected[2]
    return projected[:2]


def _point(tile: dict[str, Any], world: np.ndarray) -> tuple[float, float, float]:
    x, y, width, height = (float(value) for value in tile["bbox"])
    centre = _project(world, x + width / 2.0, y + height / 2.0)
    horizontal = _project(world, x + width, y + height / 2.0)
    vertical = _project(world, x + width / 2.0, y + height)
    size = max(min(np.linalg.norm(horizontal - centre), np.linalg.norm(vertical - centre)) * 2.0, 1.0)
    return float(centre[0]), float(centre[1]), float(size)


def _stable_state(
    summaries: Sequence[dict[str, Any]],
    worlds: Sequence[np.ndarray],
    start: float,
    end: float,
    config: OfflineEventConfig,
) -> list[_StableTile]:
    indices = [index for index, summary in enumerate(summaries) if start <= float(summary.get("ts", 0.0)) <= end]
    if not indices:
        return []

    observations: dict[int, _TrackObservations] = {}
    for index in indices:
        summary = summaries[index]
        for tile in summary.get("tiles", []):
            if tile.get("zone") != "river" or not tile.get("visible", False) or not tile.get("label"):
                continue
            track_id = int(tile.get("track_id", -1))
            observation = observations.setdefault(
                track_id,
                _TrackObservations(Counter(), [], [], [], [], [], set()),
            )
            x, y, size = _point(tile, worlds[index])
            observation.labels[str(tile["label"])] += 1
            observation.points.append((x, y))
            observation.sizes.append(size)
            observation.confidences.append(float(tile.get("confidence", 0.0)))
            observation.frame_indices.append(int(summary.get("frame_idx", index)))
            observation.timestamps.append(float(summary.get("ts", 0.0)))
            observation.frames_seen.add(index)

    grouped: list[_TrackObservations] = []
    for observation in observations.values():
        point = np.median(np.asarray(observation.points), axis=0)
        size = float(np.median(observation.sizes))
        target: _TrackObservations | None = None
        for group in grouped:
            group_point = np.median(np.asarray(group.points), axis=0)
            group_size = float(np.median(group.sizes))
            if np.linalg.norm(point - group_point) <= max(size, group_size) * config.match_distance_ratio:
                target = group
                break
        if target is None:
            grouped.append(observation)
            continue
        target.labels.update(observation.labels)
        target.points.extend(observation.points)
        target.sizes.extend(observation.sizes)
        target.confidences.extend(observation.confidences)
        target.frame_indices.extend(observation.frame_indices)
        target.timestamps.extend(observation.timestamps)
        target.frames_seen.update(observation.frames_seen)

    required = max(1, int(np.ceil(len(indices) * config.min_presence_ratio)))
    stable: list[_StableTile] = []
    for observation in grouped:
        if len(observation.frames_seen) < required:
            continue
        label, _ = observation.labels.most_common(1)[0]
        points = np.asarray(observation.points)
        stable.append(
            _StableTile(
                label=label,
                x=float(np.median(points[:, 0])),
                y=float(np.median(points[:, 1])),
                size=float(np.median(observation.sizes)),
                confidence=float(np.mean(observation.confidences)),
                support=min(len(observation.frames_seen) / len(indices), 1.0),
                frame_idx=int(np.median(observation.frame_indices)),
                ts=float(np.median(observation.timestamps)),
            )
        )
    return stable


def _distance(left: _StableTile, right: _StableTile) -> float:
    return float(np.hypot(left.x - right.x, left.y - right.y))


def _match(previous: Sequence[_StableTile], current: Sequence[_StableTile], ratio: float) -> set[int]:
    pairs: list[tuple[float, int, int]] = []
    for previous_index, old in enumerate(previous):
        for current_index, new in enumerate(current):
            threshold = max(old.size, new.size) * ratio
            distance = _distance(old, new)
            if distance <= threshold:
                pairs.append((distance, previous_index, current_index))
    matched_previous: set[int] = set()
    matched_current: set[int] = set()
    for _, previous_index, current_index in sorted(pairs):
        if previous_index in matched_previous or current_index in matched_current:
            continue
        matched_previous.add(previous_index)
        matched_current.add(current_index)
    return matched_current


def reconstruct_events_with_landings(
    summaries: Sequence[dict[str, Any]],
    homographies: Sequence[Any] | None = None,
    config: OfflineEventConfig | None = None,
) -> tuple[list[GameEvent], list[LandingPoint]]:
    config = config or OfflineEventConfig()
    indexed_frames = [
        (index, summary)
        for index, summary in enumerate(summaries)
        if summary.get("type", "frame_summary") == "frame_summary"
    ]
    timestamps = [float(summary.get("ts", 0.0)) for _, summary in indexed_frames]
    if homographies is not None and any(left > right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("summaries must be sorted when homographies are incremental")
    indexed_frames.sort(key=lambda item: float(item[1].get("ts", 0.0)))
    frames = [summary for _, summary in indexed_frames]
    if not frames:
        return [], []
    if config.start_player is not None and config.start_player not in SEATS:
        raise ValueError(f"start_player must be one of {SEATS}")

    ordered_homographies = None
    if homographies is not None:
        ordered_homographies = [homographies[index] for index, _ in indexed_frames if index < len(homographies)]
    worlds = _world_matrices(ordered_homographies, len(frames))
    first_ts = float(frames[0].get("ts", 0.0))
    last_ts = float(frames[-1].get("ts", 0.0))
    window = max(config.stable_window_s, 0.01)
    step = max(config.sample_step_s, 0.01)
    initial_window = max(config.initial_window_s, 0.01)
    initial_config = OfflineEventConfig(
        **{
            field: value
            for field, value in vars(config).items()
            if field != "min_presence_ratio"
        },
        min_presence_ratio=0.0,
    )
    known = _stable_state(frames, worlds, first_ts, min(last_ts, first_ts + initial_window), initial_config)
    initial_counts: Counter[str] = Counter(tile.label for tile in known)
    candidates: list[_StableTile] = []
    cursor = first_ts + initial_window
    while cursor + window <= last_ts + 1e-9:
        current = _stable_state(frames, worlds, cursor, cursor + window, config)
        matched = _match(known, current, config.match_distance_ratio)
        for index, tile in enumerate(current):
            if index in matched:
                continue
            duplicate = any(
                _distance(tile, candidate) <= max(tile.size, candidate.size) * config.match_distance_ratio
                for candidate in candidates
            )
            if not duplicate:
                candidates.append(tile)
                known.append(tile)
        cursor += step

    candidates.sort(key=lambda tile: tile.ts)
    events: list[GameEvent] = []
    landings: list[LandingPoint] = []
    counts = initial_counts.copy()
    last_event_ts = -float("inf")
    turn_index = SEATS.index(config.start_player) if config.start_player is not None else 0
    for candidate in candidates:
        if candidate.ts - last_event_ts < config.min_gap_s or counts[candidate.label] >= 4:
            continue
        player = SEATS[turn_index] if config.start_player is not None else "unknown"
        events.append(
            GameEvent(
                seq=len(events) + 1,
                event_type="discard",
                player=player,
                tile=candidate.label,
                ts=candidate.ts,
                frame_idx=candidate.frame_idx,
                confidence=min(candidate.confidence, candidate.support),
            )
        )
        landings.append(LandingPoint(candidate.x, candidate.y, candidate.size))
        counts[candidate.label] += 1
        last_event_ts = candidate.ts
        if config.start_player is not None:
            turn_index = (turn_index + 1) % len(SEATS)
    return events, landings


def reconstruct_events(
    summaries: Sequence[dict[str, Any]],
    homographies: Sequence[Any] | None = None,
    config: OfflineEventConfig | None = None,
) -> list[GameEvent]:
    events, _ = reconstruct_events_with_landings(summaries, homographies, config)
    return events
