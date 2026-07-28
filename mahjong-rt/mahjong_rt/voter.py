"""Multi-frame class voting per track (Phase 4 task 4).

Single-frame classification is ~99.5% accurate but independent per frame, so a raw feed
flickers. Voting turns a stream of guesses about one physical tile into one stable
answer, and hysteresis stops that answer from flapping on a single bad frame while
still allowing a genuine change (a tile flipped face-up) to come through quickly.

The decision logic is a pure function of the observation window: same observations in,
same verdict out. That is what makes it unit-testable and cheap to sweep offline —
`replay.py` can re-run voting over recorded observations without touching the models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Observation:
    """One classifier read on one track."""

    label: str
    confidence: float
    short_side: float  # crop's shorter side in px; bigger crops are more trustworthy


@dataclass(frozen=True)
class Verdict:
    label: str | None  # None == undecided
    confidence: float
    votes: float
    total: float
    effective: int


def size_weight(short_side: float, full_weight_px: float = 40.0, floor: float = 0.35) -> float:
    """A 12px crop and a 90px crop should not carry the same vote."""
    if short_side >= full_weight_px:
        return 1.0
    return floor + (1.0 - floor) * max(0.0, short_side) / full_weight_px


def decide(
    observations: Sequence[Observation],
    *,
    min_conf: float = 0.5,
    min_effective: int = 3,
    majority_ratio: float = 0.6,
    full_weight_px: float = 40.0,
) -> Verdict:
    """Weighted majority over a window. Pure function — no state, no side effects."""
    weights: dict[str, float] = {}
    total = 0.0
    effective = 0
    for obs in observations:
        # Below the abstain threshold the classifier is guessing; a guess should not
        # get a vote at all, otherwise noise accumulates into a confident wrong answer.
        if obs.confidence < min_conf:
            continue
        weight = obs.confidence * size_weight(obs.short_side, full_weight_px)
        weights[obs.label] = weights.get(obs.label, 0.0) + weight
        total += weight
        effective += 1

    if effective < min_effective or total <= 0:
        return Verdict(None, 0.0, 0.0, total, effective)

    label, votes = max(weights.items(), key=lambda kv: kv[1])
    ratio = votes / total
    if ratio < majority_ratio:
        return Verdict(None, ratio, votes, total, effective)
    return Verdict(label, ratio, votes, total, effective)


class TrackVoter:
    """Sliding-window voting plus hysteresis for one track."""

    def __init__(
        self,
        window: int = 7,
        min_conf: float = 0.5,
        min_effective: int = 3,
        majority_ratio: float = 0.6,
        hysteresis: int = 4,
        full_weight_px: float = 40.0,
    ) -> None:
        self.window = window
        self.min_conf = min_conf
        self.min_effective = min_effective
        self.majority_ratio = majority_ratio
        self.hysteresis = hysteresis
        self.full_weight_px = full_weight_px
        self.observations: deque[Observation] = deque(maxlen=window)
        self.label: str | None = None
        self.confidence: float = 0.0
        self._challenger: str | None = None
        self._challenger_streak: int = 0

    def add(self, observation: Observation) -> tuple[str | None, bool]:
        """Feed one classification. Returns (current label, changed_this_frame)."""
        self.observations.append(observation)
        verdict = decide(
            self.observations,
            min_conf=self.min_conf,
            min_effective=self.min_effective,
            majority_ratio=self.majority_ratio,
            full_weight_px=self.full_weight_px,
        )
        if verdict.label is None:
            return self.label, False

        if self.label is None:
            self.label = verdict.label
            self.confidence = verdict.confidence
            self._challenger = None
            self._challenger_streak = 0
            return self.label, True

        if verdict.label == self.label:
            self.confidence = verdict.confidence
            self._challenger = None
            self._challenger_streak = 0
            return self.label, False

        # A different winner: make it prove itself over consecutive observations before
        # overturning a settled answer. One bad frame must never flip the display.
        if verdict.label == self._challenger:
            self._challenger_streak += 1
        else:
            self._challenger = verdict.label
            self._challenger_streak = 1
        if self._challenger_streak >= self.hysteresis:
            self.label = verdict.label
            self.confidence = verdict.confidence
            self._challenger = None
            self._challenger_streak = 0
            return self.label, True
        return self.label, False
