"""Voting logic tests — the four scenarios Phase 4 task 4 lists for acceptance."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.voter import Observation, TrackVoter, decide, size_weight


def obs(label: str, conf: float = 0.95, short: float = 60.0) -> Observation:
    return Observation(label=label, confidence=conf, short_side=short)


def test_converges_on_consistent_reads():
    voter = TrackVoter()
    label = None
    for _ in range(3):
        label, _ = voter.add(obs("w5"))
    assert label == "w5"


def test_undecided_before_min_effective():
    voter = TrackVoter(min_effective=3)
    label, changed = voter.add(obs("w5"))
    assert label is None and not changed
    label, _ = voter.add(obs("w5"))
    assert label is None


def test_low_confidence_reads_are_abstained_not_voted():
    # Six guesses below the abstain threshold must never produce a verdict.
    voter = TrackVoter(min_conf=0.5)
    for _ in range(6):
        label, _ = voter.add(obs("t3", conf=0.30))
    assert label is None


def test_split_window_stays_undecided():
    # No class reaches the 60% majority, so the voter refuses to commit.
    verdict = decide([obs("w1"), obs("w2"), obs("w3"), obs("w4")])
    assert verdict.label is None


def test_hysteresis_blocks_single_frame_flip():
    voter = TrackVoter(hysteresis=4)
    for _ in range(5):
        voter.add(obs("w5"))
    assert voter.label == "w5"
    label, changed = voter.add(obs("w9"))
    assert label == "w5" and not changed


def test_hysteresis_allows_genuine_change():
    # A tile really did flip: sustained disagreement must eventually win.
    voter = TrackVoter(hysteresis=4, window=7)
    for _ in range(5):
        voter.add(obs("w5"))
    label = voter.label
    for _ in range(10):
        label, _ = voter.add(obs("w9"))
    assert label == "w9"


def test_small_crops_carry_less_weight():
    assert size_weight(10.0) < size_weight(40.0)
    assert size_weight(80.0) == 1.0
    # Two confident big reads should outweigh three marginal tiny ones.
    verdict = decide(
        [obs("w5", 0.9, 80), obs("w5", 0.9, 80), obs("w9", 0.6, 10), obs("w9", 0.6, 10), obs("w9", 0.6, 10)],
        min_effective=3,
    )
    assert verdict.label == "w5"


def test_decide_is_deterministic():
    seq = [obs("b2"), obs("b2"), obs("b4"), obs("b2")]
    assert decide(seq) == decide(seq)
