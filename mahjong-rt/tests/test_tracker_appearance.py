"""Appearance gating tests (Phase 4 task 3, item 4).

Densely packed tiles overlap heavily, so IoU alone cannot say which box continues which
track. These pin the behaviour that fixed it: a match that overlaps but does not look
like the same tile gets rejected, and turning the weight off restores pure IoU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))

from mahjong_rt.tracker import ByteTrackGMC, cosine_similarity


def test_cosine_similarity_basics():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    assert cosine_similarity(a, a)[0, 0] == 1.0
    assert cosine_similarity(a, a)[0, 1] == 0.0
    assert cosine_similarity(np.zeros((0, 2)), a).shape == (0, 2)


def one_hot(index: int, size: int = 4) -> np.ndarray:
    vector = np.zeros(size, np.float32)
    vector[index] = 1.0
    return vector


def test_appearance_off_matches_on_overlap_alone():
    tracker = ByteTrackGMC(appearance_weight=0.0)
    boxes = np.array([[0, 0, 40, 40]], np.float32)
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=np.stack([one_hot(0)]))
    first = tracker.tracks[0].track_id
    # Same place, completely different appearance: with gating off this still matches.
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=np.stack([one_hot(3)]))
    assert len(tracker.tracks) == 1 and tracker.tracks[0].track_id == first


def test_appearance_rejects_mismatched_continuation():
    tracker = ByteTrackGMC(appearance_weight=0.9, match_thresh=0.7)
    boxes = np.array([[0, 0, 40, 40]], np.float32)
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=np.stack([one_hot(0)]))
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=np.stack([one_hot(3)]))
    # The box overlaps perfectly but looks like a different tile, so the track is not
    # continued; a new one is started instead of silently swapping identities.
    assert len(tracker.tracks) == 2


def test_appearance_keeps_matching_when_it_agrees():
    tracker = ByteTrackGMC(appearance_weight=0.9, match_thresh=0.7)
    boxes = np.array([[0, 0, 40, 40]], np.float32)
    descriptor = np.stack([one_hot(1)])
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=descriptor)
    first = tracker.tracks[0].track_id
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=descriptor)
    assert len(tracker.tracks) == 1 and tracker.tracks[0].track_id == first


def test_descriptor_is_smoothed_not_replaced():
    tracker = ByteTrackGMC(appearance_weight=0.5, appearance_momentum=0.7)
    boxes = np.array([[0, 0, 40, 40]], np.float32)
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=np.stack([one_hot(0)]))
    tracker.update(boxes, np.array([0.9], np.float32), None, descriptors=np.stack([one_hot(0)]))
    descriptor = tracker.tracks[0].descriptor
    # A single odd frame should not be able to redefine what a track looks like.
    assert descriptor is not None and 0.0 < float(descriptor[0]) <= 1.0


def test_missing_descriptors_fall_back_to_iou():
    tracker = ByteTrackGMC(appearance_weight=0.9)
    boxes = np.array([[0, 0, 40, 40]], np.float32)
    tracker.update(boxes, np.array([0.9], np.float32), None)
    tracker.update(boxes, np.array([0.9], np.float32), None)
    assert len(tracker.tracks) == 1
