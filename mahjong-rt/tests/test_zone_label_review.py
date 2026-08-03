from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.review_zone_labels import (
    CANONICAL_ZONES,
    apply_reviews,
    validate_canonical_labels,
)


def labels():
    return [
        {
            "image": "a.jpg",
            "boxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "zones": ["river", "opponent_wall"],
            "classes": ["w1", "w2"],
        }
    ]


def review(reviewer, zone, source=None):
    from scripts.review_zone_labels import labels_digest

    return {
        "schema_version": 1,
        "reviewer": reviewer,
        "labels_sha256": labels_digest(source or labels()),
        "records": [
            {
                "image": "a.jpg",
                "box_index": 1,
                "old_label": "opponent_wall",
                "label": zone,
                "rationale": "牌位于对家牌墙内侧并面向对家。",
            }
        ],
    }


def test_canonical_zone_contract_has_exactly_five_zones():
    assert CANONICAL_ZONES == {
        "my_hand",
        "seat_left",
        "seat_across",
        "seat_right",
        "river",
    }


def test_legacy_zone_fails_canonical_validation():
    with pytest.raises(ValueError, match="opponent_wall"):
        validate_canonical_labels(labels())


def test_reviews_must_be_independent():
    with pytest.raises(ValueError, match="different"):
        apply_reviews(labels(), review("same", "seat_across"), review("same", "seat_across"))


def test_disagreement_aborts_without_mutation():
    source = labels()
    before = copy.deepcopy(source)
    with pytest.raises(ValueError, match="disagree"):
        apply_reviews(source, review("alice", "seat_across"), review("bob", "river"))
    assert source == before


def test_agreed_review_changes_only_target_zone_and_writes_audit():
    source = labels()
    migrated, audit = apply_reviews(
        source,
        review("alice", "seat_across"),
        review("bob", "seat_across"),
        expected_labels_sha256=review("alice", "seat_across")["labels_sha256"],
    )
    assert migrated[0]["zones"] == ["river", "seat_across"]
    assert migrated[0]["boxes"] == source[0]["boxes"]
    assert migrated[0]["classes"] == source[0]["classes"]
    assert audit["changes"] == [
        {
            "image": "a.jpg",
            "box_index": 1,
            "old_label": "opponent_wall",
            "reviewer_a": "alice",
            "reviewer_a_label": "seat_across",
            "reviewer_b": "bob",
            "reviewer_b_label": "seat_across",
            "final_label": "seat_across",
            "rationale_a": "牌位于对家牌墙内侧并面向对家。",
            "rationale_b": "牌位于对家牌墙内侧并面向对家。",
        }
    ]


def test_ambiguous_canonical_sample_cannot_be_omitted():
    source = labels()
    source[0]["zone_ambiguous"] = [True, False]
    with pytest.raises(ValueError, match="all required samples"):
        apply_reviews(
            source,
            review("alice", "seat_across", source),
            review("bob", "seat_across", source),
        )


def test_empty_rationale_is_rejected():
    second = review("bob", "seat_across")
    second["records"][0]["rationale"] = ""
    with pytest.raises(ValueError, match="rationale"):
        apply_reviews(labels(), review("alice", "seat_across"), second)


def test_review_cannot_change_non_target_fields():
    bad = review("bob", "seat_across")
    bad["records"][0]["old_label"] = "river"
    with pytest.raises(ValueError, match="old_label"):
        apply_reviews(labels(), review("alice", "seat_across"), bad)
