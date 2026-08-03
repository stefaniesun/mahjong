from __future__ import annotations

import numpy as np
import pytest

from mahjong_rt.zone_types import (
    OrientationBatch,
    OrientationProvenance,
    OrientationScore,
    TableGeometry,
    TileZoneDiagnostic,
    ZoneAnalysisContext,
)


def test_table_geometry_requires_four_finite_points():
    with pytest.raises(ValueError, match="four finite points"):
        TableGeometry(corners=np.asarray([[0.0, 0.0], [1.0, 1.0]], np.float32))


def test_table_geometry_copies_and_freezes_corners():
    corners = np.asarray([[0, 10], [0, 0], [10, 0], [10, 10]], np.float32)
    geometry = TableGeometry(corners)
    corners[0, 0] = 99
    assert geometry.corners[0, 0] == 0
    with pytest.raises(ValueError, match="read-only"):
        geometry.corners[0, 0] = 3


def test_orientation_score_normalizes_and_reports_margin():
    score = OrientationScore.from_logits([1.0, 3.0, 0.0, 0.0])
    assert np.isclose(sum(score.probabilities), 1.0)
    assert score.best_rotation == 90
    assert score.margin > 0.0


def test_orientation_score_rejects_inconsistent_direct_construction():
    with pytest.raises(ValueError, match="must match probabilities"):
        OrientationScore((0.7, 0.1, 0.1, 0.1), 90, 0.6)


def test_orientation_score_copies_mutable_probabilities():
    probabilities = [0.7, 0.1, 0.1, 0.1]
    score = OrientationScore(probabilities, 0, 0.6)
    probabilities[0] = 0.1
    assert score.probabilities == (0.7, 0.1, 0.1, 0.1)


def test_zone_context_rejects_box_count_mismatch():
    with pytest.raises(ValueError, match="orientation scores"):
        ZoneAnalysisContext(
            table=TableGeometry.unit_square(1280, 720),
            orientation_batch=OrientationBatch(
                scores=(),
                provenance=OrientationProvenance("a" * 64, "cls-v1", "entropy-v1"),
            ),
            classes=("w1",),
        ).validate_for(1)


def test_zone_context_rejects_class_count_mismatch():
    with pytest.raises(ValueError, match="classes"):
        ZoneAnalysisContext(
            table=TableGeometry.unit_square(1280, 720),
            classes=(),
        ).validate_for(1)


def test_orientation_provenance_validates_hash_and_training_manifest():
    with pytest.raises(ValueError, match="model_sha256"):
        OrientationProvenance("invalid", "cls-v1", "entropy-v1")
    with pytest.raises(ValueError, match="training"):
        OrientationProvenance(
            "a" * 64,
            "cls-v1",
            "entropy-v1",
            training_manifest_sha256="b" * 64,
            training_images=None,
        )


@pytest.mark.parametrize("image", ["folder/image.jpg", "folder\\image.jpg", "", "a\nb.jpg"])
def test_orientation_provenance_rejects_non_basename_training_images(image):
    with pytest.raises(ValueError, match="training images"):
        OrientationProvenance(
            "a" * 64,
            "cls-v1",
            "entropy-v1",
            training_manifest_sha256="b" * 64,
            training_images=(image,),
        )


def test_diagnostic_allows_missing_second_candidate():
    diagnostic = TileZoneDiagnostic(
        zone="river",
        best_cost=0.2,
        second_cost=float("inf"),
        margin=float("inf"),
        evidence=(),
        ambiguous=False,
        failure=None,
    )
    assert diagnostic.second_cost == float("inf")


def test_diagnostic_is_json_serializable():
    diagnostic = TileZoneDiagnostic(
        zone="river",
        best_cost=0.2,
        second_cost=0.7,
        margin=0.5,
        evidence=("near_center",),
        ambiguous=False,
        failure=None,
    )
    assert diagnostic.to_dict()["margin"] == 0.5
    assert diagnostic.to_dict()["evidence"] == ["near_center"]
