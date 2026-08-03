from __future__ import annotations

import numpy as np
import pytest

from mahjong_rt.table_geometry import TableNormalizer, order_corners
from mahjong_rt.zone_types import TableGeometry


def test_order_corners_returns_left_bottom_clockwise_contract():
    points = np.asarray([[900, 100], [100, 600], [100, 100], [900, 600]], np.float32)
    ordered = order_corners(points)
    np.testing.assert_allclose(ordered, [[100, 600], [100, 100], [900, 100], [900, 600]])


def test_normalizer_maps_table_to_unit_square():
    table = TableGeometry(
        np.asarray([[100, 600], [250, 100], [1000, 150], [1150, 620]], np.float32)
    )
    result = TableNormalizer().normalize([[250, 250, 100, 80]], table, 1280, 720)
    assert result.homography.shape == (3, 3)
    assert result.homography.flags.writeable is False
    assert len(result.tiles) == 1
    assert 0.0 <= result.tiles[0].center[0] <= 1.0
    assert 0.0 <= result.tiles[0].center[1] <= 1.0
    assert len(result.tiles[0].edge_distances) == 4
    assert result.quality.valid


def test_degenerate_table_is_rejected():
    table = TableGeometry(np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], np.float32))
    with pytest.raises(ValueError, match="degenerate"):
        TableNormalizer().normalize([], table, 1280, 720)


def test_self_intersecting_table_is_rejected():
    table = TableGeometry(np.asarray([[0, 100], [100, 0], [0, 0], [100, 100]], np.float32))
    with pytest.raises(ValueError, match="invalid corner order|non-convex|degenerate"):
        TableNormalizer(min_area_ratio=0.0).normalize([], table, 100, 100)


def test_projective_transform_preserves_normalized_tile_geometry():
    source = TableGeometry.unit_square(1000, 1000)
    normalizer = TableNormalizer()
    original = normalizer.normalize([[400, 200, 100, 160]], source, 1000, 1000)
    matrix = np.asarray(
        [[1.0, 0.15, 120], [0.05, 0.9, 80], [0.0002, 0.0001, 1]], np.float32
    )
    transformed_table = normalizer.transform_table(source, matrix)
    transformed_boxes = normalizer.transform_boxes([[400, 200, 100, 160]], matrix)
    warped = normalizer.normalize(transformed_boxes, transformed_table, 1400, 1000)
    np.testing.assert_allclose(warped.tiles[0].center, original.tiles[0].center, atol=2e-2)
    np.testing.assert_allclose(
        warped.tiles[0].edge_distances,
        original.tiles[0].edge_distances,
        atol=2e-2,
    )


def test_invalid_box_is_rejected():
    table = TableGeometry.unit_square(1000, 1000)
    with pytest.raises(ValueError, match="boxes"):
        TableNormalizer().normalize([[1, 2, -3, 4]], table, 1000, 1000)


def test_low_table_area_is_rejected():
    table = TableGeometry(np.asarray([[10, 20], [10, 10], [20, 10], [20, 20]], np.float32))
    with pytest.raises(ValueError, match="area ratio"):
        TableNormalizer(min_area_ratio=0.08).normalize([], table, 1000, 1000)
