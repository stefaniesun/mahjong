from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Sequence

import cv2
import numpy as np

from .zone_types import TableGeometry


def order_corners(points: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.shape != (4, 2) or not np.isfinite(value).all():
        raise ValueError("table corners must contain four finite points")
    order_y = np.argsort(value[:, 1], kind="stable")
    top = value[order_y[:2]][np.argsort(value[order_y[:2], 0], kind="stable")]
    bottom = value[order_y[2:]][np.argsort(value[order_y[2:], 0], kind="stable")]
    ordered = np.asarray([bottom[0], top[0], top[1], bottom[1]], dtype=np.float32)
    ordered.setflags(write=False)
    return ordered


@dataclass(frozen=True)
class NormalizedTile:
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    width: float
    height: float
    angle_deg: float
    edge_distances: tuple[float, float, float, float]


@dataclass(frozen=True)
class NormalizationQuality:
    quadrilateral_area_ratio: float
    homography_condition: float
    outside_tile_fraction: float
    valid: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedLayout:
    tiles: tuple[NormalizedTile, ...]
    homography: np.ndarray
    quality: NormalizationQuality

    def __post_init__(self) -> None:
        matrix = np.asarray(self.homography, dtype=np.float64).copy()
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("homography must be a finite 3x3 matrix")
        matrix.setflags(write=False)
        object.__setattr__(self, "homography", matrix)
        object.__setattr__(self, "tiles", tuple(self.tiles))


class TableNormalizer:
    _DESTINATION = np.asarray([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float32)

    def __init__(self, max_condition: float = 1e6, min_area_ratio: float = 0.08) -> None:
        if max_condition <= 0 or not np.isfinite(max_condition):
            raise ValueError("max_condition must be finite and positive")
        if not 0.0 <= min_area_ratio <= 1.0:
            raise ValueError("min_area_ratio must be between zero and one")
        self.max_condition = float(max_condition)
        self.min_area_ratio = float(min_area_ratio)

    def normalize(
        self,
        boxes: Sequence[Sequence[float]],
        table: TableGeometry,
        frame_w: int,
        frame_h: int,
    ) -> NormalizedLayout:
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame dimensions must be positive")
        corners = np.asarray(table.corners, dtype=np.float32)
        self._validate_ordered_corners(corners)
        area = abs(float(cv2.contourArea(corners)))
        area_ratio = area / float(frame_w * frame_h)
        if area_ratio <= 1e-8:
            raise ValueError("table quadrilateral is degenerate")
        if area_ratio < self.min_area_ratio:
            raise ValueError("table quadrilateral area ratio is below minimum")

        homography = cv2.getPerspectiveTransform(corners, self._DESTINATION)
        condition = float(np.linalg.cond(homography))
        if not np.isfinite(condition) or condition > self.max_condition:
            raise ValueError("table homography is degenerate or ill-conditioned")

        box_values = self._validate_boxes(boxes)
        tiles = tuple(self._map_box(box, homography) for box in box_values)
        outside = sum(
            not (0.0 <= tile.center[0] <= 1.0 and 0.0 <= tile.center[1] <= 1.0)
            for tile in tiles
        )
        outside_fraction = outside / len(tiles) if tiles else 0.0
        failures = ("tile_centers_outside_table",) if outside else ()
        quality = NormalizationQuality(
            quadrilateral_area_ratio=area_ratio,
            homography_condition=condition,
            outside_tile_fraction=outside_fraction,
            valid=not failures,
            failures=failures,
        )
        return NormalizedLayout(tiles=tiles, homography=homography, quality=quality)

    def transform_table(self, table: TableGeometry, matrix: np.ndarray) -> TableGeometry:
        transformed = self._transform_points(np.asarray(table.corners), matrix)
        return TableGeometry(transformed, source=table.source, confidence=table.confidence)

    def transform_boxes(
        self, boxes: Sequence[Sequence[float]], matrix: np.ndarray
    ) -> list[list[float]]:
        values = self._validate_boxes(boxes)
        transformed: list[list[float]] = []
        for box in values:
            corners = self._box_corners(box)
            mapped = self._transform_points(corners, matrix)
            minimum = mapped.min(axis=0)
            maximum = mapped.max(axis=0)
            center = (minimum + maximum) / 2.0
            size = maximum - minimum
            transformed.append(
                [float(center[0]), float(center[1]), float(size[0]), float(size[1])]
            )
        return transformed

    @staticmethod
    def _validate_ordered_corners(corners: np.ndarray) -> None:
        contour = corners.reshape(-1, 1, 2)
        signed_area = float(cv2.contourArea(contour, oriented=True))
        if abs(signed_area) <= 1e-8:
            raise ValueError("table quadrilateral is degenerate")
        if not cv2.isContourConvex(contour):
            raise ValueError("table quadrilateral is non-convex or has invalid corner order")
        if signed_area <= 0:
            raise ValueError("table quadrilateral has invalid corner order")
        edges = np.roll(corners, -1, axis=0) - corners
        if np.any(np.linalg.norm(edges, axis=1) <= 1e-6):
            raise ValueError("table quadrilateral is degenerate")

    @staticmethod
    def _validate_boxes(boxes: Sequence[Sequence[float]]) -> np.ndarray:
        value = np.asarray(boxes, dtype=np.float64)
        if value.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all():
            raise ValueError("boxes must be a finite Nx4 array")
        if np.any(value[:, 2:] <= 0):
            raise ValueError("boxes must have positive width and height")
        return value

    @staticmethod
    def _box_corners(box: Sequence[float]) -> np.ndarray:
        center_x, center_y, width, height = (float(value) for value in box)
        half_w = width / 2.0
        half_h = height / 2.0
        return np.asarray(
            [
                [center_x - half_w, center_y - half_h],
                [center_x + half_w, center_y - half_h],
                [center_x + half_w, center_y + half_h],
                [center_x - half_w, center_y + half_h],
            ],
            dtype=np.float32,
        )

    @classmethod
    def _map_box(cls, box: Sequence[float], homography: np.ndarray) -> NormalizedTile:
        mapped = cls._transform_points(cls._box_corners(box), homography)
        top = np.linalg.norm(mapped[1] - mapped[0])
        bottom = np.linalg.norm(mapped[2] - mapped[3])
        left = np.linalg.norm(mapped[3] - mapped[0])
        right = np.linalg.norm(mapped[2] - mapped[1])
        center = mapped.mean(axis=0)
        axis = ((mapped[1] - mapped[0]) + (mapped[2] - mapped[3])) / 2.0
        angle = degrees(atan2(float(axis[1]), float(axis[0])))
        center_x, center_y = (float(value) for value in center)
        return NormalizedTile(
            corners=tuple((float(point[0]), float(point[1])) for point in mapped),
            center=(center_x, center_y),
            width=float((top + bottom) / 2.0),
            height=float((left + right) / 2.0),
            angle_deg=float(angle),
            edge_distances=(center_x, center_y, 1.0 - center_x, 1.0 - center_y),
        )

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("transform must be a finite 3x3 matrix")
        value = np.asarray(points, dtype=np.float32)
        mapped = cv2.perspectiveTransform(value.reshape(1, -1, 2), transform)[0]
        if not np.isfinite(mapped).all():
            raise ValueError("perspective transform produced non-finite points")
        return mapped
