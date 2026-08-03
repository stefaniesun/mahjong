from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

import numpy as np

CALIBRATION_SCHEMA_VERSION = 1
ROTATIONS = (0, 90, 180, 270)


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TableGeometry:
    corners: np.ndarray
    source: str = "annotated"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        value = np.asarray(self.corners, dtype=np.float32).copy()
        if value.shape != (4, 2) or not np.isfinite(value).all():
            raise ValueError("table corners must contain four finite points")
        if not self.source:
            raise ValueError("table geometry source must not be empty")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("table geometry confidence must be between zero and one")
        value.setflags(write=False)
        object.__setattr__(self, "corners", value)

    @classmethod
    def unit_square(cls, width: int, height: int) -> "TableGeometry":
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        return cls(
            np.asarray(
                [[0, height], [0, 0], [width, 0], [width, height]],
                dtype=np.float32,
            )
        )


@dataclass(frozen=True)
class OrientationScore:
    probabilities: tuple[float, float, float, float]
    best_rotation: int
    margin: float

    def __post_init__(self) -> None:
        probs = np.asarray(self.probabilities, dtype=np.float64)
        if (
            probs.shape != (4,)
            or not np.isfinite(probs).all()
            or (probs < 0).any()
            or not np.isclose(probs.sum(), 1.0)
        ):
            raise ValueError("orientation probabilities must be normalized")
        normalized = tuple(float(value) for value in probs)
        object.__setattr__(self, "probabilities", normalized)
        order = np.argsort(probs, kind="stable")
        expected_best = ROTATIONS[int(order[-1])]
        expected_margin = float(probs[order[-1]] - probs[order[-2]])
        if self.best_rotation != expected_best or not np.isclose(self.margin, expected_margin):
            raise ValueError("orientation best_rotation and margin must match probabilities")

    @classmethod
    def from_logits(cls, logits: Sequence[float]) -> "OrientationScore":
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("orientation logits must contain four finite values")
        shifted = values - values.max()
        exponentials = np.exp(shifted)
        return cls.from_probabilities(exponentials / exponentials.sum())

    @classmethod
    def from_probabilities(cls, probabilities: Sequence[float]) -> "OrientationScore":
        probs = np.asarray(probabilities, dtype=np.float64)
        if (
            probs.shape != (4,)
            or not np.isfinite(probs).all()
            or (probs < 0).any()
            or probs.sum() <= 0
        ):
            raise ValueError(
                "orientation probabilities must contain four finite non-negative values"
            )
        probs = probs / probs.sum()
        order = np.argsort(probs, kind="stable")
        best = int(order[-1])
        return cls(
            tuple(float(value) for value in probs),
            ROTATIONS[best],
            float(probs[order[-1]] - probs[order[-2]]),
        )


@dataclass(frozen=True)
class OrientationProvenance:
    model_sha256: str
    preprocessing_version: str
    scorer_version: str
    training_manifest_sha256: str | None = None
    training_images: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.model_sha256, "model_sha256")
        if not self.preprocessing_version:
            raise ValueError("preprocessing_version must not be empty")
        if not self.scorer_version:
            raise ValueError("scorer_version must not be empty")
        if (self.training_manifest_sha256 is None) != (self.training_images is None):
            raise ValueError("training manifest digest and training images must be provided together")
        if self.training_images is None:
            return
        _validate_sha256(self.training_manifest_sha256 or "", "training_manifest_sha256")
        images = tuple(self.training_images)
        if not images or images != tuple(sorted(set(images))):
            raise ValueError("training images must be a non-empty sorted unique tuple")
        if any(
            not image
            or image in {".", ".."}
            or "/" in image
            or "\\" in image
            or any(ord(character) < 32 for character in image)
            for image in images
        ):
            raise ValueError("training images must contain safe basenames only")
        canonical = ("\n".join(images) + "\n").encode("utf-8")
        if sha256(canonical).hexdigest() != self.training_manifest_sha256:
            raise ValueError("training manifest digest does not match training images")
        object.__setattr__(self, "training_images", images)


@dataclass(frozen=True)
class OrientationBatch:
    scores: tuple[OrientationScore, ...]
    provenance: OrientationProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", tuple(self.scores))


@dataclass(frozen=True)
class TileZoneDiagnostic:
    zone: str
    best_cost: float
    second_cost: float
    margin: float
    evidence: tuple[str, ...]
    ambiguous: bool
    failure: str | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.best_cost):
            raise ValueError("diagnostic best cost must be finite")
        if np.isnan(self.second_cost) or np.isnan(self.margin):
            raise ValueError("diagnostic second cost and margin must not be NaN")
        if self.second_cost < self.best_cost:
            raise ValueError("diagnostic second cost must not be below best cost")
        expected_margin = self.second_cost - self.best_cost
        if not np.isclose(self.margin, expected_margin):
            raise ValueError("diagnostic margin must equal second cost minus best cost")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "best_cost": self.best_cost,
            "second_cost": self.second_cost,
            "margin": self.margin,
            "evidence": list(self.evidence),
            "ambiguous": self.ambiguous,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class ZoneAnalysisContext:
    table: TableGeometry
    orientation_batch: OrientationBatch | None = None
    classes: tuple[str, ...] | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        if self.classes is not None:
            object.__setattr__(self, "classes", tuple(self.classes))

    def validate_for(self, box_count: int) -> None:
        if box_count < 0:
            raise ValueError("box count must be non-negative")
        if self.orientation_batch is not None and len(self.orientation_batch.scores) != box_count:
            raise ValueError("orientation scores length must match boxes")
        if self.classes is not None and len(self.classes) != box_count:
            raise ValueError("classes length must match boxes")
