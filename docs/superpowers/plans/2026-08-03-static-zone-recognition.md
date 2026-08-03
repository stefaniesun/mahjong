# Static Zone Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibrated, perspective-invariant static-image zone recognizer that combines table geometry, tile orientation, constrained groups, and global scoring, with 100% accuracy and coverage as a release gate on the current 899 boxes and every sealed test set.

**Architecture:** Keep `mahjong_rt.zones` as the backward-compatible facade and add focused modules for table normalization, calibration, orientation scoring, layout grouping, and structured solving. The strict path requires table geometry and a calibration profile; the legacy heuristic remains explicitly selectable for replay compatibility. Implement in measurable stages so geometry, orientation, and structure are evaluated independently before their signals are combined.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, ONNX Runtime, PyYAML, pytest; no SciPy, NetworkX, or integer-programming dependency.

---

## Scope and release gates

This plan implements the complete static-image path described in `docs/superpowers/specs/2026-08-03-static-zone-recognition-design.md`. Automatic table-border detection and manual/annotated corners share the same `TableGeometry` interface. The current 20-image dataset does not yet contain table corners. It also contains five legacy `opponent_wall` labels even though the accepted task contract has five zones and excludes walls. The first data deliverable therefore includes deterministic corner annotation, independent double-review of all legacy/ambiguous labels, and an auditable label-migration report before model work starts.

The implementation must not overfit silently. There are four distinct gates:

1. **Data-readiness gate:** all 899 labels conform to the five-zone contract, every image has valid corners, two reviewers agree on migrated/ambiguous samples, and the migration audit passes.
2. **Development gate:** unit, compatibility, schema, synthetic-perspective, and image-level five-fold tests pass.
3. **Current-set gate:** all 899 boxes receive out-of-fold predictions from profiles that never saw their image and reach 899/899 with 100% coverage. A separate frozen-profile holdout result reports its actual evaluated count and is never mislabeled as 899/899.
4. **Generalization gate:** a frozen profile fitted on a new scene's calibration split reaches N/N on its sealed image-level test split. A failed sealed set becomes development data, and a new sealed set is required after changes.

A task may improve the implementation without satisfying gates 3 or 4. No task may claim the 100% goal is achieved until the out-of-fold 899/899 gate and a genuinely sealed N/N gate are both demonstrated by the evaluator. A fixed profile evaluated on images used to fit it is diagnostic only and cannot satisfy either gate.

## File map

### New production files

- `mahjong-rt/mahjong_rt/zone_types.py` — immutable shared types, strict context, diagnostics, and schema constants.
- `mahjong-rt/mahjong_rt/table_geometry.py` — corner ordering, validation, homography, mapped tile geometry, and automatic border locator.
- `mahjong-rt/mahjong_rt/zone_calibration.py` — load/save/version validation and fitting of calibration profiles.
- `mahjong-rt/mahjong_rt/tile_orientation.py` — classifier adapter and four-rotation orientation scores.
- `mahjong-rt/mahjong_rt/layout_graph.py` — constrained adjacency graph and non-chaining groups.
- `mahjong-rt/mahjong_rt/zone_solver.py` — explainable unary costs, group consistency, deterministic global assignment, and margins.

### New scripts and tests

- `mahjong-rt/scripts/annotate_table_corners.py` — annotate four corners in the existing 20 images.
- `mahjong-rt/scripts/review_zone_labels.py` — collect two independent reviews and generate the five-zone migration audit.
- `mahjong-rt/scripts/calibrate_zones.py` — fit a versioned profile from an explicit calibration manifest.
- `mahjong-rt/scripts/eval_zones.py` — immutable evaluation/reporting entry point.
- `mahjong-rt/scripts/eval_zone_orientation.py` — measure whether the existing classifier contains useful rotation signal.
- `mahjong-rt/tests/test_table_geometry.py`
- `mahjong-rt/tests/test_zone_calibration.py`
- `mahjong-rt/tests/test_tile_orientation.py`
- `mahjong-rt/tests/test_layout_graph.py`
- `mahjong-rt/tests/test_zone_solver.py`
- `mahjong-rt/tests/test_zone_pipeline.py`
- `mahjong-rt/tests/test_eval_zones.py`

### Modified files

- `mahjong-rt/mahjong_rt/zones.py` — preserve legacy implementation and dispatch to strict structured path when context is supplied.
- `mahjong-rt/mahjong_rt/pipeline.py` — pass frame/corners/profile only when strict static zones are configured; preserve default event schema.
- `mahjong-rt/mahjong_rt/replay.py` — reject strict mode when recordings lack required static-image context.
- `mahjong-rt/configs/pipeline.yaml` — add explicit `mode`, profile path, table-locator, and failure-policy keys.
- `mahjong-rt/tests/test_zones.py` — pin legacy compatibility and move strict 100% assertion to the dedicated evaluator.
- `output/zone_annotation/zone_labels_with_class.json` — add `table_corners` only through the annotation script; do not change boxes, zones, classes, or image order.

---

### Task 1: Shared strict-zone contracts

**Files:**
- Create: `mahjong-rt/mahjong_rt/zone_types.py`
- Test: `mahjong-rt/tests/test_zone_types.py`

- [ ] **Step 1: Write failing tests for immutable context and diagnostics**

```python
from __future__ import annotations

import numpy as np
import pytest

from mahjong_rt.zone_types import (
    OrientationScore,
    TableGeometry,
    TileZoneDiagnostic,
    ZoneAnalysisContext,
)


def test_table_geometry_requires_four_finite_points():
    with pytest.raises(ValueError, match="four finite points"):
        TableGeometry(corners=np.asarray([[0.0, 0.0], [1.0, 1.0]], np.float32))


def test_orientation_score_normalizes_and_reports_margin():
    score = OrientationScore.from_logits([1.0, 3.0, 0.0, 0.0])
    assert np.isclose(sum(score.probabilities), 1.0)
    assert score.best_rotation == 90
    assert score.margin > 0.0


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
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run from `mahjong-rt`:

```powershell
python -m pytest tests/test_zone_types.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'mahjong_rt.zone_types'`.

- [ ] **Step 3: Implement the shared contracts**

Create these public contracts in `mahjong_rt/zone_types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

CALIBRATION_SCHEMA_VERSION = 1
ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class TableGeometry:
    corners: np.ndarray
    source: str = "annotated"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        value = np.asarray(self.corners, dtype=np.float32).copy()
        if value.shape != (4, 2) or not np.isfinite(value).all():
            raise ValueError("table corners must contain four finite points")
        value.setflags(write=False)
        object.__setattr__(self, "corners", value)

    @classmethod
    def unit_square(cls, width: int, height: int) -> "TableGeometry":
        return cls(np.asarray([[0, height], [0, 0], [width, 0], [width, height]], np.float32))


@dataclass(frozen=True)
class OrientationScore:
    probabilities: tuple[float, float, float, float]
    best_rotation: int
    margin: float

    def __post_init__(self) -> None:
        probs = np.asarray(self.probabilities, dtype=np.float64)
        if probs.shape != (4,) or not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1.0):
            raise ValueError("orientation probabilities must be normalized")
        order = np.argsort(probs)
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
        probs = np.exp(shifted) / np.exp(shifted).sum()
        return cls.from_probabilities(probs)

    @classmethod
    def from_probabilities(cls, probabilities: Sequence[float]) -> "OrientationScore":
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (4,) or not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError("orientation probabilities must contain four finite non-negative values")
        probs = probs / probs.sum()
        order = np.argsort(probs)
        best = int(order[-1])
        return cls(tuple(float(x) for x in probs), ROTATIONS[best], float(probs[order[-1]] - probs[order[-2]]))


@dataclass(frozen=True)
class OrientationProvenance:
    model_sha256: str
    preprocessing_version: str
    scorer_version: str
    training_manifest_sha256: str | None = None
    training_images: tuple[str, ...] | None = None


@dataclass(frozen=True)
class OrientationBatch:
    scores: tuple[OrientationScore, ...]
    provenance: OrientationProvenance


@dataclass(frozen=True)
class TileZoneDiagnostic:
    zone: str
    best_cost: float
    second_cost: float
    margin: float
    evidence: tuple[str, ...]
    ambiguous: bool
    failure: str | None

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

    def validate_for(self, box_count: int) -> None:
        if self.orientation_batch is not None and len(self.orientation_batch.scores) != box_count:
            raise ValueError("orientation scores length must match boxes")
        if self.classes is not None and len(self.classes) != box_count:
            raise ValueError("classes length must match boxes")
```

- [ ] **Step 4: Run the focused test and all existing zone tests**

```powershell
python -m pytest tests/test_zone_types.py tests/test_zones.py -q
```

Expected: all tests pass; existing `assign_zones` behavior is unchanged.

- [ ] **Step 5: Commit the contracts**

```powershell
git add mahjong_rt/zone_types.py tests/test_zone_types.py
git commit -m "feat(zones): define structured zone contracts"
```

---

### Task 2: Data readiness, label migration, and perspective normalization

**Files:**
- Create: `mahjong-rt/mahjong_rt/table_geometry.py`
- Create: `mahjong-rt/tests/test_table_geometry.py`
- Create: `mahjong-rt/tests/test_zone_label_review.py`
- Create: `mahjong-rt/scripts/annotate_table_corners.py`
- Create: `mahjong-rt/scripts/review_zone_labels.py`
- Create: `output/zone_annotation/zone_label_migration_audit.json`
- Modify: `output/zone_annotation/zone_labels_with_class.json`

- [ ] **Step 1: Write failing geometry tests**

```python
import numpy as np
import pytest

from mahjong_rt.table_geometry import TableNormalizer, order_corners
from mahjong_rt.zone_types import TableGeometry


def test_order_corners_returns_left_bottom_clockwise_contract():
    points = np.asarray([[900, 100], [100, 600], [100, 100], [900, 600]], np.float32)
    ordered = order_corners(points)
    np.testing.assert_allclose(ordered, [[100, 600], [100, 100], [900, 100], [900, 600]])


def test_normalizer_maps_table_to_unit_square():
    table = TableGeometry(np.asarray([[100, 600], [250, 100], [1000, 150], [1150, 620]], np.float32))
    result = TableNormalizer().normalize([[250, 250, 100, 80]], table, 1280, 720)
    assert result.homography.shape == (3, 3)
    assert len(result.tiles) == 1
    assert 0.0 <= result.tiles[0].center[0] <= 1.0
    assert 0.0 <= result.tiles[0].center[1] <= 1.0
    assert len(result.tiles[0].edge_distances) == 4


def test_degenerate_table_is_rejected():
    table = TableGeometry(np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], np.float32))
    with pytest.raises(ValueError, match="degenerate"):
        TableNormalizer().normalize([], table, 1280, 720)


def test_projective_transform_preserves_normalized_tile_geometry():
    source = TableGeometry.unit_square(1000, 1000)
    normalizer = TableNormalizer()
    original = normalizer.normalize([[400, 200, 100, 160]], source, 1000, 1000)
    matrix = np.asarray([[1.0, 0.15, 120], [0.05, 0.9, 80], [0.0002, 0.0001, 1]], np.float32)
    transformed_table = normalizer.transform_table(source, matrix)
    transformed_boxes = normalizer.transform_boxes([[400, 200, 100, 160]], matrix)
    warped = normalizer.normalize(transformed_boxes, transformed_table, 1400, 1000)
    np.testing.assert_allclose(warped.tiles[0].center, original.tiles[0].center, atol=2e-2)
```

- [ ] **Step 2: Run tests and verify missing implementation failure**

```powershell
python -m pytest tests/test_table_geometry.py -q
```

Expected: collection fails because `mahjong_rt.table_geometry` does not exist.

- [ ] **Step 3: Implement deterministic homography and mapped tile geometry**

In `mahjong_rt/table_geometry.py`, define:

```python
@dataclass(frozen=True)
class NormalizedTile:
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    width: float
    height: float
    angle_deg: float
    edge_distances: tuple[float, float, float, float]  # left, top, right, bottom


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


class TableNormalizer:
    def __init__(self, max_condition: float = 1e6, min_area_ratio: float = 0.08) -> None: ...
    def normalize(self, boxes, table: TableGeometry, frame_w: int, frame_h: int) -> NormalizedLayout: ...
    def transform_table(self, table: TableGeometry, matrix: np.ndarray) -> TableGeometry: ...
    def transform_boxes(self, boxes, matrix: np.ndarray) -> list[list[float]]: ...
```

Use `cv2.getPerspectiveTransform` and `cv2.perspectiveTransform`. Map each `xywh` rectangle's four corners. Compute mapped width and height as opposite-edge means and compute orientation with `atan2`. Reject self-intersecting, near-zero-area, non-convex, or ill-conditioned table quadrilaterals. Keep all normalized coordinates in `[0, 1]` space; do not hard-code a pixel output size.

- [ ] **Step 4: Add a deterministic corner annotation script**

Implement `scripts/annotate_table_corners.py` with this CLI:

```powershell
python scripts/annotate_table_corners.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images
```

Behavior:

- Open each image with OpenCV.
- Show any existing corners.
- Collect exactly four clicks in order: left-bottom, left-top, right-top, right-bottom.
- `R` resets the current image, `S` saves and advances, `Q` exits without modifying the current item.
- Save atomically through a sibling `.tmp` file and `Path.replace`.
- Modify only a `table_corners` field; preserve every existing field and list order.
- Before saving, construct `TableGeometry` and call `TableNormalizer.normalize` to reject invalid corners.

The saved field is:

```json
"table_corners": [[x_lb, y_lb], [x_lt, y_lt], [x_rt, y_rt], [x_rb, y_rb]]
```

- [ ] **Step 5: Write failing tests for five-zone schema and independent review**

`tests/test_zone_label_review.py` must assert:

- only `my_hand`, `seat_left`, `seat_across`, `seat_right`, and `river` are accepted in the canonical dataset;
- the current five `opponent_wall` samples make validation fail before migration;
- two review files are required and reviewers cannot share an ID;
- a disagreement cannot be migrated automatically;
- the audit records image, box index, old label, reviewer-A label, reviewer-B label, final label, and rationale;
- boxes, classes, image order, and all unrelated fields are unchanged by migration.

Run:

```powershell
python -m pytest tests/test_zone_label_review.py -q
```

Expected: collection fails because `scripts.review_zone_labels` does not exist.

- [ ] **Step 6: Implement and perform independent double-review**

Implement `scripts/review_zone_labels.py` with two separate phases:

```powershell
python scripts/review_zone_labels.py collect --labels ../output/zone_annotation/zone_labels_with_class.json --reviewer <reviewer-a> --output ../output/zone_annotation/review_a.json
python scripts/review_zone_labels.py collect --labels ../output/zone_annotation/zone_labels_with_class.json --reviewer <reviewer-b> --output ../output/zone_annotation/review_b.json
python scripts/review_zone_labels.py apply --labels ../output/zone_annotation/zone_labels_with_class.json --review-a ../output/zone_annotation/review_a.json --review-b ../output/zone_annotation/review_b.json --audit ../output/zone_annotation/zone_label_migration_audit.json
```

The collection UI must show every `opponent_wall` sample plus all samples explicitly marked ambiguous during review. Review B must not see review A's answer. `apply` may update a label only when both reviewers independently select the same canonical zone and provide a non-empty rationale. Disagreements abort without modifying the canonical file. The audit stores the input/output SHA-256 digests and every changed sample. Do not infer `opponent_wall → seat_across` automatically.

- [ ] **Step 7: Annotate all 20 images and validate the data-readiness gate**

Run the corner annotation and completed double-review workflow, then run:

```powershell
python -c "import json; p='../output/zone_annotation/zone_labels_with_class.json'; d=json.load(open(p,encoding='utf-8')); allowed={'my_hand','seat_left','seat_across','seat_right','river'}; assert len(d)==20 and all(len(x.get('table_corners',[]))==4 for x in d); assert all(z in allowed for x in d for z in x['zones']); assert sum(len(x['zones']) for x in d)==899; print('ready:',len(d),899)"
```

Expected: `ready: 20 899`. If reviewers cannot agree from static visual evidence, stop and revise the annotation contract; do not proceed to calibration.

- [ ] **Step 8: Run geometry, review, and legacy regression tests**

```powershell
python -m pytest tests/test_table_geometry.py tests/test_zone_label_review.py tests/test_zones.py -q
```

Expected: all tests pass. The audit proves that corner annotation changed only `table_corners`, while label migration changed only independently agreed `zones` entries.

- [ ] **Step 9: Commit normalized geometry and audited labels**

```powershell
git add mahjong_rt/table_geometry.py tests/test_table_geometry.py tests/test_zone_label_review.py scripts/annotate_table_corners.py scripts/review_zone_labels.py ../output/zone_annotation/zone_labels_with_class.json ../output/zone_annotation/zone_label_migration_audit.json
git commit -m "feat(zones): prepare audited perspective-normalized data"
```

---

### Task 3: Versioned calibration profiles

**Files:**
- Create: `mahjong-rt/mahjong_rt/zone_calibration.py`
- Create: `mahjong-rt/tests/test_zone_calibration.py`
- Create: `mahjong-rt/scripts/calibrate_zones.py`
- Create: `mahjong-rt/configs/zones/current_v1.json`

- [ ] **Step 1: Write failing profile tests**

```python
import json

import pytest

from mahjong_rt.zone_calibration import CalibrationSample, ZoneCalibrator, ZoneProfile


def sample(image, center, zone):
    return CalibrationSample(
        image=image,
        tile_index=0,
        zone=zone,
        center=center,
        edge_distances=(center[0], center[1], 1-center[0], 1-center[1]),
        size=(0.08, 0.12),
        angle_deg=0.0,
        nearest_neighbor=0.1,
    )


def test_profile_round_trip_is_stable(tmp_path):
    profile = ZoneCalibrator().fit([
        sample("a.jpg", (0.5, 0.9), "my_hand"),
        sample("b.jpg", (0.1, 0.5), "seat_left"),
        sample("c.jpg", (0.5, 0.1), "seat_across"),
        sample("d.jpg", (0.9, 0.5), "seat_right"),
        sample("e.jpg", (0.5, 0.5), "river"),
    ], calibration_id="camera-a")
    path = tmp_path / "profile.json"
    profile.save(path)
    assert ZoneProfile.load(path) == profile


def test_unknown_schema_version_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        ZoneProfile.load(path)


def test_fit_records_image_level_provenance():
    profile = ZoneCalibrator().fit([
        sample("a.jpg", (0.5, 0.9), "my_hand"),
        sample("b.jpg", (0.2, 0.4), "seat_left"),
        sample("c.jpg", (0.5, 0.1), "seat_across"),
        sample("d.jpg", (0.8, 0.4), "seat_right"),
        sample("e.jpg", (0.5, 0.5), "river"),
    ], calibration_id="camera-a")
    assert profile.calibration_images == ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg")
    assert profile.calibration_digest
```

- [ ] **Step 2: Verify the tests fail**

```powershell
python -m pytest tests/test_zone_calibration.py -q
```

Expected: import failure for `mahjong_rt.zone_calibration`.

- [ ] **Step 3: Implement profile fitting without test-set leakage**

Define these public objects:

```python
@dataclass(frozen=True)
class CalibrationSample:
    image: str
    tile_index: int
    zone: str
    center: tuple[float, float]
    edge_distances: tuple[float, float, float, float]
    size: tuple[float, float]
    angle_deg: float
    nearest_neighbor: float
    orientation_probabilities: tuple[float, float, float, float] | None = None
    orientation_margin: float | None = None

    def __post_init__(self) -> None:
        # Require a basename-only image ID, non-negative tile index, canonical zone,
        # finite geometry, normalized coordinates/distances, positive size, and
        # paired orientation fields. Reconstruct OrientationScore from probabilities
        # and require orientation_margin to equal its computed margin.
        ...


@dataclass(frozen=True)
class FeatureDistribution:
    median: float
    mad: float
    low: float
    high: float


@dataclass(frozen=True)
class ZonePrior:
    center_x: FeatureDistribution
    center_y: FeatureDistribution
    edge_distance: FeatureDistribution
    width: FeatureDistribution
    height: FeatureDistribution
    nearest_neighbor: FeatureDistribution
    angle_deg: FeatureDistribution
    orientation_probabilities: tuple[FeatureDistribution, FeatureDistribution, FeatureDistribution, FeatureDistribution] | None
    orientation_margin: FeatureDistribution | None
    core_polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ZoneProfile:
    schema_version: int
    calibration_id: str
    calibration_images: tuple[str, ...]
    calibration_digest: str
    zones: Mapping[str, ZonePrior]
    weights: Mapping[str, float]
    solver: Mapping[str, float]
    orientation_provenance: OrientationProvenance | None = None
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "ZoneProfile": ...


class ZoneCalibrator:
    def fit(self, samples: Sequence[CalibrationSample], calibration_id: str) -> ZoneProfile: ...
```

Require at least one sample for each of the five canonical zones; fail instead of inventing priors for a missing zone. Use median, median absolute deviation with a `1e-4` floor, and 5th/95th percentiles. Build `core_polygon` with `cv2.convexHull`; for one or two samples use a clipped rectangle expanded by `0.02`. Compute a SHA-256 digest over sorted image names and serialized samples. Reject unknown zones and empty calibration IDs.

Treat the profile JSON as a strict schema: require exactly the five canonical zone keys; reject missing or unknown top-level/nested fields, NaN/Infinity, invalid hashes, absent weight/solver keys, unknown keys, negative weights, and solver values outside declared ranges. Define exactly these weight keys and defaults/ranges: `center=1.0 [0,10]`, `edge=1.0 [0,10]`, `size=0.25 [0,10]`, `nearest_neighbor=0.5 [0,10]`, `angle=0.25 [0,10]`, `polygon=1.0 [0,10]`, `orientation=0.0 [0,10]`. Define exactly these solver keys and defaults/ranges: `mismatch_penalty=0.05 [0,2]`, `meld_multiplier=2.0 [1,4]`, `top_band=0.35 [0,1]`, `bottom_band=0.65 [0,1]`, `side_band=0.35 [0,1]`. Require `top_band < bottom_band`. Deep-freeze loaded `zones`, `weights`, and `solver` with `MappingProxyType`; every nested collection is a tuple or frozen dataclass. If any orientation prior exists, require all five zones to contain it and require matching `orientation_provenance`; otherwise require provenance to be `null`. Add tests for each rejection and for mutation attempts raising `TypeError`.

- [ ] **Step 4: Implement an explicit-manifest calibration CLI**

`calibrate_zones.py` accepts geometry-only calibration or an explicit orientation artifact:

```powershell
python scripts/calibrate_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images z01.jpg,z02.jpg,z03.jpg --calibration-id current-v1 --output configs/zones/current_v1.json
python scripts/calibrate_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images z01.jpg,z02.jpg,z03.jpg --calibration-id current-v1 --orientation-scores ../output/orientation_scores_train.json --orientation-model ../output/cls_final_v2/best.onnx --preprocessing-version cls-v1 --scorer-version entropy-v1 --output configs/zones/current_v1.json
```

It must:

1. Require an explicit comma-separated image allow-list.
2. Reject duplicate/missing image names.
3. Require `table_corners` on every selected image.
4. Normalize all selected boxes with `TableNormalizer`.
5. Derive nearest-neighbor distance in normalized coordinates.
6. Fit and save the profile.
7. When `--orientation-scores` is supplied, require exactly one score for every selected `(image, tile_index)`, reject scores from any unselected image, verify the artifact's model hash/preprocessing/scorer metadata against CLI inputs, and populate the sample orientation fields.
8. Print only selected calibration image names, sample counts by zone, profile path, and digest.

It must never accept a test manifest or infer “all remaining images.” Geometry-only mode must write `null` orientation priors/provenance.

- [ ] **Step 5: Fit a development profile and verify serialization**

Use an explicit image-level calibration split, initially odd-numbered images:

```powershell
python scripts/calibrate_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images z01.jpg,z03.jpg,z05.jpg,z07.jpg,z09.jpg,z11.jpg,z13.jpg,z15.jpg,z17.jpg,z19.jpg --calibration-id current-v1 --output configs/zones/current_v1.json
python -c "from mahjong_rt.zone_calibration import ZoneProfile; p=ZoneProfile.load('configs/zones/current_v1.json'); print(p.calibration_id, len(p.calibration_images))"
```

Expected: `current-v1 10`.

- [ ] **Step 6: Run profile tests**

```powershell
python -m pytest tests/test_zone_calibration.py tests/test_table_geometry.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit calibration support**

```powershell
git add mahjong_rt/zone_calibration.py tests/test_zone_calibration.py scripts/calibrate_zones.py configs/zones/current_v1.json
git commit -m "feat(zones): fit versioned calibration profiles"
```

---

### Task 4: Calibrated geometric scorer and strict facade

**Files:**
- Create: `mahjong-rt/mahjong_rt/zone_solver.py`
- Create: `mahjong-rt/tests/test_zone_solver.py`
- Modify: `mahjong-rt/mahjong_rt/zones.py`
- Modify: `mahjong-rt/tests/test_zones.py`

- [ ] **Step 1: Write failing unary-cost and compatibility tests**

```python
from pathlib import Path

from mahjong_rt.events import Zone
from mahjong_rt.zone_calibration import ZoneProfile
from mahjong_rt.zone_solver import StructuredZoneSolver
from mahjong_rt.zone_types import TableGeometry, ZoneAnalysisContext
from mahjong_rt.zones import ZoneConfig, analyze_layout


def test_solver_prefers_nearest_calibrated_zone():
    profile = ZoneProfile.load(Path("configs/zones/current_v1.json"))
    solver = StructuredZoneSolver(profile)
    costs, evidence = solver.unary_costs(center=(0.5, 0.05), edge_distances=(0.5, 0.05, 0.5, 0.95), size=(0.05, 0.08), angle_deg=0.0, nearest_neighbor=0.1, orientation=None)
    assert min(costs, key=costs.get) == Zone.SEAT_ACROSS.value
    assert evidence[Zone.SEAT_ACROSS.value]


def test_legacy_call_stays_byte_for_byte_compatible():
    boxes = [[600, 600, 100, 100], [500, 300, 30, 30]]
    before = analyze_layout(boxes, 1280, 720, ZoneConfig())
    after = analyze_layout(boxes, 1280, 720, ZoneConfig(), context=None, profile=None)
    assert after == before


def test_strict_call_requires_profile():
    context = ZoneAnalysisContext(table=TableGeometry.unit_square(1280, 720))
    try:
        analyze_layout([[500, 300, 30, 30]], 1280, 720, ZoneConfig(mode="structured"), context=context)
    except ValueError as exc:
        assert "profile" in str(exc)
    else:
        raise AssertionError("strict structured mode accepted a missing profile")
```

- [ ] **Step 2: Verify focused tests fail**

```powershell
python -m pytest tests/test_zone_solver.py tests/test_zones.py -q
```

Expected: `StructuredZoneSolver` is missing and `ZoneConfig` does not accept `mode`.

- [ ] **Step 3: Implement deterministic unary costs and diagnostics**

`StructuredZoneSolver.unary_costs` must compute robust normalized distances against each `ZonePrior`:

```python
def robust_distance(value: float, distribution: FeatureDistribution) -> float:
    scale = max(distribution.mad * 1.4826, 1e-4)
    return min(abs(value - distribution.median) / scale, 12.0)
```

For each zone, combine profile weights for center, relevant table edge, size, nearest-neighbor, angle, and—when calibrated in Task 5—the distance between the observed four-way `OrientationScore.probabilities`/`margin` and that zone's orientation distributions. Add a polygon penalty of zero inside the core polygon and the Euclidean distance to its nearest edge outside. Return a per-zone evidence list naming the three smallest contributing terms. Before Task 5 populates orientation priors, orientation cost is exactly zero.

`solve_unary(layout, orientation_batch=None)` returns zones and one `TileZoneDiagnostic` per tile. Sort equal costs by this fixed order to guarantee determinism:

```python
ZONE_ORDER = ("my_hand", "seat_left", "seat_across", "seat_right", "river")
```

- [ ] **Step 4: Extend `zones.py` without breaking existing callers**

Extend `ZoneConfig` with:

```python
mode: str = "legacy"  # legacy | structured
strict_failure: bool = True
min_margin: float = 0.0
```

Change signatures to:

```python
def analyze_layout(
    boxes,
    frame_w,
    frame_h,
    config,
    *,
    context: ZoneAnalysisContext | None = None,
    profile: ZoneProfile | None = None,
) -> tuple[list[str], dict]: ...


def assign_zones(
    boxes,
    frame_w,
    frame_h,
    config,
    *,
    context: ZoneAnalysisContext | None = None,
    profile: ZoneProfile | None = None,
) -> list[str]: ...
```

Rules:

- `mode="legacy"` executes the existing implementation unchanged.
- `mode="structured"` requires `context` and `profile`.
- Validate context length before normalization.
- Normalization failure raises `ValueError` when `strict_failure=True`; otherwise return `unknown_zone` and diagnostics.
- Structured debug output contains `mode`, normalization quality, profile digest, and serialized per-tile diagnostics.
- Do not modify `events.py`.

- [ ] **Step 5: Run focused and full unit tests**

```powershell
python -m pytest tests/test_zone_solver.py tests/test_zones.py tests/test_table_geometry.py tests/test_zone_calibration.py -q
python -m pytest tests/ -q
```

Expected: all tests pass and existing legacy assertions are unchanged.

- [ ] **Step 6: Commit calibrated geometric classification**

```powershell
git add mahjong_rt/zone_solver.py mahjong_rt/zones.py tests/test_zone_solver.py tests/test_zones.py
git commit -m "feat(zones): add calibrated geometric solver"
```

---

### Task 5: Measure and expose four-rotation orientation evidence

**Files:**
- Create: `mahjong-rt/mahjong_rt/tile_orientation.py`
- Create: `mahjong-rt/tests/test_tile_orientation.py`
- Create: `mahjong-rt/scripts/eval_zone_orientation.py`
- Modify: `mahjong-rt/mahjong_rt/pipeline.py`
- Modify: `mahjong-rt/mahjong_rt/zone_calibration.py`
- Modify: `mahjong-rt/scripts/calibrate_zones.py`
- Modify: `mahjong-rt/configs/zones/current_v1.json`

- [ ] **Step 1: Write failing orientation tests with a fake classifier**

```python
import numpy as np

from mahjong_rt.tile_orientation import TileOrientationEstimator


class FakeClassifier:
    def predict_probabilities(self, patches):
        rows = []
        for patch in patches:
            marker = int(patch[0, 0, 0])
            rows.append([0.9, 0.1] if marker == 7 else [0.55, 0.45])
        return np.asarray(rows, np.float32)


def test_estimator_batches_four_rotations_per_crop():
    crop = np.zeros((8, 12, 3), np.uint8)
    crop[0, 0, 0] = 7
    estimator = TileOrientationEstimator(FakeClassifier(), OrientationProvenance("a" * 64, "test", "entropy-v1"))
    result = estimator.score([crop])
    assert len(result.scores) == 1
    assert result.scores[0].best_rotation == 0
    assert result.scores[0].margin > 0
    assert result.provenance == estimator.provenance


def test_empty_crops_return_empty_scores():
    estimator = TileOrientationEstimator(FakeClassifier(), OrientationProvenance("a" * 64, "test", "entropy-v1"))
    assert estimator.score([]).scores == ()
```

- [ ] **Step 2: Verify tests fail**

```powershell
python -m pytest tests/test_tile_orientation.py -q
```

Expected: import failure for `mahjong_rt.tile_orientation`.

- [ ] **Step 3: Implement a reusable probability adapter and orientation scorer**

Define:

```python
class ProbabilityClassifier(Protocol):
    def predict_probabilities(self, patches: Sequence[np.ndarray]) -> np.ndarray: ...


class TileOrientationEstimator:
    def __init__(self, classifier: ProbabilityClassifier, provenance: OrientationProvenance) -> None: ...
    @property
    def provenance(self) -> OrientationProvenance: ...
    def score(self, crops: Sequence[np.ndarray]) -> OrientationBatch: ...
```

For each crop, generate rotations with `np.rot90(crop, k)` for `k=0..3`. Batch all rotations in one classifier call. For each rotation compute:

```python
rotation_score = log(max_class_probability + 1e-8) - normalized_entropy(probabilities)
```

Convert the four entropy-adjusted logits with `OrientationScore.from_logits`, wrap all scores in `OrientationBatch`, and copy the estimator's immutable provenance into the result. Keep orientation independent from the predicted tile class. `StructuredZoneSolver` must reject a non-null batch unless its provenance exactly equals `ZoneProfile.orientation_provenance`; add focused tests for mismatched model hash, preprocessing version, and scorer version.

Extract the ONNX preprocessing and probability computation currently in `Pipeline._classify` into a private adapter implementing `predict_probabilities`; keep `_classify` output identical by constructing `Observation` from those probabilities.

- [ ] **Step 4: Implement orientation signal evaluation before enabling its weight**

`eval_zone_orientation.py` loads images, GT boxes/classes/zones, and classifier weights. It writes a reusable `--scores-output` artifact with `schema_version=1`, `labels_sha256`, `boxes_sha256`, `model_sha256`, `training_images` as a sorted basename-only list and `training_manifest_sha256` over its canonical newline-joined UTF-8 bytes (both required when orientation is used for an OOF claim), `preprocessing_version`, `scorer_version`, and a `records` list. Each record contains basename-only `image`, non-negative `tile_index`, the canonicalized source `xywh` rounded to six decimals as `box_identity`, four normalized finite `probabilities`, and computed `margin`. Records are sorted by `(image, tile_index)`; duplicates, missing selected boxes, extra boxes, digest mismatch, unknown fields, or non-canonical JSON are rejected. Canonical serialization is UTF-8 JSON with sorted keys, compact separators, and `allow_nan=False`. It reports:

- best rotation distribution per zone;
- median orientation margin per zone;
- four-class orientation consistency within each stable player row;
- `seat_across` versus `river` binary AUC for each rotation score;
- image-level five-fold performance of geometry-only versus geometry-plus-orientation, fitting the orientation weight on training folds only.

Run:

```powershell
python scripts/eval_zone_orientation.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --weights ../output/cls_final_v2/best.onnx --profile configs/zones/current_v1.json --scores-output ../output/orientation_scores_all.json --output ../output/zone_orientation_report.json
```

Acceptance rule: for each CV fold, filter the score artifact to that fold's training images, call `ZoneCalibrator.fit` with those orientation samples, and evaluate only the held-out images. Set the default orientation weight above zero only if held-out image-level CV improves and `seat_across` recall does not regress. After acceptance, regenerate the declared development profile with `calibrate_zones.py --orientation-scores ...`; its `ZonePrior` entries must contain four probability distributions and an orientation-margin distribution, and runtime must reject an estimator whose model hash/preprocessing/scorer versions differ from profile provenance. Otherwise keep those fields/provenance `null`, retain weight `0.0`, and record that the existing classifier does not provide a reliable new signal; do not tune on the final test split.

- [ ] **Step 5: Run orientation and pipeline regression tests**

```powershell
python -m pytest tests/test_tile_orientation.py tests/test_zone_solver.py tests/test_zones.py -q
python -m pytest tests/ -q
```

Expected: all tests pass; `_classify` still produces the same labels/confidences for a fixed probability matrix.

- [ ] **Step 6: Commit orientation evidence support**

```powershell
git add mahjong_rt/tile_orientation.py mahjong_rt/pipeline.py mahjong_rt/zone_calibration.py tests/test_tile_orientation.py scripts/eval_zone_orientation.py scripts/calibrate_zones.py configs/zones/current_v1.json
git commit -m "feat(zones): calibrate tile orientation evidence"
```

---

### Task 6: Constrained layout graph without single-link chaining

**Files:**
- Create: `mahjong-rt/mahjong_rt/layout_graph.py`
- Create: `mahjong-rt/tests/test_layout_graph.py`

- [ ] **Step 1: Write failing anti-chaining and meld tests**

```python
from mahjong_rt.layout_graph import LayoutGraphBuilder
from mahjong_rt.table_geometry import NormalizedTile


def tile(x, y, angle=0.0):
    return NormalizedTile(
        corners=((x-.02,y-.03),(x+.02,y-.03),(x+.02,y+.03),(x-.02,y+.03)),
        center=(x, y), width=.04, height=.06, angle_deg=angle,
        edge_distances=(x, y, 1-x, 1-y),
    )


def test_bridge_tiles_do_not_chain_across_group_into_river():
    across = [tile(.40, .12), tile(.46, .12), tile(.52, .12)]
    bridge = [tile(.55, .18), tile(.58, .24), tile(.61, .30)]
    groups = LayoutGraphBuilder(max_group_span=.22).build(across + bridge)
    assert max(len(group.members) for group in groups) < 6
    assert any(group.members == (0, 1, 2) for group in groups)


def test_three_equal_classes_form_meld_candidate():
    groups = LayoutGraphBuilder().build(
        [tile(.3,.2), tile(.35,.2), tile(.4,.2)],
        classes=("w1", "w1", "w1"),
    )
    assert any(group.kind == "meld" and group.members == (0,1,2) for group in groups)
```

- [ ] **Step 2: Verify tests fail**

```powershell
python -m pytest tests/test_layout_graph.py -q
```

Expected: import failure for `mahjong_rt.layout_graph`.

- [ ] **Step 3: Implement constrained adjacency and maximal groups**

Define:

```python
@dataclass(frozen=True)
class TileGroup:
    members: tuple[int, ...]
    kind: str  # row | column | meld
    axis_deg: float
    span: float
    consistency: float
    centroid: tuple[float, float]


class LayoutGraphBuilder:
    def __init__(self, max_gap=0.09, max_angle_diff=18.0, max_size_ratio=1.8, max_group_span=0.35): ...
    def build(self, tiles, classes=None, orientations=None) -> tuple[TileGroup, ...]: ...
```

Create candidate edges only when gap, orientation, size ratio, and projected row/column offset all pass. Grow a group only while:

- total span remains below `max_group_span`;
- every new node is compatible with the group's fitted axis;
- no internal projected gap exceeds `max_gap`;
- median size ratio remains below `max_size_ratio`.

Enumerate seeds deterministically by tile index and sort final groups by `(-len(members), members)`. Mark 3–4 equal-class aligned groups as `meld`. Do not merge connected components solely because a bridge edge exists.

- [ ] **Step 4: Add invariance tests under input-preserving transforms**

Add tests that translate/scale all normalized tile geometry and assert identical group membership. Add a shuffled-input test that maps indices back to the original order and asserts identical groups.

- [ ] **Step 5: Run graph and geometry tests**

```powershell
python -m pytest tests/test_layout_graph.py tests/test_table_geometry.py -q
```

Expected: all tests pass, including the explicit six-tile bridge case.

- [ ] **Step 6: Commit constrained grouping**

```powershell
git add mahjong_rt/layout_graph.py tests/test_layout_graph.py
git commit -m "feat(zones): build constrained tile layout groups"
```

---

### Task 7: Deterministic structured global solver

**Files:**
- Modify: `mahjong-rt/mahjong_rt/zone_solver.py`
- Modify: `mahjong-rt/mahjong_rt/zones.py`
- Modify: `mahjong-rt/tests/test_zone_solver.py`
- Create: `mahjong-rt/tests/test_zone_pipeline.py`

- [ ] **Step 1: Write failing group-consistency and margin tests**

```python
from mahjong_rt.layout_graph import TileGroup
from mahjong_rt.zone_solver import StructuredZoneSolver


def test_group_consistency_rescues_one_boundary_tile(profile, normalized_layout):
    solver = StructuredZoneSolver(profile)
    unary = [
        {"seat_across": .1, "river": .8},
        {"seat_across": .2, "river": .7},
        {"seat_across": .55, "river": .50},
    ]
    result = solver.solve_costs(unary, (TileGroup((0,1,2), "row", 0.0, .15, .95, (.5, .1)),))
    assert result.zones == ("seat_across", "seat_across", "seat_across")


def test_river_group_is_not_forced_to_one_player_zone(profile):
    solver = StructuredZoneSolver(profile)
    unary = [
        {"seat_across": .45, "river": .10},
        {"seat_across": .10, "river": .45},
    ]
    result = solver.solve_costs(unary, ())
    assert result.zones == ("river", "seat_across")


def test_diagnostics_report_best_second_and_margin(profile):
    result = StructuredZoneSolver(profile).solve_costs([{"river": .2, "seat_across": .6}], ())
    diagnostic = result.diagnostics[0]
    assert diagnostic.best_cost == .2
    assert diagnostic.second_cost == .6
    assert diagnostic.margin == .4
```

Fixtures load a temporary minimal `ZoneProfile`; they must not depend on the generated repository profile.

- [ ] **Step 2: Verify the structured tests fail**

```powershell
python -m pytest tests/test_zone_solver.py -q
```

Expected: `solve_costs` or the expected structured result type is missing.

- [ ] **Step 3: Implement exact group-label enumeration**

Define:

```python
@dataclass(frozen=True)
class StructuredSolveResult:
    zones: tuple[str, ...]
    diagnostics: tuple[TileZoneDiagnostic, ...]
    total_cost: float
```

For each non-overlapping candidate group, enumerate all five labels per member (at most \(5^4=625\) assignments for a four-tile group) and minimize unary plus a split penalty. Missing zone keys in synthetic/test unary dictionaries have cost `math.inf`; production unary dictionaries must contain exactly all five canonical zones. Store `mismatch_penalty=0.05` in the required solver schema with an allowed range `[0.0, 2.0]`, and construct the boundary-rescue fixture with that explicit value:

```python
multiplier = 2.0 if group.kind == "meld" else 1.0
pairwise_split_cost = multiplier * mismatch_penalty * sum(
    1 for i, j in combinations(group.members, 2) if labels[i] != labels[j]
)
group_cost = sum(unary[i][labels[i]] for i in group.members) + pairwise_split_cost
```

This penalizes splitting a coherent group rather than penalizing the unified candidate, so the boundary-tile rescue test is mathematically achievable. The solver reports both unary and split-cost components.

Apply these explicit constraints:

- `row` near the top edge may use `seat_across` or `river`.
- `row` near the bottom edge may use `my_hand` or `river`.
- left/right edge-aligned groups may use the corresponding side seat or `river`.
- `meld` uses `2.0 * mismatch_penalty` for every unequal label pair, making the multiplier explicit and testable.
- A group whose admissible label set is only `river` skips pairwise penalties entirely. Otherwise pairwise penalties apply symmetrically to unequal labels, including river/player splits; document this as a split penalty rather than claiming that unified `river` receives no relative benefit.
- Overlapping candidate groups are processed in deterministic priority order and a tile may be claimed by only one accepted group.

Do not introduce a general optimizer dependency. The number of labels per local group is five, so exact local enumeration is sufficient and auditable.

- [ ] **Step 4: Connect layout groups and orientation to the strict facade**

In structured `analyze_layout`:

1. Normalize boxes.
2. Build groups using classes and orientation scores from `ZoneAnalysisContext`.
3. Calculate unary costs.
4. Run `solve_costs`.
5. If any margin is below `config.min_margin`, set `ambiguous=True` in diagnostics. In strict release evaluation, keep the chosen label so coverage remains measurable; in non-strict development mode allow `unknown_zone`.
6. Return group membership, total cost, and tile diagnostics in debug output.

- [ ] **Step 5: Add end-to-end synthetic layout tests**

`tests/test_zone_pipeline.py` creates a trapezoidal table with five unambiguous groups and verifies:

- all five zones appear;
- perspective-warping the table and boxes leaves all labels unchanged;
- shuffling box order only shuffles corresponding outputs;
- strict mode raises on invalid table geometry;
- non-strict mode returns `unknown_zone` plus failure diagnostics.

- [ ] **Step 6: Run all zone tests**

```powershell
python -m pytest tests/test_zone_solver.py tests/test_layout_graph.py tests/test_zone_pipeline.py tests/test_zones.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the structured solver**

```powershell
git add mahjong_rt/zone_solver.py mahjong_rt/zones.py tests/test_zone_solver.py tests/test_zone_pipeline.py
git commit -m "feat(zones): solve zone labels with layout constraints"
```

---

### Task 8: Immutable evaluation, CV, and 100% release report

**Files:**
- Create: `mahjong-rt/scripts/eval_zones.py`
- Create: `mahjong-rt/tests/test_eval_zones.py`
- Modify: `mahjong-rt/tests/test_zones.py`

- [ ] **Step 1: Write failing evaluator tests**

```python
import json
from pathlib import Path

import pytest

from scripts.eval_zones import evaluate_predictions, validate_disjoint_images


def test_split_validation_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_images({"a.jpg"}, {"a.jpg", "b.jpg"})


def test_cross_validation_requires_each_image_once():
    folds = ({"a.jpg", "b.jpg"}, {"b.jpg", "c.jpg"})
    with pytest.raises(ValueError, match="exactly once"):
        validate_cross_validation_folds(folds, {"a.jpg", "b.jpg", "c.jpg"})


def test_unknown_counts_as_wrong_and_uncovered():
    report = evaluate_predictions(
        truths=["river", "seat_across"],
        predictions=["river", "unknown_zone"],
        records=[("a.jpg", 0), ("a.jpg", 1)],
        diagnostics=[{}, {"failure": "low_margin"}],
    )
    assert report["correct"] == 1
    assert report["total"] == 2
    assert report["accuracy"] == .5
    assert report["coverage"] == .5
    assert report["passed"] is False


def test_perfect_report_requires_every_zone_recall_present():
    report = evaluate_predictions(
        truths=["my_hand", "river", "seat_left", "seat_across", "seat_right"],
        predictions=["my_hand", "river", "seat_left", "seat_across", "seat_right"],
        records=[("a.jpg", i) for i in range(5)],
        diagnostics=[{} for _ in range(5)],
    )
    assert report["passed"] is True
```

- [ ] **Step 2: Verify evaluator tests fail**

```powershell
python -m pytest tests/test_eval_zones.py -q
```

Expected: import failure for `scripts.eval_zones`.

- [ ] **Step 3: Implement evaluator and actionable failure output**

The CLI is:

```powershell
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --profile configs/zones/current_v1.json --test-images z02.jpg,z04.jpg,z06.jpg,z08.jpg,z10.jpg,z12.jpg,z14.jpg,z16.jpg,z18.jpg,z20.jpg --report ../output/zone_eval_current_v1.json --require-perfect
```

Requirements:

- In fixed-profile mode, never fit or mutate a profile. In `--cross-validate` mode, call the pure `ZoneCalibrator.fit` separately inside each fold and persist its provenance; never overwrite the supplied or repository profile.
- Verify calibration images from the profile are disjoint from `--test-images`.
- Require `table_corners` for each evaluated image.
- Run `analyze_layout` in structured strict mode.
- Report correct/total, accuracy, coverage, confusion matrix, per-zone recall, per-image accuracy, `seat_across` errors, and all failures.
- Each error includes image, box index, GT, prediction, best/second costs, margin, evidence, normalized features, group membership, and failure category.
- `--require-perfect` exits `0` only when accuracy=1, coverage=1, and recall=1 for every zone present in GT; otherwise exit `1`.
- Outside CV mode, exactly one of `--test-images` and `--test-manifest` is required. `--test-manifest` must be non-empty and contain one basename per non-empty line; reject absolute paths, path separators, `.`/`..`, and duplicates.
- Require every normalized manifest entry to appear exactly once in labels, require labels basenames to be globally unique, and require its image file to exist and decode successfully. Missing, unresolved, or multiply matched entries abort the entire evaluation; no entry may be skipped.
- Compute and report both SHA-256 of the original manifest bytes and SHA-256 of the canonical UTF-8 newline-joined normalized basename list; persist that normalized list in the report.
- A result is labeled `profile_out_of_fold` only when every dataset image appears in exactly one held-out fold and no held-out image appears in that fold's fitted profile provenance. It may be promoted to `full_out_of_fold` only when the report additionally proves that frozen algorithm/grid/locator versions and every learned component's training provenance exclude all corresponding held-out images. A fixed-profile subset result is labeled `holdout_subset`; a profile evaluated on any provenance image is labeled `diagnostic_leaky` and can never pass a release gate.
- Store CLI arguments, profile digest, labels SHA-256, git commit, and UTC timestamp in the JSON report.
- Never fit or mutate a profile in fixed-profile mode; CV fold fitting is the sole exception and must produce in-memory fold-local profiles with recorded digests.

- [ ] **Step 4: Implement honest image-level five-fold development evaluation**

Add optional `--cross-validate 5 --seed 0`. In each fold:

1. Fit a fresh profile from four folds only.
2. If orientation weight is enabled, require an orientation artifact; verify its labels/boxes/model digests, filter scores to training images for prior fitting, and pass held-out scores only at prediction time. Validate the canonical `training_images` list against `training_manifest_sha256` and require its set to be disjoint from held-out zone images; otherwise mark the fold `profile_oof_only`, not eligible for the 899/899 OOF gate.
3. Select any solver weights from a finite declared grid on those training images only.
4. Evaluate the held-out images once.
5. Aggregate predictions across folds before calculating overall and per-zone metrics.

Print both mean fold accuracy and aggregate box accuracy. Persist every fold's training image names, held-out names, fitted profile digest, selected weights, orientation provenance, and predictions. Validate that the union of held-out images equals all 20 images and that held-out folds are pairwise disjoint. Because these 20 images have already informed algorithm design, label review, locator thresholds, and the candidate grid, name this report `profile_out_of_fold`, not a statistically untouched generalization result. It is the engineering current-set 899/899 gate only; the genuinely sealed new-scene gate remains the sole untouched generalization claim. Keep the fixed-profile holdout result separate and report its actual held-out box count rather than calling it 899/899.

- [ ] **Step 5: Replace the misleading loose strict regression**

Keep `test_accuracy_on_labelled_set` as the legacy `>= 0.92` compatibility test. Add a new marked integration test that invokes the evaluator only when corners and `configs/zones/current_v1.json` exist:

```python
@pytest.mark.integration
def test_structured_release_gate_on_current_set():
    report = run_current_structured_cross_validation()
    assert report["evaluation_kind"] in {"profile_out_of_fold", "full_out_of_fold"}
    assert report["total"] == 899
    assert report["correct"] == 899, format_failures(report)
    assert report["coverage"] == 1.0
```

Do not weaken or skip this test once the corner annotations and profile are committed.

- [ ] **Step 6: Run evaluator tests and current held-out split**

```powershell
python -m pytest tests/test_eval_zones.py tests/test_zones.py -q
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --profile configs/zones/current_v1.json --test-images z02.jpg,z04.jpg,z06.jpg,z08.jpg,z10.jpg,z12.jpg,z14.jpg,z16.jpg,z18.jpg,z20.jpg --report ../output/zone_eval_current_v1.json
```

Expected: tests pass. The evaluation command always writes a report; it may expose errors at this stage and must not be described as a 100% pass unless its JSON says `passed: true`.

- [ ] **Step 7: Run cross-validation and classify every remaining error**

```powershell
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --cross-validate 5 --seed 0 --report ../output/zone_eval_cv_v1.json
```

For each error, assign exactly one category from the design: table localization, perspective mapping, annotation ambiguity, across/river overlap, orientation evidence, grouping, solver constraint, or bad input box. Record category counts in the report rather than adding data-specific hard-coded exceptions.

- [ ] **Step 8: Commit the immutable evaluator**

```powershell
git add scripts/eval_zones.py tests/test_eval_zones.py tests/test_zones.py
git commit -m "test(zones): add sealed perfect-accuracy evaluator"
```

---

### Task 9: Runtime configuration and explicit failure behavior

**Files:**
- Modify: `mahjong-rt/mahjong_rt/pipeline.py`
- Modify: `mahjong-rt/mahjong_rt/replay.py`
- Modify: `mahjong-rt/configs/pipeline.yaml`
- Modify: `mahjong-rt/tests/test_zone_pipeline.py`

- [ ] **Step 1: Write failing configuration and replay tests**

```python
import pytest

from mahjong_rt.pipeline import PipelineConfig
from mahjong_rt.replay import replay


def test_structured_config_requires_profile_path():
    with pytest.raises(ValueError, match="profile"):
        PipelineConfig(det_weights="d.pt", cls_weights="c.onnx", zones={"mode": "structured"}).validate()


def test_replay_rejects_structured_mode_without_table_geometry(recording):
    with pytest.raises(ValueError, match="table geometry"):
        replay(recording, zones_cfg={"mode": "structured", "profile_path": "configs/zones/current_v1.json"})
```

- [ ] **Step 2: Verify focused tests fail**

```powershell
python -m pytest tests/test_zone_pipeline.py -q
```

Expected: `PipelineConfig.validate` is missing or strict replay does not reject missing geometry.

- [ ] **Step 3: Add explicit structured configuration**

Add these keys under `zones` in `configs/pipeline.yaml`:

```yaml
zones:
  mode: legacy
  strict_failure: true
  min_margin: 0.0
  profile_path: null
  table_locator:
    mode: annotated  # annotated | automatic
    min_confidence: 0.95
    min_area_ratio: 0.08
    max_homography_condition: 1000000.0
```

The default remains `legacy` because the existing realtime video pipeline does not provide per-frame corner annotations and its event contract must not change accidentally. Static evaluation explicitly selects structured mode.

Add `PipelineConfig.validate()` and call it before model loading. Resolve `profile_path` relative to the YAML file in CLI loaders, not the process working directory. Load a `ZoneProfile` once during `Pipeline` initialization.

- [ ] **Step 4: Wire context into pipeline only when available**

Add a `table_geometry_provider` dependency to `Pipeline.__init__`:

```python
def __init__(self, config: PipelineConfig, table_geometry_provider: Callable[[np.ndarray], TableGeometry] | None = None) -> None:
```

In structured mode:

1. Require the provider.
2. Obtain table geometry for the frame.
3. Reuse current crops and compute orientation scores when profile orientation weight is nonzero.
4. Build `ZoneAnalysisContext` with table, classes, and orientations.
5. Call `assign_zones(..., context=context, profile=self.zone_profile)`.
6. Propagate failures in strict mode; never silently call legacy mode.

Keep all published event fields and protocol version unchanged.

- [ ] **Step 5: Make replay limitations explicit**

Because current recordings contain boxes, classes, probabilities, and GMC homographies but no table corners or source frame, `replay` must raise a clear error for structured mode. Legacy replay remains unchanged. A future recording schema extension is outside this static-image scope.

- [ ] **Step 6: Run all tests**

```powershell
python -m pytest tests/ -q
```

Expected: all tests pass, including existing tracker, voter, metrics, zones, and replay behavior.

- [ ] **Step 7: Commit runtime wiring**

```powershell
git add mahjong_rt/pipeline.py mahjong_rt/replay.py configs/pipeline.yaml tests/test_zone_pipeline.py
git commit -m "feat(zones): wire strict calibrated mode explicitly"
```

---

### Task 10: Automatic table-border locator and final release gates

**Files:**
- Modify: `mahjong-rt/mahjong_rt/table_geometry.py`
- Modify: `mahjong-rt/tests/test_table_geometry.py`
- Modify: `mahjong-rt/scripts/eval_zones.py`
- Create: `mahjong-rt/tests/fixtures/table_locator/expected.json`

- [ ] **Step 1: Write failing locator confidence tests**

```python
import cv2
import numpy as np

from mahjong_rt.table_geometry import TableBorderLocator


def synthetic_table():
    image = np.zeros((720, 1280, 3), np.uint8)
    polygon = np.asarray([[120,620],[260,90],[1030,110],[1170,630]], np.int32)
    cv2.fillConvexPoly(image, polygon, (55, 115, 70))
    cv2.polylines(image, [polygon], True, (210, 210, 210), 8)
    return image


def test_locator_finds_four_ordered_corners():
    result = TableBorderLocator().locate(synthetic_table())
    assert result.geometry.corners.shape == (4, 2)
    assert result.geometry.confidence >= .95
    assert result.failures == ()


def test_locator_rejects_blank_image():
    result = TableBorderLocator().locate(np.zeros((720,1280,3), np.uint8))
    assert result.geometry is None
    assert "no_table_quadrilateral" in result.failures
```

- [ ] **Step 2: Verify tests fail**

```powershell
python -m pytest tests/test_table_geometry.py -q
```

Expected: `TableBorderLocator` is missing.

- [ ] **Step 3: Implement an explainable OpenCV locator**

Implement this deterministic pipeline:

1. Downscale only for detection and retain scale factors.
2. Convert to HSV and Lab.
3. Build table-surface candidates from saturation and local color consistency.
4. Apply morphological close/open with kernels proportional to image size.
5. Find external contours.
6. Approximate convex quadrilaterals with `cv2.approxPolyDP`.
7. Score candidates by area ratio, convexity, edge support from Canny, opposite-edge consistency, and percentage of tile centers inside when boxes are supplied.
8. Refine each edge using `cv2.fitLine` on nearby edge pixels and intersect adjacent lines.
9. Return the best valid `TableGeometry` plus component scores and failures.

Confidence is a weighted score with weights fixed in configuration. Do not fit locator thresholds on sealed test labels.

- [ ] **Step 4: Add real-image locator fixtures**

Select at least one image from each distinct source/camera represented in the 20-image set. Store only image names and expected annotated corners in `tests/fixtures/table_locator/expected.json`; reuse the existing images rather than copying binaries. Assert:

- all fixtures return a valid quadrilateral;
- mean corner error is at most 2% of image diagonal;
- every GT tile center lies within the located table or an explicitly configured edge tolerance;
- any confidence below `0.95` fails the test.

- [ ] **Step 5: Compare automatic and annotated geometry**

Run the evaluator twice on the development split:

```powershell
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --profile configs/zones/current_v1.json --table-source annotated --test-images z02.jpg,z04.jpg,z06.jpg,z08.jpg,z10.jpg,z12.jpg,z14.jpg,z16.jpg,z18.jpg,z20.jpg --report ../output/zone_eval_annotated.json
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --profile configs/zones/current_v1.json --table-source automatic --test-images z02.jpg,z04.jpg,z06.jpg,z08.jpg,z10.jpg,z12.jpg,z14.jpg,z16.jpg,z18.jpg,z20.jpg --report ../output/zone_eval_automatic.json
```

Any label difference must be attributed to a measured corner displacement or normalization failure. Do not compensate with per-image zone exceptions.

- [ ] **Step 6: Run current-set release gate**

After error analysis and only generic algorithm/profile changes, fit the declared production calibration set and run the disjoint current test split with `--require-perfect`:

```powershell
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --profile configs/zones/current_v1.json --table-source automatic --test-images z02.jpg,z04.jpg,z06.jpg,z08.jpg,z10.jpg,z12.jpg,z14.jpg,z16.jpg,z18.jpg,z20.jpg --report ../output/zone_eval_release.json --require-perfect
```

Expected for this holdout diagnostic: exit code `0`, `correct == total`, `coverage == 1.0`, and each present zone recall equals `1.0`. The report must be marked `holdout_subset` and show its actual number of boxes; it does not by itself satisfy the current-set 899/899 gate. Then run the five-fold command from Task 8 with `--require-perfect`; only a `profile_out_of_fold` or stronger `full_out_of_fold` report with `correct=total=899` satisfies the engineering current-set gate. Neither substitutes for the sealed new-scene generalization gate. If either check fails, retain the reports and return to the owning module based on the error category; do not mark this step complete.

- [ ] **Step 7: Run a genuinely sealed new-scene gate**

Before revealing any results, create and freeze a new-scene inventory containing the exact calibration and sealed test basenames, exact expected image count, exact expected GT box count, labels SHA-256, canonical test-manifest SHA-256, image-content SHA-256 values, frozen algorithm/profile/model hashes, and an independent reviewer/signature field. The inventory itself must be committed or stored in an append-only location before evaluation. Use 20–30 calibration images and 30–50 sealed test images from a new camera configuration, then run:

```powershell
python scripts/eval_zones.py --labels <sealed-labels.json> --images <sealed-images-dir> --profile <frozen-profile.json> --table-source automatic --sealed-inventory <frozen-inventory.json> --test-manifest <sealed-test-images.txt> --report <sealed-report.json> --require-perfect
```

The evaluator must require exact equality between the normalized manifest and the frozen inventory's sealed-test set, and exact matches for expected image count, expected box count, labels hash, every image hash, algorithm/profile/model hashes, and precommitted manifest hash. Any missing or extra entry aborts before inference. Expected for release: exit code `0` and N/N. If labels are opened to diagnose a failure, retire that sealed set, move it to development data, and collect a new sealed test set for the next claim.

- [ ] **Step 8: Run full verification**

```powershell
python -m pytest tests/ -q
python -m pytest tests/ -q -m integration
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --cross-validate 5 --seed 0 --report ../output/zone_eval_cv_final.json
```

Expected: all unit and integration tests pass. Record the CV metrics without describing them as a sealed-test guarantee.

- [ ] **Step 9: Commit only after both release gates pass**

```powershell
git add mahjong_rt/table_geometry.py tests/test_table_geometry.py tests/fixtures/table_locator/expected.json scripts/eval_zones.py configs/zones/current_v1.json
git commit -m "feat(zones): locate table borders for calibrated recognition"
```

---

## Plan self-review

### Spec coverage

- Perspective normalization and quality checks: Tasks 2 and 10.
- Independent calibration and versioned profile: Task 3.
- Four-rotation orientation evidence: Task 5.
- Non-chaining layout graph and meld evidence: Task 6.
- Structured global solving, margins, evidence, ambiguity: Tasks 4 and 7.
- Backward-compatible facade and explicit strict failure: Tasks 4 and 9.
- Annotation rules, legacy `opponent_wall` migration, and independent double-review audit: Task 2.
- Image-level isolation, cross-validation, accuracy/coverage, per-zone recall: Task 8.
- Current out-of-fold 899/899 and genuinely sealed N/N gates: Tasks 8 and 10.
- No video temporal logic, detector retraining, or embedded region head: preserved throughout.

### Type consistency

- `TableGeometry`, `OrientationScore`, `TileZoneDiagnostic`, and `ZoneAnalysisContext` originate in Task 1 and are reused unchanged.
- `NormalizedTile` and `NormalizedLayout` originate in Task 2.
- `ZoneProfile` originates in Task 3.
- `StructuredZoneSolver` begins with unary scoring in Task 4 and gains `solve_costs` in Task 7.
- `assign_zones` retains its original positional parameters; all new context is keyword-only.

### Dependency check

All algorithms use existing NumPy/OpenCV/ONNX Runtime dependencies. No new package is required.
