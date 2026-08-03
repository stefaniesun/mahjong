# Adaptive Tile-Group Zone Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pixel-threshold and single-link zone assignment with a deterministic, table-corner-free recognizer that derives tile rows, layout groups, player anchors, and final zones from each image's existing boxes and optional classes.

**Architecture:** Keep `mahjong_rt.zones` as the backward-compatible facade. Add focused modules for frame-relative tile relations, direction-constrained row extraction, semantic layout groups, adaptive anchors, and exact group-level assignment; every stage consumes only existing boxes, image dimensions, and optional classes. The old heuristic remains explicitly selectable, while the new `adaptive_groups` mode is evaluated without table corners, calibration profiles, external models, or per-image exceptions.

**Tech Stack:** Python 3.10+, NumPy, pytest, standard library; no new runtime dependency and no table-corner/homography input.

---

## Scope and non-negotiable constraints

This plan replaces the superseded perspective-normalization plan and implements `docs/superpowers/specs/2026-08-03-static-zone-recognition-design.md`.

- Own seat remains below, left seat left, across seat above, and right seat right.
- Camera position and perspective may vary, but that directional relationship remains stable.
- Production inputs are `xywh` boxes, frame dimensions, and optional existing tile classes.
- No table corners, table-border detector, homography, calibration profile, orientation model, or new manual annotation enters prediction.
- Scene quantities use per-image medians, robust spreads, and ranks; fixed values express only dimensionless algorithm safeguards or Mahjong structure.
- Existing `legacy` behavior remains unchanged for callers that do not select `adaptive_groups`.
- Existing experimental `table_geometry.py` and annotation tooling are left isolated; they are not imported by the adaptive path. Removing them is outside this implementation because deletion would erase already tested experimental work without benefiting runtime behavior.
- The five historical `opponent_wall` labels must be resolved under the canonical five-zone contract before a 899/899 claim. The existing independent-review tool remains the mechanism; no automatic migration is allowed.

## Release gates

Before Task 8 current-set evaluation, generate and commit `output/zone_test_manifest.txt` as the sorted list of all 20 unique `image` fields from the top-level annotation JSON array after canonical label review. Record its SHA-256 and assert exactly 20 records and 899 boxes. The manifest is an inventory, not a source of zones.

1. **Compatibility:** all existing legacy tests pass byte-for-byte.
2. **Structural correctness:** synthetic row, anti-chaining, group, anchor, missing-seat, and transform-invariance tests pass.
3. **Current set:** all canonical boxes receive predictions; accuracy, coverage, and every present-zone recall are 100%.
4. **Sealed set:** after code/config freeze, a newly sealed image set reaches N/N. Human labels act only as post-prediction truth and never as algorithm input.

No implementation task may claim 100% unless the evaluator report itself records a passing gate.

## File map

### New production files

- `mahjong-rt/mahjong_rt/tile_relations.py` — immutable frame-normalized tile features and pairwise relation matrix.
- `mahjong-rt/mahjong_rt/tile_lines.py` — deterministic direction-constrained candidate row extraction without single-link chaining.
- `mahjong-rt/mahjong_rt/layout_groups.py` — hand, meld, river-block, and singleton candidates.
- `mahjong-rt/mahjong_rt/layout_anchors.py` — robust layout center, directional player anchors, and inside/outside relationships.
- `mahjong-rt/mahjong_rt/adaptive_zone_solver.py` — group-level candidate costs, exact assignment, singleton attachment, and diagnostics.

### New scripts and tests

- `mahjong-rt/scripts/eval_zones.py` — immutable evaluator for current and sealed manifests whose records identify the unique `image` field of each annotation-array record; images are optional metadata, not inference input.
- `mahjong-rt/tests/test_tile_relations.py`
- `mahjong-rt/tests/test_tile_lines.py`
- `mahjong-rt/tests/test_layout_groups.py`
- `mahjong-rt/tests/test_layout_anchors.py`
- `mahjong-rt/tests/test_adaptive_zone_solver.py`
- `mahjong-rt/tests/test_adaptive_zone_pipeline.py`
- `mahjong-rt/tests/test_eval_zones.py`

### Modified files

- `mahjong-rt/mahjong_rt/zone_types.py` — remove `TableGeometry` as a required member of `ZoneAnalysisContext`; retain old geometry/orientation types only for experimental compatibility.
- `mahjong-rt/mahjong_rt/zones.py` — preserve legacy path and dispatch explicit `adaptive_groups` mode.
- `mahjong-rt/mahjong_rt/pipeline.py` — optionally pass already available classes; never request table geometry.
- `mahjong-rt/mahjong_rt/replay.py` — allow adaptive mode using recorded boxes/classes when present.
- `mahjong-rt/configs/pipeline.yaml` — expose only adaptive dimensionless safeguards; no table locator/profile keys.
- `mahjong-rt/tests/test_zone_types.py`
- `mahjong-rt/tests/test_zones.py`

---

### Task 1: Replace strict context with existing-input context

**Files:**
- Modify: `mahjong-rt/mahjong_rt/zone_types.py`
- Modify: `mahjong-rt/tests/test_zone_types.py`

- [ ] **Step 1: Write failing tests for a table-free context**

Add:

```python
def test_zone_context_does_not_require_table_geometry():
    context = ZoneAnalysisContext(classes=("w1", "w1", "w1"), strict=True)
    context.validate_for(3)
    assert context.classes == ("w1", "w1", "w1")


def test_zone_context_rejects_class_count_mismatch():
    context = ZoneAnalysisContext(classes=("w1",))
    with pytest.raises(ValueError, match="classes length"):
        context.validate_for(2)
```

Change existing context construction so it no longer supplies `table=`. Keep the independent tests for `TableGeometry` and orientation value objects because those experimental types still exist.

- [ ] **Step 2: Run the focused test and verify failure**

```powershell
python -m pytest tests/test_zone_types.py -q
```

Expected: FAIL because `ZoneAnalysisContext.table` is required.

- [ ] **Step 3: Make context table-free**

Replace only the context dataclass with:

```python
@dataclass(frozen=True)
class ZoneAnalysisContext:
    classes: tuple[str, ...] | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        if self.classes is not None:
            object.__setattr__(self, "classes", tuple(self.classes))

    def validate_for(self, box_count: int) -> None:
        if box_count < 0:
            raise ValueError("box count must be non-negative")
        if self.classes is not None and len(self.classes) != box_count:
            raise ValueError("classes length must match boxes")
```

Do not delete `TableGeometry`, `OrientationScore`, or `OrientationBatch`; they remain isolated experimental contracts and must not be imported by the adaptive path.

- [ ] **Step 4: Run contract and legacy tests**

```powershell
python -m pytest tests/test_zone_types.py tests/test_zones.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add mahjong_rt/zone_types.py tests/test_zone_types.py
git commit -m "refactor(zones): make structured context table-free"
```

---

### Task 2: Build per-image tile relations

**Files:**
- Create: `mahjong-rt/mahjong_rt/tile_relations.py`
- Create: `mahjong-rt/tests/test_tile_relations.py`

- [ ] **Step 1: Write failing relation and invariance tests**

```python
import numpy as np
import pytest

from mahjong_rt.tile_relations import build_tile_relations


def test_relations_use_frame_median_scale():
    boxes = [[100, 100, 20, 40], [130, 100, 20, 40], [200, 200, 40, 80]]
    result = build_tile_relations(boxes, 400, 300)
    assert result.scale == pytest.approx(20.0)
    assert result.tiles[0].center == pytest.approx((0.275, 0.4))
    assert result.gaps[0, 1] == pytest.approx(0.5)
    assert result.size_ratios[0, 2] == pytest.approx(0.5)


def test_translation_scale_and_resolution_preserve_relations():
    boxes = [[100, 100, 20, 40], [130, 100, 20, 40], [160, 100, 20, 40]]
    scaled = [[x * 2 + 50, y * 2 + 30, w * 2, h * 2] for x, y, w, h in boxes]
    a = build_tile_relations(boxes, 400, 300)
    b = build_tile_relations(scaled, 850, 630)
    np.testing.assert_allclose(a.gaps, b.gaps)
    np.testing.assert_allclose(a.size_ratios, b.size_ratios)
    np.testing.assert_allclose(a.axes_deg, b.axes_deg)


def test_invalid_boxes_are_rejected():
    with pytest.raises(ValueError, match="positive width and height"):
        build_tile_relations([[1, 2, 0, 4]], 100, 100)
```

- [ ] **Step 2: Verify missing module failure**

```powershell
python -m pytest tests/test_tile_relations.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement immutable relation types**

Create:

```python
@dataclass(frozen=True)
class TileFeature:
    index: int
    center: tuple[float, float]       # frame-normalized only for direction naming
    size: tuple[float, float]         # width/scale, height/scale
    short_ratio: float
    aspect_ratio: float


@dataclass(frozen=True)
class TileRelations:
    tiles: tuple[TileFeature, ...]
    scale: float
    center_distances: np.ndarray      # divided by image median short side
    gaps: np.ndarray                  # minimum rectangle-edge gap / scale
    axes_deg: np.ndarray              # undirected angle in [0, 180)
    size_ratios: np.ndarray           # min(short_i, short_j)/max(...)
    cross_offsets: np.ndarray         # perpendicular center offset / scale
```

`build_tile_relations(boxes, frame_w, frame_h)` must:

- validate positive frame dimensions and finite positive `xywh` boxes;
- return an empty immutable result for no boxes;
- use median short side as `scale`;
- calculate symmetric matrices with zero diagonals;
- copy arrays and call `setflags(write=False)`;
- calculate rectangle-edge gap as `hypot(max(abs(dx)-(w_i+w_j)/2,0), max(abs(dy)-(h_i+h_j)/2,0)) / scale`;
- avoid image- or camera-specific constants.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest tests/test_tile_relations.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add mahjong_rt/tile_relations.py tests/test_tile_relations.py
git commit -m "feat(zones): derive adaptive tile relations"
```

---

### Task 3: Extract direction-constrained tile lines

**Files:**
- Create: `mahjong-rt/mahjong_rt/tile_lines.py`
- Create: `mahjong-rt/tests/test_tile_lines.py`

- [ ] **Step 1: Write failing row and anti-chaining tests**

```python
from mahjong_rt.tile_lines import LineExtractionConfig, extract_tile_lines
from mahjong_rt.tile_relations import build_tile_relations


def line_boxes(points, size=(20, 30)):
    return [[x, y, size[0], size[1]] for x, y in points]


def test_extracts_horizontal_vertical_and_sloped_lines():
    boxes = line_boxes([(20, 100), (45, 100), (70, 100)])
    boxes += line_boxes([(200, 20), (200, 50), (200, 80)])
    boxes += line_boxes([(300, 50), (325, 60), (350, 70)])
    lines = extract_tile_lines(build_tile_relations(boxes, 500, 300))
    assert {line.members for line in lines} >= {(0, 1, 2), (3, 4, 5), (6, 7, 8)}


def test_bridge_does_not_single_link_two_lines():
    boxes = line_boxes([(100, 50), (130, 50), (160, 50)])
    boxes += line_boxes([(170, 75), (180, 100)])
    boxes += line_boxes([(190, 130), (220, 130), (250, 130)])
    lines = extract_tile_lines(build_tile_relations(boxes, 400, 240))
    assert (0, 1, 2) in {line.members for line in lines}
    assert (5, 6, 7) in {line.members for line in lines}
    assert all(len(line.members) < 8 for line in lines)


def test_shuffled_input_preserves_geometric_membership():
    boxes = line_boxes([(20, 100), (45, 100), (70, 100), (95, 100)])
    order = [2, 0, 3, 1]
    original = extract_tile_lines(build_tile_relations(boxes, 200, 160))
    shuffled = extract_tile_lines(build_tile_relations([boxes[i] for i in order], 200, 160))
    restored = {tuple(sorted(order[i] for i in row.members)) for row in shuffled}
    assert tuple(sorted(original[0].members)) in restored
```

- [ ] **Step 2: Verify missing module failure**

```powershell
python -m pytest tests/test_tile_lines.py -q
```

Expected: collection FAIL.

- [ ] **Step 3: Implement deterministic row extraction**

Define:

```python
@dataclass(frozen=True)
class TileLine:
    members: tuple[int, ...]
    axis_deg: float
    centroid: tuple[float, float]
    span: float
    median_gap: float
    gap_cv: float
    fit_error: float
    size_consistency: float


@dataclass(frozen=True)
class LineExtractionConfig:
    max_gap_iqr_factor: float = 1.5
    max_cross_offset_ratio: float = 0.55
    min_size_similarity: float = 0.60
    max_fit_error_ratio: float = 0.45
    min_members: int = 2
```

Algorithm:

1. A pair is initially compatible when `size_ratio >= min_size_similarity`. For every tile, select the compatible neighbor with the smallest rectangle-edge gap; ties use neighbor index. Collect those finite nearest gaps as `G`.
2. Let `Q1,Q3` use NumPy's linear percentile and `IQR=Q3-Q1`. The candidate gap limit is `Q3 + max_gap_iqr_factor*IQR`. If `len(G)<4` or `IQR=0`, use `max(G)`; if `G` is empty, emit only singleton handling downstream and no line seeds. All gaps are already divided by the image median short side.
3. Keep compatible pairs whose rectangle-edge gap is at most that limit and sort seeds by `(gap,min_index,max_index)`.
4. Grow in both projected directions along the seed axis. For candidate tile `j`, project each oriented rectangle onto the current unit row axis. The projected gap is the non-negative distance between the candidate interval and the nearest terminal member's interval; it is not center distance.
5. Compute accepted row projected gaps plus the candidate gap. The candidate gap limit uses the same `Q3 + max_gap_iqr_factor*IQR` formula; with fewer than four samples or zero IQR, use `max(existing_gaps)`, and for the first growth step use the global candidate gap limit from step 2.
6. Accept only when size similarity passes, perpendicular centroid residual divided by the row's median normalized short size is at most `max_cross_offset_ratio`, candidate projected gap is within the row limit, and refitting does not violate the safeguards below.
7. Refit the row axis with the first principal component after each accepted tile. Orient its sign lexicographically from the smaller endpoint center toward the larger; equal eigenvalues retain the prior seed axis.
8. Define fit error as maximum perpendicular centroid residual divided by median normalized short size. Define gap coefficient of variation as population standard deviation divided by `max(mean_gap,1e-9)`, with zero for fewer than two gaps. Reject growth when fit error exceeds `max_fit_error_ratio` or gap CV exceeds `max_gap_iqr_factor`; this reuses an existing dimensionless safeguard instead of adding a camera parameter.
9. Deduplicate identical member sets, remove strict subsets with no lower fit error, and sort by `(-member_count,members)`.

- [ ] **Step 4: Add mild-affine invariance test**

Apply `x'=1.15x+0.08y+20`, `y'=0.05x+0.9y+10` to box centers while preserving positive sizes. Assert the same dominant member sets; this matches the directional-stability premise and does not claim arbitrary projective invariance.

- [ ] **Step 5: Run tests**

```powershell
python -m pytest tests/test_tile_lines.py tests/test_tile_relations.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add mahjong_rt/tile_lines.py tests/test_tile_lines.py
git commit -m "feat(zones): extract constrained tile lines"
```

---

### Task 4: Build hand, meld, river-block, and singleton groups

**Files:**
- Create: `mahjong-rt/mahjong_rt/layout_groups.py`
- Create: `mahjong-rt/tests/test_layout_groups.py`

- [ ] **Step 1: Write failing semantic-group tests**

```python
from mahjong_rt.layout_groups import build_layout_groups
from mahjong_rt.tile_lines import extract_tile_lines
from mahjong_rt.tile_relations import build_tile_relations


def build(boxes, classes=None):
    relations = build_tile_relations(boxes, 800, 600)
    return build_layout_groups(relations, extract_tile_lines(relations), classes)


def test_long_regular_line_is_hand_candidate():
    boxes = [[100 + i * 28, 500, 22, 34] for i in range(9)]
    groups = build(boxes)
    assert any(g.kind == "hand" and len(g.members) == 9 for g in groups)


def test_three_equal_tiles_raise_meld_evidence_without_being_required():
    boxes = [[100 + i * 25, 200, 22, 34] for i in range(3)]
    equal = build(boxes, ("w1", "w1", "w1"))
    unknown = build(boxes, None)
    assert any(g.kind == "meld" and g.class_evidence for g in equal)
    assert any(g.kind == "meld" for g in unknown)


def test_multiple_short_parallel_lines_form_disjoint_river_candidates():
    boxes = []
    for y in (220, 260, 300):
        boxes += [[300 + i * 28, y, 22, 34] for i in range(4)]
    groups = build(boxes)
    river_members = {m for g in groups if g.kind in {"river_block", "row"} for m in g.members}
    assert river_members == set(range(12))
    assert any(g.kind == "river_block" and len(g.members) == 8 for g in groups)


def test_unclaimed_tile_becomes_singleton():
    groups = build([[100, 100, 20, 30]])
    assert groups[0].kind == "singleton"
    assert groups[0].members == (0,)
```

- [ ] **Step 2: Verify missing module failure**

```powershell
python -m pytest tests/test_layout_groups.py -q
```

Expected: collection FAIL.

- [ ] **Step 3: Implement group contracts and deterministic precedence**

```python
@dataclass(frozen=True)
class LayoutGroup:
    group_id: int
    members: tuple[int, ...]
    kind: str  # hand | meld | river_block | row | singleton
    centroid: tuple[float, float]
    axis_deg: float | None
    radial_span: float
    regularity: float
    class_evidence: bool
    source_lines: tuple[int, ...]
```

`build_layout_groups(relations, lines, classes=None)` produces a **disjoint partition**, preventing combinatorial exact-cover search:

1. Compute normalized line features in `[0,1]`: `L` and `S` use empirical CDF rank `(count(value ≤ x)-1)/max(1,n-1)`, so ties receive the same upper rank; regularity is `R = max(0, 1-(min(1,gap_cv)+min(1,fit_error))/2)`.
2. A line's hand score is `H = (L + R + S) / 3`. Process lines by descending `(H, member_count, R)` and then ascending `members`; accept a hand candidate when `H` is at or above the median candidate score and it has at least five members. The five-member floor is a Mahjong structural safeguard, not a camera parameter.
3. From still-unclaimed tiles, process 3–4 member lines by descending `(R, member_count)` then ascending `members` and accept them as `meld`. Three/four equal optional classes set `class_evidence=True` but classes never gate creation.
4. From remaining disjoint short rows, evaluate all row pairs. For a pair, use their mean unit axis; `parallel_gap` is the non-negative gap between their projected intervals, `orthogonal_gap` is centroid separation on the perpendicular axis, and `row_gap_ratio = orthogonal_gap / max(median within-row projected gap, 1e-9)`. Axis eligibility uses the upper Tukey fence of all pairwise axis differences; gap eligibility uses the upper Tukey fence of all row-gap ratios. With fewer than four samples or zero IQR, use the observed maximum as the fence. Projected intervals must overlap by a positive amount.
5. Sort eligible row pairs by descending `B = mean(R) + min(1, orthogonal_span/max(parallel_span,1e-9))`, then ascending member union. Greedily create a two-row river block only when neither row was consumed and `B > mean(R)`. Additional rows are not incrementally appended; three-row blocks emerge later as separate adjacent river groups with the same final zone. This prevents ambiguous overlapping block enumeration.
6. Remaining valid disjoint lines become `row`; unclaimed tiles become `singleton`.
7. At every earlier overlap, keep the candidate with larger `(H, member_count, R)`; exact ties keep lexicographically smaller `members`. Remove claimed members before the next candidate.
8. Every tile belongs to exactly one output group; each group has sorted unique members and deterministic ordering.

This deterministic partition intentionally gives up overlapping hypotheses in exchange for bounded runtime and clear failure ownership. Erroneous early grouping is exposed as a row/group test failure rather than hidden inside a combinatorial solver.

- [ ] **Step 4: Add coverage and no-mutation tests**

Assert every input tile appears in at least one group, inputs remain unchanged, and optional classes must match box count.

- [ ] **Step 5: Run tests**

```powershell
python -m pytest tests/test_layout_groups.py tests/test_tile_lines.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add mahjong_rt/layout_groups.py tests/test_layout_groups.py
git commit -m "feat(zones): classify adaptive layout groups"
```

---

### Task 5: Infer robust center and directional player anchors

**Files:**
- Create: `mahjong-rt/mahjong_rt/layout_anchors.py`
- Create: `mahjong-rt/tests/test_layout_anchors.py`

- [ ] **Step 1: Write failing center, direction, and missing-seat tests**

```python
from mahjong_rt.layout_anchors import infer_layout_anchors
from mahjong_rt.layout_groups import LayoutGroup


def group(i, kind, x, y, n=5, regularity=.9):
    return LayoutGroup(i, tuple(range(i * 20, i * 20 + n)), kind, (x, y), 0.0,
                       .2, regularity, False, ())


def test_group_center_is_not_weighted_by_tile_count():
    groups = (
        group(0, "hand", .5, .9, 14), group(1, "hand", .1, .5, 4),
        group(2, "hand", .5, .1, 4), group(3, "hand", .9, .5, 4),
        group(4, "river_block", .5, .5, 12),
    )
    result = infer_layout_anchors(groups)
    assert result.center == pytest.approx((.5, .5), abs=.05)


def test_anchor_directions_map_to_fixed_seats():
    groups = (
        group(0, "hand", .5, .9), group(1, "hand", .1, .5),
        group(2, "hand", .5, .1), group(3, "hand", .9, .5),
    )
    result = infer_layout_anchors(groups)
    assert result.by_zone["my_hand"] == 0
    assert result.by_zone["seat_left"] == 1
    assert result.by_zone["seat_across"] == 2
    assert result.by_zone["seat_right"] == 3


def test_missing_across_anchor_is_not_invented():
    groups = (group(0, "hand", .5, .9), group(1, "hand", .1, .5), group(2, "hand", .9, .5))
    result = infer_layout_anchors(groups)
    assert "seat_across" not in result.by_zone
    assert "missing_seat_across_anchor" in result.failures
```

- [ ] **Step 2: Verify missing module failure**

```powershell
python -m pytest tests/test_layout_anchors.py -q
```

Expected: collection FAIL.

- [ ] **Step 3: Implement unweighted robust center and anchor ranking**

```python
@dataclass(frozen=True)
class AnchorResult:
    center: tuple[float, float]
    by_zone: Mapping[str, int]
    radial_ranks: Mapping[int, float]
    inside_scores: Mapping[int, float]
    failures: tuple[str, ...]
```

Implementation requirements:

- Task 4 supplies a disjoint partition, so each non-singleton group contributes one centroid exactly once regardless of tile count; singletons are excluded from center estimation when any non-singleton exists.
- Use coordinate medians for the initial center and iterate the Weiszfeld geometric median to tolerance `1e-9` or 64 iterations with an epsilon guard.
- Normalize radius to empirical-CDF rank `r∈[0,1]`, regularity to `q∈[0,1]`, and outer-hull membership to `h∈{0,1}`. Compute the outer hull with monotonic chain on group centroids; collinear boundary points count as hull points. For fewer than three unique centroids, every unique point is on the hull. Use exact normalized coordinates and lexicographic group-ID tie-breaking, with no pixel tolerance.
- Score anchor candidates only from `hand`, `meld`, and `row`; never select `river_block` or `singleton` as an anchor.
- Match at most one anchor to each fixed direction vector: bottom `(0,1)`, left `(-1,0)`, top `(0,-1)`, right `(1,0)`.
- For candidate `g` and direction `d`, define direction alignment `a=max(0,dot(unit(g-center),d))` and anchor score `A=(2a+r+q+h)/5`. Candidate-direction pairs with `a < median positive alignment for that direction` are ineligible. This threshold is derived from the current image.
- Exhaustively assign at most four directions over at most the top four eligible candidates per direction. Each group may serve at most one direction. Missing direction contributes zero and is preferred over any ineligible candidate. Maximize summed `A`, tie-breaking by direction order and group ID. This search is bounded by `5^4=625` states.
- Build mappings with `MappingProxyType`.
- For anchor `p` and layout center `c`, define its inward half-plane as `{x | dot(x-p, c-p) >= 0}`; boundary points are inside. For each group centroid `x`, start with observation `1-r`, append `1` or `0` for every available anchor half-plane, and set `inside_score` to their arithmetic mean. Missing anchors add no boundary and no invented evidence.

- [ ] **Step 4: Add translation/scale and candidate-order invariance tests**

Assert unchanged anchor group identities after uniform frame transforms and after shuffling candidate group order.

- [ ] **Step 5: Run tests**

```powershell
python -m pytest tests/test_layout_anchors.py tests/test_layout_groups.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add mahjong_rt/layout_anchors.py tests/test_layout_anchors.py
git commit -m "feat(zones): infer adaptive player anchors"
```

---

### Task 6: Solve group labels and attach singletons globally

**Files:**
- Create: `mahjong-rt/mahjong_rt/adaptive_zone_solver.py`
- Create: `mahjong-rt/tests/test_adaptive_zone_solver.py`

- [ ] **Step 1: Write failing group consistency and ambiguity tests**

```python
from mahjong_rt.adaptive_zone_solver import AdaptiveZoneSolver, GroupCandidateCost


def test_group_assignment_keeps_boundary_tile_with_row():
    costs = (
        GroupCandidateCost(0, (0, 1, 2), "row", {
            "seat_across": .20, "river": .55, "my_hand": 9, "seat_left": 9, "seat_right": 9,
        }),
    )
    result = AdaptiveZoneSolver().solve(3, costs)
    assert result.zones == ("seat_across", "seat_across", "seat_across")


def test_partitioned_groups_choose_their_lowest_cost_labels():
    costs = (
        GroupCandidateCost(0, (0, 1, 2), "hand", {
            "seat_across": .25, "river": 1.2, "my_hand": 9, "seat_left": 9, "seat_right": 9,
        }),
        GroupCandidateCost(1, (3, 4, 5), "river_block", {
            "river": .30, "seat_across": 1.4, "my_hand": 9, "seat_left": 9, "seat_right": 9,
        }),
    )
    result = AdaptiveZoneSolver().solve(6, costs)
    assert result.zones[:3] == ("seat_across",) * 3
    assert result.zones[3:] == ("river",) * 3


def test_solver_reports_best_second_margin_and_reason():
    result = AdaptiveZoneSolver().solve(1, (
        GroupCandidateCost(0, (0,), "singleton", {
            "river": .2, "seat_across": .6, "my_hand": 1, "seat_left": 1, "seat_right": 1,
        }),
    ))
    d = result.diagnostics[0]
    assert d.best_cost == .2
    assert d.second_cost == .6
    assert d.margin == .4
    assert d.evidence
```

- [ ] **Step 2: Verify missing module failure**

```powershell
python -m pytest tests/test_adaptive_zone_solver.py -q
```

Expected: collection FAIL.

- [ ] **Step 3: Implement cost and result contracts**

```python
ZONE_ORDER = ("my_hand", "seat_left", "seat_across", "seat_right", "river")

@dataclass(frozen=True)
class GroupCandidateCost:
    group_id: int
    members: tuple[int, ...]
    kind: str
    costs: Mapping[str, float]
    evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

@dataclass(frozen=True)
class AdaptiveSolveResult:
    zones: tuple[str, ...]
    diagnostics: tuple[TileZoneDiagnostic, ...]
    selected_group_by_tile: tuple[int, ...]
    total_cost: float
```

- [ ] **Step 4: Implement partitioned group labeling and singleton assignment**

- Validate exactly five finite non-negative costs for every group.
- Validate groups form a disjoint, complete partition of `range(tile_count)`; reject overlaps, gaps, duplicate members, and out-of-range indexes instead of searching alternate covers.
- For each group choose one zone for all members. `river_block` receives a type cost favoring `river`; `hand` and `meld` receive anchor-direction costs; `row` and `singleton` rely on the same normalized structural evidence without changing membership.
- Minimize the sum of group costs. Because groups are partitioned, the optimum is deterministic per group and runtime is linear in group count.
- Resolve equal costs by `ZONE_ORDER`, then group ID.
- Build per-tile diagnostics from the chosen group-zone cost and the second-lowest zone cost for that same group.
- `AdaptiveZoneSolver.solve(tile_count, costs, min_margin=0.0)` validates finite non-negative `min_margin`. Never emit `unknown_zone`; set `ambiguous=True` exactly when `second_cost-best_cost < min_margin`, retaining the best label for measurable coverage.

- [ ] **Step 5: Add malformed cost, determinism, and partition-validation tests**

Test missing zone keys, NaN/Infinity, duplicate/overlapping members, coverage gaps, out-of-range tile indexes, candidate order shuffling, and an all-singleton image.

- [ ] **Step 6: Run tests**

```powershell
python -m pytest tests/test_adaptive_zone_solver.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add mahjong_rt/adaptive_zone_solver.py tests/test_adaptive_zone_solver.py
git commit -m "feat(zones): solve adaptive group labels"
```

---

### Task 7: Integrate the table-free adaptive pipeline

**Files:**
- Modify: `mahjong-rt/mahjong_rt/zones.py`
- Modify: `mahjong-rt/mahjong_rt/pipeline.py`
- Modify: `mahjong-rt/mahjong_rt/replay.py`
- Modify: `mahjong-rt/configs/pipeline.yaml`
- Modify: `mahjong-rt/tests/test_zones.py`
- Create: `mahjong-rt/tests/test_adaptive_zone_pipeline.py`

- [ ] **Step 1: Write failing facade and end-to-end tests**

```python
from mahjong_rt.zone_types import ZoneAnalysisContext
from mahjong_rt.zones import ZoneConfig, analyze_layout


def test_legacy_mode_is_unchanged():
    boxes = [[600, 600, 100, 100], [500, 300, 30, 30]]
    before = analyze_layout(boxes, 1280, 720, ZoneConfig())
    after = analyze_layout(boxes, 1280, 720, ZoneConfig(mode="legacy"))
    assert after == before


def test_adaptive_mode_needs_no_table_or_profile():
    boxes = [[300 + i * 45, 620, 38, 60] for i in range(8)]
    zones, debug = analyze_layout(
        boxes, 1000, 800, ZoneConfig(mode="adaptive_groups"),
        context=ZoneAnalysisContext(classes=("unknown",) * len(boxes)),
    )
    assert zones == ["my_hand"] * len(boxes)
    assert debug["mode"] == "adaptive_groups"
    assert "groups" in debug and "anchors" in debug


def test_five_group_layout_maps_fixed_directions():
    boxes = synthetic_five_zone_boxes()
    zones, _ = analyze_layout(boxes, 1000, 800, ZoneConfig(mode="adaptive_groups"))
    assert zones_for_members(zones, "bottom") == {"my_hand"}
    assert zones_for_members(zones, "left") == {"seat_left"}
    assert zones_for_members(zones, "top") == {"seat_across"}
    assert zones_for_members(zones, "right") == {"seat_right"}
    assert zones_for_members(zones, "center") == {"river"}
```

The test helper must construct explicit bottom/left/top/right long rows and three central short river rows; do not read repository labels.

- [ ] **Step 2: Verify tests fail**

```powershell
python -m pytest tests/test_adaptive_zone_pipeline.py tests/test_zones.py -q
```

Expected: FAIL because `ZoneConfig.mode` and adaptive dispatch do not exist.

- [ ] **Step 3: Extend facade without modifying legacy implementation**

Extend `ZoneConfig`:

```python
mode: str = "legacy"  # legacy | adaptive_groups
line_max_gap_iqr_factor: float = 1.5
line_max_cross_offset_ratio: float = 0.55
line_min_size_similarity: float = 0.60
line_max_fit_error_ratio: float = 0.45
min_margin: float = 0.0
```

Change signatures with keyword-only context:

```python
def analyze_layout(boxes, frame_w, frame_h, config, *, context=None) -> tuple[list[str], dict]: ...
def assign_zones(boxes, frame_w, frame_h, config, *, context=None) -> list[str]: ...
```

Move the current body unchanged into `_analyze_legacy`. Adaptive dispatch must:

1. validate context count;
2. call `build_tile_relations`;
3. call `extract_tile_lines`;
4. call `build_layout_groups`;
5. call `infer_layout_anchors`;
6. construct group-zone costs from normalized evidence using the explicit formula below;
7. call `AdaptiveZoneSolver.solve`;

For group `g`, normalize regularity `q`, radial rank `r`, inside score `i`, optional class evidence `c∈{0,1}`, and directional alignments `a_z∈[0,1]`. Let `k_river`, `k_hand`, and `k_meld` be one-hot group-kind indicators. Define scores:

```python
river_score = (2 * i + 2 * k_river + (1 - r) + q) / 6
player_score[z] = (2 * a_z + r + q + anchor_match_z + k_hand + .5 * k_meld + .5 * c) / 7
cost[z] = 1.0 - clip(score[z], 0.0, 1.0)
```

`anchor_match_z` is 1 only when this group is the selected anchor for zone `z`; otherwise 0. For a meld and an available zone anchor `p_z`, define alignment to that anchor as `max(0, dot(unit(g-center), unit(p_z-center)))`; if the anchor is missing, alignment is 0. Replace `anchor_match_z` with the maximum of selected-anchor match and this alignment. The constants are dimensionless evidence weights shared by every image; no values may depend on image identity. Every production rule added after evaluation must first be expressed as a general structural test. `AdaptiveZoneSolver.solve(..., min_margin=config.min_margin)` marks a tile ambiguous exactly when `second_cost-best_cost < min_margin`; equality is not ambiguous.
8. return JSON-safe debug data for tiles, lines, groups, center, anchors, selected groups, diagnostics, and failures.

No adaptive module may import `table_geometry`, and no adaptive call accepts a profile or corner argument.

- [ ] **Step 4: Wire only existing classes into runtime**

In `pipeline.py`, derive class strings from current observations when available and call:

```python
context = ZoneAnalysisContext(
    classes=tuple(obs.label if obs is not None else "unknown" for obs in observations),
    strict=True,
)
zones = assign_zones(xywh, frame.shape[1], frame.shape[0], self.zone_config, context=context)
```

In `replay.py`, pass recorded class labels when available; when unavailable, pass `context=None`. Adaptive grouping must still work geometry-only. Preserve event and recording schemas.

- [ ] **Step 5: Add explicit configuration**

Under `zones` in `configs/pipeline.yaml`, add:

```yaml
  mode: legacy
  line_max_gap_iqr_factor: 1.5
  line_max_cross_offset_ratio: 0.55
  line_min_size_similarity: 0.60
  line_max_fit_error_ratio: 0.45
  min_margin: 0.0
```

Do not add `profile_path`, `table_locator`, corner source, or homography settings.

- [ ] **Step 6: Add transform, missing-seat, class-error, and shuffle tests**

For the synthetic five-zone layout assert:

- uniform translation/scale/resolution changes preserve labels;
- mild affine distortion preserves labels;
- removing all top-seat tiles does not invent `seat_across` on river tiles;
- replacing every class with `unknown` preserves spatially unambiguous labels;
- shuffling boxes/classes only shuffles corresponding outputs;
- empty input returns `[]` and complete debug structure;
- invalid boxes fail explicitly.

- [ ] **Step 7: Run all tests**

```powershell
python -m pytest tests/test_adaptive_zone_pipeline.py tests/test_zones.py tests/test_zone_types.py -q
python -m pytest tests/ -q
```

Expected: PASS; the legacy accuracy floor and all prior non-zone tests remain unchanged.

- [ ] **Step 8: Commit**

```powershell
git add mahjong_rt/zones.py mahjong_rt/pipeline.py mahjong_rt/replay.py configs/pipeline.yaml tests/test_zones.py tests/test_adaptive_zone_pipeline.py
git commit -m "feat(zones): integrate adaptive group recognition"
```

---

### Task 8: Add immutable evaluation and honest 100% gates

**Files:**
- Create: `mahjong-rt/scripts/eval_zones.py`
- Create: `mahjong-rt/tests/test_eval_zones.py`
- Modify: `mahjong-rt/tests/test_zones.py`

- [ ] **Step 1: Write failing metric and manifest tests**

```python
import pytest

from scripts.eval_zones import evaluate_predictions, load_test_manifest


def test_unknown_is_wrong_and_uncovered():
    report = evaluate_predictions(
        truths=["river", "seat_across"], predictions=["river", "unknown_zone"],
        records=[("a.jpg", 0), ("a.jpg", 1)], diagnostics=[{}, {}],
    )
    assert report["correct"] == 1
    assert report["total"] == 2
    assert report["accuracy"] == .5
    assert report["coverage"] == .5
    assert report["passed"] is False


def test_perfect_requires_every_present_zone_recall():
    zones = ["my_hand", "seat_left", "seat_across", "seat_right", "river"]
    report = evaluate_predictions(zones, zones, [("a.jpg", i) for i in range(5)], [{}] * 5)
    assert report["passed"] is True


def test_manifest_rejects_paths_and_duplicates(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("batch/a.jpg\nfolder/b.jpg\nbatch/a.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="key|duplicate"):
        load_test_manifest(path)
```

- [ ] **Step 2: Verify missing evaluator failure**

```powershell
python -m pytest tests/test_eval_zones.py -q
```

Expected: collection FAIL.

- [ ] **Step 3: Implement strict evaluator primitives**

Before implementing the integration helper, create `output/zone_test_manifest.txt` from the canonical annotation array's unique `image` fields only after independent review has removed all non-five-zone labels. Sort values lexicographically, require 20 unique records and 899 total boxes, write atomically, and expose a test that recomputes those counts and the manifest SHA-256. The loader builds `{record["image"]: record}` and rejects duplicate `image` values. It maps current fields `w→frame_width`, `h→frame_height`, and `cls→classes`; `boxes` and `zones` retain their names. `source`, `heuristic`, and `hit` are metadata only and never enter inference.

`evaluate_predictions` must report:

- correct/total, accuracy, coverage, confusion matrix;
- per-zone recall and per-image accuracy;
- every error with image, box index, GT, prediction, candidate line/group, selected group, best/second costs, margin, evidence, and failure category;
- `passed=True` only for total > 0, accuracy 1, coverage 1, and recall 1 for every zone present in truth.

Manifest entries must exactly match annotation-array `image` values such as `video/frame.jpg`; validation rejects empty entries, absolute paths, `.`/`..` segments, backslashes, control characters, duplicates, missing label records, and any boxes/zones/classes length mismatch. If the optional `--images` argument is supplied, it additionally rejects missing or undecodable images resolved beneath that root. Image pixels are never passed to zone inference.

- [ ] **Step 4: Implement table-free CLI and immutable report**

```powershell
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --test-manifest ../output/zone_test_manifest.txt --report ../output/zone_eval_adaptive.json --require-perfect
```

Requirements:

- force `ZoneConfig(mode="adaptive_groups")`;
- never read `table_corners`, calibration profiles, or non-manifest annotation records;
- require canonical five-zone truth; report legacy labels and abort before inference;
- run every manifest annotation record exactly once and lock predictions before scoring;
- store normalized annotation `image` values, original/canonical manifest SHA-256, labels SHA-256, selected configuration, git commit, and UTC timestamp;
- write canonical UTF-8 JSON atomically with sorted keys and `allow_nan=False`;
- exit `0` under `--require-perfect` only when `passed=True`, otherwise `1`.

- [ ] **Step 5: Add current-set integration gate**

Add:

```python
@pytest.mark.integration
def test_adaptive_release_gate_on_current_set():
    report = run_current_adaptive_evaluation(
        labels_path=Path("../output/zone_annotation/zone_labels_with_class.json"),
        manifest_path=Path("../output/zone_test_manifest.txt"),
    )
    assert report["total"] == 899
    assert report["correct"] == 899, format_failures(report)
    assert report["coverage"] == 1.0
    assert all(value == 1.0 for value in report["recall"].values())
```

Do not skip failures. If five `opponent_wall` labels remain, the helper must fail with a canonical-label error rather than silently dropping or remapping them.

- [ ] **Step 6: Run evaluator tests and current diagnostic**

```powershell
python -m pytest tests/test_eval_zones.py tests/test_zones.py -q
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --test-manifest ../output/zone_test_manifest.txt --report ../output/zone_eval_adaptive.json
```

Expected: unit tests PASS. The CLI always writes an honest report after successful input validation; do not claim success unless it says `passed: true`.

- [ ] **Step 7: Classify errors and improve only general structural rules**

Every error receives exactly one category:

- bad input box;
- row extraction;
- erroneous group merge;
- erroneous row split;
- group type;
- center/anchor;
- seat direction;
- singleton attachment;
- global cost/constraint;
- truth ambiguity.

For implementation errors, return to the owning task with a minimal failing test derived from the structural pattern, not an image-name or box-index exception. Truth ambiguity uses the existing independent review workflow.

- [ ] **Step 8: Run the current 899/899 gate**

```powershell
python -m pytest tests/ -q
python -m pytest tests/test_zones.py -q -m integration
python scripts/eval_zones.py --labels ../output/zone_annotation/zone_labels_with_class.json --images ../output/zone_annotation/images --test-manifest ../output/zone_test_manifest.txt --report ../output/zone_eval_adaptive_final.json --require-perfect
```

Expected for completion: all tests PASS and evaluator exits `0` with `correct=total=899`, coverage 1, and every present-zone recall 1. If not, retain the report and do not mark the release gate complete.

- [ ] **Step 9: Run a genuinely sealed new-set gate**

Before scoring, freeze algorithm commit, configuration, exact annotation `image` values, image SHA-256 values when images are archived, expected record count, expected box count, and manifest SHA-256 in a sealed inventory. Do **not** put truth-label SHA-256 into the pre-prediction inventory because truth is still hidden. Run predictions first using a labels-free inference annotation array whose records contain only `image`, `w`, `h`, `boxes`, and optional `cls`, then persist and hash the prediction artifact. Only afterward reveal truth labels and record their SHA-256 in the final report:

```powershell
python scripts/eval_zones.py predict --inputs <sealed-inference-inputs.json> --test-manifest <sealed-manifest.txt> --sealed-inventory <sealed-inventory.json> --predictions <locked-predictions.json>
python scripts/eval_zones.py score --labels <revealed-sealed-labels.json> --test-manifest <sealed-manifest.txt> --sealed-inventory <sealed-inventory.json> --predictions <locked-predictions.json> --report <sealed-report.json> --require-perfect
```

`predict` must reject truth-bearing inputs and validate the frozen commit/config/manifest/counts. It writes canonical prediction JSON bytes atomically with sorted keys, compact separators, UTF-8, and `allow_nan=False`, then writes the SHA-256 of those exact bytes to `<predictions>.sha256`; the digest is not embedded in the JSON. `score` must never rerun inference; it validates the sidecar digest against the exact prediction bytes, then compares revealed truth and records truth SHA-256 in the report. Any inventory/hash/count mismatch aborts. Passing requires N/N with no corner annotations or other human inference input. Once a failed set is inspected for development, retire it and collect a new sealed set for the next generalization claim.

- [ ] **Step 10: Commit evaluator only after verified behavior**

```powershell
git add scripts/eval_zones.py tests/test_eval_zones.py tests/test_zones.py
git commit -m "test(zones): add table-free perfect-accuracy gates"
```

Do not commit generated reports unless the repository's existing output policy explicitly tracks them.

---

## Plan self-review

### Spec coverage

- Existing boxes/classes only and no new manual/model input: Tasks 1, 2, and 7.
- Frame-adaptive relative measurements: Task 2.
- Direction-constrained rows and anti-chaining: Task 3.
- Hand, meld, river-block, and singleton structure: Task 4.
- Robust center, relative outskirts, fixed seat directions, and missing seats: Task 5.
- Whole-group assignment over a disjoint partition, singleton handling, margins, and evidence: Task 6.
- Legacy compatibility and runtime/replay integration: Task 7.
- No table corners in current/new validation and honest 100% gates: Task 8.
- Label ambiguity and five-zone migration remain independently reviewed, never inferred automatically: scope plus Task 8.

### Placeholder scan

The plan contains no `TBD`, `TODO`, per-image exception, unspecified profile, or deferred implementation step. Every code-producing task defines its public contracts, failure command, passing command, and commit boundary.

### Type consistency

- `ZoneAnalysisContext` becomes table-free in Task 1 and is used unchanged in Task 7.
- `TileRelations` originates in Task 2 and feeds Tasks 3–5.
- `TileLine` originates in Task 3 and feeds Task 4.
- `LayoutGroup` originates in Task 4 and feeds Tasks 5–7.
- `AnchorResult` originates in Task 5 and feeds Task 7 cost construction.
- `GroupCandidateCost` and `AdaptiveSolveResult` originate in Task 6 and are used by Task 7.
- `assign_zones` retains its four positional arguments; optional context remains keyword-only.

### Dependency and scope check

All production algorithms use existing NumPy and the standard library. The plan adds one cohesive static-zone subsystem and does not add table detection, model training, temporal reasoning, or unrelated refactoring.
