from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2

CANONICAL_ZONES = {"my_hand", "seat_left", "seat_across", "seat_right", "river"}
REVIEW_SCHEMA_VERSION = 1
LEGACY_ZONES = {"opponent_wall"}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def labels_digest(labels: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(labels)).hexdigest()


def validate_canonical_labels(labels: Sequence[dict[str, Any]]) -> None:
    for item in labels:
        image = item.get("image", "<unknown>")
        boxes = item.get("boxes")
        zones = item.get("zones")
        classes = item.get("classes")
        if not isinstance(boxes, list) or not isinstance(zones, list) or not isinstance(classes, list):
            raise ValueError(f"{image}: boxes, zones and classes must be lists")
        if not (len(boxes) == len(zones) == len(classes)):
            raise ValueError(f"{image}: boxes, zones and classes lengths must match")
        for index, zone in enumerate(zones):
            if zone not in CANONICAL_ZONES:
                raise ValueError(f"{image}[{index}]: unsupported zone {zone!r}")


def _validate_review(review: dict[str, Any], expected_digest: str | None) -> None:
    required = {"schema_version", "reviewer", "labels_sha256", "records"}
    if set(review) != required:
        raise ValueError("review fields do not match the strict schema")
    if review["schema_version"] != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported review schema_version")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError("reviewer must not be empty")
    digest = review["labels_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("labels_sha256 must be a SHA-256 digest")
    if expected_digest is not None and digest != expected_digest:
        raise ValueError("review labels_sha256 does not match source labels")
    if not isinstance(review["records"], list):
        raise ValueError("review records must be a list")


def required_review_keys(labels: Sequence[dict[str, Any]]) -> set[tuple[str, int]]:
    required: set[tuple[str, int]] = set()
    for item in labels:
        image = item.get("image")
        zones = item.get("zones")
        if not isinstance(image, str) or not image or not isinstance(zones, list):
            raise ValueError("labels must contain image and zones")
        ambiguous = item.get("zone_ambiguous", [False] * len(zones))
        if not isinstance(ambiguous, list) or len(ambiguous) != len(zones):
            raise ValueError("zone_ambiguous must match zones length")
        required.update(
            (image, index)
            for index, zone in enumerate(zones)
            if zone not in CANONICAL_ZONES or bool(ambiguous[index])
        )
    return required


def _review_records(review: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    expected_fields = {"image", "box_index", "old_label", "label", "rationale"}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in review["records"]:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError("review record fields do not match the strict schema")
        image = record["image"]
        box_index = record["box_index"]
        if not isinstance(image, str) or not image or "/" in image or "\\" in image:
            raise ValueError("review image must be a basename")
        if not isinstance(box_index, int) or box_index < 0:
            raise ValueError("review box_index must be non-negative")
        if record["label"] not in CANONICAL_ZONES:
            raise ValueError("review label must be a canonical zone")
        if not isinstance(record["rationale"], str) or not record["rationale"].strip():
            raise ValueError("review rationale must not be empty")
        key = (image, box_index)
        if key in records:
            raise ValueError("review contains duplicate records")
        records[key] = record
    return records


def apply_reviews(
    labels: Sequence[dict[str, Any]],
    review_a: dict[str, Any],
    review_b: dict[str, Any],
    expected_labels_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = copy.deepcopy(list(labels))
    digest = expected_labels_sha256 or labels_digest(source)
    _validate_review(review_a, digest)
    _validate_review(review_b, digest)
    if review_a["reviewer"].strip() == review_b["reviewer"].strip():
        raise ValueError("reviews must be completed by different reviewers")
    records_a = _review_records(review_a)
    records_b = _review_records(review_b)
    if set(records_a) != set(records_b):
        raise ValueError("reviewers must review the same samples")
    required = required_review_keys(source)
    if set(records_a) != required:
        raise ValueError("reviews must cover all required samples exactly once")

    item_by_image = {item.get("image"): item for item in source}
    if len(item_by_image) != len(source):
        raise ValueError("label image names must be unique")
    changes: list[dict[str, Any]] = []
    for image, box_index in sorted(records_a):
        first = records_a[(image, box_index)]
        second = records_b[(image, box_index)]
        if first["old_label"] != second["old_label"]:
            raise ValueError("review old_label values disagree")
        if first["label"] != second["label"]:
            raise ValueError("reviewers disagree on final label")
        item = item_by_image.get(image)
        if item is None or box_index >= len(item.get("zones", [])):
            raise ValueError("review references a missing image or box")
        old_label = item["zones"][box_index]
        if first["old_label"] != old_label:
            raise ValueError("review old_label does not match source labels")
        item["zones"][box_index] = first["label"]
        if "zone_ambiguous" in item:
            item["zone_ambiguous"][box_index] = False
        changes.append(
            {
                "image": image,
                "box_index": box_index,
                "old_label": old_label,
                "reviewer_a": review_a["reviewer"],
                "reviewer_a_label": first["label"],
                "reviewer_b": review_b["reviewer"],
                "reviewer_b_label": second["label"],
                "final_label": first["label"],
                "rationale_a": first["rationale"],
                "rationale_b": second["rationale"],
            }
        )

    unresolved = [
        (item["image"], index, zone)
        for item in source
        for index, zone in enumerate(item.get("zones", []))
        if zone not in CANONICAL_ZONES
    ]
    if unresolved:
        raise ValueError(f"unreviewed non-canonical labels remain: {unresolved}")
    audit = {
        "schema_version": 1,
        "input_labels_sha256": digest,
        "output_labels_sha256": labels_digest(source),
        "reviewers": [review_a["reviewer"], review_b["reviewer"]],
        "changes": changes,
    }
    return source, audit


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def collect_review(labels_path: Path, images_dir: Path, reviewer: str, output: Path) -> None:
    labels = _load_json(labels_path)
    digest = labels_digest(labels)
    records: list[dict[str, Any]] = []
    keys = required_review_keys(labels)
    item_by_image = {item["image"]: item for item in labels}
    candidates = [
        (item_by_image[image], index, item_by_image[image]["zones"][index])
        for image, index in sorted(keys)
    ]
    key_map = {
        ord("1"): "my_hand",
        ord("2"): "seat_left",
        ord("3"): "seat_across",
        ord("4"): "seat_right",
        ord("5"): "river",
    }
    for item, index, old_label in candidates:
        image = cv2.imread(str(images_dir / item["image"]))
        if image is None:
            raise ValueError(f"cannot decode image {item['image']}")
        x, y, width, height = item["boxes"][index]
        first = (round(x - width / 2), round(y - height / 2))
        second = (round(x + width / 2), round(y + height / 2))
        cv2.rectangle(image, first, second, (0, 0, 255), 3)
        cv2.putText(image, f"{item['image']} #{index} {old_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("zone review: 1=my 2=left 3=across 4=right 5=river q=quit", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return
        if key not in key_map:
            raise ValueError("review aborted: expected a zone key")
        rationale = input(f"Rationale for {item['image']} #{index}: ").strip()
        if not rationale:
            raise ValueError("review rationale must not be empty")
        records.append(
            {
                "image": item["image"],
                "box_index": index,
                "old_label": old_label,
                "label": key_map[key],
                "rationale": rationale,
            }
        )
    cv2.destroyAllWindows()
    _write_json_atomic(
        output,
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "reviewer": reviewer,
            "labels_sha256": digest,
            "records": records,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and apply independent zone reviews")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--labels", type=Path, required=True)
    collect.add_argument("--images", type=Path, required=True)
    collect.add_argument("--reviewer", required=True)
    collect.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--labels", type=Path, required=True)
    apply.add_argument("--review-a", type=Path, required=True)
    apply.add_argument("--review-b", type=Path, required=True)
    apply.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect_review(args.labels, args.images, args.reviewer, args.output)
        return 0
    labels = _load_json(args.labels)
    migrated, audit = apply_reviews(
        labels,
        _load_json(args.review_a),
        _load_json(args.review_b),
        expected_labels_sha256=labels_digest(labels),
    )
    _write_json_atomic(args.labels, migrated)
    _write_json_atomic(args.audit, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
