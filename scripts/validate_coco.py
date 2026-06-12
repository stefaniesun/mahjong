"""Validate COCO annotations exported from CVAT and generate review previews.

Example:
    python scripts/validate_coco.py --annotations data/test_set_v1/annotations/instances_default.json \
        --images-root data/test_set_v1/images --report data/test_set_v1/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import yaml

try:
    from pycocotools.coco import COCO as PycocoCOCO
except ImportError:  # pragma: no cover - depends on local environment
    PycocoCOCO = None


@dataclass
class AnnotationIssue:
    image_id: int
    image_file: str
    annotation_id: int
    category_name: str
    bbox: list[float]
    problem_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_file": self.image_file,
            "annotation_id": self.annotation_id,
            "category_name": self.category_name,
            "bbox": self.bbox,
            "problem_type": self.problem_type,
        }


class CocoDataset:
    def __init__(self, dataset: dict[str, Any]) -> None:
        self.dataset = dataset
        self._annotations_by_id = {
            int(annotation["id"]): annotation for annotation in dataset.get("annotations", [])
        }

    @classmethod
    def from_file(cls, annotations_path: Path) -> "CocoDataset":
        if PycocoCOCO is not None:
            coco = PycocoCOCO(str(annotations_path))
            return cls(coco.dataset)
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        return cls(payload)

    def getAnnIds(self, imgIds: Sequence[int]) -> list[int]:
        image_ids = {int(image_id) for image_id in imgIds}
        return [
            int(annotation["id"])
            for annotation in self.dataset.get("annotations", [])
            if int(annotation["image_id"]) in image_ids
        ]

    def loadAnns(self, ann_ids: Sequence[int]) -> list[dict[str, Any]]:
        return [self._annotations_by_id[int(annotation_id)] for annotation_id in ann_ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a COCO dataset converted from X-AnyLabeling annotations.")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to the COCO annotations JSON file.")
    parser.add_argument("--images-root", type=Path, required=True, help="Directory containing dataset images.")
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("configs/classes.yaml"),
        help="Path to classes.yaml used to validate category ids and names.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/test_set_v1/validation_report.json"),
        help="Where to write the validation report JSON.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("data/test_set_v1/validation_preview"),
        help="Directory for rendered preview images.",
    )
    parser.add_argument("--preview-count", type=int, default=30, help="Number of images to render for manual review.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for preview sampling.")
    return parser


def load_expected_categories(classes_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for group in payload.get("classification", []):
        names.extend(group)
    return [{"id": index + 1, "name": name} for index, name in enumerate(names)]


def validate_categories(actual_categories: Sequence[dict[str, Any]], expected_categories: Sequence[dict[str, Any]]) -> tuple[bool, str]:
    actual_pairs = [(int(item["id"]), str(item["name"])) for item in actual_categories]
    expected_pairs = [(int(item["id"]), str(item["name"])) for item in expected_categories]
    if actual_pairs != expected_pairs:
        return False, f"Category mismatch. Expected {expected_pairs}, got {actual_pairs}"
    return True, ""


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def bucket_short_side(width: float, height: float) -> str:
    short_side = min(width, height)
    if short_side < 20:
        return "lt20"
    if short_side <= 40:
        return "20to40"
    return "gt40"


def collect_issues(
    annotations: Sequence[dict[str, Any]],
    images_by_id: dict[int, dict[str, Any]],
    category_names: dict[int, str],
) -> list[AnnotationIssue]:
    issues: list[AnnotationIssue] = []
    grouped_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        grouped_by_image[int(annotation["image_id"])].append(annotation)

    for annotation in annotations:
        image = images_by_id[int(annotation["image_id"])]
        bbox = [float(value) for value in annotation["bbox"]]
        x, y, width, height = bbox
        area = width * height
        category_name = category_names[int(annotation["category_id"])]
        base = AnnotationIssue(
            image_id=int(annotation["image_id"]),
            image_file=str(image["file_name"]),
            annotation_id=int(annotation["id"]),
            category_name=category_name,
            bbox=bbox,
            problem_type="",
        )

        if area < 25:
            issues.append(AnnotationIssue(**{**base.__dict__, "problem_type": "tiny_box"}))

        if width <= 0 or height <= 0:
            issues.append(AnnotationIssue(**{**base.__dict__, "problem_type": "out_of_bounds"}))
        else:
            aspect_ratio = width / height
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                issues.append(AnnotationIssue(**{**base.__dict__, "problem_type": "extreme_aspect_ratio"}))

        if x < 0 or y < 0 or x + width > float(image["width"]) or y + height > float(image["height"]):
            issues.append(AnnotationIssue(**{**base.__dict__, "problem_type": "out_of_bounds"}))

    for image_id, image_annotations in grouped_by_image.items():
        for index, left in enumerate(image_annotations):
            for right in image_annotations[index + 1 :]:
                if int(left["category_id"]) != int(right["category_id"]):
                    continue
                if compute_iou(left["bbox"], right["bbox"]) > 0.95:
                    image = images_by_id[image_id]
                    category_name = category_names[int(left["category_id"])]
                    issues.append(
                        AnnotationIssue(
                            image_id=image_id,
                            image_file=str(image["file_name"]),
                            annotation_id=int(right["id"]),
                            category_name=category_name,
                            bbox=[float(value) for value in right["bbox"]],
                            problem_type="duplicate_box",
                        )
                    )

    return issues


def render_previews(
    coco: CocoDataset,
    images: Sequence[dict[str, Any]],
    images_root: Path,
    preview_dir: Path,
    category_names: dict[int, str],
) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image in images:
        image_path = images_root / image["file_name"]
        canvas = cv2.imread(str(image_path))
        if canvas is None:
            continue
        annotation_ids = coco.getAnnIds(imgIds=[int(image["id"])] )
        annotations = coco.loadAnns(annotation_ids)
        for annotation in annotations:
            x, y, width, height = [int(round(value)) for value in annotation["bbox"]]
            color = (0, 200, 0)
            cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
            label = category_names[int(annotation["category_id"])]
            text_origin = (max(0, x), max(18, y - 6))
            cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        output_path = preview_dir / Path(str(image["file_name"])).name
        cv2.imwrite(str(output_path), canvas)
        count += 1
    return count


def build_report(
    coco: CocoDataset,
    categories: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    issues: Sequence[AnnotationIssue],
    preview_images: int,
) -> dict[str, Any]:
    class_distribution = {str(category["name"]): 0 for category in categories}
    size_buckets = {"lt20": 0, "20to40": 0, "gt40": 0}

    category_lookup = {int(category["id"]): str(category["name"]) for category in categories}
    for annotation in annotations:
        category_name = category_lookup[int(annotation["category_id"])]
        class_distribution[category_name] += 1
        _, _, width, height = [float(value) for value in annotation["bbox"]]
        size_buckets[bucket_short_side(width, height)] += 1

    missing_categories = [name for name, count in class_distribution.items() if count == 0]
    image_count = len(coco.dataset.get("images", []))
    average_boxes_per_image = round(len(annotations) / image_count, 4) if image_count else 0.0

    return {
        "summary": {
            "total_images": image_count,
            "total_annotations": len(annotations),
            "average_boxes_per_image": average_boxes_per_image,
            "preview_images": preview_images,
            "issue_count": len(issues),
        },
        "class_distribution": class_distribution,
        "size_buckets": size_buckets,
        "warnings": {
            "missing_categories": missing_categories,
        },
        "issues": [issue.to_dict() for issue in issues],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    annotations_path = args.annotations.resolve()
    images_root = args.images_root.resolve()
    classes_path = args.classes.resolve()
    report_path = args.report.resolve()
    preview_dir = args.preview_dir.resolve()

    if not annotations_path.exists():
        print(f"Annotations file not found: {annotations_path}", file=sys.stderr)
        return 1
    if not images_root.exists():
        print(f"Images root not found: {images_root}", file=sys.stderr)
        return 1
    if not classes_path.exists():
        print(f"Classes config not found: {classes_path}", file=sys.stderr)
        return 1

    coco = CocoDataset.from_file(annotations_path)
    expected_categories = load_expected_categories(classes_path)
    actual_categories = list(coco.dataset.get("categories", []))
    valid, error_message = validate_categories(actual_categories, expected_categories)
    if not valid:
        print(error_message, file=sys.stderr)
        return 1

    categories = sorted(actual_categories, key=lambda item: int(item["id"]))
    category_names = {int(category["id"]): str(category["name"]) for category in categories}
    images = list(coco.dataset.get("images", []))
    annotations = list(coco.dataset.get("annotations", []))
    images_by_id = {int(image["id"]): image for image in images}

    issues = collect_issues(annotations, images_by_id, category_names)

    preview_count = min(max(args.preview_count, 0), len(images))
    preview_images = 0
    if preview_count > 0:
        sampler = random.Random(args.seed)
        sampled_images = sampler.sample(images, preview_count)
        preview_images = render_previews(coco, sampled_images, images_root, preview_dir, category_names)

    report = build_report(coco, categories, annotations, issues, preview_images)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Validated {len(images)} images and {len(annotations)} annotations. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
