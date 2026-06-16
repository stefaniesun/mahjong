"""Convert X-AnyLabeling labels to YOLO detection data and classification crops.

Example:
    python scripts/convert_labels.py --input-root data/labeled --output-root output/validation_run_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_DISCARD_LABELS = ("back", "tile_back")
SIZE_BUCKETS = ("lt20", "20to40", "gt40")
SIZE_BUCKET_LABELS = {"lt20": "<20px", "20to40": "20~40px", "gt40": ">40px"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert X-AnyLabeling labels into YOLO detection data and class crops.")
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"), help="Path config YAML.")
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"), help="Class config YAML.")
    parser.add_argument("--input-root", type=Path, default=None, help="Directory containing labeled images and JSON files.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root directory.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic train/val split.")
    parser.add_argument("--crop-margin", type=float, default=0.08, help="Crop margin ratio around each box.")
    parser.add_argument("--report", type=Path, default=None, help="Output convert_report.json path.")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def resolve_config_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_class_config(classes_path: Path) -> tuple[list[str], set[str]]:
    payload = load_yaml(classes_path)
    discard_labels = {str(label) for label in payload.get("discard_labels", DEFAULT_DISCARD_LABELS)}
    labels: list[str] = []
    for group in payload.get("classification", []):
        for label in group:
            label_name = str(label)
            if label_name not in discard_labels and label_name not in labels:
                labels.append(label_name)
    return labels, discard_labels


def image_paths_from_root(input_root: Path) -> list[Path]:
    return [path for path in sorted(input_root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def bbox_from_points(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    if len(points) < 2:
        raise ValueError("A rectangle/polygon shape needs at least two points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def clip_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    return x1, y1, x2, y2


def yolo_line(bbox: tuple[float, float, float, float], image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    cx = x1 + box_width / 2.0
    cy = y1 + box_height / 2.0
    values = (cx / image_width, cy / image_height, box_width / image_width, box_height / image_height)
    return "0 " + " ".join(f"{value:.6f}" for value in values)


def size_bucket(width: float, height: float) -> str:
    short_side = min(width, height)
    if short_side < 20:
        return "lt20"
    if short_side <= 40:
        return "20to40"
    return "gt40"


def safe_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("").as_posix()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
    if len(safe_name) <= 96:
        return safe_name
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    return f"{safe_name[:80]}__{digest}"



def crop_with_margin(
    image: Any,
    bbox: tuple[float, float, float, float],
    margin_ratio: float,
) -> tuple[Any, tuple[int, int, int, int]] | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    margin_x = box_width * margin_ratio
    margin_y = box_height * margin_ratio
    left = max(0, int(round(x1 - margin_x)))
    top = max(0, int(round(y1 - margin_y)))
    right = min(width, int(round(x2 + margin_x)))
    bottom = min(height, int(round(y2 + margin_y)))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right], (left, top, right, bottom)


def split_images(image_paths: list[Path], val_ratio: float, seed: int) -> tuple[set[Path], set[Path]]:
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    if len(shuffled) > 1 and val_ratio > 0:
        val_count = max(1, min(len(shuffled) - 1, val_count))
    val_paths = set(shuffled[:val_count])
    train_paths = set(shuffled[val_count:])
    return train_paths, val_paths


def prepare_output_dirs(output_root: Path) -> tuple[Path, Path]:
    yolo_root = output_root / "yolo_det"
    crops_root = output_root / "cls_crops"
    for path in (yolo_root, crops_root):
        if path.exists():
            shutil.rmtree(path)
    for subset in ("train", "val"):
        (yolo_root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / subset).mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)
    return yolo_root, crops_root


def write_data_yaml(yolo_root: Path) -> None:
    payload = {
        "path": yolo_root.resolve().as_posix(),
        "train": (yolo_root / "images" / "train").resolve().as_posix(),
        "val": (yolo_root / "images" / "val").resolve().as_posix(),
        "nc": 1,
        "names": ["tile_face"],
    }
    (yolo_root / "data.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def convert_dataset(
    input_root: Path,
    output_root: Path,
    classes_path: Path,
    *,
    val_ratio: float,
    seed: int,
    crop_margin: float,
) -> dict[str, Any]:
    class_labels, discard_labels = load_class_config(classes_path)
    if not class_labels:
        raise ValueError(f"No classification labels found in {classes_path}")
    class_set = set(class_labels)
    yolo_root, crops_root = prepare_output_dirs(output_root)
    for label in class_labels:
        (crops_root / label).mkdir(parents=True, exist_ok=True)

    all_images = image_paths_from_root(input_root)
    missing_json: list[str] = []
    paired_images: list[Path] = []
    for image_path in all_images:
        if image_path.with_suffix(".json").exists():
            paired_images.append(image_path)
        else:
            missing_json.append(image_path.relative_to(input_root).as_posix())

    train_paths, val_paths = split_images(paired_images, val_ratio, seed)
    class_counts = Counter({label: 0 for label in class_labels})
    bucket_counts = Counter({bucket: 0 for bucket in SIZE_BUCKETS})
    invalid_labels: list[dict[str, Any]] = []
    discarded_boxes = 0
    total_shapes = 0
    kept_boxes = 0
    crop_count = 0
    yolo_label_files = 0

    for image_path in paired_images:
        json_path = image_path.with_suffix(".json")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_height, image_width = image.shape[:2]
        width = int(payload.get("imageWidth") or image_width)
        height = int(payload.get("imageHeight") or image_height)
        subset = "val" if image_path in val_paths else "train"
        output_stem = safe_stem(image_path, input_root)
        output_image = yolo_root / "images" / subset / f"{output_stem}{image_path.suffix.lower()}"
        output_label = yolo_root / "labels" / subset / f"{output_stem}.txt"
        shutil.copy2(image_path, output_image)

        lines: list[str] = []
        for shape_index, shape in enumerate(payload.get("shapes", []), start=1):
            total_shapes += 1
            label = str(shape.get("label", ""))
            raw_bbox = bbox_from_points(shape.get("points", []))
            bbox = clip_bbox(raw_bbox, width, height)
            box_width = bbox[2] - bbox[0]
            box_height = bbox[3] - bbox[1]
            if label in discard_labels:
                discarded_boxes += 1
                continue
            if label not in class_set:
                invalid_labels.append(
                    {
                        "image": image_path.relative_to(input_root).as_posix(),
                        "label": label,
                        "bbox": [round(value, 2) for value in raw_bbox],
                    }
                )
                continue
            if box_width <= 0 or box_height <= 0:
                invalid_labels.append(
                    {
                        "image": image_path.relative_to(input_root).as_posix(),
                        "label": label,
                        "bbox": [round(value, 2) for value in raw_bbox],
                        "reason": "empty_box_after_clipping",
                    }
                )
                continue

            lines.append(yolo_line(bbox, width, height))
            class_counts[label] += 1
            bucket = size_bucket(box_width, box_height)
            bucket_counts[bucket] += 1
            kept_boxes += 1

            crop_result = crop_with_margin(image, bbox, crop_margin)
            if crop_result is not None:
                crop, (left, top, right, bottom) = crop_result
                crop_name = f"{output_stem}__box{shape_index:03d}__{bucket}__{right-left}x{bottom-top}.jpg"
                crop_path = crops_root / label / crop_name
                if not cv2.imwrite(str(crop_path), crop):
                    raise OSError(f"Failed to write crop: {crop_path}")
                crop_count += 1


        output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        yolo_label_files += 1

    write_data_yaml(yolo_root)

    warnings: list[str] = []
    for label in class_labels:
        count = class_counts[label]
        if count == 0:
            warnings.append(f"Class {label} has 0 boxes; please check whether it is absent or mislabeled.")
        elif count < 5:
            warnings.append(f"Class {label} has only {count} boxes; data may be too sparse.")
    if invalid_labels:
        warnings.append(f"Found {len(invalid_labels)} invalid labels; fix them in X-AnyLabeling before training.")
    if missing_json:
        warnings.append(f"Skipped {len(missing_json)} images without matching JSON files.")

    return {
        "summary": {
            "input_root": input_root.resolve().as_posix(),
            "output_root": output_root.resolve().as_posix(),
            "total_images": len(all_images),
            "paired_images": len(paired_images),
            "skipped_images_without_json": len(missing_json),
            "total_shapes": total_shapes,
            "kept_boxes": kept_boxes,
            "discarded_back_boxes": discarded_boxes,
            "invalid_label_boxes": len(invalid_labels),
            "crop_files": crop_count,
            "yolo_label_files": yolo_label_files,
            "train_images": len(train_paths),
            "val_images": len(val_paths),
            "val_ratio": val_ratio,
            "seed": seed,
        },
        "outputs": {
            "yolo_data_yaml": (yolo_root / "data.yaml").resolve().as_posix(),
            "yolo_root": yolo_root.resolve().as_posix(),
            "cls_crops_root": crops_root.resolve().as_posix(),
        },
        "missing_json_images": missing_json,
        "invalid_labels": invalid_labels,
        "class_distribution": {label: class_counts[label] for label in class_labels},
        "size_buckets": {bucket: bucket_counts[bucket] for bucket in SIZE_BUCKETS},
        "size_bucket_labels": SIZE_BUCKET_LABELS,
        "discard_labels": sorted(discard_labels),
        "warnings": warnings,
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Convert labels summary")
    print(f"  images: {summary['paired_images']}/{summary['total_images']} paired, {summary['skipped_images_without_json']} skipped")
    print(f"  boxes: {summary['kept_boxes']} kept, {summary['discarded_back_boxes']} discarded, {summary['invalid_label_boxes']} invalid")
    print(f"  split: {summary['train_images']} train / {summary['val_images']} val")
    print("  class distribution:")
    for label, count in report["class_distribution"].items():
        bar = "#" * min(40, int(count))
        print(f"    {label:7s} {count:4d} {bar}")
    print("  size buckets:")
    for bucket, count in report["size_buckets"].items():
        print(f"    {SIZE_BUCKET_LABELS[bucket]:7s} {count}")
    if report["warnings"]:
        print("  warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path.cwd().resolve()
    paths = load_yaml((project_root / args.paths).resolve() if not args.paths.is_absolute() else args.paths.resolve())

    input_root = args.input_root or resolve_config_path(
        paths.get("validation_labeled") or paths.get("labeled_dir"), base=project_root
    ) or Path("data/labeled")
    output_root = args.output_root or resolve_config_path(
        paths.get("validation_output") or paths.get("output_dir"), base=project_root
    ) or Path("output/validation_run_v1")
    classes_path = args.classes
    report_path = args.report

    input_root = (project_root / input_root).resolve() if not input_root.is_absolute() else input_root.resolve()
    output_root = (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    classes_path = (project_root / classes_path).resolve() if not classes_path.is_absolute() else classes_path.resolve()
    report_path = report_path or output_root / "convert_report.json"
    report_path = (project_root / report_path).resolve() if not report_path.is_absolute() else report_path.resolve()

    if not input_root.exists():
        print(f"Input root not found: {input_root}", file=sys.stderr)
        return 1
    if not classes_path.exists():
        print(f"Classes config not found: {classes_path}", file=sys.stderr)
        return 1
    if not 0 <= args.val_ratio < 1:
        print("--val-ratio must be in [0, 1)", file=sys.stderr)
        return 1

    try:
        report = convert_dataset(
            input_root,
            output_root,
            classes_path,
            val_ratio=args.val_ratio,
            seed=args.seed,
            crop_margin=args.crop_margin,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
