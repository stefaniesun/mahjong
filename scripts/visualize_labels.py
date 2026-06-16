"""Render X-AnyLabeling label previews and confusing-class crop sheets.

Example:
    python scripts/visualize_labels.py --paths configs/paths.yaml --classes configs/classes.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from scripts.convert_labels import (
        bbox_from_points,
        clip_bbox,
        crop_with_margin,
        image_paths_from_root,
        load_class_config,
        load_yaml,
        resolve_config_path,
        safe_stem,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from convert_labels import (
        bbox_from_points,
        clip_bbox,
        crop_with_margin,
        image_paths_from_root,
        load_class_config,
        load_yaml,
        resolve_config_path,
        safe_stem,
    )


UNKNOWN_COLOR = (160, 160, 160)
INVALID_COLOR = (255, 0, 255)
SUIT_COLORS = {
    "w": (60, 170, 255),
    "t": (80, 210, 80),
    "b": (255, 120, 70),
    "unknown": UNKNOWN_COLOR,
}
DEFAULT_CONFUSING_CLASSES = ["t4", "t5", "t6", "t9", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw X-AnyLabeling boxes for manual label review.")
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"), help="Path config YAML.")
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"), help="Class config YAML.")
    parser.add_argument("--input-root", type=Path, default=None, help="Directory containing labeled images and JSON files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Preview output directory.")
    parser.add_argument("--count", type=int, default=20, help="Number of full-image previews to render.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--crop-margin", type=float, default=0.08, help="Crop margin ratio for confusing-class crops.")
    parser.add_argument("--per-class", type=int, default=5, help="Crops per confusing class in the contact sheet.")
    parser.add_argument(
        "--confusing-classes",
        nargs="*",
        default=DEFAULT_CONFUSING_CLASSES,
        help="Classes to sample for the confusing-class crop sheet.",
    )
    return parser


def label_color(label: str, *, valid: bool = True) -> tuple[int, int, int]:
    if not valid:
        return INVALID_COLOR
    if label == "unknown":
        return UNKNOWN_COLOR
    return SUIT_COLORS.get(label[:1], UNKNOWN_COLOR)


def sample_items(items: Sequence[Path], count: int, seed: int) -> list[Path]:
    sampled = list(items)
    random.Random(seed).shuffle(sampled)
    return sampled[: max(0, min(count, len(sampled)))]


def load_shapes(json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return list(payload.get("shapes", []))


def text_box_origin(
    desired_x: int,
    desired_y: int,
    text_size: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> tuple[int, int, tuple[int, int, int, int]]:
    text_width, text_height = text_size
    x = max(0, min(desired_x, image_width - text_width - 4))
    y = max(text_height + 4, min(desired_y, image_height - 2))
    for _ in range(80):
        rect = (x, y - text_height - 4, x + text_width + 4, y + 4)
        if not any(overlaps(rect, other) for other in occupied):
            return x, y, rect
        y += text_height + 8
        if y > image_height - 2:
            y = text_height + 4
            x = min(image_width - text_width - 4, x + 24)
    rect = (x, max(0, y - text_height - 4), x + text_width + 4, min(image_height, y + 4))
    return x, min(y, image_height - 2), rect


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def draw_labeled_image(
    image: Any,
    shapes: Sequence[dict[str, Any]],
    class_labels: set[str],
    discard_labels: set[str],
) -> tuple[Any, int]:
    rendered = image.copy()
    image_height, image_width = rendered.shape[:2]
    occupied: list[tuple[int, int, int, int]] = []
    drawn = 0
    sorted_shapes = sorted(shapes, key=lambda item: (bbox_from_points(item.get("points", []))[1], bbox_from_points(item.get("points", []))[0]))
    for shape in sorted_shapes:
        label = str(shape.get("label", ""))
        if label in discard_labels:
            continue
        bbox = clip_bbox(bbox_from_points(shape.get("points", [])), image_width, image_height)
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        valid = label in class_labels
        color = label_color(label, valid=valid)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        text = label if valid else f"INVALID:{label}"
        (text_width, text_height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx, ty, rect = text_box_origin(x1, max(text_height + 4, y1 - 4), (text_width, text_height), occupied, image_width, image_height)
        occupied.append(rect)
        cv2.rectangle(rendered, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(rendered, text, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        drawn += 1
    return rendered, drawn


def render_preview_images(
    input_root: Path,
    output_dir: Path,
    class_labels: set[str],
    discard_labels: set[str],
    *,
    count: int,
    seed: int,
) -> list[Path]:
    preview_dir = output_dir / "images"
    preview_dir.mkdir(parents=True, exist_ok=True)
    paired_images = [path for path in image_paths_from_root(input_root) if path.with_suffix(".json").exists()]
    outputs: list[Path] = []
    for image_path in sample_items(paired_images, count, seed):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        rendered, drawn = draw_labeled_image(image, load_shapes(image_path.with_suffix(".json")), class_labels, discard_labels)
        out_path = preview_dir / f"{safe_stem(image_path, input_root)}__{drawn:03d}boxes.jpg"
        if not cv2.imwrite(str(out_path), rendered):
            raise OSError(f"Failed to write preview: {out_path}")
        outputs.append(out_path)
    return outputs


def collect_confusing_crops(
    input_root: Path,
    class_labels: set[str],
    discard_labels: set[str],
    confusing_classes: Sequence[str],
    *,
    per_class: int,
    seed: int,
    crop_margin: float,
) -> dict[str, list[Any]]:
    requested = [label for label in confusing_classes if label in class_labels]
    samples: dict[str, list[tuple[str, Any]]] = {label: [] for label in requested}
    for image_path in image_paths_from_root(input_root):
        if not image_path.with_suffix(".json").exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_height, image_width = image.shape[:2]
        for index, shape in enumerate(load_shapes(image_path.with_suffix(".json")), start=1):
            label = str(shape.get("label", ""))
            if label not in samples or label in discard_labels:
                continue
            bbox = clip_bbox(bbox_from_points(shape.get("points", [])), image_width, image_height)
            crop_result = crop_with_margin(image, bbox, crop_margin)
            if crop_result is None:
                continue
            crop, _coords = crop_result
            samples[label].append((f"{safe_stem(image_path, input_root)} #{index}", crop))
    rng = random.Random(seed)
    result: dict[str, list[Any]] = {}
    for label, crops in samples.items():
        rng.shuffle(crops)
        result[label] = [crop for _name, crop in crops[:per_class]]
    return result


def make_contact_sheet(crops_by_label: dict[str, list[Any]], *, cell_size: int = 120, label_width: int = 70) -> Any:
    labels = list(crops_by_label)
    columns = max(1, max((len(crops) for crops in crops_by_label.values()), default=1))
    height = max(1, len(labels)) * cell_size
    width = label_width + columns * cell_size
    sheet = np.full((height, width, 3), 245, dtype=np.uint8)
    for row, label in enumerate(labels):
        y = row * cell_size
        color = label_color(label)
        cv2.rectangle(sheet, (0, y), (label_width - 1, y + cell_size - 1), color, -1)
        cv2.putText(sheet, label, (8, y + cell_size // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        for col, crop in enumerate(crops_by_label[label]):
            resized = resize_to_cell(crop, cell_size - 12)
            top = y + (cell_size - resized.shape[0]) // 2
            left = label_width + col * cell_size + (cell_size - resized.shape[1]) // 2
            sheet[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
            cv2.rectangle(sheet, (label_width + col * cell_size, y), (label_width + (col + 1) * cell_size - 1, y + cell_size - 1), (210, 210, 210), 1)
    return sheet


def resize_to_cell(image: Any, max_side: int) -> Any:
    height, width = image.shape[:2]
    scale = min(max_side / max(width, height), 1.0 if max(width, height) <= max_side else max_side / max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def write_confusing_sheet(
    input_root: Path,
    output_dir: Path,
    class_labels: set[str],
    discard_labels: set[str],
    confusing_classes: Sequence[str],
    *,
    per_class: int,
    seed: int,
    crop_margin: float,
) -> tuple[Path, dict[str, int]]:
    crops_by_label = collect_confusing_crops(
        input_root,
        class_labels,
        discard_labels,
        confusing_classes,
        per_class=per_class,
        seed=seed,
        crop_margin=crop_margin,
    )
    sheet = make_contact_sheet(crops_by_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "confusing_classes_contact_sheet.jpg"
    if not cv2.imwrite(str(out_path), sheet):
        raise OSError(f"Failed to write confusing-class contact sheet: {out_path}")
    return out_path, {label: len(crops) for label, crops in crops_by_label.items()}


def visualize_dataset(
    input_root: Path,
    output_dir: Path,
    classes_path: Path,
    *,
    count: int,
    seed: int,
    crop_margin: float,
    per_class: int,
    confusing_classes: Sequence[str],
) -> dict[str, Any]:
    class_labels_list, discard_labels = load_class_config(classes_path)
    class_labels = set(class_labels_list)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_paths = render_preview_images(
        input_root,
        output_dir,
        class_labels,
        discard_labels,
        count=count,
        seed=seed,
    )
    sheet_path, crop_counts = write_confusing_sheet(
        input_root,
        output_dir,
        class_labels,
        discard_labels,
        confusing_classes,
        per_class=per_class,
        seed=seed,
        crop_margin=crop_margin,
    )
    report = {
        "input_root": input_root.resolve().as_posix(),
        "output_dir": output_dir.resolve().as_posix(),
        "preview_images": [path.resolve().as_posix() for path in preview_paths],
        "preview_count": len(preview_paths),
        "confusing_contact_sheet": sheet_path.resolve().as_posix(),
        "confusing_crop_counts": crop_counts,
        "discard_labels": sorted(discard_labels),
    }
    (output_dir / "visualize_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path.cwd().resolve()
    paths_path = (project_root / args.paths).resolve() if not args.paths.is_absolute() else args.paths.resolve()
    paths = load_yaml(paths_path)
    input_root = args.input_root or resolve_config_path(
        paths.get("validation_labeled") or paths.get("labeled_dir"), base=project_root
    ) or Path("data/labeled")
    output_dir = args.output_dir or resolve_config_path(paths.get("label_preview_output"), base=project_root) or Path("output/label_preview")
    classes_path = args.classes
    input_root = (project_root / input_root).resolve() if not input_root.is_absolute() else input_root.resolve()
    output_dir = (project_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    classes_path = (project_root / classes_path).resolve() if not classes_path.is_absolute() else classes_path.resolve()

    if not input_root.exists():
        print(f"Input root not found: {input_root}", file=sys.stderr)
        return 1
    if not classes_path.exists():
        print(f"Classes config not found: {classes_path}", file=sys.stderr)
        return 1
    if args.count < 0 or args.per_class < 0:
        print("--count and --per-class must be non-negative", file=sys.stderr)
        return 1

    try:
        report = visualize_dataset(
            input_root,
            output_dir,
            classes_path,
            count=args.count,
            seed=args.seed,
            crop_margin=args.crop_margin,
            per_class=args.per_class,
            confusing_classes=args.confusing_classes,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Label visualization summary")
    print(f"  preview images: {report['preview_count']}")
    print(f"  output dir: {output_dir}")
    print(f"  confusing sheet: {report['confusing_contact_sheet']}")
    print("  confusing crop counts:")
    for label, count in report["confusing_crop_counts"].items():
        print(f"    {label:4s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
