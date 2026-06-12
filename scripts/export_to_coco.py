"""Convert X-AnyLabeling per-image JSON files into a COCO annotations file.

Example:
    python scripts/export_to_coco.py --input-dir data/frames_selected --output data/test_set_v1/annotations/instances_default.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert X-AnyLabeling annotations into a COCO dataset.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing images and X-AnyLabeling JSON files.")
    parser.add_argument("--output", type=Path, required=True, help="Output COCO annotations JSON path.")
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("configs/classes.yaml"),
        help="Path to classes.yaml used to define category ids and names.",
    )
    return parser


def load_categories(classes_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for group in payload.get("classification", []):
        names.extend(str(name) for name in group)
    return [{"id": index + 1, "name": name} for index, name in enumerate(names)]


def extract_source(file_name: str) -> str:
    parts = Path(file_name).name.split("__")
    return parts[0] if parts else "unknown"


def normalize_bbox(points: Sequence[Sequence[float]]) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def build_coco_dataset(input_dir: Path, classes_path: Path) -> dict[str, Any]:
    categories = load_categories(classes_path)
    category_ids = {str(category["name"]): int(category["id"]) for category in categories}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    json_paths = sorted(input_dir.glob("*.json"))
    for image_id, json_path in enumerate(json_paths, start=1):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_file = str(payload.get("imagePath") or f"{json_path.stem}.jpg")
        image_path = input_dir / image_file
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension for {image_file}")
        if not image_path.exists():
            raise ValueError(f"Image file referenced by annotation does not exist: {image_path}")

        width = int(payload["imageWidth"])
        height = int(payload["imageHeight"])
        images.append(
            {
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
                "source": extract_source(image_file),
            }
        )

        for shape in payload.get("shapes", []):
            label = str(shape["label"])
            if label not in category_ids:
                raise ValueError(f"Unknown label '{label}' in {json_path.name}")
            bbox = normalize_bbox(shape["points"])
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[label],
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    classes_path = args.classes.resolve()

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1
    if not classes_path.exists():
        print(f"Classes config not found: {classes_path}", file=sys.stderr)
        return 1

    try:
        dataset = build_coco_dataset(input_dir, classes_path)
    except (KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote COCO annotations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
