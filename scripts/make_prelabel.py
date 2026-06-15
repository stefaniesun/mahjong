"""Generate X-AnyLabeling prelabels from the temporary YOLO prelabeler.

Example:
    python scripts/make_prelabel.py --input-root data/frames_selected --paths configs/paths.yaml --prelabel-map configs/prelabel_map.yaml
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
    parser = argparse.ArgumentParser(description="Generate X-AnyLabeling JSON prelabels from the YOLO prelabeler.")
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing selected frames.")
    parser.add_argument("--model", type=Path, default=None, help="Path to the YOLO prelabeler model. Overrides --paths.")
    parser.add_argument(
        "--paths",
        type=Path,
        default=Path("configs/paths.yaml"),
        help="Path to paths.yaml containing prelabeler_onnx.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("configs/classes.yaml"),
        help="Path to classes.yaml used to validate and map labels.",
    )
    parser.add_argument(
        "--prelabel-map",
        type=Path,
        default=Path("configs/prelabel_map.yaml"),
        help="27-class prelabeler to project-label mapping YAML.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference.")
    return parser


def load_categories(classes_path: Path) -> set[str]:
    payload = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for group in payload.get("classification", []):
        names.update(str(name) for name in group)
    return names


def load_prelabel_map(map_path: Path) -> dict[str, str]:
    payload = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
    raw_map = payload.get("map", {})
    if not isinstance(raw_map, dict):
        raise ValueError(f"Invalid prelabel map: {map_path}")
    mapped: dict[str, str] = {}
    for raw_label, value in raw_map.items():
        if not isinstance(value, dict) or "cls" not in value:
            raise ValueError(f"Invalid mapping for {raw_label}: {value}")
        mapped[str(raw_label)] = str(value["cls"])
    return mapped


def load_model_path_from_paths(paths_path: Path) -> Path:
    payload = yaml.safe_load(paths_path.read_text(encoding="utf-8")) or {}
    model = payload.get("prelabeler_onnx") or payload.get("prelabeler_pt")
    if not model:
        raise ValueError(f"prelabeler_onnx not found in paths config: {paths_path}")
    return Path(str(model))


def load_model(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("ultralytics is required to run make_prelabel.py") from exc
    return YOLO(str(model_path))



def image_paths_from_root(input_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]



def map_label(raw_label: str, valid_labels: set[str], prelabel_map: dict[str, str] | None = None) -> str:
    mapped_label = prelabel_map.get(raw_label, raw_label) if prelabel_map else raw_label
    return mapped_label if mapped_label in valid_labels else "unknown"



def build_shape(label: str, bbox: Sequence[float]) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
    }


def write_label_file(image_path: Path, width: int, height: int, shapes: Sequence[dict[str, Any]]) -> None:
    payload = {
        "version": "2.4.0",
        "flags": {},
        "shapes": list(shapes),
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    label_path = image_path.with_suffix(".json")
    label_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_prelabels(
    model: Any,
    image_paths: Sequence[Path],
    valid_labels: set[str],
    conf: float,
    prelabel_map: dict[str, str] | None = None,
) -> None:
    for image_path in image_paths:
        results = model.predict(source=str(image_path), conf=conf, verbose=False)
        result = results[0]
        height, width = [int(value) for value in result.orig_shape]
        names = getattr(model, "names", {})
        boxes = result.boxes
        xyxy_items = boxes.xyxy.cpu().tolist() if boxes is not None else []
        cls_items = boxes.cls.cpu().tolist() if boxes is not None else []

        shapes = []
        for bbox, cls_idx in zip(xyxy_items, cls_items):
            raw_label = str(names[int(cls_idx)])
            label = map_label(raw_label, valid_labels, prelabel_map)
            shapes.append(build_shape(label, bbox))

        write_label_file(image_path, width, height, shapes)



def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_root = args.input_root.resolve()
    classes_path = args.classes.resolve()
    paths_path = args.paths.resolve()
    prelabel_map_path = args.prelabel_map.resolve()

    if not input_root.exists():
        print(f"Input root not found: {input_root}", file=sys.stderr)
        return 1
    if not classes_path.exists():
        print(f"Classes config not found: {classes_path}", file=sys.stderr)
        return 1
    if not prelabel_map_path.exists():
        print(f"Prelabel map not found: {prelabel_map_path}", file=sys.stderr)
        return 1

    try:
        model_path = args.model.resolve() if args.model else load_model_path_from_paths(paths_path).resolve()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    image_paths = image_paths_from_root(input_root)
    if not image_paths:
        print(f"No input images found under: {input_root}", file=sys.stderr)
        return 1

    valid_labels = load_categories(classes_path)
    valid_labels.add("unknown")
    prelabel_map = load_prelabel_map(prelabel_map_path)
    model = load_model(model_path)
    generate_prelabels(model, image_paths, valid_labels, args.conf, prelabel_map)
    print(f"Wrote {len(image_paths)} X-AnyLabeling prelabel files to {input_root}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
