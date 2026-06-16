"""Prepare a versioned single-class YOLO mix dataset for the iterative prelabeler.

This script converts two sources into one single-class detector dataset:

* real X-AnyLabeling labels -> class 0 ``tile_face``; discard ``back``.
* Roboflow YOLO labels -> class 0 ``tile_face``.

The training mix is controlled by ``real_effective_ratio``. Real images are sampled
with replacement into the train image list, while image files are linked instead of
copied. The fixed real validation split is stored under ``real_val`` and remains
stable across runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import yaml

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DISCARD_LABELS = {"back", "tile_back"}
SIZE_BUCKETS = ("lt20", "20to40", "gt40")
SIZE_BUCKET_LABELS = {"lt20": "<20px", "20to40": "20~40px", "gt40": ">40px"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare real+Roboflow single-class datasets for an iterative Mahjong prelabeler."
    )
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"), help="Path to paths.yaml.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_iter.yaml"), help="Path to train_iter.yaml.")
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"), help="Path to classes.yaml.")
    parser.add_argument("--version", type=int, default=None, help="Dataset version; defaults to config version.")
    parser.add_argument("--force", action="store_true", help="Overwrite datasets/mix_v{N} if it already exists.")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def safe_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("").as_posix()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
    if len(safe) <= 120:
        return safe
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:100]}__{digest}"


def stable_float(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)


def author_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    match = re.match(r"(bili_\d+|dy_[^_]+|douyin_[^_]+)", path.stem)
    if match:
        return match.group(1)
    return path.stem.split("__", 1)[0]


def bbox_from_points(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    if len(points) < 2:
        raise ValueError("shape needs at least two points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def clip_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def yolo_line_from_xyxy(bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return "0 " + " ".join(f"{value:.6f}" for value in (cx / width, cy / height, box_w / width, box_h / height))


def size_bucket(width: float, height: float) -> str:
    short = min(width, height)
    if short < 20:
        return "lt20"
    if short <= 40:
        return "20to40"
    return "gt40"


def load_valid_labels(classes_path: Path) -> set[str]:
    payload = load_yaml(classes_path)
    labels: set[str] = set()
    for group in payload.get("classification", []):
        labels.update(str(label) for label in group)
    return labels


def ensure_clean_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {path}. Use --force or increment version.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def paired_real_images(labeled_root: Path) -> tuple[list[Path], list[str]]:
    missing: list[str] = []
    paired: list[Path] = []
    for image_path in image_paths(labeled_root):
        if image_path.with_suffix(".json").exists():
            paired.append(image_path)
        else:
            missing.append(image_path.relative_to(labeled_root).as_posix())
    return paired, missing


def load_or_update_real_val_manifest(
    real_images: Sequence[Path],
    labeled_root: Path,
    manifest_path: Path,
    *,
    ratio: float,
    min_per_author: int,
    seed: int,
) -> tuple[set[str], dict[str, Any]]:
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_val = {str(item) for item in previous.get("val_images", [])}
    existing_rels = {path.relative_to(labeled_root).as_posix() for path in real_images}
    val_rels = previous_val & existing_rels

    by_author: dict[str, list[str]] = defaultdict(list)
    for path in real_images:
        rel = path.relative_to(labeled_root).as_posix()
        by_author[author_from_path(path, labeled_root)].append(rel)

    author_counts: dict[str, dict[str, int]] = {}
    for author, rels in by_author.items():
        rels = sorted(rels)
        target = int(math.ceil(len(rels) * ratio)) if ratio > 0 else 0
        if len(rels) > 1 and ratio > 0:
            target = max(min_per_author, min(len(rels) - 1, target))
        current = sorted(rel for rel in rels if rel in val_rels)
        if len(current) < target:
            candidates = [rel for rel in rels if rel not in val_rels]
            candidates.sort(key=lambda rel: stable_float(f"val:{author}:{rel}", seed))
            val_rels.update(candidates[: target - len(current)])
        author_counts[author] = {
            "total": len(rels),
            "val": sum(1 for rel in rels if rel in val_rels),
            "train": sum(1 for rel in rels if rel not in val_rels),
        }

    manifest = {
        "seed": seed,
        "ratio": ratio,
        "min_per_author": min_per_author,
        "val_images": sorted(val_rels),
        "authors": dict(sorted(author_counts.items())),
    }
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return val_rels, manifest


def convert_real_image(
    image_path: Path,
    labeled_root: Path,
    label_out: Path,
    valid_labels: set[str],
) -> tuple[int, int, list[dict[str, Any]], Counter[str]]:
    payload = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_h, image_w = image.shape[:2]
    width = int(payload.get("imageWidth") or image_w)
    height = int(payload.get("imageHeight") or image_h)
    lines: list[str] = []
    invalid: list[dict[str, Any]] = []
    buckets: Counter[str] = Counter({bucket: 0 for bucket in SIZE_BUCKETS})
    kept = 0
    discarded = 0
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", ""))
        raw_bbox = bbox_from_points(shape.get("points", []))
        bbox = clip_bbox(raw_bbox, width, height)
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        rel_image = image_path.relative_to(labeled_root).as_posix()
        if label in DISCARD_LABELS:
            discarded += 1
            continue
        if label not in valid_labels:
            invalid.append({"image": rel_image, "label": label, "bbox": [round(value, 2) for value in raw_bbox]})
            continue
        if box_w <= 0 or box_h <= 0:
            invalid.append(
                {"image": rel_image, "label": label, "bbox": [round(value, 2) for value in raw_bbox], "reason": "empty_box"}
            )
            continue
        lines.append(yolo_line_from_xyxy(bbox, width, height))
        buckets[size_bucket(box_w, box_h)] += 1
        kept += 1
    write_text(label_out, "\n".join(lines) + ("\n" if lines else ""))
    return kept, discarded, invalid, buckets


def find_roboflow_data(paths_cfg: dict[str, Any], project_root: Path) -> Path | None:
    candidates = [
        resolve_path(paths_cfg.get("prelabel_roboflow_data"), project_root),
        resolve_path(paths_cfg.get("prelabel_source_data"), project_root),
    ]
    root = resolve_path(paths_cfg.get("prelabel_roboflow_root"), project_root)
    if root is not None:
        candidates.append(root / "data.yaml")
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return None


def roboflow_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def resolve_dataset_item(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    parts = list(path.parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    fallback = base / Path(*parts) if parts else base
    return fallback if fallback.exists() else candidate


def roboflow_images(data_yaml: Path, sample_limit: int | None = None) -> list[Path]:
    payload = load_yaml(data_yaml)
    base = resolve_path(payload.get("path"), data_yaml.parent) or data_yaml.parent
    paths: list[Path] = []
    for key in ("train", "val", "valid", "test"):
        value = payload.get(key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            source = resolve_dataset_item(item, base)
            if source.is_file() and source.suffix.lower() == ".txt":
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        paths.append(resolve_dataset_item(line.strip(), source.parent))
            elif source.is_dir():
                paths.extend(image_paths(source))
            elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(source)
    unique = sorted({path.resolve() for path in paths if path.exists()})
    return unique[:sample_limit] if sample_limit else unique



def flatten_roboflow_label(src_label: Path, dst_label: Path) -> tuple[int, int]:
    if not src_label.exists():
        write_text(dst_label, "")
        return 0, 1
    lines: list[str] = []
    for raw in src_label.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) >= 5:
            lines.append("0 " + " ".join(parts[1:5]))
    write_text(dst_label, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines), 0


def recommended_ratio(real_count: int) -> str:
    if real_count < 150:
        return "0.30~0.35"
    if real_count < 400:
        return "0.40~0.55"
    if real_count < 800:
        return "0.60~0.75"
    return "0.80~1.00"


def repeat_to_ratio(real_items: Sequence[Path], roboflow_items: Sequence[Path], ratio: float, seed: int) -> list[Path]:
    if not real_items or ratio <= 0:
        return []
    if not roboflow_items or ratio >= 1:
        return list(real_items)
    target_real = int(round((ratio * len(roboflow_items)) / (1.0 - ratio)))
    target_real = max(len(real_items), target_real)
    ordered = sorted(real_items, key=lambda path: stable_float(f"sample:{path.as_posix()}", seed))
    return [ordered[index % len(ordered)] for index in range(target_real)]


def write_data_yaml(mix_root: Path, train_list: Path, val_list: Path) -> None:
    payload = {
        "path": mix_root.resolve().as_posix(),
        "train": train_list.resolve().as_posix(),
        "val": val_list.resolve().as_posix(),
        "nc": 1,
        "names": ["tile_face"],
    }
    write_text(mix_root / "data.yaml", yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def prepare_dataset(
    paths_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    classes_path: Path,
    project_root: Path,
    *,
    version: int,
    force: bool,
) -> dict[str, Any]:
    seed = int(train_cfg.get("seed", 42))
    ratio = float(train_cfg.get("real_effective_ratio", 0.35))
    if not 0 <= ratio <= 1:
        raise ValueError("real_effective_ratio must be in [0, 1]")
    val_ratio = float(train_cfg.get("real_val_ratio_per_author", 0.15))
    min_val = int(train_cfg.get("min_val_per_author", 1))
    use_roboflow = bool(train_cfg.get("use_roboflow", True))
    roboflow_limit = train_cfg.get("roboflow_sample_limit")
    roboflow_limit = int(roboflow_limit) if roboflow_limit is not None else None

    labeled_root = resolve_path(paths_cfg.get("prelabel_labeled_root") or paths_cfg.get("validation_labeled"), project_root)
    datasets_root = resolve_path(paths_cfg.get("prelabel_datasets_root"), project_root) or project_root / "datasets"
    real_val_root = resolve_path(paths_cfg.get("prelabel_real_val_root"), project_root) or datasets_root / "real_val"
    if labeled_root is None or not labeled_root.exists():
        raise FileNotFoundError(f"Labeled root not found: {labeled_root}")
    valid_labels = load_valid_labels(classes_path)
    if not valid_labels:
        raise ValueError(f"No classification labels found in {classes_path}")

    mix_root = datasets_root / f"mix_v{version}"
    ensure_clean_dir(mix_root, force=force)
    for subdir in ("images/train", "images/val", "labels/train", "labels/val", "manifests"):
        (mix_root / subdir).mkdir(parents=True, exist_ok=True)
    real_val_root.mkdir(parents=True, exist_ok=True)

    real_images, missing_json = paired_real_images(labeled_root)
    val_rels, val_manifest = load_or_update_real_val_manifest(
        real_images,
        labeled_root,
        real_val_root / "manifest.json",
        ratio=val_ratio,
        min_per_author=min_val,
        seed=seed,
    )

    real_train_links: list[Path] = []
    real_val_links: list[Path] = []
    real_train_source_by_link: dict[Path, Path] = {}
    invalid_labels: list[dict[str, Any]] = []
    author_dist: Counter[str] = Counter()
    size_buckets: Counter[str] = Counter({bucket: 0 for bucket in SIZE_BUCKETS})
    link_modes: Counter[str] = Counter()
    real_kept_boxes = 0
    discarded_boxes = 0

    for image_path in real_images:
        rel = image_path.relative_to(labeled_root).as_posix()
        subset = "val" if rel in val_rels else "train"
        author_dist[author_from_path(image_path, labeled_root)] += 1
        stem = "real__" + safe_stem(image_path, labeled_root)
        linked_image = mix_root / "images" / subset / f"{stem}{image_path.suffix.lower()}"
        linked_label = mix_root / "labels" / subset / f"{stem}.txt"
        link_modes[link_file(image_path, linked_image)] += 1
        kept, discarded, invalid, buckets = convert_real_image(image_path, labeled_root, linked_label, valid_labels)
        real_kept_boxes += kept
        discarded_boxes += discarded
        invalid_labels.extend(invalid)
        size_buckets.update(buckets)
        if subset == "train":
            real_train_links.append(linked_image)
            real_train_source_by_link[linked_image] = image_path
        else:
            real_val_links.append(linked_image)

    roboflow_links: list[Path] = []
    roboflow_boxes = 0
    roboflow_missing_labels = 0
    roboflow_data = find_roboflow_data(paths_cfg, project_root) if use_roboflow else None
    if roboflow_data is not None:
        for idx, image_path in enumerate(roboflow_images(roboflow_data, roboflow_limit), start=1):
            stem = f"rf__{idx:06d}__{safe_stem(image_path, roboflow_data.parent)}"
            linked_image = mix_root / "images" / "train" / f"{stem}{image_path.suffix.lower()}"
            linked_label = mix_root / "labels" / "train" / f"{stem}.txt"
            link_modes[link_file(image_path, linked_image)] += 1
            boxes, missing = flatten_roboflow_label(roboflow_label_path(image_path) or image_path.with_suffix(".txt"), linked_label)
            roboflow_boxes += boxes
            roboflow_missing_labels += missing
            roboflow_links.append(linked_image)

    sampled_real = repeat_to_ratio(real_train_links, roboflow_links, ratio, seed)
    train_entries = sampled_real + roboflow_links
    actual_real_ratio = (len(sampled_real) / len(train_entries)) if train_entries else 0.0

    train_list = mix_root / "train.txt"
    val_list = mix_root / "real_val.txt"
    write_text(train_list, "\n".join(path.resolve().as_posix() for path in train_entries) + ("\n" if train_entries else ""))
    write_text(val_list, "\n".join(path.resolve().as_posix() for path in real_val_links) + ("\n" if real_val_links else ""))
    write_data_yaml(mix_root, train_list, val_list)

    snapshot = {
        "version": version,
        "config": train_cfg,
        "paths": paths_cfg,
        "real_train_sources": sorted(path.relative_to(labeled_root).as_posix() for path in real_train_source_by_link.values()),
        "real_val_sources": sorted(val_rels),
        "roboflow_data": roboflow_data.as_posix() if roboflow_data else None,
    }
    write_text(mix_root / "manifests" / "snapshot.json", json.dumps(snapshot, ensure_ascii=False, indent=2))

    report = {
        "version": version,
        "outputs": {
            "mix_root": mix_root.resolve().as_posix(),
            "data_yaml": (mix_root / "data.yaml").resolve().as_posix(),
            "train_list": train_list.resolve().as_posix(),
            "real_val_list": val_list.resolve().as_posix(),
            "real_val_manifest": (real_val_root / "manifest.json").resolve().as_posix(),
        },
        "summary": {
            "real_total_images": len(real_images),
            "real_train_unique_images": len(real_train_links),
            "real_val_images": len(real_val_links),
            "roboflow_images": len(roboflow_links),
            "sampled_real_train_entries": len(sampled_real),
            "sampled_roboflow_train_entries": len(roboflow_links),
            "train_entries": len(train_entries),
            "target_real_effective_ratio": ratio,
            "actual_real_effective_ratio": round(actual_real_ratio, 6),
            "ratio_abs_error": round(abs(actual_real_ratio - ratio), 6),
            "recommended_real_effective_ratio": recommended_ratio(len(real_images)),
            "real_kept_boxes": real_kept_boxes,
            "discarded_back_boxes": discarded_boxes,
            "roboflow_boxes": roboflow_boxes,
            "invalid_label_boxes": len(invalid_labels),
            "missing_real_json_images": len(missing_json),
            "roboflow_missing_label_files": roboflow_missing_labels,
        },
        "real_val": val_manifest,
        "author_distribution": dict(sorted(author_dist.items())),
        "size_buckets": {bucket: size_buckets[bucket] for bucket in SIZE_BUCKETS},
        "size_bucket_labels": SIZE_BUCKET_LABELS,
        "invalid_labels": invalid_labels,
        "missing_real_json_images": missing_json,
        "link_modes": dict(link_modes),
        "warnings": [],
    }
    if roboflow_data is None and use_roboflow:
        report["warnings"].append("Roboflow data.yaml not found; mixed dataset contains real data only.")
    if abs(actual_real_ratio - ratio) > 0.03 and roboflow_links:
        report["warnings"].append("Actual real effective ratio differs from target by more than 3%.")
    if invalid_labels:
        report["warnings"].append("Invalid labels found; fix them before serious training.")
    if any(mode == "copy" for mode in link_modes):
        report["warnings"].append("Some files were copied because symlink/hardlink creation was unavailable.")

    write_text(mix_root / "prepare_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"Prepared mix_v{report['version']}: {report['outputs']['data_yaml']}")
    print(
        "  real: "
        f"{summary['real_total_images']} total, {summary['real_train_unique_images']} train, "
        f"{summary['real_val_images']} real_val"
    )
    print(f"  roboflow: {summary['roboflow_images']} images, {summary['roboflow_boxes']} boxes")
    print(
        "  effective ratio: "
        f"target={summary['target_real_effective_ratio']:.3f}, "
        f"actual={summary['actual_real_effective_ratio']:.3f}"
    )
    print(f"  recommended ratio for current real size: {summary['recommended_real_effective_ratio']}")
    if report["warnings"]:
        print("  warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path.cwd().resolve()
    paths_path = args.paths.resolve() if args.paths.is_absolute() else (project_root / args.paths).resolve()
    config_path = args.config.resolve() if args.config.is_absolute() else (project_root / args.config).resolve()
    classes_path = args.classes.resolve() if args.classes.is_absolute() else (project_root / args.classes).resolve()
    try:
        paths_cfg = load_yaml(paths_path)
        train_cfg = load_yaml(config_path)
        version = int(args.version if args.version is not None else train_cfg.get("version", 1))
        report = prepare_dataset(
            paths_cfg,
            train_cfg,
            classes_path,
            project_root,
            version=version,
            force=args.force,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
