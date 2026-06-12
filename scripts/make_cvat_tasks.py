"""Create CVAT-friendly zip task bundles from selected frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack selected frames into CVAT upload zip tasks and write a task manifest."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/frames_selected"),
        help="Directory containing selected frame images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cvat_tasks"),
        help="Directory where task zip files will be written.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the generated task manifest JSON. Defaults to <output-root>/task_manifest.json.",
    )
    parser.add_argument(
        "--task-size",
        type=int,
        default=50,
        help="Maximum number of images per CVAT task zip.",
    )
    parser.add_argument(
        "--task-prefix",
        type=str,
        default="mahjong-task",
        help="Prefix for generated task names and zip filenames.",
    )
    return parser


def collect_images(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def chunk_paths(paths: Sequence[Path], chunk_size: int) -> list[list[Path]]:
    return [list(paths[index : index + chunk_size]) for index in range(0, len(paths), chunk_size)]


def write_task_zip(zip_path: Path, images: Sequence[Path]) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in images:
            archive.write(image_path, arcname=image_path.name)


def build_manifest(tasks: list[dict[str, object]], total_images: int, task_size: int, input_root: Path) -> dict[str, object]:
    return {
        "input_root": str(input_root),
        "summary": {
            "total_images": total_images,
            "task_count": len(tasks),
            "task_size": task_size,
        },
        "tasks": tasks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.task_size <= 0:
        parser.error("--task-size must be a positive integer")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else output_root / "task_manifest.json"

    if not input_root.exists() or not input_root.is_dir():
        print(f"Input root does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 1

    images = collect_images(input_root)
    if not images:
        print(f"No images found under {input_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    task_chunks = chunk_paths(images, args.task_size)
    task_entries: list[dict[str, object]] = []

    total_tasks = len(task_chunks)
    index_width = max(3, len(str(total_tasks))) if total_tasks else 3

    for index, chunk in enumerate(task_chunks, start=1):
        task_name = f"{args.task_prefix}-{index:0{index_width}d}"
        zip_name = f"{task_name}.zip"
        zip_path = output_root / zip_name
        write_task_zip(zip_path, chunk)
        task_entries.append(
            {
                "task_index": index,
                "task_name": task_name,
                "zip_path": str(zip_path),
                "image_count": len(chunk),
                "images": [path.name for path in chunk],
            }
        )

    manifest_payload = build_manifest(task_entries, total_images=len(images), task_size=args.task_size, input_root=input_root)
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Packed {len(images)} images into {len(task_entries)} task(s). Manifest: {manifest_path}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
