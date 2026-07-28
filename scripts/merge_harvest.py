"""Merge hand-verified harvest crops into the classifier training set (Phase 3 task 4).

Only `review/` is merged: those labels are ones you confirmed by eye. `auto/` holds the
model's own guesses, and folding those back in is self-training — it re-teaches the model
what it already believes and can cement the very confusions we are trying to fix.

Crops land in the train split only; the existing val split is left untouched so that
before/after numbers stay comparable.

Example:
    python scripts/merge_harvest.py --harvest output/harvest_v1 --dataset output/cls_train_v1 \
        --out output/cls_train_v2 --only-classes w
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Sequence

SKIP_DIRS = {"_bad", "_dup", "unknown"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge verified harvest crops into the training set.")
    parser.add_argument("--harvest", type=Path, required=True, help="Harvest root produced by harvest_crops.py.")
    parser.add_argument("--dataset", type=Path, required=True, help="Existing dataset root with train/ and val/.")
    parser.add_argument("--out", type=Path, required=True, help="New dataset root to create.")
    parser.add_argument("--only-classes", default="", help="Comma-separated class names or prefixes to merge (e.g. 'w'). Empty merges every class.")
    parser.add_argument("--include-auto", action="store_true", help="Also merge auto/ (model-guessed labels). Off by default on purpose.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be merged without writing.")
    return parser


def wanted(name: str, rules: Sequence[str]) -> bool:
    if not rules:
        return True
    return any(name == rule or name.startswith(rule) for rule in rules)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rules = [r.strip() for r in args.only_classes.split(",") if r.strip()]

    lanes = ["review"] + (["auto"] if args.include_auto else [])
    added: Counter[str] = Counter()
    for lane in lanes:
        lane_dir = args.harvest / lane
        if not lane_dir.is_dir():
            continue
        for class_dir in sorted(lane_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name in SKIP_DIRS:
                continue
            if not wanted(class_dir.name, rules):
                continue
            added[class_dir.name] += len(list(class_dir.glob("*.jpg")))

    if not added:
        print("没有可合并的 crop，检查 --harvest 路径与 --only-classes")
        return 1

    print(f"将合并 {sum(added.values())} 张 crop（来源 lane: {', '.join(lanes)}）")
    for name, count in sorted(added.items()):
        print(f"  {name:8s} +{count}")

    if args.dry_run:
        print("\n--dry-run：未写入任何文件")
        return 0

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.dataset, args.out)

    copied = 0
    for lane in lanes:
        lane_dir = args.harvest / lane
        if not lane_dir.is_dir():
            continue
        for class_dir in sorted(lane_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name in SKIP_DIRS:
                continue
            if not wanted(class_dir.name, rules):
                continue
            target = args.out / "train" / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for crop in class_dir.glob("*.jpg"):
                shutil.copy2(crop, target / f"harvest__{crop.name}")
                copied += 1

    print(f"\n已写入 {args.out}（新增 {copied} 张到 train/）")
    for split in ("train", "val"):
        split_dir = args.out / split
        if not split_dir.is_dir():
            continue
        total = sum(len(list(d.glob("*.jpg"))) for d in split_dir.iterdir() if d.is_dir())
        print(f"  {split}: {total} 张")
    print("\n各类 train 数量：")
    train_dir = args.out / "train"
    counts = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(train_dir.iterdir()) if d.is_dir()}
    for name, count in counts.items():
        mark = f"  (+{added[name]})" if name in added else ""
        print(f"  {name:8s} {count:5d}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
