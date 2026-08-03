"""Apply a reviewed patch to the zone label set.

Takes the JSON exported by review.html and rewrites zone_labels_with_class.json (and the
class-free zone_labels.json alongside it, which the tests read). Keeps a .bak of each.

Three kinds of entry:
  {"image":..,"box":3,"to":"my_hand"}      relabel an existing box
  {"image":..,"box":3,"to":"__delete__"}   drop a box that is not a tile
  {"image":..,"box":-1,"bbox":[..],"to":"river"}   add a box that was missed

Deletions and additions are applied after relabels and in descending index order, so the
indices in the patch stay valid throughout.

    python scripts/apply_zone_fixes.py --patch ../output/zone_annotation/zone_fixes.json
    python scripts/apply_zone_fixes.py --patch ... --dry-run     # report only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "output" / "zone_annotation"


def apply(data: list[dict], patch: list[dict]) -> tuple[list[dict], dict[str, int]]:
    by_image: dict[str, list[dict]] = {}
    for p in patch:
        by_image.setdefault(p["image"], []).append(p)
    tally = {"relabel": 0, "delete": 0, "add": 0}

    for item in data:
        ops = by_image.get(item["image"], [])
        if not ops:
            continue
        for p in ops:
            if p["box"] >= 0 and p["to"] != "__delete__":
                item["zones"][p["box"]] = p["to"]
                tally["relabel"] += 1
        for p in sorted([o for o in ops if o["box"] >= 0 and o["to"] == "__delete__"],
                        key=lambda o: -o["box"]):
            i = p["box"]
            for key in ("boxes", "zones", "cls", "heuristic", "hit"):
                if key in item and isinstance(item[key], list) and len(item[key]) > i:
                    item[key].pop(i)
            tally["delete"] += 1
        for p in [o for o in ops if o["box"] < 0 and o["to"] != "__delete__"]:
            item["boxes"].append([round(float(v), 1) for v in p["bbox"]])
            item["zones"].append(p["to"])
            for key, filler in (("cls", "?"), ("heuristic", "river"), ("hit", False)):
                if key in item and isinstance(item[key], list):
                    item[key].append(filler)
            tally["add"] += 1
    return data, tally


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=ANN / "zone_labels_with_class.json")
    ap.add_argument("--also", type=Path, default=ANN / "zone_labels.json",
                    help="Second copy kept in sync (the tests read this one).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    patch = json.loads(args.patch.read_text(encoding="utf-8"))
    data = json.loads(args.labels.read_text(encoding="utf-8"))
    before = sum(len(d["boxes"]) for d in data)
    data, tally = apply(data, patch)
    after = sum(len(d["boxes"]) for d in data)

    print(f"改标 {tally['relabel']}  删除 {tally['delete']}  新增 {tally['add']}   框数 {before} -> {after}")
    if args.dry_run:
        print("(dry-run,未写入)")
        return 0

    shutil.copy(args.labels, args.labels.with_suffix(".json.bak"))
    args.labels.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.also and args.also.exists():
        shutil.copy(args.also, args.also.with_suffix(".json.bak"))
        slim = [{k: v for k, v in d.items() if k != "cls"} for d in data]
        args.also.write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {args.labels.name}(原文件备份为 .json.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
