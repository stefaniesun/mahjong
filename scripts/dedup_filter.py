"""dedup_filter.py

从 `data/frames_candidate/` 中去重并均衡采样，输出最终待标注帧到 `data/frames_selected/`。

示例：
    python scripts/dedup_filter.py --input-root data/frames_candidate --output-root data/frames_selected --report data/frames_selected/selection_report.json --total 500
    python scripts/dedup_filter.py --dry-run --filename-style B
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imagehash
from PIL import Image
from tqdm import tqdm

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_TOTAL = 500
DEFAULT_PHASH_DISTANCE_THRESHOLD = 8
DEFAULT_MAX_SOURCE_RATIO = 0.25
DEFAULT_FILENAME_STYLE = "B"


class DedupFilterError(RuntimeError):
    """去重与采样脚本异常。"""


@dataclass(frozen=True)
class FrameCandidate:
    path: Path
    source: str
    video_id: str
    blur_score: float
    phash_hex: str

    @property
    def frame_token(self) -> str:
        match = re.search(r"_f(\d+)$", self.path.stem)
        if match:
            return f"f{match.group(1)}"
        return self.path.stem

    def phash(self) -> imagehash.ImageHash:
        return imagehash.hex_to_hash(self.phash_hex)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对候选帧做感知哈希去重与均衡采样")
    parser.add_argument("--input-root", default="data/frames_candidate", help="候选帧输入根目录")
    parser.add_argument("--output-root", default="data/frames_selected", help="选中帧输出目录")
    parser.add_argument("--report", default="data/frames_selected/selection_report.json", help="筛选报告 JSON 路径")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="目标总选帧数")
    parser.add_argument("--phash-distance-threshold", type=int, default=DEFAULT_PHASH_DISTANCE_THRESHOLD, help="感知哈希判重的最大汉明距离")
    parser.add_argument("--max-source-ratio", type=float, default=DEFAULT_MAX_SOURCE_RATIO, help="单博主占比上限，默认 0.25")
    parser.add_argument("--max-per-video", type=int, default=0, help="单视频最多选中的帧数，0 表示自动按配额推导")
    parser.add_argument("--filename-style", default=DEFAULT_FILENAME_STYLE, choices=["A", "B"], help="输出文件命名风格，默认 B")
    parser.add_argument("--dry-run", action="store_true", help="仅生成报告，不复制图片")
    return parser.parse_args(argv)


def compute_blur_score(image_path: Path) -> float:
    image = cv2.imread(str(image_path))
    if image is None:
        raise DedupFilterError(f"无法读取图片: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_phash(image_path: Path) -> str:
    with Image.open(image_path) as image:
        return str(imagehash.phash(image))


def collect_frame_candidates(input_root: Path) -> list[FrameCandidate]:
    if not input_root.exists():
        raise DedupFilterError(f"输入目录不存在: {input_root}")

    files = [path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    if not files:
        raise DedupFilterError(f"输入目录下未找到候选帧: {input_root}")

    candidates: list[FrameCandidate] = []
    for path in tqdm(sorted(files), desc="scan_frames", unit="image"):
        relative = path.relative_to(input_root)
        if len(relative.parts) < 2:
            raise DedupFilterError(f"候选帧路径缺少博主目录层级: {path}")
        source = relative.parts[0]
        if len(relative.parts) >= 3:
            video_id = relative.parts[1]
        else:
            stem = path.stem
            if "_f" in stem:
                video_id = stem.rsplit("_f", 1)[0]
            else:
                video_id = stem
        candidates.append(
            FrameCandidate(
                path=path,
                source=source,
                video_id=video_id,
                blur_score=compute_blur_score(path),
                phash_hex=compute_phash(path),
            )
        )
    return candidates




def group_candidates_by_source(candidates: list[FrameCandidate]) -> dict[str, list[FrameCandidate]]:
    grouped: dict[str, list[FrameCandidate]] = {}
    for item in candidates:
        grouped.setdefault(item.source, []).append(item)
    return grouped


def deduplicate_source_candidates(
    candidates_by_source: dict[str, list[FrameCandidate]],
    *,
    phash_distance_threshold: int,
) -> dict[str, list[FrameCandidate]]:
    deduped: dict[str, list[FrameCandidate]] = {}
    for source, items in candidates_by_source.items():
        sorted_items = sorted(items, key=lambda item: item.blur_score, reverse=True)
        kept: list[FrameCandidate] = []
        kept_hashes: list[imagehash.ImageHash] = []
        for item in sorted_items:
            current_hash = item.phash()
            is_duplicate = any(current_hash - existing_hash <= phash_distance_threshold for existing_hash in kept_hashes)
            if is_duplicate:
                continue
            kept.append(item)
            kept_hashes.append(current_hash)
        deduped[source] = sorted(kept, key=lambda item: (item.video_id, item.path.name))
    return deduped


def _compute_source_caps(source_names: list[str], *, total: int, max_source_ratio: float) -> dict[str, int]:
    if total <= 0:
        raise DedupFilterError("--total 必须大于 0")
    if not source_names:
        raise DedupFilterError("没有可分配的博主来源")
    if not 0 < max_source_ratio <= 1:
        raise DedupFilterError("--max-source-ratio 必须在 0 到 1 之间")

    equal_quota = max(total // len(source_names), 1)
    ratio_cap = max(int(total * max_source_ratio), 1)
    return {source: max(equal_quota, ratio_cap) for source in source_names}



def plan_selection(
    items_by_source: dict[str, list[FrameCandidate]],
    *,
    total: int,
    max_source_ratio: float,
    max_per_video: int,
) -> dict[str, list[FrameCandidate]]:
    source_caps = _compute_source_caps(list(items_by_source.keys()), total=total, max_source_ratio=max_source_ratio)
    selected: dict[str, list[FrameCandidate]] = {source: [] for source in items_by_source}

    per_source_video_limit: dict[str, int] = {}
    for source, items in items_by_source.items():
        videos = {item.video_id for item in items}
        auto_limit = max(1, source_caps[source] // max(len(videos), 1))
        per_source_video_limit[source] = max_per_video if max_per_video > 0 else auto_limit

    remaining = total
    made_progress = True
    while remaining > 0 and made_progress:
        made_progress = False
        for source in sorted(items_by_source):
            if remaining <= 0:
                break
            current = selected[source]
            if len(current) >= source_caps[source]:
                continue

            counts_by_video: dict[str, int] = {}
            for item in current:
                counts_by_video[item.video_id] = counts_by_video.get(item.video_id, 0) + 1

            candidates = [item for item in items_by_source[source] if item not in current]
            candidates.sort(
                key=lambda item: (
                    counts_by_video.get(item.video_id, 0),
                    item.video_id in counts_by_video,
                    -item.blur_score,
                    item.video_id,
                    item.path.name,
                )
            )


            picked: FrameCandidate | None = None
            for item in candidates:
                if counts_by_video.get(item.video_id, 0) >= per_source_video_limit[source]:
                    continue
                picked = item
                break
            if picked is None:
                continue

            current.append(picked)

            remaining -= 1
            made_progress = True

    return {source: sorted(items, key=lambda item: (item.video_id, item.path.name)) for source, items in selected.items() if items}


def build_output_name(item: FrameCandidate, *, filename_style: str) -> str:
    if filename_style == "A":
        return f"{item.source}_{item.path.name}"
    safe_video_id = re.sub(r"[^0-9A-Za-z_-]+", "_", item.video_id)
    return f"{item.source}__{safe_video_id}__{item.frame_token}{item.path.suffix.lower()}"


def copy_selected_files(
    selection_plan: dict[str, list[FrameCandidate]],
    *,
    output_root: Path,
    filename_style: str,
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for source, items in selection_plan.items():
        source_dir = output_root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            target_name = build_output_name(item, filename_style=filename_style)
            target_path = source_dir / target_name
            shutil.copy2(item.path, target_path)
            copied.append({
                "source": source,
                "video_id": item.video_id,
                "original_path": str(item.path),
                "target_path": str(target_path),
                "target_name": target_name,
            })
    return copied


def build_report(
    *,
    candidates_by_source: dict[str, list[FrameCandidate]],
    deduped_by_source: dict[str, list[FrameCandidate]],
    selection_plan: dict[str, list[FrameCandidate]],
    selected_files: list[dict[str, str]],
    dry_run: bool,
    total_requested: int,
) -> dict[str, Any]:
    sources_payload: dict[str, Any] = {}
    for source in sorted(candidates_by_source):
        source_candidates = candidates_by_source[source]
        source_deduped = deduped_by_source.get(source, [])
        source_selected = selection_plan.get(source, [])

        video_candidate_counts: dict[str, int] = {}
        video_selected_counts: dict[str, int] = {}
        for item in source_candidates:
            video_candidate_counts[item.video_id] = video_candidate_counts.get(item.video_id, 0) + 1
        for item in source_selected:
            video_selected_counts[item.video_id] = video_selected_counts.get(item.video_id, 0) + 1

        sources_payload[source] = {
            "candidate_frames": len(source_candidates),
            "deduped_frames": len(source_deduped),
            "selected_frames": len(source_selected),
            "videos": [
                {
                    "video_id": video_id,
                    "candidate_frames": video_candidate_counts[video_id],
                    "selected_frames": video_selected_counts.get(video_id, 0),
                }
                for video_id in sorted(video_candidate_counts)
            ],
        }

    return {
        "dry_run": dry_run,
        "summary": {
            "total_requested": total_requested,
            "candidate_frames": sum(len(items) for items in candidates_by_source.values()),
            "deduped_frames": sum(len(items) for items in deduped_by_source.values()),
            "selected_frames": sum(len(items) for items in selection_plan.values()),
            "selected_sources": sum(1 for items in selection_plan.values() if items),
        },
        "sources": sources_payload,
        "selected_files": selected_files,
    }


def save_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    candidates = collect_frame_candidates(input_root)
    candidates_by_source = group_candidates_by_source(candidates)
    deduped_by_source = deduplicate_source_candidates(
        candidates_by_source,
        phash_distance_threshold=args.phash_distance_threshold,
    )
    selection_plan = plan_selection(
        deduped_by_source,
        total=args.total,
        max_source_ratio=args.max_source_ratio,
        max_per_video=args.max_per_video,
    )

    selected_files: list[dict[str, str]] = []
    for source, items in selection_plan.items():
        for item in items:
            selected_files.append(
                {
                    "source": source,
                    "video_id": item.video_id,
                    "original_path": str(item.path),
                    "target_name": build_output_name(item, filename_style=args.filename_style),
                }
            )

    if not args.dry_run:
        copied = copy_selected_files(selection_plan, output_root=output_root, filename_style=args.filename_style)
        copied_map = {entry["target_name"]: entry["target_path"] for entry in copied}
        for entry in selected_files:
            entry["target_path"] = copied_map[entry["target_name"]]

    report_payload = build_report(
        candidates_by_source=candidates_by_source,
        deduped_by_source=deduped_by_source,
        selection_plan=selection_plan,
        selected_files=selected_files,
        dry_run=args.dry_run,
        total_requested=args.total,
    )
    save_report(report_path, report_payload)
    print(f"筛选完成：选中 {report_payload['summary']['selected_frames']} 张，报告写入 {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DedupFilterError as exc:
        print(f"[dedup_filter] ERROR: {exc}")
        raise SystemExit(1)
