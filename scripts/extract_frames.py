"""extract_frames.py

从 `data/raw_videos/` 批量抽帧，并基于模糊度与曝光规则过滤低质量帧。

示例：
    python scripts/extract_frames.py --input-root data/raw_videos --output-root data/frames_candidate
    python scripts/extract_frames.py --fps 0.5 --blur-threshold 100 --report data/frames_candidate/extract_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_FPS = 0.5
DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_MIN_BRIGHTNESS = 30.0
DEFAULT_MAX_BRIGHTNESS = 225.0


class ExtractFramesError(RuntimeError):
    """抽帧业务异常。"""


@dataclass
class VideoExtractReport:
    """单视频抽帧统计。"""

    video: str
    source: str
    sampled_frames: int = 0
    saved_frames: int = 0
    dropped_blur: int = 0
    dropped_dark: int = 0
    dropped_bright: int = 0
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "source": self.source,
            "sampled_frames": self.sampled_frames,
            "saved_frames": self.saved_frames,
            "dropped_blur": self.dropped_blur,
            "dropped_dark": self.dropped_dark,
            "dropped_bright": self.dropped_bright,
            "output_files": self.output_files,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量抽取视频帧并过滤模糊/曝光异常帧")
    parser.add_argument("--input-root", default="data/raw_videos", help="输入视频根目录")
    parser.add_argument("--output-root", default="data/frames_candidate", help="输出帧根目录")
    parser.add_argument("--report", default="data/frames_candidate/extract_report.json", help="抽帧报告 JSON 路径")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="每秒抽取帧数，默认 0.5")
    parser.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD, help="Laplacian 方差阈值，低于该值视为模糊")
    parser.add_argument("--min-brightness", type=float, default=DEFAULT_MIN_BRIGHTNESS, help="平均亮度下限，低于该值视为欠曝")
    parser.add_argument("--max-brightness", type=float, default=DEFAULT_MAX_BRIGHTNESS, help="平均亮度上限，高于该值视为过曝")
    parser.add_argument("--limit-videos", type=int, default=0, help="仅处理前 N 个视频，便于联调")
    return parser.parse_args(argv)


def collect_video_files(root: Path) -> list[Path]:
    if not root.exists():
        raise ExtractFramesError(f"输入目录不存在: {root}")
    videos = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
    if not videos:
        raise ExtractFramesError(f"输入目录下未找到视频文件: {root}")
    return sorted(videos)


def compute_blur_score(frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def should_keep_frame(
    frame: Any,
    *,
    blur_threshold: float,
    min_brightness: float,
    max_brightness: float,
) -> tuple[bool, str | None, float, float]:
    blur_score = compute_blur_score(frame)
    brightness = compute_brightness(frame)
    if blur_score < blur_threshold:
        return False, "blur", blur_score, brightness
    if brightness < min_brightness:
        return False, "dark", blur_score, brightness
    if brightness > max_brightness:
        return False, "bright", blur_score, brightness
    return True, None, blur_score, brightness


def build_output_path(output_root: Path, input_root: Path, video_path: Path, frame_index: int) -> Path:
    relative_parent = video_path.parent.relative_to(input_root)
    stem = video_path.stem
    return output_root / relative_parent / f"{stem}_f{frame_index:06d}.jpg"


def extract_video_frames(video_path: Path, args: argparse.Namespace, input_root: Path, output_root: Path) -> VideoExtractReport:
    source = video_path.parent.name
    report = VideoExtractReport(video=str(video_path), source=source)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExtractFramesError(f"无法打开视频: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = 25.0
    sample_interval = max(int(round(native_fps / args.fps)), 1)

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % sample_interval != 0:
                continue
            report.sampled_frames += 1
            keep, reason, _, _ = should_keep_frame(
                frame,
                blur_threshold=args.blur_threshold,
                min_brightness=args.min_brightness,
                max_brightness=args.max_brightness,
            )
            if not keep:
                if reason == "blur":
                    report.dropped_blur += 1
                elif reason == "dark":
                    report.dropped_dark += 1
                elif reason == "bright":
                    report.dropped_bright += 1
                continue
            output_path = build_output_path(output_root, input_root, video_path, frame_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), frame):
                raise ExtractFramesError(f"写入帧失败: {output_path}")
            report.saved_frames += 1
            report.output_files.append(str(output_path))
    finally:
        capture.release()
    return report


def save_report(path: Path, reports: list[VideoExtractReport]) -> None:
    payload = {
        "videos": [item.to_dict() for item in reports],
        "summary": {
            "total_videos": len(reports),
            "sampled_frames": sum(item.sampled_frames for item in reports),
            "saved_frames": sum(item.saved_frames for item in reports),
            "dropped_blur": sum(item.dropped_blur for item in reports),
            "dropped_dark": sum(item.dropped_dark for item in reports),
            "dropped_bright": sum(item.dropped_bright for item in reports),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fps <= 0:
        raise ExtractFramesError("--fps 必须大于 0")
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    videos = collect_video_files(input_root)
    if args.limit_videos > 0:
        videos = videos[: args.limit_videos]

    reports: list[VideoExtractReport] = []
    for video_path in tqdm(videos, desc="extract_frames", unit="video"):
        reports.append(extract_video_frames(video_path, args, input_root, output_root))

    save_report(report_path, reports)
    print(f"抽帧完成：处理 {len(reports)} 个视频，报告写入 {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtractFramesError as exc:
        print(f"[extract_frames] ERROR: {exc}")
        raise SystemExit(1)
