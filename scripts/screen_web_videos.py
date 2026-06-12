"""screen_web_videos.py

对抓取回来的网络视频执行粗筛：统计含牌帧占比、标记疑似录屏视频、生成有效片段区间，并输出 JSON + HTML 预览页。

示例：
    python scripts/screen_web_videos.py --input-root data/raw_videos --report data/web_screen/screen_report.json --preview data/web_screen/preview.html --clips-root data/web_clips --skip-ffmpeg
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_SAMPLE_FPS = 1.0
DEFAULT_MIN_TILES_PER_FRAME = 3
DEFAULT_MIN_ACTIVE_RATIO = 0.3
DEFAULT_MAX_GAP_SECONDS = 3.0
DEFAULT_MIN_CLIP_DURATION = 2.0
DEFAULT_SCREEN_BORDER_THRESHOLD = 0.3
DEFAULT_STATIC_BACKGROUND_THRESHOLD = 0.9
DEFAULT_GRID_LAYOUT_THRESHOLD = 0.8


class ScreenWebVideosError(RuntimeError):
    """网络视频筛选异常。"""


@dataclass
class ClipRange:
    start_seconds: float
    end_seconds: float
    output_path: str = ""

    @property
    def duration_seconds(self) -> float:
        return max(self.end_seconds - self.start_seconds, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "output_path": self.output_path,
        }


@dataclass
class VideoMetrics:
    sampled_frames: int = 0
    active_tile_frames: int = 0
    active_timestamps: list[float] = field(default_factory=list)
    static_background_ratio: float = 0.0
    border_ui_ratio: float = 0.0
    grid_layout_ratio: float = 0.0
    honor_ratio: float = 0.0

    @property
    def active_ratio(self) -> float:
        if self.sampled_frames <= 0:
            return 0.0
        return self.active_tile_frames / self.sampled_frames


@dataclass
class VideoDecision:
    low_value: bool
    suspected_screen_recording: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_value": self.low_value,
            "suspected_screen_recording": self.suspected_screen_recording,
            "reasons": self.reasons,
        }


@dataclass
class AnalysisConfig:
    sample_fps: float = DEFAULT_SAMPLE_FPS
    min_tiles_per_frame: int = DEFAULT_MIN_TILES_PER_FRAME
    min_active_ratio: float = DEFAULT_MIN_ACTIVE_RATIO
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
    min_clip_duration: float = DEFAULT_MIN_CLIP_DURATION
    screen_border_threshold: float = DEFAULT_SCREEN_BORDER_THRESHOLD
    static_background_threshold: float = DEFAULT_STATIC_BACKGROUND_THRESHOLD
    grid_layout_threshold: float = DEFAULT_GRID_LAYOUT_THRESHOLD


@dataclass
class VideoAnalysisResult:
    video_path: Path
    source: str
    metrics: VideoMetrics
    decision: VideoDecision
    clips: list[ClipRange]
    thumbnails: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": str(self.video_path),
            "source": self.source,
            "metrics": {
                "sampled_frames": self.metrics.sampled_frames,
                "active_tile_frames": self.metrics.active_tile_frames,
                "active_ratio": round(self.metrics.active_ratio, 4),
                "static_background_ratio": round(self.metrics.static_background_ratio, 4),
                "border_ui_ratio": round(self.metrics.border_ui_ratio, 4),
                "grid_layout_ratio": round(self.metrics.grid_layout_ratio, 4),
                "honor_ratio": round(self.metrics.honor_ratio, 4),
            },
            "decision": self.decision.to_dict(),
            "clips": [clip.to_dict() for clip in self.clips],
            "thumbnails": [str(path) for path in self.thumbnails],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="粗筛网络视频并输出片段预览")
    parser.add_argument("--input-root", default="data/raw_videos", help="输入视频根目录")
    parser.add_argument("--report", default="data/web_screen/screen_report.json", help="筛选报告 JSON 路径")
    parser.add_argument("--preview", default="data/web_screen/preview.html", help="HTML 预览页路径")
    parser.add_argument("--clips-root", default="data/web_clips", help="切片输出目录")
    parser.add_argument("--thumbnails-root", default="data/web_screen/thumbnails", help="缩略图输出目录")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS, help="每秒采样帧数")
    parser.add_argument("--min-tiles-per-frame", type=int, default=DEFAULT_MIN_TILES_PER_FRAME, help="判定含牌帧的最小牌数阈值")
    parser.add_argument("--min-active-ratio", type=float, default=DEFAULT_MIN_ACTIVE_RATIO, help="低价值视频判定阈值")
    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS, help="片段合并允许的最大间断秒数")
    parser.add_argument("--min-clip-duration", type=float, default=DEFAULT_MIN_CLIP_DURATION, help="保留片段最小时长")
    parser.add_argument("--screen-border-threshold", type=float, default=DEFAULT_SCREEN_BORDER_THRESHOLD, help="疑似录屏边框阈值")
    parser.add_argument("--static-background-threshold", type=float, default=DEFAULT_STATIC_BACKGROUND_THRESHOLD, help="疑似录屏静态背景阈值")
    parser.add_argument("--grid-layout-threshold", type=float, default=DEFAULT_GRID_LAYOUT_THRESHOLD, help="疑似录屏网格布局阈值")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg 可执行文件路径")
    parser.add_argument("--skip-ffmpeg", action="store_true", help="只输出报告，不实际切片")
    parser.add_argument("--limit-videos", type=int, default=0, help="仅处理前 N 个视频，便于联调")
    return parser.parse_args(argv)


def collect_video_files(root: Path) -> list[Path]:
    if not root.exists():
        raise ScreenWebVideosError(f"输入目录不存在: {root}")
    videos = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
    if not videos:
        raise ScreenWebVideosError(f"输入目录下未找到视频文件: {root}")
    return sorted(videos)


def merge_active_ranges(timestamps: list[float], *, max_gap_seconds: float, min_duration_seconds: float) -> list[tuple[float, float]]:
    if not timestamps:
        return []
    merged: list[tuple[float, float]] = []
    sorted_timestamps = sorted(timestamps)
    start = end = sorted_timestamps[0]
    for ts in sorted_timestamps[1:]:
        if ts - end <= max_gap_seconds:
            end = ts
            continue
        if end - start >= min_duration_seconds:
            merged.append((start, end))
        start = end = ts
    if end - start >= min_duration_seconds:
        merged.append((start, end))
    return merged


def classify_video(
    metrics: VideoMetrics,
    *,
    min_active_ratio: float,
    screen_border_threshold: float,
    static_background_threshold: float,
    grid_layout_threshold: float,
) -> VideoDecision:
    reasons: list[str] = []
    low_value = metrics.active_ratio < min_active_ratio
    suspected = (
        metrics.border_ui_ratio >= screen_border_threshold
        and metrics.static_background_ratio >= static_background_threshold
        and metrics.grid_layout_ratio >= grid_layout_threshold
    )
    if low_value:
        reasons.append("active_ratio_below_threshold")
    if suspected:
        reasons.append("screen_recording_heuristic")
    if metrics.honor_ratio >= 0.3:
        reasons.append("suspected_non_sichuan_mahjong")
    return VideoDecision(low_value=low_value, suspected_screen_recording=suspected, reasons=reasons)


def _edge_density(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _border_ratio(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    band_h = max(h // 8, 1)
    band_w = max(w // 8, 1)
    border = np.zeros((h, w), dtype=bool)
    border[:band_h, :] = True
    border[-band_h:, :] = True
    border[:, :band_w] = True
    border[:, -band_w:] = True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray[border].std() < 12.0)


def _grid_ratio(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    x_strength = float(np.mean(np.abs(sobel_x)))
    y_strength = float(np.mean(np.abs(sobel_y)))
    denom = max(x_strength, y_strength, 1e-6)
    return min(x_strength, y_strength) / denom


def _estimate_tile_count(frame: np.ndarray) -> int:
    edge_density = _edge_density(frame)
    if edge_density >= 0.03:
        return 4
    if edge_density >= 0.015:
        return 2
    return 0


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def _thumbnail_src(path: Path, preview_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(preview_dir).as_posix()
    except ValueError:
        return resolved.as_posix()


def _save_thumbnail(frame: np.ndarray, *, thumbnails_root: Path, source: str, video_path: Path, timestamp: float, index: int) -> Path:
    target_dir = thumbnails_root / source
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{_safe_stem(video_path)}_{index:02d}_{int(round(timestamp * 1000)):06d}ms.jpg"
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        raise ScreenWebVideosError(f"缩略图写入失败: {output_path}")
    return output_path


def analyze_video(

    video_path: Path,
    *,
    config: AnalysisConfig,
    detector: object | None,
    thumbnails_root: Path | None = None,
) -> VideoAnalysisResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ScreenWebVideosError(f"无法打开视频: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = 25.0
    sample_interval = max(int(round(native_fps / config.sample_fps)), 1)

    metrics = VideoMetrics()
    thumbnails: list[Path] = []
    border_hits = 0
    grid_scores: list[float] = []
    static_flags: list[float] = []
    previous_gray: np.ndarray | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % sample_interval != 0:
                continue

            metrics.sampled_frames += 1
            timestamp = (frame_index - 1) / native_fps
            tile_count = _estimate_tile_count(frame) if detector is None else 0
            if tile_count >= config.min_tiles_per_frame:
                metrics.active_tile_frames += 1
                metrics.active_timestamps.append(timestamp)
                if thumbnails_root is not None and len(thumbnails) < 3:
                    thumbnails.append(
                        _save_thumbnail(
                            frame,
                            thumbnails_root=thumbnails_root,
                            source=video_path.parent.name,
                            video_path=video_path,
                            timestamp=timestamp,
                            index=len(thumbnails) + 1,
                        )
                    )

            border_hits += int(_border_ratio(frame) >= 1.0)
            grid_scores.append(_grid_ratio(frame))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                static_flags.append(float(diff.mean() < 2.0))
            previous_gray = gray
    finally:
        capture.release()


    metrics.border_ui_ratio = border_hits / metrics.sampled_frames if metrics.sampled_frames else 0.0
    metrics.grid_layout_ratio = float(np.mean(grid_scores)) if grid_scores else 0.0
    metrics.static_background_ratio = float(np.mean(static_flags)) if static_flags else 0.0

    decision = classify_video(
        metrics,
        min_active_ratio=config.min_active_ratio,
        screen_border_threshold=config.screen_border_threshold,
        static_background_threshold=config.static_background_threshold,
        grid_layout_threshold=config.grid_layout_threshold,
    )
    clips = [ClipRange(start, end) for start, end in merge_active_ranges(metrics.active_timestamps, max_gap_seconds=config.max_gap_seconds, min_duration_seconds=config.min_clip_duration)]
    return VideoAnalysisResult(
        video_path=video_path,
        source=video_path.parent.name,
        metrics=metrics,
        decision=decision,
        clips=clips,
        thumbnails=thumbnails,
    )


def export_clips(result: VideoAnalysisResult, *, clips_root: Path, ffmpeg_bin: str) -> None:
    target_dir = clips_root / result.source
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, clip in enumerate(result.clips, start=1):
        output_path = target_dir / f"{result.video_path.stem}_clip{index}.mp4"
        command = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{clip.start_seconds:.3f}",
            "-to",
            f"{clip.end_seconds:.3f}",
            "-i",
            str(result.video_path),
            "-c",
            "copy",
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ScreenWebVideosError(f"ffmpeg 切片失败: {completed.stderr.strip()}")
        clip.output_path = str(output_path)


def write_outputs(results: list[VideoAnalysisResult], *, report_path: Path, preview_path: Path) -> None:
    payload = {
        "videos": [result.to_dict() for result in results],
        "summary": {
            "total_videos": len(results),
            "low_value": sum(1 for item in results if item.decision.low_value),
            "suspected_screen_recording": sum(1 for item in results if item.decision.suspected_screen_recording),
            "total_clips": sum(len(item.clips) for item in results),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_dir = preview_path.parent.resolve()
    rows: list[str] = []
    for item in results:
        status = "疑似录屏" if item.decision.suspected_screen_recording else ("低价值" if item.decision.low_value else "保留")
        clips_desc = "<br>".join(f"{clip.start_seconds:.1f}s - {clip.end_seconds:.1f}s" for clip in item.clips) or "无片段"
        reasons = ", ".join(item.decision.reasons) or "无"
        thumbs_html = "<br>".join(
            f"<img src='{html.escape(_thumbnail_src(Path(path), preview_dir))}' alt='{html.escape(item.video_path.name)}' style='max-width:220px;max-height:140px;border:1px solid #ccc;'>"
            for path in item.thumbnails
        ) or "无截图"

        rows.append(
            "<tr>"
            f"<td>{html.escape(item.source)}</td>"
            f"<td>{html.escape(item.video_path.name)}</td>"
            f"<td>{status}</td>"
            f"<td>{item.metrics.active_ratio:.2%}</td>"
            f"<td>{html.escape(reasons)}</td>"
            f"<td>{html.escape(clips_desc)}</td>"
            f"<td>{thumbs_html}</td>"
            "</tr>"
        )
    html_text = (
        "<html><head><meta charset='utf-8'><title>screen_web_videos</title></head><body>"
        "<h1>网络视频筛选预览</h1>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<tr><th>来源</th><th>视频</th><th>判定</th><th>含牌帧占比</th><th>理由</th><th>片段</th><th>截图</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(html_text, encoding="utf-8")



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_fps <= 0:
        raise ScreenWebVideosError("--sample-fps 必须大于 0")
    if args.min_clip_duration < 0:
        raise ScreenWebVideosError("--min-clip-duration 不能小于 0")

    input_root = Path(args.input_root).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    preview_path = Path(args.preview).expanduser().resolve()
    clips_root = Path(args.clips_root).expanduser().resolve()
    thumbnails_root = Path(args.thumbnails_root).expanduser().resolve()
    videos = collect_video_files(input_root)

    if args.limit_videos > 0:
        videos = videos[: args.limit_videos]

    config = AnalysisConfig(
        sample_fps=args.sample_fps,
        min_tiles_per_frame=args.min_tiles_per_frame,
        min_active_ratio=args.min_active_ratio,
        max_gap_seconds=args.max_gap_seconds,
        min_clip_duration=args.min_clip_duration,
        screen_border_threshold=args.screen_border_threshold,
        static_background_threshold=args.static_background_threshold,
        grid_layout_threshold=args.grid_layout_threshold,
    )

    results: list[VideoAnalysisResult] = []
    for video_path in tqdm(videos, desc="screen_web_videos", unit="video"):
        result = analyze_video(video_path, config=config, detector=None, thumbnails_root=thumbnails_root)
        if not args.skip_ffmpeg:
            export_clips(result, clips_root=clips_root, ffmpeg_bin=args.ffmpeg_bin)
        results.append(result)


    write_outputs(results, report_path=report_path, preview_path=preview_path)
    print(f"筛选完成：处理 {len(results)} 个视频，报告写入 {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScreenWebVideosError as exc:
        print(f"[screen_web_videos] ERROR: {exc}")
        raise SystemExit(1)
