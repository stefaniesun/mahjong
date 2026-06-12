# Screen Web Videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 `scripts/screen_web_videos.py`，对抓取回来的网络视频执行粗筛、疑似录屏判定、有效片段切割，并生成 JSON + HTML 预览结果，支撑 Phase 0 任务 2.5。

**Architecture:** 采用单脚本 CLI 方案，沿用现有 `scripts/extract_frames.py` 的风格：遍历 `raw_videos` 下的视频，用 OpenCV 按固定采样率读取帧、计算启发式指标，并在可选模型推理结果的辅助下判定视频价值与片段区间。报告层输出结构化 `screen_report.json` 与本地静态 HTML 预览页；切片调用本机 `ffmpeg` 执行无损复制。

**Tech Stack:** Python 3.10+、OpenCV、NumPy、subprocess、json、pathlib、argparse、unittest、可选 Ultralytics/YOLO 推理。

---

### File Structure

- Create: `scripts/screen_web_videos.py`
  - CLI 入口
  - 视频遍历与采样
  - 含牌帧统计与启发式判定
  - 连续片段合并与 `ffmpeg` 切片
  - JSON / HTML 报告生成
- Create: `tests/test_screen_web_videos.py`
  - 纯函数单测（区间合并、录屏启发式、报告汇总）
  - 小型合成视频冒烟测试
- Modify: `README.md`
  - 增加任务 2.5 的运行示例、输入输出说明、`ffmpeg` 依赖说明
- Optional Modify: `requirements.txt`
  - 若脚本新增仓库中尚未声明的轻量依赖，再补充依赖声明；优先复用现有依赖避免新增

### Task 1: 搭建脚本骨架与核心数据结构

**Files:**
- Create: `scripts/screen_web_videos.py`
- Test: `tests/test_screen_web_videos.py`

- [ ] **Step 1: 写失败测试，固定输入扫描与片段合并接口**

```python
from pathlib import Path

from scripts import screen_web_videos


def test_collect_video_files_and_merge_segments(tmp_path: Path) -> None:
    root = tmp_path / "raw_videos"
    source = root / "bili_demo"
    source.mkdir(parents=True)
    video = source / "sample.mp4"
    video.write_bytes(b"demo")

    videos = screen_web_videos.collect_video_files(root)
    assert videos == [video]

    merged = screen_web_videos.merge_active_ranges(
        [0.0, 1.0, 4.0, 5.0, 11.0],
        max_gap_seconds=3.0,
        min_duration_seconds=1.5,
    )
    assert merged == [(0.0, 5.0)]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_screen_web_videos.py::test_collect_video_files_and_merge_segments -v`
Expected: FAIL with `ImportError` or missing function errors for `screen_web_videos`

- [ ] **Step 3: 写最小实现骨架**

```python
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


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
    start = end = timestamps[0]
    for ts in timestamps[1:]:
        if ts - end <= max_gap_seconds:
            end = ts
            continue
        if end - start >= min_duration_seconds:
            merged.append((start, end))
        start = end = ts
    if end - start >= min_duration_seconds:
        merged.append((start, end))
    return merged
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_screen_web_videos.py::test_collect_video_files_and_merge_segments -v`
Expected: PASS

### Task 2: 实现录屏启发式与视频级判定

**Files:**
- Modify: `scripts/screen_web_videos.py`
- Test: `tests/test_screen_web_videos.py`

- [ ] **Step 1: 写失败测试，覆盖录屏启发式和低价值判定**

```python
def test_classify_video_flags_low_value_and_screen_capture() -> None:
    metrics = screen_web_videos.VideoMetrics(
        sampled_frames=10,
        active_tile_frames=2,
        active_timestamps=[0.0, 1.0],
        static_background_ratio=0.96,
        border_ui_ratio=0.41,
        grid_layout_ratio=0.88,
        honor_ratio=0.0,
    )
    decision = screen_web_videos.classify_video(
        metrics,
        min_active_ratio=0.3,
        screen_border_threshold=0.3,
        static_background_threshold=0.9,
        grid_layout_threshold=0.8,
    )
    assert decision.low_value is True
    assert decision.suspected_screen_recording is True
    assert "active_ratio_below_threshold" in decision.reasons
    assert "screen_recording_heuristic" in decision.reasons
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_screen_web_videos.py::test_classify_video_flags_low_value_and_screen_capture -v`
Expected: FAIL with missing `VideoMetrics` / `classify_video`

- [ ] **Step 3: 写最小实现**

```python
from dataclasses import dataclass, field


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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_screen_web_videos.py::test_classify_video_flags_low_value_and_screen_capture -v`
Expected: PASS

### Task 3: 实现无模型模式的采样分析与片段切割

**Files:**
- Modify: `scripts/screen_web_videos.py`
- Test: `tests/test_screen_web_videos.py`

- [ ] **Step 1: 写失败测试，验证单视频分析与切片计划输出**

```python
def test_analyze_video_without_model_produces_clip_ranges(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    write_test_video(video_path)

    config = screen_web_videos.AnalysisConfig(
        sample_fps=1.0,
        min_tiles_per_frame=3,
        min_active_ratio=0.2,
        max_gap_seconds=3.0,
        min_clip_duration=1.0,
    )
    result = screen_web_videos.analyze_video(video_path, config=config, detector=None)
    assert result.metrics.sampled_frames >= 3
    assert result.clips
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_screen_web_videos.py::test_analyze_video_without_model_produces_clip_ranges -v`
Expected: FAIL with missing `AnalysisConfig` / `analyze_video`

- [ ] **Step 3: 写最小实现，先支持启发式无模型模式**

```python
@dataclass
class AnalysisConfig:
    sample_fps: float = 1.0
    min_tiles_per_frame: int = 3
    min_active_ratio: float = 0.3
    max_gap_seconds: float = 3.0
    min_clip_duration: float = 2.0


def analyze_video(video_path: Path, *, config: AnalysisConfig, detector: object | None) -> VideoAnalysisResult:
    capture = cv2.VideoCapture(str(video_path))
    # 读取视频 fps，按 sample_fps 抽样
    # 若 detector is None，则使用亮度变化 + 边缘密度近似估计 active frames
    # 记录 active_timestamps
    # 调 merge_active_ranges 生成 clips
    # 返回 VideoAnalysisResult
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_screen_web_videos.py::test_analyze_video_without_model_produces_clip_ranges -v`
Expected: PASS

### Task 4: 实现报告落盘与 HTML 预览页

**Files:**
- Modify: `scripts/screen_web_videos.py`
- Test: `tests/test_screen_web_videos.py`

- [ ] **Step 1: 写失败测试，验证 JSON 汇总与 HTML 生成**

```python
def test_write_outputs_creates_report_and_preview(tmp_path: Path) -> None:
    result = screen_web_videos.VideoAnalysisResult.example(tmp_path)
    report_path = tmp_path / "screen_report.json"
    preview_path = tmp_path / "preview.html"

    screen_web_videos.write_outputs([result], report_path=report_path, preview_path=preview_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_videos"] == 1
    html = preview_path.read_text(encoding="utf-8")
    assert "疑似录屏" in html
    assert result.video_path.name in html
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_screen_web_videos.py::test_write_outputs_creates_report_and_preview -v`
Expected: FAIL with missing `write_outputs`

- [ ] **Step 3: 写最小实现**

```python
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

    rows = []
    for item in results:
        rows.append(f"<tr><td>{item.video_path.name}</td><td>{'疑似录屏' if item.decision.suspected_screen_recording else '保留'}</td></tr>")
    html = "<html><body><table>" + "".join(rows) + "</table></body></html>"
    preview_path.write_text(html, encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_screen_web_videos.py::test_write_outputs_creates_report_and_preview -v`
Expected: PASS

### Task 5: 打通 CLI、`ffmpeg` 切片与 README 文档

**Files:**
- Modify: `scripts/screen_web_videos.py`
- Modify: `README.md`
- Test: `tests/test_screen_web_videos.py`

- [ ] **Step 1: 写失败测试，验证 `main()` 端到端输出**

```python
def test_main_runs_end_to_end(tmp_path: Path) -> None:
    input_root = tmp_path / "raw_videos"
    source_dir = input_root / "bili_demo"
    source_dir.mkdir(parents=True)
    write_test_video(source_dir / "sample.mp4")

    report_path = tmp_path / "screen_report.json"
    preview_path = tmp_path / "preview.html"
    clips_root = tmp_path / "web_clips"

    exit_code = screen_web_videos.main(
        [
            "--input-root", str(input_root),
            "--report", str(report_path),
            "--preview", str(preview_path),
            "--clips-root", str(clips_root),
            "--sample-fps", "1",
            "--min-active-ratio", "0.1",
            "--min-clip-duration", "1",
            "--skip-ffmpeg",
        ]
    )
    assert exit_code == 0
    assert report_path.exists()
    assert preview_path.exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_screen_web_videos.py::test_main_runs_end_to_end -v`
Expected: FAIL with CLI plumbing missing

- [ ] **Step 3: 实现 CLI 与文档更新**

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AnalysisConfig(...)
    videos = collect_video_files(Path(args.input_root).resolve())
    results = [analyze_video(video, config=config, detector=None) for video in videos]
    if not args.skip_ffmpeg:
        for result in results:
            export_clips(result, clips_root=Path(args.clips_root).resolve(), ffmpeg_bin=args.ffmpeg_bin)
    write_outputs(results, report_path=Path(args.report).resolve(), preview_path=Path(args.preview).resolve())
    print(f"筛选完成：处理 {len(results)} 个视频")
    return 0
```

```markdown
## 任务 2.5：网络视频粗筛

```bash
python scripts/screen_web_videos.py --input-root data/raw_videos --report data/web_screen/screen_report.json --preview data/web_screen/preview.html --clips-root data/web_clips --skip-ffmpeg
```

- 依赖本机 `ffmpeg`；若只联调判定逻辑，可先传 `--skip-ffmpeg`
- 默认按 1 fps 采样，输出视频级判定、片段区间和 HTML 预览页
```

- [ ] **Step 4: 运行目标测试与相关回归**

Run: `pytest tests/test_screen_web_videos.py -v`
Expected: PASS

Run: `pytest tests/test_extract_frames.py tests/test_fetch_videos.py -v`
Expected: PASS
