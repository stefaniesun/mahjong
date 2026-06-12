import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts import screen_web_videos


class ScreenWebVideosTests(unittest.TestCase):
    def test_collect_video_files_and_merge_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "raw_videos"
            source = root / "bili_demo"
            source.mkdir(parents=True)
            video = source / "sample.mp4"
            video.write_bytes(b"demo")

            videos = screen_web_videos.collect_video_files(root)
            self.assertEqual(videos, [video])

            merged = screen_web_videos.merge_active_ranges(
                [0.0, 1.0, 4.0, 5.0, 11.0],
                max_gap_seconds=3.0,
                min_duration_seconds=1.5,
            )
            self.assertEqual(merged, [(0.0, 5.0)])

    def test_classify_video_flags_low_value_and_screen_capture(self) -> None:
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
        self.assertTrue(decision.low_value)
        self.assertTrue(decision.suspected_screen_recording)
        self.assertIn("active_ratio_below_threshold", decision.reasons)
        self.assertIn("screen_recording_heuristic", decision.reasons)

    def test_analyze_video_without_model_produces_clip_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "sample.mp4"
            thumbnails_root = root / "thumbnails"
            self._write_test_video(video_path)

            config = screen_web_videos.AnalysisConfig(
                sample_fps=1.0,
                min_tiles_per_frame=1,
                min_active_ratio=0.2,
                max_gap_seconds=3.0,
                min_clip_duration=1.0,
            )
            result = screen_web_videos.analyze_video(
                video_path,
                config=config,
                detector=None,
                thumbnails_root=thumbnails_root,
            )
            self.assertGreaterEqual(result.metrics.sampled_frames, 3)
            self.assertTrue(result.clips)
            self.assertTrue(result.thumbnails)
            self.assertTrue(result.thumbnails[0].exists())


    def test_write_outputs_creates_report_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "screen_report.json"
            preview_path = root / "preview.html"
            thumb = root / "thumb.jpg"
            thumb.write_bytes(b"thumb")

            result = screen_web_videos.VideoAnalysisResult(
                video_path=root / "sample.mp4",
                source="bili_demo",
                metrics=screen_web_videos.VideoMetrics(sampled_frames=10, active_tile_frames=5),
                decision=screen_web_videos.VideoDecision(
                    low_value=False,
                    suspected_screen_recording=True,
                    reasons=["screen_recording_heuristic"],
                ),
                clips=[screen_web_videos.ClipRange(start_seconds=0.0, end_seconds=5.0)],
                thumbnails=[thumb],
            )

            screen_web_videos.write_outputs([result], report_path=report_path, preview_path=preview_path)

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_videos"], 1)
            self.assertEqual(payload["videos"][0]["thumbnails"], [str(thumb)])
            html = preview_path.read_text(encoding="utf-8")
            self.assertIn("疑似录屏", html)
            self.assertIn("sample.mp4", html)
            self.assertIn("<img", html)
            self.assertIn("thumb.jpg", html)


    def test_main_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "raw_videos"
            source_dir = input_root / "bili_demo"
            source_dir.mkdir(parents=True)
            self._write_test_video(source_dir / "sample.mp4")

            report_path = root / "screen_report.json"
            preview_path = root / "preview.html"
            clips_root = root / "web_clips"
            thumbnails_root = root / "thumbnails"

            exit_code = screen_web_videos.main(
                [
                    "--input-root",
                    str(input_root),
                    "--report",
                    str(report_path),
                    "--preview",
                    str(preview_path),
                    "--clips-root",
                    str(clips_root),
                    "--thumbnails-root",
                    str(thumbnails_root),
                    "--sample-fps",
                    "1",
                    "--min-active-ratio",
                    "0.1",
                    "--min-clip-duration",
                    "1",
                    "--skip-ffmpeg",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(report_path.exists())
            self.assertTrue(preview_path.exists())
            self.assertTrue(thumbnails_root.exists())
            self.assertTrue(list(thumbnails_root.rglob("*.jpg")))


    def _write_test_video(self, path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            4.0,
            (96, 96),
        )
        if not writer.isOpened():
            self.fail("测试视频写入器未能打开")
        try:
            for idx in range(12):
                frame = np.zeros((96, 96, 3), dtype=np.uint8)
                if 2 <= idx <= 8:
                    frame[16:80, 16:80] = 180
                    frame[::4, ::4] = 255
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
