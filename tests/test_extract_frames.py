import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts import extract_frames


class ExtractFramesTests(unittest.TestCase):
    def test_should_keep_frame_filters_blur_and_exposure(self) -> None:
        sharp = np.zeros((64, 64, 3), dtype=np.uint8)
        sharp[:, :32] = 255
        keep, reason, _, _ = extract_frames.should_keep_frame(
            sharp,
            blur_threshold=10,
            min_brightness=10,
            max_brightness=245,
        )
        self.assertTrue(keep)
        self.assertIsNone(reason)

        blurred = np.full((64, 64, 3), 128, dtype=np.uint8)
        keep, reason, _, _ = extract_frames.should_keep_frame(
            blurred,
            blur_threshold=10,
            min_brightness=10,
            max_brightness=245,
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "blur")

        dark = np.zeros((64, 64, 3), dtype=np.uint8)
        dark[0, 0] = 255
        keep, reason, _, _ = extract_frames.should_keep_frame(
            dark,
            blur_threshold=0,
            min_brightness=10,
            max_brightness=245,
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "dark")

        bright = np.full((64, 64, 3), 255, dtype=np.uint8)
        keep, reason, _, _ = extract_frames.should_keep_frame(
            bright,
            blur_threshold=0,
            min_brightness=10,
            max_brightness=245,
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "bright")

    def test_main_extracts_frames_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "raw_videos"
            source_dir = input_root / "bili_demo"
            source_dir.mkdir(parents=True)
            video_path = source_dir / "sample.mp4"
            self._write_test_video(video_path)

            output_root = root / "frames_candidate"
            report_path = root / "extract_report.json"

            exit_code = extract_frames.main(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--report",
                    str(report_path),
                    "--fps",
                    "2",
                    "--blur-threshold",
                    "10",
                    "--min-brightness",
                    "10",
                    "--max-brightness",
                    "245",
                ]
            )

            self.assertEqual(exit_code, 0)
            images = sorted(output_root.rglob("*.jpg"))
            self.assertGreaterEqual(len(images), 2)

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_videos"], 1)
            self.assertEqual(payload["summary"]["saved_frames"], len(images))
            self.assertIn("videos", payload)
            self.assertEqual(payload["videos"][0]["source"], "bili_demo")

    def test_main_extracts_frames_from_downloaded_bilibili_sample(self) -> None:
        source_root = Path("d:/mahjong/data/raw_videos_smoke")
        if not source_root.exists():
            self.skipTest("本地 smoke 下载样本不存在，跳过真实联调测试")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "raw_videos_smoke"
            shutil.copytree(source_root, input_root)
            output_root = root / "frames_candidate"
            report_path = root / "extract_report.json"

            exit_code = extract_frames.main(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--report",
                    str(report_path),
                    "--fps",
                    "0.2",
                    "--blur-threshold",
                    "20",
                    "--min-brightness",
                    "5",
                    "--max-brightness",
                    "250",
                    "--limit-videos",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            images = sorted(output_root.rglob("*.jpg"))
            self.assertGreaterEqual(len(images), 1)

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_videos"], 1)
            self.assertEqual(payload["summary"]["saved_frames"], len(images))
            self.assertGreaterEqual(payload["videos"][0]["sampled_frames"], len(images))
            self.assertIn("bili_1014433798", payload["videos"][0]["source"])

    def _write_test_video(self, path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            4.0,
            (64, 64),
        )
        if not writer.isOpened():
            self.fail("测试视频写入器未能打开")
        try:
            for idx in range(8):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[:, : 16 + idx * 4] = 255
                frame[::2, ::2] = 64
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()

