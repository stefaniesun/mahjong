import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts import dedup_filter


class DedupFilterTests(unittest.TestCase):
    def test_group_duplicates_keeps_sharpest_frame_across_videos_for_same_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "frames_candidate" / "bili_123"
            video_a = source_root / "video_a"
            video_b = source_root / "video_b"
            video_a.mkdir(parents=True)
            video_b.mkdir(parents=True)

            soft = video_a / "video_a_f000001.jpg"
            sharp = video_b / "video_b_f000002.jpg"
            self._write_pattern_image(soft, stripe_step=12)
            self._write_pattern_image(sharp, stripe_step=4)

            candidates = dedup_filter.collect_frame_candidates(root / "frames_candidate")
            unique_items = dedup_filter.deduplicate_source_candidates(
                candidates_by_source={"bili_123": candidates},
                phash_distance_threshold=8,
            )["bili_123"]

            self.assertEqual(len(unique_items), 1)
            self.assertEqual(unique_items[0].path, sharp)
            self.assertGreater(unique_items[0].blur_score, 0.0)

    def test_plan_selection_balances_sources_and_videos_with_caps(self) -> None:
        items = []
        for source_name in ["bili_1", "bili_2", "bili_3", "bili_4"]:
            for video_idx in range(3):
                for frame_idx in range(3):
                    items.append(
                        dedup_filter.FrameCandidate(
                            path=Path(f"{source_name}/video_{video_idx}/frame_{frame_idx}.jpg"),
                            source=source_name,
                            video_id=f"video_{video_idx}",
                            blur_score=100.0 - frame_idx,
                            phash_hex=f"{source_name}-{video_idx}-{frame_idx}",
                        )
                    )

        plan = dedup_filter.plan_selection(
            items_by_source={
                source: [item for item in items if item.source == source]
                for source in ["bili_1", "bili_2", "bili_3", "bili_4"]
            },
            total=8,
            max_source_ratio=0.25,
            max_per_video=1,
        )

        self.assertEqual(sum(len(value) for value in plan.values()), 8)
        for source, selected in plan.items():
            self.assertLessEqual(len(selected), 2)
            self.assertEqual(len({item.video_id for item in selected}), len(selected))

    def test_main_dry_run_writes_report_without_copying_files_and_uses_format_b_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "frames_candidate"
            output_root = root / "frames_selected"
            report_path = root / "selection_report.json"

            self._write_candidate_set(input_root / "bili_111", video_count=2, frames_per_video=1)
            self._write_candidate_set(input_root / "dy_222", video_count=2, frames_per_video=1)

            exit_code = dedup_filter.main(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--report",
                    str(report_path),
                    "--total",
                    "4",
                    "--max-source-ratio",
                    "0.5",
                    "--max-per-video",
                    "1",
                    "--filename-style",
                    "B",
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(report_path.exists())
            self.assertFalse(output_root.exists() and list(output_root.rglob("*.jpg")))

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["summary"]["selected_frames"], 4)
            self.assertEqual(payload["summary"]["selected_sources"], 2)
            first_name = payload["selected_files"][0]["target_name"]
            self.assertRegex(first_name, r"^(bili|dy)_\d+__video_\d+__f\d{6}\.jpg$")


    def test_main_copies_selected_files_and_report_matches_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "frames_candidate"
            output_root = root / "frames_selected"
            report_path = root / "selection_report.json"

            self._write_candidate_set(input_root / "bili_111", video_count=2, frames_per_video=3)
            self._write_candidate_set(input_root / "dy_222", video_count=2, frames_per_video=3)

            exit_code = dedup_filter.main(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--report",
                    str(report_path),
                    "--total",
                    "4",
                    "--max-source-ratio",
                    "0.5",
                    "--max-per-video",
                    "2",
                    "--filename-style",
                    "B",
                ]
            )

            self.assertEqual(exit_code, 0)
            copied_files = sorted(output_root.rglob("*.jpg"))
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(copied_files), 4)
            self.assertEqual(payload["summary"]["selected_frames"], len(copied_files))
            self.assertEqual({path.name for path in copied_files}, {item["target_name"] for item in payload["selected_files"]})

    def _write_candidate_set(self, source_dir: Path, *, video_count: int, frames_per_video: int) -> None:
        for video_idx in range(video_count):
            video_dir = source_dir / f"video_{video_idx}"
            video_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(frames_per_video):
                path = video_dir / f"video_{video_idx}_f{frame_idx + 1:06d}.jpg"
                self._write_pattern_image(path, stripe_step=4 + frame_idx + video_idx)

    def _write_pattern_image(self, path: Path, *, stripe_step: int) -> None:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[:, :] = 80
        image[:, ::stripe_step] = 240
        image[::stripe_step, :] = 200
        cv2.imwrite(str(path), image)


if __name__ == "__main__":
    unittest.main()
