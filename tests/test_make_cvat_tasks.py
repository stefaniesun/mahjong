import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import make_cvat_tasks


class MakeCvatTasksTests(unittest.TestCase):
    def test_main_batches_frames_into_zip_tasks_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "frames_selected"
            output_root = root / "cvat_tasks"
            manifest_path = output_root / "task_manifest.json"
            self._write_frames(input_root, total=120)

            exit_code = make_cvat_tasks.main(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--manifest",
                    str(manifest_path),
                    "--task-size",
                    "50",
                    "--task-prefix",
                    "mahjong-v1",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(manifest_path.exists())

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_images"], 120)
            self.assertEqual(payload["summary"]["task_count"], 3)
            self.assertEqual(payload["summary"]["task_size"], 50)
            self.assertEqual([task["image_count"] for task in payload["tasks"]], [50, 50, 20])
            self.assertEqual(payload["tasks"][0]["task_name"], "mahjong-v1-001")
            self.assertEqual(payload["tasks"][2]["task_name"], "mahjong-v1-003")

            zip_paths = sorted(output_root.glob("*.zip"))
            self.assertEqual([path.name for path in zip_paths], [
                "mahjong-v1-001.zip",
                "mahjong-v1-002.zip",
                "mahjong-v1-003.zip",
            ])

            with zipfile.ZipFile(zip_paths[0]) as archive:
                names = sorted(archive.namelist())
            self.assertEqual(len(names), 50)
            self.assertEqual(names[0], "bili_001__video_000__f000001.jpg")
            self.assertEqual(names[-1], "bili_001__video_004__f000050.jpg")

    def test_main_rejects_empty_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "frames_selected"
            output_root = root / "cvat_tasks"
            input_root.mkdir(parents=True)

            exit_code = make_cvat_tasks.main(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(output_root.exists())

    def _write_frames(self, input_root: Path, *, total: int) -> None:
        input_root.mkdir(parents=True, exist_ok=True)
        for index in range(total):
            video_idx = index // 10
            frame_path = input_root / f"bili_001__video_{video_idx:03d}__f{index + 1:06d}.jpg"
            frame_path.write_bytes(f"frame-{index}".encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
