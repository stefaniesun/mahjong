import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts import validate_coco


class ValidateCocoTests(unittest.TestCase):
    def test_main_detects_invalid_boxes_and_writes_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True)
            annotations_path = root / "instances.json"
            report_path = root / "validation_report.json"
            preview_dir = root / "validation_preview"
            classes_path = root / "classes.yaml"
            classes_path.write_text(
                "detection:\n  0: tile_face\n  1: tile_back\n\nclassification:\n  - [w1, w2]\n  - [back]\n  - [unknown]\n",
                encoding="utf-8",
            )

            self._write_image(images_dir / "img_001.jpg", width=100, height=100)
            self._write_image(images_dir / "img_002.jpg", width=120, height=120)

            payload = {
                "images": [
                    {"id": 1, "file_name": "img_001.jpg", "width": 100, "height": 100},
                    {"id": 2, "file_name": "img_002.jpg", "width": 120, "height": 120},
                ],
                "categories": [
                    {"id": 1, "name": "w1"},
                    {"id": 2, "name": "w2"},
                    {"id": 3, "name": "back"},
                    {"id": 4, "name": "unknown"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 4, 4], "area": 16, "iscrowd": 0},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 60, 8], "area": 480, "iscrowd": 0},
                    {"id": 3, "image_id": 1, "category_id": 2, "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 0},
                    {"id": 4, "image_id": 1, "category_id": 2, "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 0},
                    {"id": 5, "image_id": 2, "category_id": 3, "bbox": [110, 110, 20, 20], "area": 400, "iscrowd": 0},
                ],
            }
            annotations_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            exit_code = validate_coco.main(
                [
                    "--annotations",
                    str(annotations_path),
                    "--images-root",
                    str(images_dir),
                    "--classes",
                    str(classes_path),
                    "--report",
                    str(report_path),
                    "--preview-dir",
                    str(preview_dir),
                    "--preview-count",
                    "2",
                    "--seed",
                    "123",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(report_path.exists())
            self.assertTrue(preview_dir.exists())
            self.assertEqual(len(list(preview_dir.glob("*.jpg"))), 2)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["total_images"], 2)
            self.assertEqual(report["summary"]["total_annotations"], 5)
            self.assertEqual(report["summary"]["preview_images"], 2)
            self.assertEqual(report["class_distribution"]["unknown"], 0)
            self.assertIn("unknown", report["warnings"]["missing_categories"])
            self.assertEqual(report["size_buckets"]["lt20"], 2)
            self.assertEqual(report["size_buckets"]["20to40"], 3)

            problem_types = {item["problem_type"] for item in report["issues"]}
            self.assertIn("tiny_box", problem_types)
            self.assertIn("extreme_aspect_ratio", problem_types)
            self.assertIn("duplicate_box", problem_types)
            self.assertIn("out_of_bounds", problem_types)

    def test_main_rejects_category_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True)
            annotations_path = root / "instances.json"
            report_path = root / "validation_report.json"
            classes_path = root / "classes.yaml"
            classes_path.write_text(
                "detection:\n  0: tile_face\n  1: tile_back\n\nclassification:\n  - [w1]\n  - [back]\n",
                encoding="utf-8",
            )

            self._write_image(images_dir / "img_001.jpg", width=64, height=64)
            payload = {
                "images": [{"id": 1, "file_name": "img_001.jpg", "width": 64, "height": 64}],
                "categories": [
                    {"id": 1, "name": "w1"},
                    {"id": 2, "name": "back"},
                    {"id": 3, "name": "extra"},
                ],
                "annotations": [],
            }
            annotations_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            exit_code = validate_coco.main(
                [
                    "--annotations",
                    str(annotations_path),
                    "--images-root",
                    str(images_dir),
                    "--classes",
                    str(classes_path),
                    "--report",
                    str(report_path),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(report_path.exists())

    def _write_image(self, path: Path, *, width: int, height: int) -> None:
        image = np.full((height, width, 3), 220, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (width - 10, height - 10), (0, 120, 0), 2)
        cv2.imwrite(str(path), image)


if __name__ == "__main__":
    unittest.main()
