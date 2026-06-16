import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts import convert_labels


class ConvertLabelsTests(unittest.TestCase):
    def test_bbox_from_points_uses_outer_rectangle(self) -> None:
        bbox = convert_labels.bbox_from_points([[20, 30], [10, 60], [40, 50], [15, 25]])
        self.assertEqual(bbox, (10.0, 25.0, 40.0, 60.0))

    def test_yolo_line_normalizes_single_tile_face_class(self) -> None:
        line = convert_labels.yolo_line((10, 20, 50, 80), image_width=100, image_height=200)
        self.assertEqual(line, "0 0.300000 0.250000 0.400000 0.300000")

    def test_main_exports_yolo_crops_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "labeled"
            output_root = root / "output"
            classes_path = root / "classes.yaml"
            paths_path = root / "paths.yaml"
            input_root.mkdir(parents=True)
            classes_path.write_text(
                "detection:\n  0: tile_face\n\nclassification:\n  - [w1, w2]\n  - [unknown]\n\ndiscard_labels: [back, tile_back]\n",
                encoding="utf-8",
            )
            paths_path.write_text(
                yaml.safe_dump({"validation_labeled": str(input_root), "validation_output": str(output_root)}, sort_keys=False),
                encoding="utf-8",
            )

            self._write_image(input_root / "img_a.jpg", width=100, height=100)
            self._write_label(
                input_root / "img_a.json",
                image_path="img_a.jpg",
                width=100,
                height=100,
                shapes=[
                    {"label": "w1", "points": [[10, 20], [50, 80]]},
                    {"label": "back", "points": [[1, 2], [10, 12]]},
                    {"label": "typo", "points": [[5, 6], [15, 16]]},
                ],
            )
            self._write_image(input_root / "img_b.jpg", width=120, height=120)
            self._write_label(
                input_root / "img_b.json",
                image_path="img_b.jpg",
                width=120,
                height=120,
                shapes=[{"label": "unknown", "points": [[20, 20], [42, 42]]}],
            )
            self._write_image(input_root / "missing_json.jpg", width=80, height=80)

            exit_code = convert_labels.main(
                [
                    "--paths",
                    str(paths_path),
                    "--classes",
                    str(classes_path),
                    "--val-ratio",
                    "0.5",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(exit_code, 0)
            report_path = output_root / "convert_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["total_images"], 3)
            self.assertEqual(report["summary"]["paired_images"], 2)
            self.assertEqual(report["summary"]["skipped_images_without_json"], 1)
            self.assertEqual(report["summary"]["total_shapes"], 4)
            self.assertEqual(report["summary"]["kept_boxes"], 2)
            self.assertEqual(report["summary"]["discarded_back_boxes"], 1)
            self.assertEqual(report["summary"]["invalid_label_boxes"], 1)
            self.assertEqual(report["class_distribution"]["w1"], 1)
            self.assertEqual(report["class_distribution"]["unknown"], 1)
            self.assertEqual(report["invalid_labels"][0]["label"], "typo")
            self.assertEqual(report["missing_json_images"], ["missing_json.jpg"])

            data_yaml = yaml.safe_load((output_root / "yolo_det" / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data_yaml["names"], ["tile_face"])
            self.assertEqual(data_yaml["nc"], 1)
            yolo_txt = list((output_root / "yolo_det" / "labels").rglob("*.txt"))
            self.assertEqual(len(yolo_txt), 2)
            self.assertTrue(any(path.read_text(encoding="utf-8").startswith("0 ") for path in yolo_txt))
            self.assertEqual(len(list((output_root / "cls_crops" / "w1").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output_root / "cls_crops" / "unknown").glob("*.jpg"))), 1)
            self.assertFalse((output_root / "cls_crops" / "back").exists())

    def _write_image(self, path: Path, *, width: int, height: int) -> None:
        image = np.full((height, width, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(path), image)

    def _write_label(self, path: Path, *, image_path: str, width: int, height: int, shapes: list[dict]) -> None:
        payload = {
            "version": "2.4.0",
            "flags": {},
            "shapes": [
                {
                    "label": shape["label"],
                    "points": shape["points"],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                }
                for shape in shapes
            ],
            "imagePath": image_path,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
