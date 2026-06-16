import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts import visualize_labels


class VisualizeLabelsTests(unittest.TestCase):
    def test_label_color_groups_suits_and_invalid_labels(self) -> None:
        self.assertEqual(visualize_labels.label_color("w1"), visualize_labels.SUIT_COLORS["w"])
        self.assertEqual(visualize_labels.label_color("t9"), visualize_labels.SUIT_COLORS["t"])
        self.assertEqual(visualize_labels.label_color("b3"), visualize_labels.SUIT_COLORS["b"])
        self.assertEqual(visualize_labels.label_color("unknown"), visualize_labels.UNKNOWN_COLOR)
        self.assertEqual(visualize_labels.label_color("typo", valid=False), visualize_labels.INVALID_COLOR)

    def test_draw_labeled_image_skips_discard_and_marks_invalid(self) -> None:
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        shapes = [
            {"label": "w1", "points": [[10, 20], [50, 80]]},
            {"label": "back", "points": [[60, 20], [90, 60]]},
            {"label": "typo", "points": [[100, 20], [140, 60]]},
        ]

        rendered, drawn = visualize_labels.draw_labeled_image(image, shapes, {"w1", "unknown"}, {"back", "tile_back"})

        self.assertEqual(drawn, 2)
        self.assertFalse(np.array_equal(rendered, image))

    def test_main_writes_preview_images_contact_sheet_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "labeled"
            output_dir = root / "label_preview"
            classes_path = root / "classes.yaml"
            paths_path = root / "paths.yaml"
            input_root.mkdir(parents=True)
            classes_path.write_text(
                "detection:\n  0: tile_face\n\nclassification:\n  - [w1, w2]\n  - [t4, t5]\n  - [unknown]\n\ndiscard_labels: [back, tile_back]\n",
                encoding="utf-8",
            )
            paths_path.write_text(
                yaml.safe_dump(
                    {
                        "validation_labeled": str(input_root),
                        "label_preview_output": str(output_dir),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            self._write_image(input_root / "img_a.jpg", width=160, height=120)
            self._write_label(
                input_root / "img_a.json",
                image_path="img_a.jpg",
                width=160,
                height=120,
                shapes=[
                    {"label": "w1", "points": [[10, 20], [50, 80]]},
                    {"label": "t4", "points": [[70, 20], [110, 85]]},
                    {"label": "back", "points": [[115, 20], [130, 50]]},
                ],
            )
            self._write_image(input_root / "img_b.jpg", width=180, height=120)
            self._write_label(
                input_root / "img_b.json",
                image_path="img_b.jpg",
                width=180,
                height=120,
                shapes=[
                    {"label": "w2", "points": [[15, 25], [55, 82]]},
                    {"label": "t5", "points": [[80, 25], [125, 90]]},
                    {"label": "unknown", "points": [[130, 25], [165, 95]]},
                ],
            )

            exit_code = visualize_labels.main(
                [
                    "--paths",
                    str(paths_path),
                    "--classes",
                    str(classes_path),
                    "--count",
                    "2",
                    "--per-class",
                    "1",
                    "--seed",
                    "3",
                    "--confusing-classes",
                    "t4",
                    "t5",
                    "w1",
                    "w2",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(list((output_dir / "images").glob("*.jpg"))), 2)
            self.assertTrue((output_dir / "confusing_classes_contact_sheet.jpg").exists())
            report = json.loads((output_dir / "visualize_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["preview_count"], 2)
            self.assertEqual(report["confusing_crop_counts"], {"t4": 1, "t5": 1, "w1": 1, "w2": 1})

    def _write_image(self, path: Path, *, width: int, height: int) -> None:
        image = np.full((height, width, 3), 180, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

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
