import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from scripts import validation_report


class ValidationReportTests(unittest.TestCase):
    def test_iou_and_greedy_match(self) -> None:
        ann = validation_report.Annotation("img", Path("img.jpg"), "w1", (10, 10, 30, 30), "20to40", "bili_a")
        preds = [
            validation_report.Prediction((12, 12, 31, 31), 0.9),
            validation_report.Prediction((60, 60, 80, 80), 0.8),
        ]

        self.assertGreater(validation_report.iou(ann.bbox, preds[0].bbox), 0.5)
        matches, missed, false_positive = validation_report.greedy_match([ann], preds, 0.5)

        self.assertEqual(len(matches), 1)
        self.assertEqual(missed, [])
        self.assertEqual(false_positive, [1])

    def test_loss_is_decreasing(self) -> None:
        rows = [{"train/box_loss": 2.0}, {"train/box_loss": 1.0}]
        self.assertTrue(validation_report.loss_is_decreasing(rows))
        self.assertFalse(validation_report.loss_is_decreasing([{"train/box_loss": 1.0}, {"train/box_loss": 1.2}]))

    def test_generate_report_writes_html_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labeled = root / "data" / "labeled"
            output = root / "output" / "validation_run_v1"
            train_images = output / "yolo_det" / "images" / "train"
            run_dir = root / "runs" / "val_run_v1" / "detector"
            weights = run_dir / "weights"
            configs = root / "configs"
            labeled.mkdir(parents=True)
            train_images.mkdir(parents=True)
            weights.mkdir(parents=True)
            configs.mkdir(parents=True)

            classes_path = configs / "classes.yaml"
            classes_path.write_text(
                "detection:\n  0: tile_face\nclassification:\n  - [w1, w2]\n  - [unknown]\ndiscard_labels: [back, tile_back]\n",
                encoding="utf-8",
            )
            paths_path = configs / "paths.yaml"
            paths_path.write_text(
                yaml.safe_dump(
                    {
                        "validation_labeled": str(labeled),
                        "validation_output": str(output),
                        "validation_run_dir": str(run_dir),
                        "validation_best_pt": str(weights / "best.pt"),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            image = np.full((100, 100, 3), 210, dtype=np.uint8)
            cv2.imwrite(str(labeled / "bili_a__img001.jpg"), image)
            cv2.imwrite(str(train_images / "bili_a__img001.jpg"), image)
            (labeled / "bili_a__img001.json").write_text(
                json.dumps(
                    {
                        "imageWidth": 100,
                        "imageHeight": 100,
                        "shapes": [{"label": "w1", "points": [[10, 10], [40, 40]], "shape_type": "rectangle"}],
                    }
                ),
                encoding="utf-8",
            )
            (output / "convert_report.json").write_text(
                json.dumps({"summary": {"kept_boxes": 1, "invalid_label_boxes": 0}}, ensure_ascii=False), encoding="utf-8"
            )
            (weights / "best.pt").write_text("fake", encoding="utf-8")
            with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as file_obj:
                writer = csv.DictWriter(
                    file_obj,
                    fieldnames=["epoch", "train/box_loss", "metrics/recall(B)", "metrics/mAP50(B)"],
                )
                writer.writeheader()
                writer.writerow({"epoch": 1, "train/box_loss": 2.0, "metrics/recall(B)": 0.5, "metrics/mAP50(B)": 0.4})
                writer.writerow({"epoch": 2, "train/box_loss": 1.0, "metrics/recall(B)": 0.9, "metrics/mAP50(B)": 0.8})

            def fake_predictor(best_pt: Path, images: list[Path], conf: float, device: str | None):
                return {images[0].stem: [validation_report.Prediction((11, 11, 39, 39), 0.99)]}

            with mock.patch("pathlib.Path.cwd", return_value=root):
                summary = validation_report.generate_report(
                    paths_path=paths_path,
                    classes_path=classes_path,
                    check_output=root / "output" / "check",
                    html_path=root / "output" / "validation_report.html",
                    predictor=fake_predictor,
                )

            self.assertEqual(summary["conclusion"]["status"], "合格")
            self.assertGreaterEqual(summary["match_rate"], 0.7)
            self.assertEqual(summary["diff_images"], 1)
            self.assertTrue((root / "output" / "validation_report.html").exists())
            self.assertTrue((root / "output" / "validation_report_summary.json").exists())

    def test_generate_report_flags_low_class_recall(self) -> None:
        anns = [
            validation_report.Annotation("img", Path("img.jpg"), "w1", (i * 10, 0, i * 10 + 8, 8), "lt20", "src")
            for i in range(5)
        ]
        eval_result = validation_report.evaluate_predictions(
            [Path("img.jpg")], {"img": anns}, {"img": []}, validation_report.IOU_THRESHOLD
        )
        self.assertEqual(eval_result["match_rate"], 0.0)
        self.assertEqual(eval_result["class_recall"][0]["total"], 5)


if __name__ == "__main__":
    unittest.main()
