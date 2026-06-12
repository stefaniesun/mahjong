import json
import tempfile
import unittest
from pathlib import Path

from eval import eval_detection


class EvalDetectionTests(unittest.TestCase):
    def test_main_writes_metrics_by_size_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_path = root / "gt.json"
            pred_path = root / "pred.json"
            output_dir = root / "eval_output"

            gt_payload = {
                "images": [
                    {"id": 1, "file_name": "img_a.jpg", "width": 100, "height": 100, "source": "bili_u1"},
                    {"id": 2, "file_name": "img_b.jpg", "width": 120, "height": 120, "source": "dy_u2"},
                ],
                "categories": [
                    {"id": 1, "name": "tile_face"},
                    {"id": 2, "name": "tile_back"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
                    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [40, 40, 30, 30], "area": 900, "iscrowd": 0},
                    {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 50, 50], "area": 2500, "iscrowd": 0},
                ],
            }
            pred_payload = [
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "score": 0.95},
                {"image_id": 1, "category_id": 2, "bbox": [42, 42, 28, 28], "score": 0.90},
                {"image_id": 1, "category_id": 1, "bbox": [70, 70, 10, 10], "score": 0.60},
                {"image_id": 2, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.55},
            ]
            gt_path.write_text(json.dumps(gt_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            pred_path.write_text(json.dumps(pred_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            exit_code = eval_detection.main(
                [
                    "--ground-truth",
                    str(gt_path),
                    "--predictions",
                    str(pred_path),
                    "--output-dir",
                    str(output_dir),
                    "--thresholds",
                    "0.5,0.8",
                    "--max-missed",
                    "5",
                ]
            )

            self.assertEqual(exit_code, 0)
            report_path = output_dir / "detection_metrics.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertIn("summary", report)
            self.assertIn("coco_metrics", report)
            self.assertIn("size_bucket_metrics", report)
            self.assertIn("sources", report)
            self.assertIn("pr_points", report)
            self.assertIn("missed_images", report)

            self.assertEqual(report["summary"]["total_images"], 2)
            self.assertEqual(report["summary"]["total_ground_truth"], 3)
            self.assertEqual(report["summary"]["total_predictions"], 4)

            bucket_metrics = report["size_bucket_metrics"]
            self.assertAlmostEqual(bucket_metrics["lt20"]["recall"], 1.0)
            self.assertAlmostEqual(bucket_metrics["20to40"]["recall"], 1.0)
            self.assertAlmostEqual(bucket_metrics["gt40"]["recall"], 0.0)
            self.assertAlmostEqual(bucket_metrics["lt20"]["precision"], 0.5)

            source_metrics = report["sources"]
            self.assertAlmostEqual(source_metrics["bili_u1"]["recall"], 1.0)
            self.assertAlmostEqual(source_metrics["dy_u2"]["recall"], 0.0)

            pr_points = report["pr_points"]
            self.assertEqual([point["threshold"] for point in pr_points], [0.5, 0.8])
            self.assertAlmostEqual(pr_points[0]["precision"], 0.5)
            self.assertAlmostEqual(pr_points[0]["recall"], 2 / 3, places=4)
            self.assertAlmostEqual(pr_points[1]["precision"], 1.0)
            self.assertAlmostEqual(pr_points[1]["recall"], 2 / 3, places=4)

            self.assertEqual(len(report["missed_images"]), 1)
            self.assertEqual(report["missed_images"][0]["file_name"], "img_b.jpg")
            self.assertTrue((output_dir / "missed").exists())

    def test_compute_detection_summary_matches_handcrafted_case(self) -> None:
        ground_truth = {
            "images": [
                {"id": 1, "file_name": "img.jpg", "width": 100, "height": 100, "source": "src"},
            ],
            "categories": [
                {"id": 1, "name": "tile_face"},
                {"id": 2, "name": "tile_back"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [30, 30, 30, 30]},
            ],
        }
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [31, 31, 28, 28], "score": 0.8},
            {"image_id": 1, "category_id": 1, "bbox": [70, 70, 10, 10], "score": 0.7},
        ]

        summary = eval_detection.compute_detection_report(ground_truth, predictions, thresholds=[0.5])

        self.assertEqual(summary["summary"]["matched_predictions"], 2)
        self.assertAlmostEqual(summary["size_bucket_metrics"]["lt20"]["recall"], 1.0)
        self.assertAlmostEqual(summary["size_bucket_metrics"]["20to40"]["recall"], 1.0)
        self.assertAlmostEqual(summary["overall"]["precision"], 2 / 3, places=4)
        self.assertAlmostEqual(summary["overall"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
