import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts import prepare_data


class PrepareDataTests(unittest.TestCase):
    def test_real_val_split_is_stable_when_new_images_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labeled = root / "labeled"
            datasets = root / "datasets"
            labeled.mkdir()
            self._write_real_pair(labeled / "author_a__001.jpg", "w1")
            self._write_real_pair(labeled / "author_a__002.jpg", "t1")
            self._write_real_pair(labeled / "author_a__003.jpg", "b1")
            classes = self._write_classes(root / "classes.yaml")
            paths_cfg = {"prelabel_labeled_root": str(labeled), "prelabel_datasets_root": str(datasets)}
            train_cfg = {
                "version": 1,
                "seed": 123,
                "real_effective_ratio": 1.0,
                "real_val_ratio_per_author": 0.34,
                "min_val_per_author": 1,
                "use_roboflow": False,
            }

            first = prepare_data.prepare_dataset(paths_cfg, train_cfg, classes, root, version=1, force=True)
            first_val = set(first["real_val"]["val_images"])
            self._write_real_pair(labeled / "author_a__004.jpg", "w2")
            second = prepare_data.prepare_dataset(paths_cfg, train_cfg, classes, root, version=2, force=True)
            second_val = set(second["real_val"]["val_images"])

            self.assertTrue(first_val)
            self.assertTrue(first_val.issubset(second_val))
            self.assertEqual(json.loads((datasets / "real_val" / "manifest.json").read_text())["seed"], 123)

    def test_prepare_mix_ratio_invalid_labels_and_roboflow_flattening(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labeled = root / "labeled"
            rf = root / "rf"
            datasets = root / "datasets"
            labeled.mkdir()
            self._write_real_pair(labeled / "bili_1__001.jpg", "w1")
            self._write_real_pair(labeled / "bili_1__002.jpg", "back")
            self._write_real_pair(labeled / "bili_1__003.jpg", "typo")
            self._write_roboflow_dataset(rf, count=7)
            classes = self._write_classes(root / "classes.yaml")
            paths_cfg = {
                "prelabel_labeled_root": str(labeled),
                "prelabel_datasets_root": str(datasets),
                "prelabel_roboflow_data": str(rf / "data.yaml"),
            }
            train_cfg = {
                "version": 1,
                "seed": 7,
                "real_effective_ratio": 0.35,
                "real_val_ratio_per_author": 0.0,
                "min_val_per_author": 0,
                "use_roboflow": True,
            }

            report = prepare_data.prepare_dataset(paths_cfg, train_cfg, classes, root, version=1, force=True)

            summary = report["summary"]
            self.assertEqual(summary["roboflow_images"], 7)
            self.assertAlmostEqual(summary["actual_real_effective_ratio"], 0.35, delta=0.03)
            self.assertEqual(summary["invalid_label_boxes"], 1)
            self.assertEqual(report["invalid_labels"][0]["label"], "typo")
            data_yaml = yaml.safe_load((datasets / "mix_v1" / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data_yaml["names"], ["tile_face"])
            train_entries = (datasets / "mix_v1" / "train.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(train_entries), summary["train_entries"])
            label_texts = "\n".join(path.read_text(encoding="utf-8") for path in (datasets / "mix_v1" / "labels" / "train").glob("*.txt"))
            self.assertIn("0 0.500000 0.500000 0.500000 0.500000", label_texts)

    def test_recommended_ratio_table(self) -> None:
        self.assertEqual(prepare_data.recommended_ratio(70), "0.30~0.35")
        self.assertEqual(prepare_data.recommended_ratio(200), "0.40~0.55")
        self.assertEqual(prepare_data.recommended_ratio(500), "0.60~0.75")
        self.assertEqual(prepare_data.recommended_ratio(900), "0.80~1.00")

    def _write_classes(self, path: Path) -> Path:
        path.write_text(
            "classification:\n  - [w1, w2]\n  - [t1]\n  - [b1]\n  - [unknown]\n\ndiscard_labels: [back, tile_back]\n",
            encoding="utf-8",
        )
        return path

    def _write_real_pair(self, image_path: Path, label: str) -> None:
        self._write_image(image_path)
        payload = {
            "version": "2.4.0",
            "flags": {},
            "shapes": [
                {
                    "label": label,
                    "points": [[10, 10], [50, 50]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                }
            ],
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": 80,
            "imageWidth": 80,
        }
        image_path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    def _write_roboflow_dataset(self, root: Path, count: int) -> None:
        images = root / "train" / "images"
        labels = root / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(count):
            image = images / f"rf_{index}.jpg"
            self._write_image(image)
            (labels / f"rf_{index}.txt").write_text("12 0.500000 0.500000 0.500000 0.500000\n", encoding="utf-8")
        (root / "data.yaml").write_text(
            yaml.safe_dump({"path": str(root), "train": "train/images", "nc": 27, "names": [str(i) for i in range(27)]}),
            encoding="utf-8",
        )

    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((80, 80, 3), 160, dtype=np.uint8)
        cv2.imwrite(str(path), image)


if __name__ == "__main__":
    unittest.main()
