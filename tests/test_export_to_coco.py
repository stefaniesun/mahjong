import json
import tempfile
import unittest
from pathlib import Path

from scripts import export_to_coco


class ExportToCocoTests(unittest.TestCase):
    def test_main_converts_xanylabeling_json_to_coco_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "labeled"
            output_path = root / "instances.json"
            classes_path = root / "classes.yaml"
            input_dir.mkdir(parents=True)
            classes_path.write_text(
                "detection:\n  0: tile_face\n  1: tile_back\n\nclassification:\n  - [w1, w2]\n  - [back]\n  - [unknown]\n",
                encoding="utf-8",
            )

            (input_dir / "bili_123__clip001__f000001.jpg").write_bytes(b"fake-image")
            (input_dir / "dy_456__clip010__f000008.jpg").write_bytes(b"fake-image")

            self._write_label(
                input_dir / "bili_123__clip001__f000001.json",
                image_path="bili_123__clip001__f000001.jpg",
                width=1280,
                height=720,
                shapes=[
                    {"label": "w1", "points": [[10, 20], [110, 220]]},
                    {"label": "w2", "points": [[200, 210], [240, 260]]},
                ],
            )
            self._write_label(
                input_dir / "dy_456__clip010__f000008.json",
                image_path="dy_456__clip010__f000008.jpg",
                width=640,
                height=480,
                shapes=[
                    {"label": "unknown", "points": [[5, 6], [25, 36]]},
                    {"label": "back", "points": [[40, 50], [90, 140]]},
                ],
            )

            exit_code = export_to_coco.main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(output_path),
                    "--classes",
                    str(classes_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([item["name"] for item in payload["categories"]], ["w1", "w2", "back", "unknown"])
            self.assertEqual(len(payload["images"]), 2)
            self.assertEqual(len(payload["annotations"]), 4)

            first_image = payload["images"][0]
            self.assertEqual(first_image["file_name"], "bili_123__clip001__f000001.jpg")
            self.assertEqual(first_image["source"], "bili_123")
            self.assertEqual(first_image["width"], 1280)
            self.assertEqual(first_image["height"], 720)

            first_bbox = payload["annotations"][0]["bbox"]
            self.assertEqual(first_bbox, [10.0, 20.0, 100.0, 200.0])
            self.assertEqual(payload["annotations"][0]["area"], 20000.0)

    def test_main_rejects_unknown_label_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "labeled"
            output_path = root / "instances.json"
            classes_path = root / "classes.yaml"
            input_dir.mkdir(parents=True)
            classes_path.write_text(
                "detection:\n  0: tile_face\n  1: tile_back\n\nclassification:\n  - [w1]\n  - [back]\n  - [unknown]\n",
                encoding="utf-8",
            )
            (input_dir / "img.jpg").write_bytes(b"fake-image")
            self._write_label(
                input_dir / "img.json",
                image_path="img.jpg",
                width=100,
                height=100,
                shapes=[{"label": "not-in-classes", "points": [[1, 2], [3, 4]]}],
            )

            exit_code = export_to_coco.main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(output_path),
                    "--classes",
                    str(classes_path),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(output_path.exists())

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
