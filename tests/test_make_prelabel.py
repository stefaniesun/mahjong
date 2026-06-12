import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import make_prelabel


class MakePrelabelTests(unittest.TestCase):
    def test_main_writes_xanylabeling_json_and_maps_unknown_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "frames_selected"
            classes_path = root / "classes.yaml"
            model_path = root / "legacy.pt"
            input_dir.mkdir(parents=True)
            model_path.write_bytes(b"fake-model")
            classes_path.write_text(
                "detection:\n  0: tile_face\n  1: tile_back\n\nclassification:\n  - [w1, w2]\n  - [back]\n  - [unknown]\n",
                encoding="utf-8",
            )
            image_path = input_dir / "bili_123__clip001__f000001.jpg"
            image_path.write_bytes(b"fake-image")

            with patch.object(make_prelabel, "load_model", return_value=FakeModel()):
                exit_code = make_prelabel.main(
                    [
                        "--input-root",
                        str(input_dir),
                        "--model",
                        str(model_path),
                        "--classes",
                        str(classes_path),
                        "--conf",
                        "0.25",
                    ]
                )

            self.assertEqual(exit_code, 0)
            label_path = input_dir / "bili_123__clip001__f000001.json"
            self.assertTrue(label_path.exists())

            payload = json.loads(label_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["imagePath"], "bili_123__clip001__f000001.jpg")
            self.assertEqual(payload["imageWidth"], 640)
            self.assertEqual(payload["imageHeight"], 480)
            self.assertEqual([shape["label"] for shape in payload["shapes"]], ["w1", "unknown", "back"])
            self.assertEqual(payload["shapes"][0]["points"], [[10.0, 20.0], [60.0, 120.0]])
            self.assertEqual(payload["shapes"][2]["points"], [[100.0, 110.0], [140.0, 170.0]])

    def test_main_rejects_missing_input_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "frames_selected"
            classes_path = root / "classes.yaml"
            model_path = root / "legacy.pt"
            input_dir.mkdir(parents=True)
            model_path.write_bytes(b"fake-model")
            classes_path.write_text(
                "detection:\n  0: tile_face\n  1: tile_back\n\nclassification:\n  - [w1]\n  - [back]\n  - [unknown]\n",
                encoding="utf-8",
            )

            with patch.object(make_prelabel, "load_model", return_value=FakeModel()):
                exit_code = make_prelabel.main(
                    [
                        "--input-root",
                        str(input_dir),
                        "--model",
                        str(model_path),
                        "--classes",
                        str(classes_path),
                    ]
                )

            self.assertEqual(exit_code, 1)


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([
            [10.0, 20.0, 60.0, 120.0],
            [70.0, 80.0, 90.0, 100.0],
            [100.0, 110.0, 140.0, 170.0],
        ])
        self.conf = FakeTensor([0.91, 0.77, 0.66])
        self.cls = FakeTensor([0, 5, 2])


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.orig_shape = (480, 640)


class FakeModel:
    names = {0: "w1", 2: "back", 5: "dragon_red"}

    def predict(self, source, conf, verbose):
        return [FakeResult()]


if __name__ == "__main__":
    unittest.main()
