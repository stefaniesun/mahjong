import tempfile
import unittest
from pathlib import Path

from scripts import make_prelabel


EXPECTED_LABELS = {
    **{f"{idx}B": f"t{idx}" for idx in range(1, 10)},
    **{f"{idx}C": f"w{idx}" for idx in range(1, 10)},
    **{f"{idx}D": f"b{idx}" for idx in range(1, 10)},
}


class PrelabelMapTests(unittest.TestCase):
    def test_prelabel_map_covers_27_classes_and_keeps_suits_correct(self) -> None:
        mapping = make_prelabel.load_prelabel_map(Path("configs/prelabel_map.yaml"))

        self.assertEqual(set(mapping), set(EXPECTED_LABELS))
        self.assertEqual(mapping, EXPECTED_LABELS)
        self.assertEqual(mapping["1B"], "t1")
        self.assertEqual(mapping["9B"], "t9")
        self.assertEqual(mapping["1C"], "w1")
        self.assertEqual(mapping["9C"], "w9")
        self.assertEqual(mapping["1D"], "b1")
        self.assertEqual(mapping["9D"], "b9")

    def test_load_model_path_from_paths_prefers_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = Path(tmpdir) / "paths.yaml"
            paths.write_text(
                "prelabeler_pt: runs/detect/prelabel_v1/weights/best.pt\n"
                "prelabeler_onnx: runs/detect/prelabel_v1/weights/best.onnx\n",
                encoding="utf-8",
            )

            model_path = make_prelabel.load_model_path_from_paths(paths)

        self.assertEqual(model_path.as_posix(), "runs/detect/prelabel_v1/weights/best.onnx")


if __name__ == "__main__":
    unittest.main()
