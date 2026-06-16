import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import train_v1


class TrainV1Tests(unittest.TestCase):
    def test_build_train_command_uses_validation_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "data.yaml"
            project = root / "runs" / "val_run_v1"

            command = train_v1.build_train_command(
                data=data,
                model="yolo11n.pt",
                epochs=60,
                imgsz=960,
                batch="auto",
                patience=20,
                project=project,
                name="detector",
                device="0",
                workers=2,
            )

            self.assertIn("detect", command)
            self.assertIn("train", command)
            self.assertIn(f"data={data.resolve().as_posix()}", command)
            self.assertIn("model=yolo11n.pt", command)
            self.assertIn("epochs=60", command)
            self.assertIn("imgsz=960", command)
            self.assertIn("batch=-1", command)
            self.assertIn("patience=20", command)
            self.assertIn(f"project={project.resolve().as_posix()}", command)
            self.assertIn("name=detector", command)
            self.assertIn("plots=True", command)
            self.assertIn("device=0", command)
            self.assertIn("workers=2", command)

    def test_main_dry_run_writes_config_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "output" / "validation_run_v1" / "yolo_det" / "data.yaml"
            paths = root / "configs" / "paths.yaml"
            data.parent.mkdir(parents=True)
            paths.parent.mkdir(parents=True)
            data.write_text("path: dummy\ntrain: images/train\nval: images/val\nnc: 1\nnames: [tile_face]\n", encoding="utf-8")
            paths.write_text(yaml.safe_dump({"validation_output": str(root / "output" / "validation_run_v1")}), encoding="utf-8")
            run_project = root / "runs" / "val_run_v1"

            with mock.patch("pathlib.Path.cwd", return_value=root):
                exit_code = train_v1.main(
                    [
                        "--paths",
                        str(paths),
                        "--project",
                        str(run_project),
                        "--name",
                        "detector",
                        "--epochs",
                        "1",
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            config_path = run_project / "detector" / "train_v1_config.yaml"
            command_path = run_project / "detector" / "train_v1_command.txt"
            data_snapshot = run_project / "detector" / "data_snapshot.yaml"
            self.assertTrue(config_path.exists())
            self.assertTrue(command_path.exists())
            self.assertTrue(data_snapshot.exists())
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["notice"], train_v1.NOTICE)
            self.assertEqual(config["epochs"], 1)
            self.assertEqual(config["model"], "yolo11n.pt")

    def test_update_paths_config_records_training_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = root / "configs" / "paths.yaml"
            paths.parent.mkdir(parents=True)
            paths.write_text("validation_output: output/validation_run_v1\n", encoding="utf-8")

            train_v1.update_paths_config(
                paths,
                best_pt=root / "runs" / "val_run_v1" / "detector" / "weights" / "best.pt",
                run_dir=root / "runs" / "val_run_v1" / "detector",
                data_path=root / "output" / "validation_run_v1" / "yolo_det" / "data.yaml",
            )

            payload = yaml.safe_load(paths.read_text(encoding="utf-8"))
            self.assertEqual(payload["validation_output"], "output/validation_run_v1")
            self.assertTrue(payload["validation_best_pt"].endswith("best.pt"))
            self.assertTrue(payload["validation_run_dir"].endswith("detector"))
            self.assertTrue(payload["validation_yolo_data"].endswith("data.yaml"))


if __name__ == "__main__":
    unittest.main()
