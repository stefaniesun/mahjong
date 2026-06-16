"""Train the Validation Run v1 single-class Mahjong tile detector.

Example:
    python scripts/train_v1.py --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

NOTICE = "本次为验证性训练,数据量小(~70张),精度低属正常,重点看标注质量与管线连通性"
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_EPOCHS = 60
DEFAULT_IMGSZ = 960
DEFAULT_BATCH = -1
DEFAULT_PATIENCE = 20
DEFAULT_PROJECT = Path("runs") / "val_run_v1"
DEFAULT_NAME = "detector"
DEFAULT_PATHS = Path("configs/paths.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO11n for Validation Run v1 single-class tile detection.")
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="Path config YAML.")
    parser.add_argument("--data", type=Path, default=None, help="YOLO data.yaml path. Defaults to validation_output/yolo_det/data.yaml.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics model name/path.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Training image size.")
    parser.add_argument("--batch", default=str(DEFAULT_BATCH), help="Batch size; -1/auto enables Ultralytics auto batch.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Independent experiment project directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Experiment name under --project.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=None, help="Optional dataloader workers.")
    parser.add_argument("--exist-ok", action="store_true", default=True, help="Overwrite/reuse the same experiment directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and write config snapshot without training.")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def resolve_config_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def yolo_executable() -> str:
    script_name = "yolo.exe" if sys.platform.startswith("win") else "yolo"
    local_script = Path(sys.executable).with_name("Scripts") / script_name
    if local_script.exists():
        return str(local_script)
    return "yolo"


def normalize_batch(batch: str) -> str:
    value = str(batch).strip().lower()
    if value == "auto":
        return "-1"
    return value


def quote_command(command: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def build_train_command(
    *,
    data: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: str,
    patience: int,
    project: Path,
    name: str,
    device: str | None,
    workers: int | None,
) -> list[str]:
    command = [
        yolo_executable(),
        "detect",
        "train",
        f"data={data.resolve().as_posix()}",
        f"model={model}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={normalize_batch(batch)}",
        f"patience={patience}",
        f"project={project.resolve().as_posix()}",
        f"name={name}",
        "exist_ok=True",
        "plots=True",
    ]
    if device:
        command.append(f"device={device}")
    if workers is not None:
        command.append(f"workers={workers}")
    return command


def write_config_snapshot(run_dir: Path, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_v1_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "train_v1_command.txt").write_text(config["command"] + "\n", encoding="utf-8")


def copy_dataset_snapshot(data_path: Path, run_dir: Path) -> None:
    if data_path.exists():
        shutil.copy2(data_path, run_dir / "data_snapshot.yaml")


def update_paths_config(paths_path: Path, *, best_pt: Path, run_dir: Path, data_path: Path) -> None:
    payload = load_yaml(paths_path)
    payload["validation_yolo_data"] = data_path.resolve().as_posix()
    payload["validation_run_dir"] = run_dir.resolve().as_posix()
    payload["validation_best_pt"] = best_pt.resolve().as_posix()
    paths_path.parent.mkdir(parents=True, exist_ok=True)
    paths_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def training_outputs(run_dir: Path) -> dict[str, str | bool]:
    best_pt = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    results_png = run_dir / "results.png"
    return {
        "run_dir": run_dir.resolve().as_posix(),
        "best_pt": best_pt.resolve().as_posix(),
        "best_pt_exists": best_pt.exists(),
        "results_csv": results_csv.resolve().as_posix(),
        "results_csv_exists": results_csv.exists(),
        "results_png": results_png.resolve().as_posix(),
        "results_png_exists": results_png.exists(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path.cwd().resolve()
    paths_path = (project_root / args.paths).resolve() if not args.paths.is_absolute() else args.paths.resolve()
    paths = load_yaml(paths_path)
    validation_output = resolve_config_path(paths.get("validation_output"), base=project_root) or Path("output/validation_run_v1")
    data_path = args.data or resolve_config_path(paths.get("validation_yolo_data"), base=project_root) or validation_output / "yolo_det" / "data.yaml"
    data_path = (project_root / data_path).resolve() if not data_path.is_absolute() else data_path.resolve()
    run_project = (project_root / args.project).resolve() if not args.project.is_absolute() else args.project.resolve()
    run_dir = run_project / args.name

    print(NOTICE)
    if not data_path.exists():
        print(f"YOLO data.yaml not found: {data_path}. Please run scripts/convert_labels.py first.", file=sys.stderr)
        return 1
    if args.epochs <= 0 or args.imgsz <= 0 or args.patience < 0:
        print("--epochs/--imgsz must be positive and --patience must be non-negative", file=sys.stderr)
        return 1

    command = build_train_command(
        data=data_path,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=run_project,
        name=args.name,
        device=args.device,
        workers=args.workers,
    )
    config = {
        "notice": NOTICE,
        "data": data_path.resolve().as_posix(),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": normalize_batch(args.batch),
        "patience": args.patience,
        "project": run_project.resolve().as_posix(),
        "name": args.name,
        "device": args.device,
        "workers": args.workers,
        "command": quote_command(command),
    }
    write_config_snapshot(run_dir, config)
    copy_dataset_snapshot(data_path, run_dir)
    print(config["command"])
    if args.dry_run:
        print(f"Dry run only. Config snapshot written to {run_dir / 'train_v1_config.yaml'}")
        return 0

    completed = subprocess.run(command)
    outputs = training_outputs(run_dir)
    (run_dir / "train_v1_outputs.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        print(f"Training failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Training finished but best.pt was not found: {best_pt}", file=sys.stderr)
        return 1
    update_paths_config(paths_path, best_pt=best_pt, run_dir=run_dir, data_path=data_path)
    print(f"best.pt: {best_pt.resolve()}")
    print(f"results: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
