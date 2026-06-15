#!/usr/bin/env python3
"""Train, validate, and export the temporary YOLO prelabeler.

Examples:
    py scripts/train_prelabeler.py --mode train
    py scripts/train_prelabeler.py --mode predict --source data/frames_selected
    py scripts/train_prelabeler.py --mode export
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

DEFAULT_DATA = Path("E:/360MoveData/Users/Administrator/Desktop/Mahjong.v1i.yolov11/data.yaml")
DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_NAME = "prelabel_v1"
DEFAULT_EPOCHS = 80
DEFAULT_IMGSZ = 960
DEFAULT_BATCH = 16
DEFAULT_CONF = 0.25
DEFAULT_PATHS = Path("configs/paths.yaml")
DEFAULT_BEST_PT = Path("runs/detect/prelabel_v1/weights/best.pt")
DEFAULT_BEST_ONNX = Path("runs/detect/prelabel_v1/weights/best.onnx")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export the temporary Mahjong YOLO prelabeler.")
    parser.add_argument("--mode", choices=["train", "predict", "export"], required=True, help="Operation to run.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO dataset data.yaml path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model for train, or .pt path for predict/export.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Ultralytics run name.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Train/export image size. Lower this if GPU memory is insufficient.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Training batch size. Lower this first if GPU memory is insufficient.")
    parser.add_argument("--source", type=Path, default=Path("data/frames_selected"), help="Image/video source for predict mode.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Prediction confidence threshold.")
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="paths.yaml to update after export.")
    parser.add_argument("--dry-run", action="store_true", help="Print the yolo command without running it.")
    return parser


def best_pt_from_name(name: str) -> Path:
    return Path("runs") / "detect" / name / "weights" / "best.pt"


def best_onnx_from_name(name: str) -> Path:
    return Path("runs") / "detect" / name / "weights" / "best.onnx"


def resolve_model_path(model: str, name: str) -> str:
    model_path = Path(model)
    if model_path.suffix.lower() in {".pt", ".onnx"} or model_path.exists():
        return str(model_path)
    trained = best_pt_from_name(name)
    if trained.exists():
        return str(trained)
    return model


def yolo_executable() -> str:
    script_name = "yolo.exe" if sys.platform.startswith("win") else "yolo"
    local_script = Path(sys.executable).with_name("Scripts") / script_name
    if local_script.exists():
        return str(local_script)
    return "yolo"


def run_command(command: Sequence[str], *, dry_run: bool) -> int:
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if dry_run:
        return 0
    completed = subprocess.run(list(command))
    return completed.returncode



def update_paths(paths_path: Path, *, best_pt: Path, best_onnx: Path, data_path: Path) -> None:
    payload = {}
    if paths_path.exists():
        loaded = yaml.safe_load(paths_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload.update(loaded)
    payload["prelabeler_pt"] = best_pt.as_posix()
    payload["prelabeler_onnx"] = best_onnx.as_posix()
    payload["prelabel_source_data"] = data_path.as_posix()
    paths_path.parent.mkdir(parents=True, exist_ok=True)
    paths_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "train":
        if not args.data.exists():
            print(f"Dataset data.yaml not found: {args.data}", file=sys.stderr)
            return 1
        command = [
            yolo_executable(),
            "detect",
            "train",
            f"data={args.data.as_posix()}",
            f"model={args.model}",
            f"epochs={args.epochs}",
            f"imgsz={args.imgsz}",
            f"batch={args.batch}",
            f"name={args.name}",
        ]
        return run_command(command, dry_run=args.dry_run)

    if args.mode == "predict":
        model = resolve_model_path(args.model, args.name)
        if not Path(model).exists() and model.endswith(('.pt', '.onnx')):
            print(f"Model not found: {model}", file=sys.stderr)
            return 1
        if not args.source.exists():
            print(f"Predict source not found: {args.source}", file=sys.stderr)
            return 1
        command = [yolo_executable(), "detect", "predict", f"model={model}", f"source={args.source.as_posix()}", f"conf={args.conf}"]
        return run_command(command, dry_run=args.dry_run)

    model = resolve_model_path(args.model, args.name)
    model_path = Path(model)
    if not model_path.exists():
        print(f"Model not found for export: {model_path}", file=sys.stderr)
        return 1
    command = [yolo_executable(), "export", f"model={model_path.as_posix()}", "format=onnx", f"imgsz={args.imgsz}"]
    exit_code = run_command(command, dry_run=args.dry_run)
    if exit_code == 0 and not args.dry_run:
        update_paths(args.paths, best_pt=model_path, best_onnx=model_path.with_suffix(".onnx"), data_path=args.data)
        print(f"Updated paths config: {args.paths}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
