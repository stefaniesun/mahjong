"""Run a YOLO detector over a test set and dump predictions in COCO detection format.

Closes the detection-eval loop: feed the output into eval/eval_detection.py together
with a single-class ground-truth COCO to get size-bucketed recall (the small-tile metric).

Example:
    python scripts/predict_to_coco.py \
        --model runs/val_run_v1/detector/weights/best.pt \
        --gt data/test_set_v1/annotations/instances_det_singleclass.json \
        --images-root data/test_set_v1/images \
        --output data/test_set_v1/predictions.json \
        --conf 0.001 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YOLO detector and export COCO detections.")
    parser.add_argument("--model", type=Path, required=True, help="Ultralytics .pt / .onnx detector weights.")
    parser.add_argument("--gt", type=Path, required=True, help="COCO GT json; used to map file_name -> image_id.")
    parser.add_argument("--images-root", type=Path, required=True, help="Directory holding the test images.")
    parser.add_argument("--output", type=Path, required=True, help="Output COCO detections json path.")
    parser.add_argument("--conf", type=float, default=0.001, help="Min confidence; keep low so P-R sweeps are meaningful.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold.")
    parser.add_argument("--category-id", type=int, default=1, help="Category id to stamp on every detection (match GT).")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gt_path = args.gt.resolve()
    images_root = args.images_root.resolve()
    if not gt_path.exists():
        print(f"GT file not found: {gt_path}", file=sys.stderr)
        return 1
    if not images_root.exists():
        print(f"Images root not found: {images_root}", file=sys.stderr)
        return 1

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    # Map both the bare filename and the basename so we tolerate path variations.
    name_to_id: dict[str, int] = {}
    for image in gt.get("images", []):
        file_name = str(image["file_name"])
        name_to_id[file_name] = int(image["id"])
        name_to_id[Path(file_name).name] = int(image["id"])

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install ultralytics", file=sys.stderr)
        return 1

    model = YOLO(str(args.model))
    detections: list[dict[str, Any]] = []
    missing: list[str] = []

    for image in gt.get("images", []):
        file_name = str(image["file_name"])
        image_path = images_root / Path(file_name).name
        if not image_path.exists():
            missing.append(file_name)
            continue
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        image_id = name_to_id[file_name]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                score = float(box.conf[0].item())
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": args.category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": round(score, 6),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(detections)} detections over {len(gt.get('images', []))} images to {args.output}")
    if missing:
        print(f"WARNING: {len(missing)} GT images were not found under {images_root} (first: {missing[0]})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
