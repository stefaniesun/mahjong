"""Evaluate detection results against COCO ground truth and write JSON metrics.

Example:
    python eval/eval_detection.py --ground-truth data/test_set_v1/annotations/instances_default.json \
        --predictions data/test_set_v1/predictions.json --output-dir data/test_set_v1/eval_output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2

SIZE_BUCKETS = ("lt20", "20to40", "gt40")
DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detection predictions against COCO ground truth.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to the COCO ground truth JSON.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to the COCO detections JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output"),
        help="Directory for detection metrics and missed-image visualizations.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated score thresholds for P-R operating points.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to match predictions to ground truth.",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=20,
        help="Maximum number of missed-image previews to render.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Optional image directory. If omitted, missed-image previews use blank canvases sized from COCO metadata.",
    )
    return parser


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = [float(value) for value in box_a]
    bx1, by1, bw, bh = [float(value) for value in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def bucket_short_side(width: float, height: float) -> str:
    short_side = min(float(width), float(height))
    if short_side < 20:
        return "lt20"
    if short_side <= 40:
        return "20to40"
    return "gt40"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_thresholds(raw_value: str) -> list[float]:
    thresholds = []
    for chunk in raw_value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        thresholds.append(float(stripped))
    return thresholds or list(DEFAULT_THRESHOLDS)


def empty_metric() -> dict[str, float | int]:
    return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0}


def finalize_metric(metric: dict[str, float | int]) -> dict[str, float | int]:
    tp = int(metric["tp"])
    fp = int(metric["fp"])
    fn = int(metric["fn"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def category_ap(category_id: int, ground_truth: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], iou_threshold: float) -> float:
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth:
        if int(annotation["category_id"]) == category_id:
            gt_by_image[int(annotation["image_id"])] .append(annotation)

    ranked_predictions = sorted(
        (prediction for prediction in predictions if int(prediction["category_id"]) == category_id),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    total_gt = sum(len(items) for items in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    matched_gt: dict[int, set[int]] = defaultdict(set)
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for prediction in ranked_predictions:
        image_id = int(prediction["image_id"])
        candidates = gt_by_image.get(image_id, [])
        best_match = None
        best_iou = 0.0
        for candidate in candidates:
            candidate_id = int(candidate["id"])
            if candidate_id in matched_gt[image_id]:
                continue
            iou = compute_iou(candidate["bbox"], prediction["bbox"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = candidate_id
        if best_match is not None:
            matched_gt[image_id].add(best_match)
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / total_gt)

    if not precisions:
        return 0.0
    recall_levels = [index / 10 for index in range(11)]
    ap_values = []
    for recall_level in recall_levels:
        eligible = [precision for precision, recall in zip(precisions, recalls) if recall >= recall_level]
        ap_values.append(max(eligible) if eligible else 0.0)
    return round(sum(ap_values) / len(ap_values), 4)


def match_predictions(
    ground_truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    images_by_id: dict[int, dict[str, Any]],
    *,
    threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth:
        gt_by_image[int(annotation["image_id"])] .append(annotation)

    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= threshold:
            predictions_by_image[int(prediction["image_id"])] .append(prediction)
    for image_predictions in predictions_by_image.values():
        image_predictions.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    size_metrics = {bucket: empty_metric() for bucket in SIZE_BUCKETS}
    source_metrics: dict[str, dict[str, float | int]] = defaultdict(empty_metric)
    category_metrics: dict[str, dict[str, float | int]] = defaultdict(empty_metric)
    matched_predictions = 0
    false_positives = 0
    missed_images: list[dict[str, Any]] = []

    for image_id, image in images_by_id.items():
        image_gt = gt_by_image.get(image_id, [])
        image_predictions = predictions_by_image.get(image_id, [])
        used_prediction_indexes: set[int] = set()
        missed_annotations: list[dict[str, Any]] = []
        matched_annotations = 0

        for annotation in image_gt:
            bucket = bucket_short_side(annotation["bbox"][2], annotation["bbox"][3])
            source = str(image.get("source", "unknown"))
            category_key = str(annotation["category_id"])
            best_prediction_index = None
            best_iou = 0.0
            for prediction_index, prediction in enumerate(image_predictions):
                if prediction_index in used_prediction_indexes:
                    continue
                if int(prediction["category_id"]) != int(annotation["category_id"]):
                    continue
                iou = compute_iou(annotation["bbox"], prediction["bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_prediction_index = prediction_index
            if best_prediction_index is not None:
                used_prediction_indexes.add(best_prediction_index)
                matched_predictions += 1
                matched_annotations += 1
                size_metrics[bucket]["tp"] += 1
                source_metrics[source]["tp"] += 1
                category_metrics[category_key]["tp"] += 1
            else:
                size_metrics[bucket]["fn"] += 1
                source_metrics[source]["fn"] += 1
                category_metrics[category_key]["fn"] += 1
                missed_annotations.append(annotation)

        for prediction_index, prediction in enumerate(image_predictions):
            if prediction_index in used_prediction_indexes:
                continue
            false_positives += 1
            bucket = bucket_short_side(prediction["bbox"][2], prediction["bbox"][3])
            source = str(image.get("source", "unknown"))
            category_key = str(prediction["category_id"])
            size_metrics[bucket]["fp"] += 1
            source_metrics[source]["fp"] += 1
            category_metrics[category_key]["fp"] += 1

        if missed_annotations:
            missed_images.append(
                {
                    "image_id": image_id,
                    "file_name": str(image["file_name"]),
                    "source": str(image.get("source", "unknown")),
                    "missed_count": len(missed_annotations),
                    "matched_count": matched_annotations,
                    "ground_truth_count": len(image_gt),
                }
            )

    total_gt = len(ground_truth)
    overall_precision = matched_predictions / (matched_predictions + false_positives) if (matched_predictions + false_positives) else 0.0
    overall_recall = matched_predictions / total_gt if total_gt else 0.0

    return {
        "overall": {
            "tp": matched_predictions,
            "fp": false_positives,
            "fn": total_gt - matched_predictions,
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
        },
        "matched_predictions": matched_predictions,
        "size_bucket_metrics": {bucket: finalize_metric(metric) for bucket, metric in size_metrics.items()},
        "source_metrics": {source: finalize_metric(metric) for source, metric in source_metrics.items()},
        "category_metrics": {category: finalize_metric(metric) for category, metric in category_metrics.items()},
        "missed_images": sorted(missed_images, key=lambda item: item["missed_count"], reverse=True),
    }


def compute_detection_report(
    ground_truth_payload: dict[str, Any],
    predictions_payload: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    images = ground_truth_payload.get("images", [])
    categories = ground_truth_payload.get("categories", [])
    annotations = ground_truth_payload.get("annotations", [])
    images_by_id = {int(image["id"]): image for image in images}

    base_match = match_predictions(
        annotations,
        predictions_payload,
        images_by_id,
        threshold=min(thresholds) if thresholds else 0.0,
        iou_threshold=iou_threshold,
    )

    coco_per_category: dict[str, dict[str, float]] = {}
    ap50_values: list[float] = []
    ap5095_values: list[float] = []
    for category in categories:
        category_id = int(category["id"])
        category_name = str(category["name"])
        ap50 = category_ap(category_id, annotations, predictions_payload, 0.5)
        ap_steps = [category_ap(category_id, annotations, predictions_payload, threshold) for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]
        ap5095 = round(sum(ap_steps) / len(ap_steps), 4)
        coco_per_category[category_name] = {"mAP@0.5": ap50, "mAP@0.5:0.95": ap5095}
        ap50_values.append(ap50)
        ap5095_values.append(ap5095)

    pr_points = []
    for threshold in thresholds:
        matched = match_predictions(
            annotations,
            predictions_payload,
            images_by_id,
            threshold=threshold,
            iou_threshold=iou_threshold,
        )
        pr_points.append(
            {
                "threshold": threshold,
                "precision": matched["overall"]["precision"],
                "recall": matched["overall"]["recall"],
            }
        )

    return {
        "summary": {
            "total_images": len(images),
            "total_ground_truth": len(annotations),
            "total_predictions": len(predictions_payload),
            "matched_predictions": base_match["matched_predictions"],
        },
        "overall": base_match["overall"],
        "coco_metrics": {
            "overall": {
                "mAP@0.5": round(sum(ap50_values) / len(ap50_values), 4) if ap50_values else 0.0,
                "mAP@0.5:0.95": round(sum(ap5095_values) / len(ap5095_values), 4) if ap5095_values else 0.0,
            },
            "per_category": coco_per_category,
        },
        "size_bucket_metrics": base_match["size_bucket_metrics"],
        "sources": base_match["source_metrics"],
        "categories": base_match["category_metrics"],
        "pr_points": pr_points,
        "missed_images": base_match["missed_images"],
    }


def draw_missed_images(
    report: dict[str, Any],
    ground_truth_payload: dict[str, Any],
    predictions_payload: Sequence[dict[str, Any]],
    output_dir: Path,
    images_root: Path | None,
    max_missed: int,
) -> None:
    images_by_id = {int(image["id"]): image for image in ground_truth_payload.get("images", [])}
    categories = {int(category["id"]): str(category["name"]) for category in ground_truth_payload.get("categories", [])}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth_payload.get("annotations", []):
        gt_by_image[int(annotation["image_id"])] .append(annotation)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions_payload:
        predictions_by_image[int(prediction["image_id"])] .append(prediction)

    missed_dir = output_dir / "missed"
    missed_dir.mkdir(parents=True, exist_ok=True)

    for item in report.get("missed_images", [])[:max_missed]:
        image_id = int(item["image_id"])
        image = images_by_id[image_id]
        canvas = None
        if images_root is not None:
            image_path = images_root / str(image["file_name"])
            canvas = cv2.imread(str(image_path))
        if canvas is None:
            canvas = 255 * __import__("numpy").ones((int(image["height"]), int(image["width"]), 3), dtype="uint8")

        for annotation in gt_by_image.get(image_id, []):
            x, y, w, h = [int(round(value)) for value in annotation["bbox"]]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), 2)
            label = categories.get(int(annotation["category_id"]), str(annotation["category_id"]))
            cv2.putText(canvas, f"GT:{label}", (max(0, x), max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 0), 2, cv2.LINE_AA)

        for prediction in predictions_by_image.get(image_id, []):
            x, y, w, h = [int(round(value)) for value in prediction["bbox"]]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 220), 2)
            label = categories.get(int(prediction["category_id"]), str(prediction["category_id"]))
            score = float(prediction.get("score", 0.0))
            cv2.putText(canvas, f"PR:{label}@{score:.2f}", (max(0, x), min(canvas.shape[0] - 6, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2, cv2.LINE_AA)

        output_path = missed_dir / Path(str(image["file_name"])).name
        cv2.imwrite(str(output_path), canvas)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_missed < 0:
        parser.error("--max-missed must be >= 0")
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in (0, 1]")

    ground_truth_path = args.ground_truth.resolve()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    images_root = args.images_root.resolve() if args.images_root else None

    if not ground_truth_path.exists():
        print(f"Ground truth file not found: {ground_truth_path}", file=sys.stderr)
        return 1
    if not predictions_path.exists():
        print(f"Predictions file not found: {predictions_path}", file=sys.stderr)
        return 1
    if images_root is not None and not images_root.exists():
        print(f"Images root not found: {images_root}", file=sys.stderr)
        return 1

    thresholds = parse_thresholds(args.thresholds)
    ground_truth_payload = load_json(ground_truth_path)
    predictions_payload = load_json(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = compute_detection_report(
        ground_truth_payload,
        predictions_payload,
        thresholds=thresholds,
        iou_threshold=args.iou_threshold,
    )
    report_path = output_dir / "detection_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_missed_images(report, ground_truth_payload, predictions_payload, output_dir, images_root, args.max_missed)

    print(f"Detection evaluation complete. Report: {report_path}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
