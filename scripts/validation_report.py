"""Generate Validation Run v1 quality and pipeline report.

Example:
    python scripts/validation_report.py --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from jinja2 import Template

try:
    from scripts.convert_labels import (
        SIZE_BUCKET_LABELS,
        bbox_from_points,
        clip_bbox,
        image_paths_from_root,
        load_class_config,
        load_yaml,
        resolve_config_path,
        safe_stem,
        size_bucket,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from convert_labels import (
        SIZE_BUCKET_LABELS,
        bbox_from_points,
        clip_bbox,
        image_paths_from_root,
        load_class_config,
        load_yaml,
        resolve_config_path,
        safe_stem,
        size_bucket,
    )

DEFAULT_PATHS = Path("configs/paths.yaml")
DEFAULT_CLASSES = Path("configs/classes.yaml")
IOU_THRESHOLD = 0.5
PASS_MATCH_RATE = 0.7
SYSTEMIC_CLASS_RECALL = 0.5
MIN_CLASS_BOXES_FOR_SYSTEMIC = 5


@dataclass(frozen=True)
class Annotation:
    image_key: str
    image_path: Path
    label: str
    bbox: tuple[float, float, float, float]
    bucket: str
    blogger: str


@dataclass(frozen=True)
class Prediction:
    bbox: tuple[float, float, float, float]
    conf: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Validation Run v1 HTML quality report.")
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="Path config YAML.")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES, help="Class config YAML.")
    parser.add_argument("--labeled-root", type=Path, default=None, help="Original X-AnyLabeling labeled root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Validation conversion output root.")
    parser.add_argument("--run-dir", type=Path, default=None, help="YOLO training run directory.")
    parser.add_argument("--best-pt", type=Path, default=None, help="Trained best.pt path.")
    parser.add_argument("--check-output", type=Path, default=None, help="Directory for top-difference overlay images.")
    parser.add_argument("--html", type=Path, default=None, help="Output HTML report path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="IoU threshold for matching predictions to annotations.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of worst images to draw.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics prediction device, e.g. 0 or cpu.")
    return parser


def encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{data}"


def plot_to_data_uri(title: str, xs: list[Any], ys: list[float], ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(xs, ys, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def greedy_match(
    annotations: Sequence[Annotation], predictions: Sequence[Prediction], threshold: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for ann_idx, ann in enumerate(annotations):
        for pred_idx, pred in enumerate(predictions):
            score = iou(ann.bbox, pred.bbox)
            if score >= threshold:
                candidates.append((score, ann_idx, pred_idx))
    candidates.sort(reverse=True)
    used_ann: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, ann_idx, pred_idx in candidates:
        if ann_idx in used_ann or pred_idx in used_pred:
            continue
        used_ann.add(ann_idx)
        used_pred.add(pred_idx)
        matches.append((ann_idx, pred_idx, score))
    missed = [idx for idx in range(len(annotations)) if idx not in used_ann]
    false_positive = [idx for idx in range(len(predictions)) if idx not in used_pred]
    return matches, missed, false_positive


def blogger_from_name(image_name: str) -> str:
    parts = image_name.split("__", 1)
    return parts[0] if parts else "unknown"


def collect_annotations(labeled_root: Path, classes_path: Path) -> dict[str, list[Annotation]]:
    class_labels, discard_labels = load_class_config(classes_path)
    class_set = set(class_labels)
    records: dict[str, list[Annotation]] = defaultdict(list)
    for image_path in image_paths_from_root(labeled_root):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_height, image_width = image.shape[:2]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        width = int(payload.get("imageWidth") or image_width)
        height = int(payload.get("imageHeight") or image_height)
        key = safe_stem(image_path, labeled_root)
        for shape in payload.get("shapes", []):
            label = str(shape.get("label", ""))
            if label in discard_labels or label not in class_set:
                continue
            bbox = clip_bbox(bbox_from_points(shape.get("points", [])), width, height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            records[key].append(
                Annotation(
                    image_key=key,
                    image_path=image_path,
                    label=label,
                    bbox=bbox,
                    bucket=size_bucket(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                    blogger=blogger_from_name(image_path.name),
                )
            )
    return records


def read_results_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    parsed[key.strip()] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(parsed)
    return rows


def loss_is_decreasing(rows: Sequence[dict[str, float]]) -> bool:
    if len(rows) < 2:
        return False
    first = rows[0].get("train/box_loss")
    last = rows[-1].get("train/box_loss")
    return first is not None and last is not None and last < first


def default_predictor(best_pt: Path, images: Sequence[Path], conf: float, device: str | None) -> dict[str, list[Prediction]]:
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    kwargs: dict[str, Any] = {"conf": conf, "verbose": False}
    if device:
        kwargs["device"] = device
    results = model.predict([str(path) for path in images], **kwargs)
    output: dict[str, list[Prediction]] = {}
    for image_path, result in zip(images, results):
        preds: list[Prediction] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            preds = [Prediction(tuple(map(float, box)), float(score)) for box, score in zip(xyxy, confs)]
        output[image_path.stem] = preds
    return output


def evaluate_predictions(
    image_paths: Sequence[Path],
    annotations_by_key: dict[str, list[Annotation]],
    predictions_by_key: dict[str, list[Prediction]],
    iou_threshold: float,
) -> dict[str, Any]:
    per_image: list[dict[str, Any]] = []
    class_total: Counter[str] = Counter()
    class_matched: Counter[str] = Counter()
    bucket_total: Counter[str] = Counter()
    bucket_matched: Counter[str] = Counter()
    blogger_total: Counter[str] = Counter()
    blogger_matched: Counter[str] = Counter()
    total_ann = total_pred = total_match = 0

    for image_path in image_paths:
        key = image_path.stem
        annotations = annotations_by_key.get(key, [])
        predictions = predictions_by_key.get(key, [])
        matches, missed, false_positive = greedy_match(annotations, predictions, iou_threshold)
        total_ann += len(annotations)
        total_pred += len(predictions)
        total_match += len(matches)
        matched_ann_indices = {ann_idx for ann_idx, _, _ in matches}
        for idx, ann in enumerate(annotations):
            class_total[ann.label] += 1
            bucket_total[ann.bucket] += 1
            blogger_total[ann.blogger] += 1
            if idx in matched_ann_indices:
                class_matched[ann.label] += 1
                bucket_matched[ann.bucket] += 1
                blogger_matched[ann.blogger] += 1
        per_image.append(
            {
                "image_path": image_path,
                "key": key,
                "annotations": annotations,
                "predictions": predictions,
                "matches": matches,
                "missed": missed,
                "false_positive": false_positive,
                "score": len(missed) + len(false_positive),
                "match_rate": len(matches) / len(annotations) if annotations else 1.0,
            }
        )

    return {
        "total_annotations": total_ann,
        "total_predictions": total_pred,
        "total_matches": total_match,
        "match_rate": total_match / total_ann if total_ann else 0.0,
        "false_positives": total_pred - total_match,
        "missed": total_ann - total_match,
        "per_image": per_image,
        "class_recall": recall_table(class_total, class_matched),
        "bucket_recall": recall_table(bucket_total, bucket_matched),
        "blogger_recall": recall_table(blogger_total, blogger_matched),
    }


def recall_table(total: Counter[str], matched: Counter[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(total):
        count = total[name]
        hit = matched[name]
        rows.append({"name": name, "matched": hit, "total": count, "recall": hit / count if count else 0.0})
    return rows


def draw_overlay(item: dict[str, Any], output_path: Path) -> None:
    image = cv2.imread(str(item["image_path"]))
    if image is None:
        return
    for ann in item["annotations"]:
        x1, y1, x2, y2 = [int(round(value)) for value in ann.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 190, 40), 2)
        cv2.putText(image, ann.label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 190, 40), 1, cv2.LINE_AA)
    for pred in item["predictions"]:
        x1, y1, x2, y2 = [int(round(value)) for value in pred.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 30, 230), 2)
        cv2.putText(image, f"pred {pred.conf:.2f}", (x1, min(image.shape[0] - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 230), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def pipeline_checks(convert_report: dict[str, Any], run_dir: Path, best_pt: Path, results_csv: Path, predictions_ran: bool) -> list[dict[str, str]]:
    return [
        {"name": "转换成功", "status": "PASS" if bool(convert_report) and convert_report.get("summary", {}).get("kept_boxes", 0) > 0 else "FAIL"},
        {"name": "训练完成", "status": "PASS" if results_csv.exists() and len(read_results_csv(results_csv)) > 0 else "FAIL"},
        {"name": "权重存在", "status": "PASS" if best_pt.exists() else "FAIL"},
        {"name": "评测跑通", "status": "PASS" if predictions_ran else "FAIL"},
    ]


def build_html(context: dict[str, Any]) -> str:
    template = Template(
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Validation Run v1 验证报告</title>
<style>
body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;color:#222;background:#fafafa} h1,h2{color:#111} .card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:14px 0;box-shadow:0 1px 3px #ddd}.pass{color:#12843b;font-weight:700}.fail{color:#b00020;font-weight:700}.warn{color:#9a5a00;font-weight:700} table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}th{background:#eee}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}.diff img,.chart{max-width:100%;border:1px solid #ddd;border-radius:6px}.muted{color:#666}.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#eee}.conclusion{font-size:20px;font-weight:700}.small{font-size:13px}
</style></head><body>
<h1>Validation Run v1 标注质量与管线体检报告</h1>
<div class="card"><div class="conclusion {{ 'pass' if conclusion.status == '合格' else 'fail' }}">最终结论：{{ conclusion.status }}</div><ul>{% for item in conclusion.reasons %}<li>{{ item }}</li>{% endfor %}</ul></div>
<div class="card"><h2>1. 管线连通性</h2><table><tr><th>检查项</th><th>状态</th></tr>{% for c in checks %}<tr><td>{{ c.name }}</td><td class="{{ 'pass' if c.status == 'PASS' else 'fail' }}">{{ c.status }}</td></tr>{% endfor %}</table></div>
<div class="card"><h2>2. 训练健康度</h2><p>loss 是否下降：<span class="{{ 'pass' if training.loss_down else 'fail' }}">{{ '是' if training.loss_down else '否' }}</span>；最后一轮 mAP50={{ '%.3f'|format(training.last_map50) }}，recall={{ '%.3f'|format(training.last_recall) }}。</p><img class="chart" src="{{ training.loss_chart }}"><img class="chart" src="{{ training.map_chart }}"></div>
<div class="card"><h2>3. 标注自洽性</h2><p>训练图 IoU≥{{ iou_threshold }} 匹配率：<b>{{ '%.3f'|format(eval.match_rate) }}</b>（匹配 {{ eval.total_matches }}/{{ eval.total_annotations }}，漏检 {{ eval.missed }}，误检 {{ eval.false_positives }}）。</p></div>
<div class="card"><h2>4. 系统性错误探测</h2><h3>按原始牌类召回</h3><table><tr><th>类别</th><th>匹配/总数</th><th>召回</th><th>提示</th></tr>{% for r in eval.class_recall %}<tr><td>{{ r.name }}</td><td>{{ r.matched }}/{{ r.total }}</td><td>{{ '%.3f'|format(r.recall) }}</td><td class="{{ 'warn' if r.flag else '' }}">{{ r.flag }}</td></tr>{% endfor %}</table><h3>按尺寸分桶召回</h3><table><tr><th>尺寸桶</th><th>匹配/总数</th><th>召回</th></tr>{% for r in eval.bucket_recall %}<tr><td>{{ r.label }}</td><td>{{ r.matched }}/{{ r.total }}</td><td>{{ '%.3f'|format(r.recall) }}</td></tr>{% endfor %}</table><h3>按博主/来源召回</h3><table><tr><th>来源</th><th>匹配/总数</th><th>召回</th></tr>{% for r in eval.blogger_recall %}<tr><td>{{ r.name }}</td><td>{{ r.matched }}/{{ r.total }}</td><td>{{ '%.3f'|format(r.recall) }}</td></tr>{% endfor %}</table></div>
<div class="card"><h2>5. 差异最大的 {{ diffs|length }} 张图</h2><p class="muted">绿色=人工标注，红色=模型预测。</p><div class="grid">{% for d in diffs %}<div class="diff"><div class="small"><b>{{ d.name }}</b><br>漏检 {{ d.missed }}，误检 {{ d.false_positive }}，匹配率 {{ '%.3f'|format(d.match_rate) }}</div><img src="{{ d.image_data }}"></div>{% endfor %}</div></div>
<div class="card"><h2>6. 输入产物</h2><ul><li>convert_report: {{ paths.convert_report }}</li><li>best.pt: {{ paths.best_pt }}</li><li>results.csv: {{ paths.results_csv }}</li><li>check images: {{ paths.check_output }}</li></ul></div>
</body></html>"""
    )
    return template.render(**context)


def generate_report(
    *,
    paths_path: Path,
    classes_path: Path,
    labeled_root: Path | None = None,
    output_root: Path | None = None,
    run_dir: Path | None = None,
    best_pt: Path | None = None,
    check_output: Path | None = None,
    html_path: Path | None = None,
    conf: float = 0.25,
    iou_threshold: float = IOU_THRESHOLD,
    top_k: int = 20,
    device: str | None = None,
    predictor: Callable[[Path, Sequence[Path], float, str | None], dict[str, list[Prediction]]] = default_predictor,
) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    paths = load_yaml(paths_path)
    labeled_root = labeled_root or resolve_config_path(paths.get("validation_labeled"), base=project_root) or Path("data/labeled")
    output_root = output_root or resolve_config_path(paths.get("validation_output"), base=project_root) or Path("output/validation_run_v1")
    run_dir = run_dir or resolve_config_path(paths.get("validation_run_dir"), base=project_root) or Path("runs/val_run_v1/detector")
    best_pt = best_pt or resolve_config_path(paths.get("validation_best_pt"), base=project_root) or run_dir / "weights" / "best.pt"
    check_output = check_output or resolve_config_path(paths.get("validation_check_output"), base=project_root) or Path("output/check")
    html_path = html_path or resolve_config_path(paths.get("validation_report_html"), base=project_root) or Path("output/validation_report.html")
    labeled_root = (project_root / labeled_root).resolve() if not labeled_root.is_absolute() else labeled_root.resolve()
    output_root = (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    run_dir = (project_root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    best_pt = (project_root / best_pt).resolve() if not best_pt.is_absolute() else best_pt.resolve()
    check_output = (project_root / check_output).resolve() if not check_output.is_absolute() else check_output.resolve()
    html_path = (project_root / html_path).resolve() if not html_path.is_absolute() else html_path.resolve()
    classes_path = (project_root / classes_path).resolve() if not classes_path.is_absolute() else classes_path.resolve()

    convert_report_path = output_root / "convert_report.json"
    convert_report = json.loads(convert_report_path.read_text(encoding="utf-8")) if convert_report_path.exists() else {}
    results_csv = run_dir / "results.csv"
    rows = read_results_csv(results_csv)
    yolo_train_dir = output_root / "yolo_det" / "images" / "train"
    train_images = sorted([path for path in yolo_train_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    annotations = collect_annotations(labeled_root, classes_path)
    predictions = predictor(best_pt, train_images, conf, device) if best_pt.exists() and train_images else {}
    evaluation = evaluate_predictions(train_images, annotations, predictions, iou_threshold)

    check_output.mkdir(parents=True, exist_ok=True)
    for old in check_output.glob("*.jpg"):
        old.unlink()
    worst = sorted(evaluation["per_image"], key=lambda item: (item["score"], 1.0 - item["match_rate"]), reverse=True)[:top_k]
    diffs: list[dict[str, Any]] = []
    for index, item in enumerate(worst, start=1):
        out_path = check_output / f"diff_{index:02d}__{item['key']}.jpg"
        draw_overlay(item, out_path)
        if out_path.exists():
            diffs.append(
                {
                    "name": item["image_path"].name,
                    "missed": len(item["missed"]),
                    "false_positive": len(item["false_positive"]),
                    "match_rate": item["match_rate"],
                    "image_data": encode_image(out_path),
                }
            )

    for row in evaluation["class_recall"]:
        row["flag"] = "该类可能需复查" if row["total"] >= MIN_CLASS_BOXES_FOR_SYSTEMIC and row["recall"] < SYSTEMIC_CLASS_RECALL else ""
    for row in evaluation["bucket_recall"]:
        row["label"] = SIZE_BUCKET_LABELS.get(row["name"], row["name"])

    checks = pipeline_checks(convert_report, run_dir, best_pt, results_csv, bool(predictions) or not train_images)
    all_pass = all(item["status"] == "PASS" for item in checks)
    loss_down = loss_is_decreasing(rows)
    systemic = [row["name"] for row in evaluation["class_recall"] if row.get("flag")]
    invalid_labels = convert_report.get("summary", {}).get("invalid_label_boxes", 0) if convert_report else 0
    reasons: list[str] = []
    if all_pass:
        reasons.append("管线转换、训练、权重和回测均已跑通。")
    else:
        reasons.append("存在管线 FAIL 项，请优先修复。")
    reasons.append("训练 loss 呈下降趋势，模型在学习。" if loss_down else "训练 loss 未确认下降，请检查训练曲线。")
    reasons.append(f"训练图 IoU≥{iou_threshold} 匹配率为 {evaluation['match_rate']:.3f}。")
    if systemic:
        reasons.append("以下原始牌类召回偏低，建议复查：" + ", ".join(systemic))
    else:
        reasons.append("未发现某个原始牌类系统性学不会。")
    if invalid_labels:
        reasons.append(f"转换报告仍有 {invalid_labels} 个异常 label，建议回 X-AnyLabeling 修正。")
    passed = all_pass and loss_down and evaluation["match_rate"] >= PASS_MATCH_RATE and not systemic and invalid_labels == 0
    conclusion = {"status": "合格" if passed else "需复查", "reasons": reasons}

    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(rows)]
    box_loss = [row.get("train/box_loss", 0.0) for row in rows]
    map50 = [row.get("metrics/mAP50(B)", 0.0) for row in rows]
    context = {
        "conclusion": conclusion,
        "checks": checks,
        "training": {
            "loss_down": loss_down,
            "last_map50": map50[-1] if map50 else 0.0,
            "last_recall": rows[-1].get("metrics/recall(B)", 0.0) if rows else 0.0,
            "loss_chart": plot_to_data_uri("train/box_loss", epochs, box_loss, "box loss") if rows else "",
            "map_chart": plot_to_data_uri("val mAP50", epochs, map50, "mAP50") if rows else "",
        },
        "eval": evaluation,
        "diffs": diffs,
        "iou_threshold": iou_threshold,
        "paths": {
            "convert_report": convert_report_path.as_posix(),
            "best_pt": best_pt.as_posix(),
            "results_csv": results_csv.as_posix(),
            "check_output": check_output.as_posix(),
        },
    }
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(context), encoding="utf-8")
    summary = {
        "html": html_path.as_posix(),
        "check_output": check_output.as_posix(),
        "conclusion": conclusion,
        "checks": checks,
        "match_rate": evaluation["match_rate"],
        "loss_down": loss_down,
        "systemic_low_recall_classes": systemic,
        "diff_images": len(diffs),
    }
    (html_path.parent / "validation_report_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def update_paths_config(paths_path: Path, *, html_path: Path, check_output: Path) -> None:
    payload = load_yaml(paths_path)
    payload["validation_report_html"] = html_path.resolve().as_posix()
    payload["validation_check_output"] = check_output.resolve().as_posix()
    paths_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path.cwd().resolve()
    paths_path = (project_root / args.paths).resolve() if not args.paths.is_absolute() else args.paths.resolve()
    try:
        summary = generate_report(
            paths_path=paths_path,
            classes_path=args.classes,
            labeled_root=args.labeled_root,
            output_root=args.output_root,
            run_dir=args.run_dir,
            best_pt=args.best_pt,
            check_output=args.check_output,
            html_path=args.html,
            conf=args.conf,
            iou_threshold=args.iou,
            top_k=args.top_k,
            device=args.device,
        )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    update_paths_config(paths_path, html_path=Path(summary["html"]), check_output=Path(summary["check_output"]))
    print(f"validation report: {summary['html']}")
    print(f"check images: {summary['check_output']}")
    print(f"conclusion: {summary['conclusion']['status']}")
    print(f"match_rate: {summary['match_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
