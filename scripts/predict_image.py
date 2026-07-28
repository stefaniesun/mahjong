"""End-to-end visual check: detect tiles, classify each one, draw the result.

Two modes:
  * plain — draw every detection with its predicted class and confidence.
  * compare — when the image has a matching X-AnyLabeling .json, also diff against it
    and colour-code 漏检 / 误检 / 分类错误, so you can see exactly where the models
    (or the annotations) disagree.

Example:
    python scripts/predict_image.py --image data/test_set_v1/images/xxx.jpg \
        --det output/eval_real_v1/best.pt --cls output/cls_final_v2/best.pt --out preview.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

GREEN = (0, 200, 0)
RED = (0, 0, 230)
ORANGE = (0, 165, 255)
BLUE = (230, 130, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run detector + classifier on one image and visualise.")
    parser.add_argument("--image", type=Path, required=True, help="Input frame.")
    parser.add_argument("--det", type=Path, required=True, help="Detector weights.")
    parser.add_argument("--cls", type=Path, required=True, help="Classifier checkpoint.")
    parser.add_argument("--out", type=Path, default=Path("preview.jpg"), help="Output image path.")
    parser.add_argument("--det-conf", type=float, default=0.25, help="Detector confidence floor.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--margin", type=float, default=0.08, help="Crop margin, matching training.")
    parser.add_argument("--gt", type=Path, default=None, help="X-AnyLabeling json. Defaults to the image's sibling .json when present.")
    parser.add_argument("--no-compare", action="store_true", help="Skip the ground-truth diff even if a json exists.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU used to pair predictions with ground truth.")
    parser.add_argument("--device", default="cpu")
    return parser


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for shape in payload.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        out.append({"label": str(shape.get("label", "?")), "box": [min(xs), min(ys), max(xs), max(ys)]})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from ultralytics import YOLO

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from train_classifier import build_model

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"读不到图片: {args.image}")
        return 1
    height, width = image.shape[:2]

    detector = YOLO(str(args.det))
    result = detector.predict(source=str(args.image), conf=args.det_conf, iou=0.6, imgsz=args.imgsz, device=args.device, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    dets: list[dict[str, Any]] = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            dets.append({"box": [x1, y1, x2, y2], "det_conf": float(box.conf[0].item())})

    ckpt = torch.load(args.cls, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    imgsz_cls = int(ckpt.get("imgsz", 96))
    classifier = build_model(ckpt.get("arch", "mobilenet_v3_small"), len(classes))
    classifier.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    classifier.to(device).eval()
    tf = transforms.Compose(
        [
            transforms.Resize((imgsz_cls, imgsz_cls)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if dets:
        crops = []
        for det in dets:
            x1, y1, x2, y2 = det["box"]
            mx, my = (x2 - x1) * args.margin, (y2 - y1) * args.margin
            cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            cx2, cy2 = int(min(width, x2 + mx)), int(min(height, y2 + my))
            patch = image[cy1:cy2, cx1:cx2]
            if patch.size == 0:
                patch = np.zeros((8, 8, 3), np.uint8)
            crops.append(tf(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))))
        with torch.no_grad():
            prob = torch.softmax(classifier(torch.stack(crops).to(device)), dim=1)
            conf, pred = prob.max(1)
        for det, c, p in zip(dets, conf.cpu().tolist(), pred.cpu().tolist()):
            det["cls"] = classes[p]
            det["cls_conf"] = c

    gt_path = args.gt or args.image.with_suffix(".json")
    gts = load_gt(gt_path) if (gt_path.exists() and not args.no_compare) else []

    canvas = image.copy()
    stats = {"检出": len(dets), "标注": len(gts), "匹配": 0, "分类错": 0, "漏检": 0, "误检": 0}

    if gts:
        used = set()
        for gt in gts:
            best, bi = 0.0, None
            for i, det in enumerate(dets):
                if i in used:
                    continue
                v = iou(gt["box"], det["box"])
                if v >= args.iou and v > best:
                    best, bi = v, i
            if bi is None:
                stats["漏检"] += 1
                x1, y1, x2, y2 = [int(v) for v in gt["box"]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), ORANGE, 3)
                cv2.putText(canvas, f"MISS {gt['label']}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
                continue
            used.add(bi)
            stats["匹配"] += 1
            det = dets[bi]
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            same = det.get("cls") == gt["label"]
            if not same:
                stats["分类错"] += 1
            colour = GREEN if same else RED
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            text = det.get("cls", "?") if same else f"{det.get('cls','?')}!={gt['label']}"
            cv2.putText(canvas, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        for i, det in enumerate(dets):
            if i in used:
                continue
            stats["误检"] += 1
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BLUE, 2)
            cv2.putText(canvas, f"FP {det.get('cls','?')}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1, cv2.LINE_AA)
    else:
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(canvas, f"{det.get('cls','?')} {det.get('cls_conf',0):.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)

    print(f"图片: {args.image.name}  ({width}x{height})")
    if gts:
        print("对比模式（有标注）:")
        print(f"  标注 {stats['标注']} 张牌 / 模型检出 {stats['检出']}")
        print(f"  绿框 匹配且分类正确 : {stats['匹配'] - stats['分类错']}")
        print(f"  红框 检到但分类错   : {stats['分类错']}")
        print(f"  橙框 漏检           : {stats['漏检']}")
        print(f"  蓝框 多检(标注里没有): {stats['误检']}")
    else:
        print(f"纯预测模式: 检出 {stats['检出']} 张牌（无标注可对比）")
    print(f"输出: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
