"""Audit the 20-image zone label set for errors that would corrupt evaluation.

A validation set is only as good as its labels. Before anyone tunes against these 899
boxes it is worth knowing which of them are wrong, because a method that "fails" on a
mislabelled box is being penalised for being right.

Four independent checks, none of which needs the zone algorithm (so none of them can be
gamed by agreeing with it):

1. **Dead labels.** Zones the algorithm can never emit are automatic errors.
2. **Duplicate boxes.** Two boxes on one tile double-counts it and inflates local density,
   which the algorithm reads as a feature.
3. **Geometry.** Boxes outside the frame, or far off their zone's size/position profile.
4. **Neighbour disagreement.** Mahjong tiles of one owner sit together. A box whose
   nearest neighbours all carry a different zone is either a boundary case or a slip.

Missing tiles are checked separately by --detect, which runs the detector and reports
confident detections with no ground-truth box under them.

    python scripts/audit_zone_labels.py --detect --html ../output/zone_annotation/audit.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "output" / "zone_annotation" / "zone_labels_with_class.json"
IMAGES = ROOT / "output" / "zone_annotation" / "images"

VALID_ZONES = {"my_hand", "river", "seat_left", "seat_across", "seat_right"}


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xywh boxes."""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area[:, None] + area[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def containment(boxes: np.ndarray) -> np.ndarray:
    """How much of box i sits inside box j — catches a small box drawn inside a big one,
    which IoU misses when the sizes differ a lot (a merged two-tile box, for instance)."""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1e-6)
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / area[:, None]


def audit(data: list[dict]) -> list[dict]:
    findings: list[dict] = []

    # Zone profiles over the whole set, used for the outlier check below. Built from the
    # labels themselves, so this finds boxes that disagree with their own zone's norm.
    prof: dict[str, list[tuple[float, float, float]]] = {}
    for item in data:
        arr = np.asarray(item["boxes"], dtype=np.float32).reshape(-1, 4)
        short = np.minimum(arr[:, 2], arr[:, 3])
        med = float(np.median(short))
        nx = (arr[:, 0] + arr[:, 2] / 2) / item["w"]
        ny = (arr[:, 1] + arr[:, 3] / 2) / item["h"]
        for i, z in enumerate(item["zones"]):
            prof.setdefault(z, []).append((float(nx[i]), float(ny[i]), float(short[i] / max(med, 1e-6))))
    stats = {}
    for z, rows in prof.items():
        a = np.asarray(rows, dtype=np.float32)
        stats[z] = (a.mean(axis=0), a.std(axis=0) + 1e-6, np.percentile(a, [1, 99], axis=0))

    for item in data:
        img = item["image"]
        arr = np.asarray(item["boxes"], dtype=np.float32).reshape(-1, 4)
        n = len(arr)
        zones = item["zones"]
        cls = item.get("cls", [""] * n)
        cx = arr[:, 0] + arr[:, 2] / 2
        cy = arr[:, 1] + arr[:, 3] / 2
        nx, ny = cx / item["w"], cy / item["h"]
        short = np.minimum(arr[:, 2], arr[:, 3])
        med = float(np.median(short))

        # 1. dead labels
        for i, z in enumerate(zones):
            if z not in VALID_ZONES:
                findings.append(dict(image=img, box=i, kind="dead_label", severity="certain",
                                     detail=f"标签 `{z}` 不在五区体系内,算法永不输出 → 必错",
                                     zone=z, cls=cls[i], bbox=[round(float(v), 1) for v in arr[i]]))

        # 2. duplicate / nested boxes
        if n > 1:
            iou = iou_matrix(arr)
            cont = containment(arr)
            np.fill_diagonal(iou, 0)
            np.fill_diagonal(cont, 0)
            for i in range(n):
                for j in range(i + 1, n):
                    if iou[i, j] >= 0.55:
                        findings.append(dict(image=img, box=i, kind="duplicate", severity="certain",
                                             detail=f"与 #{j} 的 IoU={iou[i,j]:.2f},同一张牌被标了两次({cls[i]} / {cls[j]})",
                                             zone=zones[i], cls=cls[i], bbox=[round(float(v), 1) for v in arr[i]], other=j))
                    elif max(cont[i, j], cont[j, i]) >= 0.85 and iou[i, j] < 0.55:
                        small, big = (i, j) if cont[i, j] > cont[j, i] else (j, i)
                        findings.append(dict(image=img, box=small, kind="nested", severity="likely",
                                             detail=f"#{small} 有 {max(cont[i,j],cont[j,i]):.0%} 落在 #{big} 内,"
                                                    f"多半是一个框套了两张牌({cls[big]} 框内含 {cls[small]})",
                                             zone=zones[small], cls=cls[small],
                                             bbox=[round(float(v), 1) for v in arr[small]], other=big))

        # 3. geometry
        for i in range(n):
            x, y, w, h = arr[i]
            if x < -2 or y < -2 or x + w > item["w"] + 2 or y + h > item["h"] + 2:
                findings.append(dict(image=img, box=i, kind="out_of_frame", severity="certain",
                                     detail=f"框越界: x∈[{x:.0f},{x+w:.0f}] y∈[{y:.0f},{y+h:.0f}],画面 {item['w']}x{item['h']}",
                                     zone=zones[i], cls=cls[i], bbox=[round(float(v), 1) for v in arr[i]]))
            if w <= 3 or h <= 3:
                findings.append(dict(image=img, box=i, kind="degenerate", severity="certain",
                                     detail=f"框尺寸退化 {w:.1f}x{h:.1f}",
                                     zone=zones[i], cls=cls[i], bbox=[round(float(v), 1) for v in arr[i]]))
            z = zones[i]
            if z in stats:
                mean, std, pct = stats[z]
                vec = np.array([nx[i], ny[i], short[i] / max(med, 1e-6)], dtype=np.float32)
                dev = np.abs(vec - mean) / std
                names = ["横向位置", "纵向位置", "相对大小"]
                worst = int(np.argmax(dev))
                if dev[worst] >= 3.0:
                    findings.append(dict(image=img, box=i, kind="zone_outlier", severity="suspect",
                                         detail=f"标为 {z},但{names[worst]} {vec[worst]:.2f} 偏离该区均值 "
                                                f"{mean[worst]:.2f} 达 {dev[worst]:.1f}σ",
                                         zone=z, cls=cls[i], bbox=[round(float(v), 1) for v in arr[i]]))

        # 4. neighbour disagreement — a tile whose three closest neighbours all belong to
        #    someone else. Tiles of one owner are placed together, so this is unusual.
        if n >= 5:
            d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
            np.fill_diagonal(d, np.inf)
            for i in range(n):
                k = np.argsort(d[i])[:3]
                # only meaningful when the neighbours are actually close by
                if d[i, k[-1]] > 3.0 * max(arr[i, 2], 1.0):
                    continue
                neigh = [zones[j] for j in k]
                if all(z != zones[i] for z in neigh) and len(set(neigh)) == 1:
                    findings.append(dict(image=img, box=i, kind="isolated_label", severity="suspect",
                                         detail=f"标为 {zones[i]},但紧邻的 3 张牌全部是 {neigh[0]}",
                                         zone=zones[i], cls=cls[i], bbox=[round(float(v), 1) for v in arr[i]],
                                         neighbour_zone=neigh[0], neighbours=[int(j) for j in k]))
    return findings


def detect_missing(data: list[dict], weights: Path, conf: float) -> list[dict]:
    """Run the detector and report confident boxes with no ground-truth box under them."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    findings: list[dict] = []
    for item in data:
        path = IMAGES / item["image"]
        if not path.exists():
            continue
        res = model.predict(str(path), conf=conf, imgsz=960, verbose=False)[0]
        pred = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        gt = np.asarray(item["boxes"], dtype=np.float32).reshape(-1, 4)
        gt_xyxy = np.concatenate([gt[:, :2], gt[:, :2] + gt[:, 2:]], axis=1) if len(gt) else np.zeros((0, 4))
        for p, s in zip(pred, scores):
            if len(gt_xyxy):
                ix1 = np.maximum(p[0], gt_xyxy[:, 0]); iy1 = np.maximum(p[1], gt_xyxy[:, 1])
                ix2 = np.minimum(p[2], gt_xyxy[:, 2]); iy2 = np.minimum(p[3], gt_xyxy[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                ap = (p[2] - p[0]) * (p[3] - p[1])
                ag = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
                best = float((inter / np.maximum(ap + ag - inter, 1e-6)).max())
            else:
                best = 0.0
            if best < 0.3:
                findings.append(dict(image=item["image"], box=-1, kind="missing_box",
                                     severity="likely" if s >= 0.6 else "suspect",
                                     detail=f"检测器在此处以 {s:.2f} 置信度发现一张牌,但没有标注框(最佳 IoU {best:.2f})",
                                     zone="-", cls="?",
                                     bbox=[round(float(p[0]), 1), round(float(p[1]), 1),
                                           round(float(p[2] - p[0]), 1), round(float(p[3] - p[1]), 1)]))
    return findings


def write_html(findings: list[dict], data: list[dict], out: Path) -> None:
    by_image: dict[str, list[dict]] = {}
    for f in findings:
        by_image.setdefault(f["image"], []).append(f)
    lookup = {d["image"]: d for d in data}

    order = {"certain": 0, "likely": 1, "suspect": 2}
    colour = {"certain": "#e5484d", "likely": "#f76b15", "suspect": "#f5d90a"}

    parts = ["""<!doctype html><meta charset="utf-8"><title>区域标注体检</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#111;color:#eee}
header{padding:16px 24px;background:#1a1a1a;position:sticky;top:0;z-index:9;border-bottom:1px solid #333}
h1{margin:0 0 6px;font-size:18px} .sub{color:#999;font-size:13px}
.img{margin:24px;border-top:1px solid #333;padding-top:16px}
.wrap{position:relative;display:inline-block}
img{max-width:900px;display:block}
.bx{position:absolute;border:2px solid;pointer-events:none}
.tag{position:absolute;font-size:10px;padding:0 3px;background:#000c;white-space:nowrap;transform:translateY(-100%)}
table{border-collapse:collapse;margin-top:10px;font-size:13px}
td,th{border:1px solid #333;padding:4px 8px;text-align:left}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
</style>"""]
    total = len(findings)
    counts = {k: sum(1 for f in findings if f["severity"] == k) for k in order}
    parts.append(f"<header><h1>区域标注体检 — {total} 处待确认</h1>"
                 f"<div class=sub>"
                 f"<span class=dot style='background:{colour['certain']}'></span>确定错误 {counts['certain']} · "
                 f"<span class=dot style='background:{colour['likely']}'></span>很可能 {counts['likely']} · "
                 f"<span class=dot style='background:{colour['suspect']}'></span>存疑 {counts['suspect']}"
                 f" &nbsp;|&nbsp; 框上标注为 <b>序号:区域</b></div></header>")

    for image in sorted(by_image):
        items = sorted(by_image[image], key=lambda f: order[f["severity"]])
        d = lookup[image]
        scale = 900 / d["w"]
        parts.append(f"<div class=img><h2>{image} &nbsp;<span class=sub>{len(items)} 处</span></h2><div class=wrap>")
        parts.append(f"<img src='images/{image}'>")
        for f in items:
            x, y, w, h = f["bbox"]
            c = colour[f["severity"]]
            label = f"{f['box'] if f['box'] >= 0 else '新'}:{f['zone']}"
            parts.append(f"<div class=bx style='left:{x*scale}px;top:{y*scale}px;width:{w*scale}px;"
                         f"height:{h*scale}px;border-color:{c}'></div>")
            parts.append(f"<div class=tag style='left:{x*scale}px;top:{y*scale}px;color:{c}'>{label}</div>")
        parts.append("</div><table><tr><th>框</th><th>类型</th><th>牌</th><th>说明</th></tr>")
        for f in items:
            c = colour[f["severity"]]
            box = f["box"] if f["box"] >= 0 else "—"
            parts.append(f"<tr><td><span class=dot style='background:{c}'></span>{box}</td>"
                         f"<td>{f['kind']}</td><td>{f['cls']}</td><td>{f['detail']}</td></tr>")
        parts.append("</table></div>")
    out.write_text("".join(parts), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--detect", action="store_true", help="also look for un-annotated tiles")
    ap.add_argument("--weights", type=Path, default=ROOT / "output" / "eval_real_v1" / "best.pt")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--json", type=Path, default=ROOT / "output" / "zone_annotation" / "audit.json")
    ap.add_argument("--html", type=Path, default=ROOT / "output" / "zone_annotation" / "audit.html")
    args = ap.parse_args(argv)

    data = json.loads(args.labels.read_text(encoding="utf-8"))
    findings = audit(data)
    if args.detect:
        findings += detect_missing(data, args.weights, args.conf)

    args.json.write_text(json.dumps(findings, ensure_ascii=False, indent=1), encoding="utf-8")
    write_html(findings, data, args.html)

    from collections import Counter
    print(f"{sum(len(d['boxes']) for d in data)} 个框,{len(findings)} 处待确认")
    for kind, cnt in Counter(f["kind"] for f in findings).most_common():
        sev = {f["severity"] for f in findings if f["kind"] == kind}
        print(f"  {kind:16s} {cnt:4d}  [{'/'.join(sorted(sev))}]")
    print(f"\n看图核对: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
