"""Harvest classifier training crops from unlabeled frames (Phase 3 task 3).

Runs the detector over frames, crops every tile it finds, runs the current classifier
on each crop, and files the crop into a folder named after the predicted class. The
human's only job afterwards is dragging misfiled images into the right folder — no
box drawing, no typing.

Two things make the human pass cheap:
  * `--review-classes` splits out the classes you actually distrust (e.g. the 万 group)
    into `review/`, leaving confident predictions in `auto/`.
  * `--review-below` sends every low-confidence crop to `review/` regardless of class.

Leakage: frames whose source video feeds the frozen test set are skipped via
`--exclude-manifest` (data/test_set_v1/split_manifest.json).

Example:
    python scripts/harvest_crops.py --frames data/frames_candidate \
        --det output/eval_real_v1/best.pt --cls output/cls_eval_v1/best.pt \
        --out output/harvest_v1 --review-classes w --review-below 0.9
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest and pre-classify tile crops from unlabeled frames.")
    parser.add_argument("--frames", type=Path, required=True, help="Directory of frames (searched recursively).")
    parser.add_argument("--include", default="", help="Only keep frames whose path matches this regex, e.g. 'dy_' to skip already-labelled bili frames.")
    parser.add_argument("--det", type=Path, required=True, help="Detector weights (.pt or .onnx).")
    parser.add_argument("--cls", type=Path, required=True, help="Classifier checkpoint from train_classifier.py.")
    parser.add_argument("--out", type=Path, required=True, help="Output root; creates auto/ and review/ subtrees.")
    parser.add_argument("--exclude-manifest", type=Path, default=Path("data/test_set_v1/split_manifest.json"), help="Frozen-test manifest whose source videos must be skipped.")
    parser.add_argument("--det-conf", type=float, default=0.5, help="Detector confidence floor.")
    parser.add_argument("--imgsz", type=int, default=960, help="Detector inference size.")
    parser.add_argument("--margin", type=float, default=0.08, help="Crop margin ratio, matching convert_labels.py.")
    parser.add_argument("--min-short-side", type=int, default=10, help="Skip boxes whose short side is below this.")
    parser.add_argument("--review-classes", default="", help="Comma-separated class names or prefixes always sent to review/ (e.g. 'w' or 'w2,w3,w5').")
    parser.add_argument("--review-below", type=float, default=0.9, help="Classifier confidence below which a crop goes to review/.")
    parser.add_argument("--resume", action="store_true", help="Skip frames that already produced crops under --out (including ones you have since re-filed).")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap frames processed (0 = all).")
    parser.add_argument("--per-class-cap", type=int, default=0, help="Stop saving a class once it reaches this many crops (0 = unlimited).")
    parser.add_argument("--device", default="cuda")
    return parser


def leaked_video_ids(manifest: Path) -> set[str]:
    """Video ids whose frames feed the frozen test set; harvesting them would leak."""
    if not manifest.exists():
        return set()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for entry in payload.get("test_videos", []):
        ids.add(entry.split("/")[-1])
    return ids


def video_id_of(name: str) -> str | None:
    match = re.search(r"(BV[0-9A-Za-z]+)", name)
    return match.group(1) if match else None


def size_bucket(short_side: float) -> str:
    if short_side < 20:
        return "lt20"
    if short_side <= 40:
        return "20to40"
    return "gt40"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    import torch
    from torchvision import transforms
    from PIL import Image
    from ultralytics import YOLO

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from train_classifier import build_model

    leaked = leaked_video_ids(args.exclude_manifest)
    print(f"排除泄漏视频 {len(leaked)} 个")

    frames = [p for p in sorted(args.frames.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    include = re.compile(args.include) if args.include else None
    kept_frames = []
    skipped_leak = 0
    skipped_filter = 0
    for path in frames:
        if include and not include.search(path.as_posix()):
            skipped_filter += 1
            continue
        vid = video_id_of(path.name)
        if vid and vid in leaked:
            skipped_leak += 1
            continue
        kept_frames.append(path)
    if include:
        print(f"--include '{args.include}' 过滤掉 {skipped_filter} 帧")

    if args.resume and args.out.exists():
        # A crop is named "<frame stem>__boxNNN__...", so any crop anywhere under --out
        # proves its frame was already processed — even if you have re-filed it by hand.
        done = {p.name.split("__box")[0] for p in args.out.rglob("*.jpg")}
        before = len(kept_frames)
        kept_frames = [p for p in kept_frames if p.stem not in done]
        print(f"--resume 跳过已处理 {before - len(kept_frames)} 帧")
    if args.max_frames:
        kept_frames = kept_frames[: args.max_frames]
    print(f"帧: 总 {len(frames)}, 排除泄漏 {skipped_leak}, 待处理 {len(kept_frames)}")

    detector = YOLO(str(args.det))
    ckpt = torch.load(args.cls, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    imgsz_cls = int(ckpt.get("imgsz", 96))
    classifier = build_model(ckpt.get("arch", "mobilenet_v3_small"), len(classes))
    classifier.load_state_dict(ckpt["model"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    classifier.to(device).eval()

    tf = transforms.Compose(
        [
            transforms.Resize((imgsz_cls, imgsz_cls)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    review_rules = [r.strip() for r in args.review_classes.split(",") if r.strip()]

    def needs_review(name: str, conf: float) -> bool:
        if conf < args.review_below:
            return True
        return any(name == rule or name.startswith(rule) for rule in review_rules)

    saved = Counter()
    routed = Counter()
    buckets = Counter()
    for index, path in enumerate(kept_frames, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        result = detector.predict(source=str(path), conf=args.det_conf, iou=0.6, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        height, width = image.shape[:2]
        crops = []
        metas = []
        for order, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < args.min_short_side:
                continue
            mx, my = bw * args.margin, bh * args.margin
            cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            cx2, cy2 = int(min(width, x2 + mx)), int(min(height, y2 + my))
            patch = image[cy1:cy2, cx1:cx2]
            if patch.size == 0:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
            metas.append((order, bw, bh, patch))
        if not crops:
            continue
        batch = torch.stack([tf(c) for c in crops]).to(device)
        with torch.no_grad():
            prob = torch.softmax(classifier(batch), dim=1)
            conf, pred = prob.max(1)
        for (order, bw, bh, patch), c, p in zip(metas, conf.cpu().tolist(), pred.cpu().tolist()):
            name = classes[p]
            if args.per_class_cap and saved[name] >= args.per_class_cap:
                continue
            bucket = size_bucket(min(bw, bh))
            lane = "review" if needs_review(name, c) else "auto"
            out_dir = args.out / lane / name
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{path.stem}__box{order:03d}__{bucket}__c{c:.2f}__{int(bw)}x{int(bh)}.jpg"
            cv2.imwrite(str(out_dir / fname), patch)
            saved[name] += 1
            routed[lane] += 1
            buckets[bucket] += 1
        if index % 200 == 0:
            print(f"  {index}/{len(kept_frames)} 帧, 已存 {sum(saved.values())} crop")

    report = {
        "frames_total": len(frames),
        "frames_skipped_leak": skipped_leak,
        "frames_processed": len(kept_frames),
        "crops_total": sum(saved.values()),
        "routed": dict(routed),
        "size_buckets": dict(buckets),
        "per_class": dict(sorted(saved.items())),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "harvest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ncrop 总数 {report['crops_total']}  auto={routed['auto']}  review={routed['review']}")
    print("尺寸分布:", dict(buckets))
    print("各类:")
    for k, v in sorted(saved.items()):
        print(f"  {k:8s} {v}")
    print(f"报告: {args.out/'harvest_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
