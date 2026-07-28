"""Evaluate the tile classifier on the frozen crop test set (Phase 3 task 5).

The headline number this project needs is **accuracy per size bucket** — a classifier
that is 99% overall but 60% on <20px tiles is useless for the far side of the table.
Bucket is read from the crop filename (`__lt20__` / `__20to40__` / `__gt40__`), which
convert_labels.py encodes at crop time.

Example:
    python scripts/eval_classifier.py --weights runs/cls_v1/best.pt \
        --data output/cls_test_v1/cls_crops --out runs/cls_v1/eval
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

BUCKETS = ("lt20", "20to40", "gt40")
BUCKET_LABEL = {"lt20": "<20px", "20to40": "20-40px", "gt40": ">40px"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the Mahjong tile classifier.")
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint from train_classifier.py.")
    parser.add_argument("--data", type=Path, required=True, help="Root with one folder per class.")
    parser.add_argument("--out", type=Path, default=Path("runs/cls_v1/eval"), help="Output directory.")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude-unknown", action="store_true", help="Report the 27-class number excluding 'unknown'.")
    return parser


def bucket_of(path: str) -> str:
    for bucket in BUCKETS:
        if f"__{bucket}__" in path:
            return bucket
    return "gt40"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    imgsz = int(ckpt.get("imgsz", 96))

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from train_classifier import build_model

    model = build_model(ckpt.get("arch", "mobilenet_v3_small"), len(classes))
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()

    eval_tf = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = ImageFolder(str(args.data), transform=eval_tf)
    if ds.classes != classes:
        print(f"WARNING: class order differs.\n  train: {classes}\n  test : {ds.classes}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    paths = [p for p, _ in ds.samples]
    preds: list[int] = []
    confs: list[float] = []
    with torch.no_grad():
        for images, _ in dl:
            logits = model(images.to(device))
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(1)
            preds.extend(pred.cpu().tolist())
            confs.extend(conf.cpu().tolist())

    targets = [t for _, t in ds.samples]
    bucket_stat: dict[str, Counter] = {b: Counter() for b in BUCKETS}
    per_class: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    correct = total = 0
    correct27 = total27 = 0
    unknown_idx = ds.classes.index("unknown") if "unknown" in ds.classes else -1

    for path, target, pred in zip(paths, targets, preds):
        bucket = bucket_of(path)
        hit = int(pred == target)
        bucket_stat[bucket]["n"] += 1
        bucket_stat[bucket]["hit"] += hit
        name = ds.classes[target]
        per_class[name]["n"] += 1
        per_class[name]["hit"] += hit
        correct += hit
        total += 1
        if target != unknown_idx:
            correct27 += hit
            total27 += 1
        if not hit:
            confusion[(name, ds.classes[pred])] += 1

    report = {
        "summary": {
            "total": total,
            "top1": round(correct / max(total, 1), 4),
            "top1_27class_excl_unknown": round(correct27 / max(total27, 1), 4),
        },
        "size_buckets": {
            b: {
                "n": bucket_stat[b]["n"],
                "acc": round(bucket_stat[b]["hit"] / max(bucket_stat[b]["n"], 1), 4),
            }
            for b in BUCKETS
        },
        "per_class": {k: {"n": v["n"], "acc": round(v["hit"] / max(v["n"], 1), 4)} for k, v in sorted(per_class.items())},
        "top_confusions": [{"true": a, "pred": b, "count": c} for (a, b), c in confusion.most_common(15)],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classification_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"total={total}  top1={report['summary']['top1']*100:.2f}%  (27class excl unknown: {report['summary']['top1_27class_excl_unknown']*100:.2f}%)")
    print("size buckets:")
    for b in BUCKETS:
        s = report["size_buckets"][b]
        print(f"  {BUCKET_LABEL[b]:9s} n={s['n']:5d}  acc={s['acc']*100:6.2f}%")
    print("worst classes:")
    for k, v in sorted(report["per_class"].items(), key=lambda x: x[1]["acc"])[:8]:
        print(f"  {k:8s} n={v['n']:4d}  acc={v['acc']*100:6.2f}%")
    print("top confusions:")
    for c in report["top_confusions"][:8]:
        print(f"  {c['true']:8s} -> {c['pred']:8s}  {c['count']}")
    print(f"report: {args.out/'classification_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
