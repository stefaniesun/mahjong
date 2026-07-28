"""Train the 28-class Mahjong tile-face classifier (Phase 3 task 3).

Backbone defaults to MobileNetV3-Small per the Phase 3 spec. Input is 96x96 to match
the deployment target. The key augmentation is **degradation**: crops are randomly
downscaled to 12~40px and blown back up, which is how a far-away tile actually reaches
the classifier after detection. Without it the model never learns to read small tiles.

Example:
    python scripts/train_classifier.py --data output/cls_train_v1 --epochs 40 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Mahjong 28-class tile classifier.")
    parser.add_argument("--data", type=Path, required=True, help="Root with train/ and val/ class folders.")
    parser.add_argument("--out", type=Path, default=Path("runs/cls_v1"), help="Output directory for weights and logs.")
    parser.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0"], help="Backbone.")
    parser.add_argument("--imgsz", type=int, default=96, help="Input size.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degrade-prob", type=float, default=0.7, help="Probability of applying the small-tile degradation.")
    parser.add_argument("--balance", action="store_true", help="Sample every class equally per epoch. Use when harvesting has skewed the class counts.")
    return parser


class Degrade:
    """Shrink to 12~40px then upscale back — simulates a distant tile's crop."""

    def __init__(self, imgsz: int, prob: float) -> None:
        self.imgsz = imgsz
        self.prob = prob

    def __call__(self, image):
        from PIL import Image

        if random.random() > self.prob:
            return image
        target = random.randint(12, 40)
        small = image.resize((target, target), Image.BILINEAR)
        return small.resize((self.imgsz, self.imgsz), Image.BILINEAR)


def build_transforms(imgsz: int, degrade_prob: float):
    from torchvision import transforms

    # ColorJitter runs after ToTensor: torchvision's PIL hue path calls np.uint8() on a
    # negative value, which raises OverflowError under numpy 2.x. The tensor path is fine.
    train_tf = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomApply([transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5)], p=0.7),
            Degrade(imgsz, degrade_prob),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model(arch: str, num_classes: int):
    from torchvision import models

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        import torch.nn as nn

        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    import torch.nn as nn

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tf, eval_tf = build_transforms(args.imgsz, args.degrade_prob)
    train_ds = ImageFolder(str(args.data / "train"), transform=train_tf)
    val_ds = ImageFolder(str(args.data / "val"), transform=eval_tf)
    classes = train_ds.classes
    print(f"classes={len(classes)} train={len(train_ds)} val={len(val_ds)}")

    if args.balance:
        # Harvesting fills some classes far faster than others; without this the model
        # picks up a prior for whichever class happened to be easiest to collect.
        from torch.utils.data import WeightedRandomSampler

        counts = [0] * len(classes)
        for _, target in train_ds.samples:
            counts[target] += 1
        weights = [1.0 / counts[target] for _, target in train_ds.samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        print(f"类别均衡采样: 最少类 {min(counts)} 张, 最多类 {max(counts)} 张, 比值 {max(counts)/max(min(counts),1):.2f}")
        train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args.arch, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for images, targets in train_dl:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
        scheduler.step()
        train_acc = correct / max(total, 1)

        model.eval()
        vt = vc = 0
        with torch.no_grad():
            for images, targets in val_dl:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                vc += (outputs.argmax(1) == targets).sum().item()
                vt += targets.size(0)
        val_acc = vc / max(vt, 1)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(total, 1), "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch:3d}/{args.epochs}  loss={loss_sum/max(total,1):.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "arch": args.arch, "imgsz": args.imgsz}, args.out / "best.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best val_acc={best_acc:.4f}  weights: {args.out/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
