"""Run the models once over the evaluation clips and store their output (Phase 4 task 9).

This is the expensive pass — everything after it (parameter sweeps, ablations, "what if
the voting window were 9") reads the recording instead of the video and finishes in
seconds.

Classifications are computed on detection crops rather than track crops so they stay
valid when tracker parameters change during a sweep, and the motion-compensation matrix
is stored because replay has no pixels to run optical flow on.

Example:
    python scripts/record_clips.py --testset ../output/video_testset_pilot \
        --det ../output/eval_real_v1/best.pt --cls ../output/cls_final_v2/best.onnx \
        --out ../output/video_testset_pilot/recordings --stride 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.recording import FrameRecord, Recording
from mahjong_rt.tracker import GlobalMotionEstimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record detector + classifier output for every evaluation clip.")
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True, help="ONNX classifier with sidecar meta.json.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--crop-margin", type=float, default=0.08)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-clips", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    import onnxruntime as ort
    from ultralytics import YOLO

    meta_path = args.cls.with_name("meta.json")
    if not meta_path.exists():
        print(f"分类器需要同目录的 meta.json: {meta_path}")
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = list(meta["classes"])
    cls_imgsz = int(meta.get("imgsz", 96))
    mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32).reshape(3, 1, 1)
    std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32).reshape(3, 1, 1)
    session = ort.InferenceSession(str(args.cls), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    detector = YOLO(str(args.det))

    manifest = json.loads((args.testset / "MANIFEST.json").read_text(encoding="utf-8"))
    clips = manifest.get("clips", [])
    if args.max_clips:
        clips = clips[: args.max_clips]

    args.out.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(clips, start=1):
        path = args.testset / clip["clip_file"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            print(f"  打不开 {path}")
            continue
        fps = clip.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recording = Recording(clip=clip["name"], classes=classes, frame_width=width, frame_height=height, fps=fps, stride=args.stride)
        gmc = GlobalMotionEstimator()

        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % max(1, args.stride):
                index += 1
                continue
            result = detector.predict(source=frame, conf=args.det_conf, iou=args.det_iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = boxes.conf.cpu().numpy().astype(np.float32)
            else:
                xyxy = np.zeros((0, 4), np.float32)
                scores = np.zeros((0,), np.float32)

            if len(xyxy):
                batch = np.empty((len(xyxy), 3, cls_imgsz, cls_imgsz), np.float32)
                for k, (x1, y1, x2, y2) in enumerate(xyxy):
                    bw, bh = x2 - x1, y2 - y1
                    mx, my = bw * args.crop_margin, bh * args.crop_margin
                    patch = frame[int(max(0, y1 - my)) : int(min(height, y2 + my)), int(max(0, x1 - mx)) : int(min(width, x2 + mx))]
                    if patch.size == 0:
                        patch = np.zeros((8, 8, 3), np.uint8)
                    resized = cv2.resize(patch, (cls_imgsz, cls_imgsz), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    batch[k] = (rgb.transpose(2, 0, 1) - mean) / std
                logits = session.run(None, {input_name: batch})[0]
                shifted = logits - logits.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                prob = exp / exp.sum(axis=1, keepdims=True)
                labels = prob.argmax(axis=1).astype(np.int16)
                confidences = prob.max(axis=1).astype(np.float32)
                probs = prob.astype(np.float32)
            else:
                labels = np.zeros((0,), np.int16)
                confidences = np.zeros((0,), np.float32)
                probs = np.zeros((0, len(classes)), np.float32)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            homography = gmc.estimate(gray, xyxy)

            recording.frames.append(
                FrameRecord(
                    frame_index=index,
                    timestamp=index / fps,
                    boxes=xyxy,
                    scores=scores,
                    labels=labels,
                    confidences=confidences,
                    probs=probs,
                    homography=homography,
                )
            )
            index += 1
        capture.release()

        out_path = args.out / f"{clip['name']}.npz"
        recording.save(out_path)
        boxes_total = sum(len(f.boxes) for f in recording.frames)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [{i}/{len(clips)}] {clip['name']}: {len(recording.frames)} 帧, {boxes_total} 框 -> {size_mb:.1f} MB", flush=True)

    print(f"\n录制完成: {args.out}")
    print("之后的参数扫描不再需要视频和模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
