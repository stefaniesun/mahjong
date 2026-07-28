"""Build the video-level evaluation set (Phase 4 task 2).

Frame-by-frame ground truth is unaffordable, so evaluation uses **sparse checkpoints**:
one frozen frame every few seconds gets fully annotated, and video-level metrics are
measured at those instants plus a handful of hand-timed appearance events.

Two subcommands:

  scan   Profile candidate videos — tile density, camera motion, far-tile share — and
         propose clip windows. Picking clips by eye is slow and biased toward whatever
         looks tidy; the profile surfaces the *hard* stretches (fast head turns, dense
         distant rivers) that the spec wants covered.

  build  Cut the chosen clips, pull checkpoint frames, and pre-label them with the
         current detector+classifier so annotation is correction rather than drawing.

Leakage: clip windows avoid frames that were harvested into classifier training
(`--harvested-root`). Contamination cannot be fully undone — a tile seen at frame N is
usually still on the table at frame N+300 — so the report states the residual honestly
rather than pretending the set is pristine.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Phase 4 video evaluation set.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Profile videos and propose clip windows.")
    scan.add_argument("--videos", type=Path, required=True, help="Directory of source videos (searched recursively).")
    scan.add_argument("--det", type=Path, required=True, help="Detector weights.")
    scan.add_argument("--out", type=Path, default=Path("video_testset/scan_report.json"))
    scan.add_argument("--sample-fps", type=float, default=1.0, help="Sampling rate for profiling.")
    scan.add_argument("--imgsz", type=int, default=960)
    scan.add_argument("--det-conf", type=float, default=0.3)
    scan.add_argument("--clip-seconds", type=float, default=45.0)
    scan.add_argument("--clips-per-video", type=int, default=1)
    scan.add_argument("--max-videos", type=int, default=0)
    scan.add_argument("--harvested-root", type=Path, default=None, help="Frames already used for training; their windows are penalised.")
    scan.add_argument("--device", default="cpu")

    build = sub.add_parser("build", help="Cut clips and extract pre-labelled checkpoint frames.")
    build.add_argument("--plan", type=Path, required=True, help="scan_report.json, optionally hand-edited.")
    build.add_argument("--out", type=Path, required=True, help="Output root for the evaluation set.")
    build.add_argument("--det", type=Path, required=True)
    build.add_argument("--cls", type=Path, required=True, help="Classifier .onnx (with meta.json) or .pt.")
    build.add_argument("--checkpoint-seconds", type=float, default=5.0, help="One annotated frame per this many seconds.")
    build.add_argument("--det-conf", type=float, default=0.15, help="Low on purpose: prelabels should over-produce, you delete faster than you draw.")
    build.add_argument("--imgsz", type=int, default=960)
    build.add_argument("--device", default="cpu")
    build.add_argument("--top-k", type=int, default=1, help="Extra class candidates to record per box.")
    return parser


# --------------------------------------------------------------------------------------
# scan


@dataclass
class VideoProfile:
    path: str
    frames: int
    fps: float
    duration: float
    samples: list[dict[str, Any]] = field(default_factory=list)
    harvested_frames: list[int] = field(default_factory=list)


def harvested_index(root: Path | None) -> dict[str, set[int]]:
    """video id -> frame indices already used in training, parsed from crop filenames."""
    if root is None or not root.exists():
        return {}
    out: dict[str, set[int]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        vid = re.search(r"__(\d{15,})_", path.name)
        frame = re.search(r"__f(\d+)", path.name)
        if vid and frame:
            out.setdefault(vid.group(1), set()).add(int(frame.group(1)))
    return out


def profile_video(path: Path, detector, args, harvested: dict[str, set[int]]) -> VideoProfile | None:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return None
    step = max(1, int(round(fps / max(args.sample_fps, 0.01))))
    vid = path.stem.split("_")[0]
    profile = VideoProfile(
        path=str(path),
        frames=total,
        fps=fps,
        duration=total / fps,
        harvested_frames=sorted(harvested.get(vid, set())),
    )

    # Read sequentially and skip, rather than seeking per sample: every
    # CAP_PROP_POS_FRAMES jump forces a re-decode from the preceding keyframe, which
    # dominated runtime and made profiling slower than the detection it was profiling.
    prev_gray = None
    idx = -1
    while True:
        ok = capture.grab()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        ok, frame = capture.retrieve()
        if not ok:
            break
        result = detector.predict(source=frame, conf=args.det_conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            shorts = np.minimum(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1])
            count = int(len(shorts))
            far_share = float((shorts < 25).mean())
            median_short = float(np.median(shorts))
        else:
            count, far_share, median_short = 0, 0.0, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        motion = 0.0 if prev_gray is None else float(np.abs(small.astype(np.float32) - prev_gray).mean())
        prev_gray = small.astype(np.float32)

        profile.samples.append(
            {"frame": idx, "t": round(idx / fps, 2), "tiles": count, "far_share": round(far_share, 3), "median_short": round(median_short, 1), "motion": round(motion, 2)}
        )
    capture.release()
    return profile


def propose_clips(profile: VideoProfile, clip_seconds: float, per_video: int) -> list[dict[str, Any]]:
    """Score sliding windows so the picked clips are the informative ones."""
    if not profile.samples:
        return []
    times = np.array([s["t"] for s in profile.samples])
    tiles = np.array([s["tiles"] for s in profile.samples], dtype=np.float32)
    far = np.array([s["far_share"] for s in profile.samples], dtype=np.float32)
    motion = np.array([s["motion"] for s in profile.samples], dtype=np.float32)
    harvested_t = np.array([f / profile.fps for f in profile.harvested_frames], dtype=np.float32)

    windows: list[dict[str, Any]] = []
    stride = max(clip_seconds / 3.0, 5.0)
    start = 0.0
    while start + clip_seconds <= profile.duration:
        mask = (times >= start) & (times < start + clip_seconds)
        if mask.sum() >= 3:
            dense = float(tiles[mask].mean())
            # Windows that also *change* are the ones with discards happening in them.
            dynamics = float(tiles[mask].std())
            turn = float(np.percentile(motion[mask], 90)) if mask.sum() > 2 else 0.0
            far_share = float(far[mask].mean())
            overlap = int(((harvested_t >= start) & (harvested_t < start + clip_seconds)).sum()) if len(harvested_t) else 0
            score = (
                min(dense / 40.0, 1.0) * 1.0
                + min(dynamics / 8.0, 1.0) * 0.7
                + min(turn / 12.0, 1.0) * 0.7
                + far_share * 0.6
                - overlap * 0.5  # penalise, do not forbid: some videos have no clean window
            )
            windows.append(
                {
                    "start": round(start, 1),
                    "end": round(start + clip_seconds, 1),
                    "mean_tiles": round(dense, 1),
                    "tile_std": round(dynamics, 2),
                    "motion_p90": round(turn, 2),
                    "far_share": round(far_share, 3),
                    "harvested_overlap": overlap,
                    "score": round(score, 3),
                }
            )
        start += stride

    windows.sort(key=lambda w: -w["score"])
    picked: list[dict[str, Any]] = []
    for window in windows:
        if any(not (window["end"] <= p["start"] or window["start"] >= p["end"]) for p in picked):
            continue
        picked.append(window)
        if len(picked) >= per_video:
            break
    return picked


def cmd_scan(args) -> int:
    from ultralytics import YOLO

    videos = [p for p in sorted(args.videos.rglob("*.mp4"))]
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        print(f"没找到视频: {args.videos}")
        return 1
    harvested = harvested_index(args.harvested_root)
    print(f"扫描 {len(videos)} 个视频 (采样 {args.sample_fps} fps)，训练已用帧索引覆盖 {len(harvested)} 个视频", flush=True)

    detector = YOLO(str(args.det))
    report: dict[str, Any] = {"videos": [], "config": {"sample_fps": args.sample_fps, "clip_seconds": args.clip_seconds}}
    for i, path in enumerate(videos, start=1):
        profile = profile_video(path, detector, args, harvested)
        if profile is None:
            print(f"  [{i}/{len(videos)}] 打不开，跳过: {path.name[:40]}", flush=True)
            continue
        clips = propose_clips(profile, args.clip_seconds, args.clips_per_video)
        mean_tiles = float(np.mean([s["tiles"] for s in profile.samples])) if profile.samples else 0.0
        report["videos"].append(
            {
                "path": profile.path,
                "duration": round(profile.duration, 1),
                "fps": profile.fps,
                "mean_tiles": round(mean_tiles, 1),
                "harvested_frames": len(profile.harvested_frames),
                "clips": clips,
            }
        )
        best = clips[0]["score"] if clips else 0.0
        print(f"  [{i}/{len(videos)}] {path.name[:34]:34s} 时长{profile.duration:6.1f}s 均牌数{mean_tiles:5.1f} 最佳片段分{best:.2f}", flush=True)

    report["videos"].sort(key=lambda v: -(v["clips"][0]["score"] if v["clips"] else 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n扫描报告: {args.out}")
    print("下一步：按 score 从高到低挑 8~12 个片段，删掉 report 里不想要的，再跑 build")
    return 0


# --------------------------------------------------------------------------------------
# build


def load_classifier(path: Path):
    import json as _json

    if path.suffix.lower() == ".onnx":
        import onnxruntime as ort

        meta = _json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return {
            "kind": "onnx",
            "session": session,
            "input": session.get_inputs()[0].name,
            "classes": meta["classes"],
            "imgsz": int(meta.get("imgsz", 96)),
            "mean": np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32).reshape(3, 1, 1),
            "std": np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32).reshape(3, 1, 1),
        }
    raise SystemExit("build 需要 ONNX 分类器（带同目录 meta.json），以匹配 Phase 4 的无 torch 约束")


def classify_batch(model: dict[str, Any], patches: list[np.ndarray], top_k: int) -> list[list[tuple[str, float]]]:
    import cv2

    if not patches:
        return []
    size = model["imgsz"]
    batch = np.empty((len(patches), 3, size, size), np.float32)
    for i, patch in enumerate(patches):
        resized = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch[i] = (rgb.transpose(2, 0, 1) - model["mean"]) / model["std"]
    logits = model["session"].run(None, {model["input"]: batch})[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(axis=1, keepdims=True)
    out: list[list[tuple[str, float]]] = []
    k = max(1, min(top_k, prob.shape[1]))
    for row in prob:
        idx = np.argsort(-row)[:k]
        out.append([(model["classes"][int(j)], float(row[int(j)])) for j in idx])
    return out


def cmd_build(args) -> int:
    import cv2
    from ultralytics import YOLO

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    entries = [(v, clip) for v in plan.get("videos", []) for clip in v.get("clips", [])]
    if not entries:
        print("plan 里没有片段，先跑 scan 或检查你编辑后的文件")
        return 1

    detector = YOLO(str(args.det))
    classifier = load_classifier(args.cls)

    clips_dir = args.out / "clips"
    ann_dir = args.out / "checkpoints"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"clips": [], "checkpoint_seconds": args.checkpoint_seconds, "leakage": {"note": "分类器见过部分本域帧，检查点分类准确率偏乐观；分类的权威数字见 test_set_v1 (99.51%)", "clips_with_harvested_overlap": 0}}
    checkpoint_total = 0

    for clip_i, (video, clip) in enumerate(entries, start=1):
        src = Path(video["path"])
        capture = cv2.VideoCapture(str(src))
        if not capture.isOpened():
            print(f"  打不开 {src.name[:40]}，跳过")
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = int(clip["start"] * fps)
        end_frame = int(clip["end"] * fps)
        clip_name = f"clip{clip_i:02d}_{src.stem[:16]}"
        clip_path = clips_dir / f"{clip_name}.mp4"
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        checkpoint_step = int(args.checkpoint_seconds * fps)
        checkpoints: list[dict[str, Any]] = []
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for idx in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            if (idx - start_frame) % checkpoint_step:
                continue

            result = detector.predict(source=frame, conf=args.det_conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            boxes = getattr(result, "boxes", None)
            shapes: list[dict[str, Any]] = []
            topk: list[list[list[Any]]] = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                patches = []
                for x1, y1, x2, y2 in xyxy:
                    bw, bh = x2 - x1, y2 - y1
                    mx, my = bw * 0.08, bh * 0.08
                    patch = frame[int(max(0, y1 - my)) : int(min(height, y2 + my)), int(max(0, x1 - mx)) : int(min(width, x2 + mx))]
                    patches.append(patch if patch.size else np.zeros((8, 8, 3), np.uint8))
                preds = classify_batch(classifier, patches, args.top_k)
                for (x1, y1, x2, y2), cands in zip(xyxy, preds):
                    shape: dict[str, Any] = {
                        "label": cands[0][0],
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                    shapes.append(shape)
                    if args.top_k > 1:
                        # Alternatives go to a sidecar, never into the shape: X-AnyLabeling
                        # renders `description` on the canvas, and 65 strings per frame
                        # buries the tiles you are trying to look at.
                        topk.append([[c, round(p, 3)] for c, p in cands])

            stem = f"{clip_name}__f{idx - start_frame:06d}"
            cv2.imwrite(str(ann_dir / f"{stem}.jpg"), frame)
            (ann_dir / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "version": "2.3.6",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": f"{stem}.jpg",
                        "imageData": None,
                        "imageHeight": height,
                        "imageWidth": width,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if topk:
                (ann_dir / f"{stem}.topk.json").write_text(json.dumps(topk, ensure_ascii=False), encoding="utf-8")
            checkpoints.append({"file": f"{stem}.jpg", "clip_frame": idx - start_frame, "clip_time": round((idx - start_frame) / fps, 2), "source_frame": idx, "prelabel_boxes": len(shapes)})
            checkpoint_total += 1
        writer.release()
        capture.release()

        overlap = int(clip.get("harvested_overlap", 0))
        manifest["clips"].append(
            {
                "name": clip_name,
                "video": str(src),
                "clip_file": str(clip_path.relative_to(args.out)),
                "start": clip["start"],
                "end": clip["end"],
                "fps": fps,
                "checkpoints": checkpoints,
                "harvested_overlap": overlap,
                "scan_score": clip.get("score"),
            }
        )
        if overlap:
            manifest["leakage"]["clips_with_harvested_overlap"] += 1
        print(f"  [{clip_i}/{len(entries)}] {clip_name}  {clip['start']}~{clip['end']}s  检查点 {len(checkpoints)} 帧")

    manifest["checkpoint_total"] = checkpoint_total
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    template = {"clips": {clip["name"]: {"events": [{"tile": "w1", "t": 0.0, "note": "示例：把这条删掉，按指引填真实事件"}]} for clip in manifest["clips"]}}
    events_path = args.out / "events_gt.json"
    if not events_path.exists():
        events_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n片段 {len(manifest['clips'])} 个，检查点 {checkpoint_total} 帧")
    print(f"输出: {args.out}")
    print(f"  clips/       评测片段视频")
    print(f"  checkpoints/ 待标注帧 + 预标注 JSON  <- 在 X-AnyLabeling 里打开这个目录")
    print(f"  events_gt.json  出现事件时间戳（按指引手填）")
    print(f"  MANIFEST.json")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_scan(args) if args.command == "scan" else cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
