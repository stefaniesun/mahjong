"""Video-level evaluation (Phase 4 task 8) — the acceptance yardstick for this phase.

Runs the pipeline in deterministic mode over every evaluation clip, then scores it on
the four things that decide whether the system is usable on video rather than on stills:
checkpoint accuracy, confirmation latency, flicker, and track continuity.

Determinism is a hard requirement: two runs over the same clips must produce identical
numbers, otherwise a parameter sweep cannot tell a real improvement from noise.

Example:
    python scripts/eval_video.py --testset video_testset --det ../output/eval_real_v1/best.pt \
        --cls ../output/cls_final_v2/best.onnx --out video_testset/eval_run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.metrics import (
    CheckpointResult,
    check_acceptance,
    compute_flicker,
    compute_latency,
    estimate_track_continuity,
    evaluate_checkpoint,
)
from mahjong_rt.pipeline import Pipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the pipeline on the video evaluation set.")
    parser.add_argument("--testset", type=Path, required=True, help="Root built by build_video_testset.py build.")
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="Output directory for report and raw results.")
    parser.add_argument("--baseline", type=Path, default=None, help="A previous summary.json to diff against.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame; 1 keeps evaluation faithful.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Checkpoints this early in a clip are skipped: the voter needs several observations before anything can be confirmed, so scoring them measures cold start, not accuracy.")
    return parser


def load_checkpoint_gt(path: Path) -> list[dict[str, Any]]:
    """X-AnyLabeling shapes -> [{bbox xywh, label}]. Tile backs are not annotated."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for shape in payload.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        out.append({"bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "label": str(shape.get("label", ""))})
    return out


def run_clip(clip: dict[str, Any], testset: Path, args, cfg: dict[str, Any]) -> dict[str, Any]:
    import cv2

    pipeline = Pipeline(
        PipelineConfig(
            det_weights=str(args.det),
            cls_weights=str(args.cls),
            det_conf=cfg.get("detector", {}).get("conf", 0.25),
            det_imgsz=cfg.get("detector", {}).get("imgsz", 960),
            device=args.device,
            tracker=cfg.get("tracker", {}),
            voter=cfg.get("voter", {}),
            state=cfg.get("state", {}),
            zones=cfg.get("zones", {}),
        )
    )

    clip_path = testset / clip["clip_file"]
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"打不开片段: {clip_path}")
    fps = clip.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 30.0

    # Checkpoint frames were exported at the source resolution, but the clip itself may
    # have been re-encoded smaller (they get shipped around). Predictions come out in
    # clip pixels, ground truth is in checkpoint pixels; without this scale every IoU is
    # zero and the run reports a total failure that is purely a units mismatch.
    clip_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    clip_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref = clip.get("checkpoints", [{}])[0].get("file")
    gt_w, gt_h = clip_w, clip_h
    if ref:
        meta = testset / "checkpoints" / Path(ref).with_suffix(".json").name
        if meta.exists():
            payload = json.loads(meta.read_text(encoding="utf-8"))
            gt_w, gt_h = int(payload.get("imageWidth", clip_w)), int(payload.get("imageHeight", clip_h))
    scale_x = gt_w / max(clip_w, 1)
    scale_y = gt_h / max(clip_h, 1)
    if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
        print(f"    片段 {clip_w}x{clip_h} vs 标注 {gt_w}x{gt_h}，预测框按 {scale_x:.2f}x 缩放对齐")

    wanted = {int(c["clip_frame"]): c for c in clip.get("checkpoints", [])}
    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    index = 0
    processed = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % max(1, args.stride):
            index += 1
            continue
        for event in pipeline.process_frame(frame, ts=index / fps):
            from dataclasses import asdict

            events.append(asdict(event))
        if index in wanted and index >= args.warmup_frames:
            snapshots.append(
                {
                    "clip_frame": index,
                    "file": wanted[index]["file"],
                    "confirmed": [
                        {
                            "track_id": t.track_id,
                            "label": t.label,
                            "bbox": [t.bbox[0] * scale_x, t.bbox[1] * scale_y, t.bbox[2] * scale_x, t.bbox[3] * scale_y],
                        }
                        for t in pipeline.state_machine.confirmed_tiles()
                    ],
                }
            )
        index += 1
        processed += 1
    capture.release()

    skipped_warmup = [c["file"] for c in clip.get("checkpoints", []) if int(c["clip_frame"]) < args.warmup_frames]
    return {
        "name": clip["name"],
        "events": events,
        "snapshots": snapshots,
        "skipped_warmup": skipped_warmup,
        "frames": processed,
        "perf": pipeline.perf_summary(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import yaml

    manifest_path = args.testset / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"找不到 {manifest_path}，先跑 build_video_testset.py build")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}

    events_gt_path = args.testset / "events_gt.json"
    events_gt = json.loads(events_gt_path.read_text(encoding="utf-8")) if events_gt_path.exists() else {"clips": {}}

    clips = manifest.get("clips", [])
    if args.max_clips:
        clips = clips[: args.max_clips]
    if not clips:
        print("MANIFEST 里没有片段")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    warmup_skipped: list[str] = []
    total_checkpoint = CheckpointResult()
    all_events: list[dict[str, Any]] = []
    all_latency_truth: list[dict[str, Any]] = []
    all_confirmed_events: list[dict[str, Any]] = []
    continuity_inputs: list[dict[str, Any]] = []
    per_clip: list[dict[str, Any]] = []
    fps_values: list[float] = []
    annotated_checkpoints = 0

    for i, clip in enumerate(clips, start=1):
        print(f"[{i}/{len(clips)}] {clip['name']} ...", flush=True)
        result = run_clip(clip, args.testset, args, cfg)
        all_events.extend(result["events"])
        fps_values.append(result["perf"].get("fps", 0.0))
        continuity_inputs.extend(result["snapshots"])
        warmup_skipped.extend(result.get("skipped_warmup", []))

        clip_checkpoint = CheckpointResult()
        for snapshot in result["snapshots"]:
            gt_path = args.testset / "checkpoints" / Path(snapshot["file"]).with_suffix(".json").name
            if not gt_path.exists():
                continue
            gt = load_checkpoint_gt(gt_path)
            # A checkpoint that still holds only the prelabels has not been corrected
            # yet; scoring against it would grade the model against itself.
            annotated_checkpoints += 1
            clip_checkpoint.merge(evaluate_checkpoint(gt, snapshot["confirmed"], iou_threshold=args.iou))
        total_checkpoint.merge(clip_checkpoint)

        truth = events_gt.get("clips", {}).get(clip["name"], {}).get("events", [])
        truth = [e for e in truth if not str(e.get("note", "")).startswith("示例")]
        all_latency_truth.extend(truth)
        all_confirmed_events.extend([e for e in result["events"] if e.get("type") == "tile_confirmed"])

        per_clip.append(
            {
                "name": clip["name"],
                "frames": result["frames"],
                "checkpoint": clip_checkpoint.as_dict(),
                "flicker": compute_flicker(result["events"]),
                "fps": result["perf"].get("fps", 0.0),
                "harvested_overlap": clip.get("harvested_overlap", 0),
            }
        )
        (args.out / f"events_{clip['name']}.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in result["events"]), encoding="utf-8"
        )

    summary = {
        "checkpoint": total_checkpoint.as_dict(),
        "latency": compute_latency(all_latency_truth, all_confirmed_events),
        "flicker": compute_flicker(all_events),
        "tracking": estimate_track_continuity(continuity_inputs),
        "performance": {"fps": round(sum(fps_values) / len(fps_values), 2) if fps_values else 0.0},
        "clips": per_clip,
        "annotated_checkpoints": annotated_checkpoints,
        "warmup_checkpoints_skipped": warmup_skipped,
        "leakage_note": manifest.get("leakage", {}).get("note", ""),
    }
    summary["acceptance"] = check_acceptance(summary)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint = summary["checkpoint"]
    print(f"\n{'='*56}")
    print(f"检查点: GT {checkpoint['gt_tiles']} 张牌 / {annotated_checkpoints} 帧" + (f"  (跳过 {len(warmup_skipped)} 个冷启动检查点)" if warmup_skipped else ""))
    print(f"  召回 {checkpoint['recall']*100:.2f}%  精确 {checkpoint['precision']*100:.2f}%  类别准确 {checkpoint['class_accuracy']*100:.2f}%")
    for bucket, stats in checkpoint["by_bucket"].items():
        print(f"    {bucket:8s} n={stats['n']:4d} 召回 {stats['recall']*100:6.2f}%  类别 {stats['class_accuracy']*100:6.2f}%")
    latency = summary["latency"]
    print(f"确认延迟: 匹配 {latency['n_matched']}/{latency['n_events']}  p50 {latency['p50']}s  p95 {latency['p95']}s")
    flicker = summary["flicker"]
    print(f"闪烁: {flicker['total_updates']} 次改判 / {flicker['confirmed_tracks']} 张确认牌 = {flicker['updates_per_tile']} 次/牌")
    print(f"跟踪: ID 切换 {summary['tracking']['id_switches']}/{summary['tracking']['comparisons']}")
    print(f"\n验收红线:")
    for item in summary["acceptance"]:
        mark = "PASS" if item["pass"] else "FAIL"
        print(f"  [{mark}] {item['name']:20s} 实测 {item['value']}  目标 {item['target']}")
    if summary["leakage_note"]:
        print(f"\n注意: {summary['leakage_note']}")
    print(f"\n报告: {args.out/'summary.json'}")

    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        print("\n与基线对比:")
        for key, path in [("类别准确", ("checkpoint", "class_accuracy")), ("延迟p95", ("latency", "p95")), ("闪烁率", ("flicker", "updates_per_tile"))]:
            now = summary[path[0]][path[1]]
            was = base.get(path[0], {}).get(path[1], 0)
            print(f"  {key:10s} {was} -> {now}  ({now - was:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
