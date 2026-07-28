"""Run the recognition pipeline over a video (Phase 4 task 7).

Example:
    python scripts/run_pipeline.py --source clip.mp4 --out-video out.mp4 --events events.jsonl \
        --det ../output/eval_real_v1/best.pt --cls ../output/cls_final_v2/best.pt --max-frames 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.events import TileState, Zone, to_json
from mahjong_rt.pipeline import Pipeline, PipelineConfig

ZONE_COLOUR = {
    Zone.MY_HAND.value: (80, 200, 255),
    Zone.RIVER.value: (120, 255, 120),
    Zone.MELD_AREA.value: (255, 160, 60),
    Zone.OPPONENT_WALL.value: (200, 120, 255),
    Zone.UNKNOWN.value: (180, 180, 180),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mahjong realtime pipeline over a video source.")
    parser.add_argument("--source", required=True, help="Video file path or camera index.")
    parser.add_argument("--det", type=Path, required=True, help="Detector weights.")
    parser.add_argument("--cls", type=Path, required=True, help="Classifier checkpoint.")
    parser.add_argument("--config", type=Path, default=None, help="pipeline.yaml; CLI flags win over it.")
    parser.add_argument("--out-video", type=Path, default=None, help="Write an annotated video here.")
    parser.add_argument("--events", type=Path, default=None, help="Write the event stream as jsonl.")
    parser.add_argument("--headless", action="store_true", help="No window; just events and stats.")
    parser.add_argument("--start-frame", type=int, default=0, help="Seek here before processing — clips often open on an intro with no tiles.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = whole video).")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame of the source.")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--classify-every", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-zones", action="store_true", help="Disable zone assignment.")
    return parser


def load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def draw(frame, tiles, stats_text: Sequence[str]):
    import cv2

    for tile in tiles:
        x, y, w, h = [int(v) for v in tile.bbox]
        confirmed = tile.state == TileState.CONFIRMED
        colour = ZONE_COLOUR.get(tile.zone, (180, 180, 180)) if confirmed else (0, 215, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2 if confirmed else 1)
        text = f"{tile.label}#{tile.track_id}" if confirmed and tile.label else f"?#{tile.track_id}"
        cv2.putText(frame, text, (x, max(11, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    for i, line in enumerate(stats_text):
        cv2.putText(frame, line, (8, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (8, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2

    cfg = load_config(args.config)
    pipeline = Pipeline(
        PipelineConfig(
            det_weights=str(args.det),
            cls_weights=str(args.cls),
            det_conf=args.det_conf,
            det_imgsz=args.imgsz,
            classify_every=args.classify_every,
            device=args.device,
            tracker=cfg.get("tracker", {}),
            voter=cfg.get("voter", {}),
            state=cfg.get("state", {}),
            zones={**cfg.get("zones", {}), **({"enabled": False} if args.no_zones else {})},
        )
    )

    source: str | int = int(args.source) if str(args.source).isdigit() else str(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"打不开视频源: {args.source}", file=sys.stderr)
        return 1
    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out_video:
        args.out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, args.stride), (width, height))

    events_handle = None
    if args.events:
        args.events.parent.mkdir(parents=True, exist_ok=True)
        events_handle = args.events.open("w", encoding="utf-8")

    counts = {"tile_confirmed": 0, "tile_updated": 0, "tile_lost": 0}
    processed = 0
    raw_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if raw_idx % max(1, args.stride) != 0:
                raw_idx += 1
                continue
            raw_idx += 1
            events = pipeline.process_frame(frame, ts=processed / (fps / max(1, args.stride)))
            for event in events:
                if event.type in counts:
                    counts[event.type] += 1
                if events_handle:
                    events_handle.write(to_json(event) + "\n")
            processed += 1

            if writer is not None or not args.headless:
                tiles = pipeline.state_machine.confirmed_tiles() + [
                    t for t in pipeline.state_machine.tiles.values() if t.state == TileState.TENTATIVE
                ]
                perf = pipeline.perf_summary()
                lines = [
                    f"frame {processed}  confirmed {len(pipeline.state_machine.confirmed_tiles())}",
                    f"det {perf.get('detect',{}).get('p50_ms',0)}ms  cls {perf.get('classify',{}).get('p50_ms',0)}ms  trk {perf.get('track',{}).get('p50_ms',0)}ms",
                    f"fps {perf.get('fps',0)}  events C{counts['tile_confirmed']} U{counts['tile_updated']} L{counts['tile_lost']}",
                ]
                canvas = draw(frame.copy(), tiles, lines)
                if writer is not None:
                    writer.write(canvas)
                if not args.headless:
                    cv2.imshow("mahjong-rt", canvas)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            if args.max_frames and processed >= args.max_frames:
                break
            if processed % 50 == 0:
                print(f"  {processed} 帧, 已确认 {len(pipeline.state_machine.confirmed_tiles())} 张牌")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if events_handle:
            events_handle.close()
        if not args.headless:
            cv2.destroyAllWindows()

    perf = pipeline.perf_summary()
    print(f"\n处理 {processed} 帧")
    print(f"事件: 确认 {counts['tile_confirmed']}  改判 {counts['tile_updated']}  消失 {counts['tile_lost']}")
    print(f"当前稳定牌数: {len(pipeline.state_machine.confirmed_tiles())}")
    print("性能:", json.dumps(perf, ensure_ascii=False))
    if args.out_video:
        print(f"标注视频: {args.out_video}")
    if args.events:
        print(f"事件流: {args.events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
