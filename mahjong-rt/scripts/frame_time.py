"""Frame index <-> timestamp conversion for logging appearance events.

Players step through clips in a frame-accurate player (PotPlayer, VLC) which shows a
frame number; `events_gt.json` wants seconds. This converts either way.

Example:
    python scripts/frame_time.py --fps 30 --frame 137
    python scripts/frame_time.py --fps 30 --time 4.57
    python scripts/frame_time.py --video video_testset/clips/clip01.mp4 --frame 137
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert between frame index and timestamp.")
    parser.add_argument("--fps", type=float, default=None, help="Frames per second; read from --video if omitted.")
    parser.add_argument("--video", type=Path, default=None, help="Read fps from this clip.")
    parser.add_argument("--frame", type=int, default=None, help="Frame index to convert to seconds.")
    parser.add_argument("--time", type=float, default=None, help="Seconds to convert to a frame index.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fps = args.fps
    if fps is None and args.video is not None:
        import cv2

        capture = cv2.VideoCapture(str(args.video))
        fps = capture.get(cv2.CAP_PROP_FPS) or None
        capture.release()
    if not fps:
        print("需要 --fps 或 --video")
        return 1
    if args.frame is None and args.time is None:
        print("需要 --frame 或 --time")
        return 1

    if args.frame is not None:
        seconds = args.frame / fps
        print(f"帧 {args.frame} @ {fps:g}fps  ->  {seconds:.2f} 秒")
    if args.time is not None:
        frame = round(args.time * fps)
        print(f"{args.time:g} 秒 @ {fps:g}fps  ->  帧 {frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
