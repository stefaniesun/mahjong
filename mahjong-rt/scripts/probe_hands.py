"""Does off-the-shelf hand tracking see anything useful in this footage?

A cheap go/no-go before building anything on it. The event layer currently follows
violent state change rather than tiles landing, and the violent state changes are
*caused by hands* — an arm crossing the table drops detections from 50 to 12 and the
extractor fires six events into the gap. If hands are visible, they turn from the noise
source into the signal: who reached in tells you whose discard it was, and where the
hand stopped tells you which cell to read the tile from.

Two questions only, both answerable without integrating anything:

1. Are the other players' hands detected at all across a table, at 1280x720? MediaPipe is
   tuned for hands near the camera, and the player's own hands dominate this view.
2. Does hand activity have the ~2.1s rhythm of the turn order, or is it constant noise?

Writes a per-frame trace so the answers can be checked against the event truth later.

    python scripts/probe_hands.py --clip clip01_7507945925261200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "output" / "video_testset_pilot"


def side_of(x: float, y: float, width: int, height: int) -> str:
    """Which player a hand at this point most likely belongs to.

    Purely positional and deliberately crude — the point of the probe is to find out
    whether the raw signal exists, not to get attribution right yet.
    """
    nx, ny = x / width, y / height
    if ny > 0.62:
        return "me"
    if nx < 0.30:
        return "left"
    if nx > 0.70:
        return "right"
    return "across"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=TESTSET)
    ap.add_argument("--clip", default="clip01_7507945925261200")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-hands", type=int, default=4)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "hand_landmarker.task")
    args = ap.parse_args(argv)

    # MediaPipe 0.10.30+ dropped the legacy `solutions` API and stopped bundling weights,
    # so the model file is downloaded separately (see docs) and driven through Tasks.
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    video = args.testset / "clips_full" / f"{args.clip}.mp4"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"打不开 {video}")
        return 1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.model)),
            running_mode=RunningMode.VIDEO,
            num_hands=args.max_hands,
            min_hand_detection_confidence=args.min_conf,
            min_tracking_confidence=args.min_conf,
        )
    )

    trace: list[dict] = []
    index = 0
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, int(index * 1000 / fps))
        found = []
        for landmarks in (result.hand_landmarks or []):
            xs = [p.x * width for p in landmarks]
            ys = [p.y * height for p in landmarks]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            # Landmark 8 is the index fingertip — the part that reaches furthest and
            # ends up nearest the tile as it is placed.
            tip = landmarks[8]
            found.append({
                "cx": round(cx, 1), "cy": round(cy, 1),
                "tip": [round(tip.x * width, 1), round(tip.y * height, 1)],
                "span": round(float(max(xs) - min(xs)), 1),
                "side": side_of(cx, cy, width, height),
            })
        trace.append({"frame": index, "ts": round(index / fps, 3), "hands": found})
        index += 1
    capture.release()
    landmarker.close()
    elapsed = time.perf_counter() - started

    out = args.out or (args.testset / "hands" / f"{args.clip}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")

    counts = [len(f["hands"]) for f in trace]
    from collections import Counter
    by_side = Counter(h["side"] for f in trace for h in f["hands"])
    print(f"{len(trace)} 帧, {elapsed:.0f} 秒 ({len(trace) / max(elapsed, 1e-9):.1f} fps)")
    print(f"检出手的帧占比: {np.mean([c > 0 for c in counts]):.1%}   每帧手数中位 {int(np.median(counts))} 最多 {max(counts)}")
    print(f"按位置归属: {dict(by_side)}")
    print(f"\n轨迹写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
