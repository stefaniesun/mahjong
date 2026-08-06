"""Render only the hand landmarks, on black — is the action visible in what the model gives?

Everything else so far has asked whether an algorithm can find the discards. This asks
whether *anything* can: strip the video away and show only the 21 points per hand that
MediaPipe reports, and see if a person watching can call the moments.

Deliberately unlabelled. No ground-truth markers, no player names — a viewer who is told
where the discards are cannot judge whether they would have found them. Only a clock, so
what they see can be reported back and scored.

Also dumps the full 21 landmarks per hand per frame, which the earlier probe did not
keep — it stored only palm centroid, index fingertip and span, so nothing about finger
articulation could be tested from it.

    python scripts/render_points_only.py
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "output" / "video_testset_pilot"
FONT_PATH = next((p for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
                  if Path(p).exists()), None)

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
         (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
# One colour per tracked hand so a viewer can follow an individual through the frame.
PALETTE = [(90, 200, 255), (120, 255, 140), (255, 150, 220), (255, 200, 90),
           (180, 180, 255), (255, 255, 255), (150, 255, 255), (255, 170, 120)]


def label(image, text, xy, size=22, colour=(230, 230, 230)):
    pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default()
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=colour)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="clip01_7507945925261200")
    ap.add_argument("--out", type=Path, default=Path.home() / "Desktop" / "只看点的轨迹.mp4")
    ap.add_argument("--dump", type=Path, default=None, help="Where to write the full landmarks.")
    ap.add_argument("--trail", type=int, default=20, help="Frames of fading fingertip trail.")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "hand_landmarker.task")
    args = ap.parse_args(argv)

    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    capture = cv2.VideoCapture(str(TESTSET / "clips_full" / f"{args.clip}.mp4"))
    if not capture.isOpened():
        print("打不开视频")
        return 1
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(args.model)),
                              running_mode=RunningMode.VIDEO, num_hands=4,
                              min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    dump: list[dict] = []
    slots: list[dict] = []            # nearest-neighbour identity, only so colours stay put
    trails: dict[int, deque] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = landmarker.detect_for_video(
            Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            int(index * 1000 / fps))

        hands = []
        for marks in (result.hand_landmarks or []):
            hands.append([(p.x * w, p.y * h) for p in marks])
        dump.append({"frame": index, "ts": round(index / fps, 3),
                     "hands": [[[round(x, 1), round(y, 1)] for x, y in pts] for pts in hands]})

        taken, assigned = set(), []
        for pts in hands:
            wrist = pts[0]
            best, best_d = None, 1e9
            for s in slots:
                if s["id"] in taken:
                    continue
                d = float(np.hypot(wrist[0] - s["wrist"][0], wrist[1] - s["wrist"][1]))
                if d < best_d:
                    best, best_d = s, d
            if best is not None and best_d < 100:
                best["wrist"] = wrist
                taken.add(best["id"])
                assigned.append((best["id"], pts))
            else:
                new = {"id": len(slots), "wrist": wrist}
                slots.append(new)
                taken.add(new["id"])
                assigned.append((new["id"], pts))

        canvas = np.zeros((h, w, 3), np.uint8)
        for hid, pts in assigned:
            colour = PALETTE[hid % len(PALETTE)]
            trail = trails.setdefault(hid, deque(maxlen=args.trail))
            trail.append(pts[8])
            for k in range(1, len(trail)):
                fade = k / len(trail)
                cv2.line(canvas, tuple(int(v) for v in trail[k - 1]), tuple(int(v) for v in trail[k]),
                         tuple(int(c * fade * 0.55) for c in colour), 2)
            for a, b in BONES:
                cv2.line(canvas, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                         tuple(int(c * 0.7) for c in colour), 2)
            for j, p in enumerate(pts):
                r = 6 if j == 8 else 4 if j in (4, 12, 16, 20) else 3
                cv2.circle(canvas, (int(p[0]), int(p[1])), r, colour, -1)
        for hid in list(trails):
            if hid not in {i for i, _ in assigned}:
                trails.pop(hid, None)

        canvas = label(canvas, f"{index / fps:5.1f} 秒", (24, 20), 30)
        canvas = label(canvas, "只显示模型输出的 21 个关键点，无原始画面",
                       (24, h - 46), 20, (120, 120, 130))
        writer.write(canvas)
        index += 1

    capture.release()
    landmarker.close()
    writer.release()
    out_json = args.dump or (TESTSET / "hands_full" / f"{args.clip}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
    print(f"{index} 帧 -> {args.out}")
    print(f"全部 21 关键点 -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
