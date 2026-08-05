"""Render what the hand model actually reports, as a video.

Every claim about hand tracking in this project so far has been a number. This draws the
raw output instead: the 21 landmarks, the skeleton, and the two derived quantities that
were tested — where the arm comes from, and how far the fingertip is from the pool.

Chosen to be shown over 17-25s of clip01, which contains four real discards (18, 20, 22,
24s) and the forearm that sweeps the table and that the model does not see.

    python scripts/render_hands_demo.py --start 17 --end 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

# OpenCV's putText cannot draw CJK at all, so the panel goes through PIL.
FONT_PATH = next((p for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
                  if Path(p).exists()), None)


def draw_text(image: np.ndarray, lines, x=14, y=30, size=20, colour=(240, 240, 240)):
    pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default()
    for i, (text, tone) in enumerate(lines):
        draw.text((x, y + i * int(size * 1.55)), text, font=font, fill=tone or colour)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "output" / "video_testset_pilot"

# MediaPipe hand topology: wrist, then four bones per finger.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
         (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
SIDE_COLOUR = {"me": (0, 255, 0), "left": (255, 0, 255), "across": (0, 165, 255),
               "right": (0, 255, 255), None: (200, 200, 200)}
TRUTH = {18: "me 打 七筒", 20: "right 打 五筒", 22: "across 打 七筒", 24: "left 打 二筒"}


def entry_side(tip, palm, w, h):
    """Extend the fingertip-to-palm vector to a frame edge: that is where the arm comes from."""
    vx, vy = palm[0] - tip[0], palm[1] - tip[1]
    hits = []
    if vx < 0:
        hits.append(((0 - palm[0]) / vx, "left"))
    if vx > 0:
        hits.append(((w - palm[0]) / vx, "right"))
    if vy > 0:
        hits.append(((h - palm[1]) / vy, "me"))
    if vy < 0:
        hits.append(((0 - palm[1]) / vy, "across"))
    hits = [x for x in hits if x[0] > 0]
    return min(hits)[1] if hits else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="clip01_7507945925261200")
    ap.add_argument("--start", type=float, default=17.0)
    ap.add_argument("--end", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=Path.home() / "Desktop" / "手部模型能看到什么.mp4")
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
    cx, cy = 0.52 * w, 0.42 * h

    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(args.model)),
                              running_mode=RunningMode.VIDEO, num_hands=4,
                              min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    PANEL = 430
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + PANEL, h))

    index, written = 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        ts = index / fps
        if ts > args.end:
            break
        if ts < args.start:
            index += 1
            continue

        result = landmarker.detect_for_video(
            Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            int(index * 1000 / fps))
        canvas = frame.copy()
        cv2.circle(canvas, (int(cx), int(cy)), 10, (255, 255, 255), 2)
        cv2.putText(canvas, "pool", (int(cx) + 14, int(cy) + 5), 0, 0.6, (255, 255, 255), 2)

        rows = []
        for n, marks in enumerate(result.hand_landmarks or []):
            pts = [(int(p.x * w), int(p.y * h)) for p in marks]
            side = entry_side(pts[8], pts[0], w, h)
            colour = SIDE_COLOUR[side]
            for a, b in BONES:
                cv2.line(canvas, pts[a], pts[b], colour, 2)
            for p in pts:
                cv2.circle(canvas, p, 3, (255, 255, 255), -1)
            # The arm direction, drawn as the model sees it.
            vx, vy = pts[0][0] - pts[8][0], pts[0][1] - pts[8][1]
            norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
            end = (int(pts[0][0] + vx / norm * 130), int(pts[0][1] + vy / norm * 130))
            cv2.arrowedLine(canvas, pts[0], end, colour, 3, tipLength=0.25)
            dist = float(np.hypot(pts[8][0] - cx, pts[8][1] - cy)) / w
            cv2.putText(canvas, f"{side}", (pts[0][0] - 20, pts[0][1] - 14), 0, 0.7, colour, 2)
            cv2.putText(canvas, f"#{n + 1}", (pts[8][0] + 8, pts[8][1]), 0, 0.6, (255, 255, 255), 2)
            rows.append(f"hand{n + 1}: 方向={side}  指尖到牌池={dist:.2f}")

        panel = np.zeros((h, PANEL, 3), np.uint8)
        lines = [(f"{ts:5.2f} 秒", (255, 255, 255)), ("", None),
                 (f"检出手数: {len(result.hand_landmarks or [])}", (255, 255, 255)), ("", None),
                 ("模型直接给出的:", (150, 220, 255)),
                 ("  · 21 个关键点坐标", None), ("  · 左手 / 右手", None), ("  · 置信度", None),
                 ("", None), ("由坐标算出来的:", (150, 220, 255))]
        lines += [(f"  {r}", None) for r in rows]
        for k, hit in TRUTH.items():
            if abs(ts - k) < 0.5:
                lines += [("", None), (f"真实事件: {hit}", (120, 255, 120))]
        panel = draw_text(panel, lines, size=20)
        panel = draw_text(panel, [("模型不认识:", (255, 170, 90)),
                                  ("  摸牌 / 打牌 / 碰牌", (255, 170, 90)),
                                  ("  小臂和袖子(挡住桌面的正是它)", (255, 170, 90))],
                          y=h - 110, size=19)

        writer.write(np.hstack([canvas, panel]))
        written += 1
        index += 1

    capture.release()
    landmarker.close()
    writer.release()
    print(f"{written} 帧 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
