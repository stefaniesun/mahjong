"""Full-frame strips over each expected draw window (previous discard -> own discard),
so the actual draw moment can be anchored visually. Dealer's first discard needs no draw.

    python tests/test1/exp_e_draw_strips.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "frames" / "draw"
OUT.mkdir(parents=True, exist_ok=True)

TESTSET = ROOT / "output" / "video_testset_pilot"
VIDEO = TESTSET / "clips_full" / "clip01_7507945925261200.mp4"
N_FRAMES = 6


def main():
    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    windows = []
    for k in range(1, len(gt)):
        t_prev = float(gt[k - 1]["t"])
        t_own = float(gt[k]["t"])
        lo, hi = t_prev + 0.2, t_own - 0.2
        if hi <= lo:
            hi = lo + 0.2
        times = np.linspace(lo, hi, N_FRAMES)
        windows.append((k, gt[k]["who"], gt[k]["tile"], times))
    wanted = {}
    for k, who, tile, times in windows:
        for t in times:
            wanted.setdefault(int(round(t * fps)), []).append((k, t))
    grabbed: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            grabbed[i] = frame
        i += 1
    cap.release()

    for k, who, tile, times in windows:
        tiles = []
        for t in times:
            img = grabbed.get(int(round(t * fps)))
            if img is None:
                img = np.zeros((720, 1280, 3), np.uint8)
            canvas = np.zeros((img.shape[0] + 30, img.shape[1], 3), np.uint8)
            canvas[30:] = img
            cv2.putText(canvas, f"{t:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(canvas)
        rows = [np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(2)]
        grid = np.vstack(rows)
        scale = 2200 / grid.shape[1]
        grid = cv2.resize(grid, (2200, int(grid.shape[0] * scale)))
        path = OUT / f"{k:02d}_draw_for_{who}_{tile}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("wrote", path.name)


if __name__ == "__main__":
    main()
