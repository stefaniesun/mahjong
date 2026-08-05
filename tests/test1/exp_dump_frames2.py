"""Zoomed pool-region frame strips around truth moments, ~0.17s spacing.

    python tests/test1/exp_dump_frames2.py
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "frames"
OUT.mkdir(exist_ok=True)

VIDEO = ROOT / "output" / "video_testset_pilot" / "clips_full" / "clip01_7507945925261200.mp4"
PROBES = [12.0, 18.0, 29.0]
# Pool crop in 1280x720 clip pixels (generous box around the central pile).
CROP = (330, 60, 850, 260)  # x1,y1,x2,y2
OFFSETS = [-1.67, -1.33, -1.0, -0.67, -0.33, 0.0, 0.33, 0.67]


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wanted = set()
    for t in PROBES:
        for off in OFFSETS:
            wanted.add(int(round((t + off) * fps)))
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

    x1, y1, x2, y2 = CROP
    for t in PROBES:
        tiles = []
        for off in OFFSETS:
            img = grabbed.get(int(round((t + off) * fps)))
            if img is None:
                img = np.zeros((720, 1280, 3), np.uint8)
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, ((x2 - x1) * 2, (y2 - y1) * 2), interpolation=cv2.INTER_CUBIC)
            canvas = np.zeros((crop.shape[0] + 30, crop.shape[1], 3), np.uint8)
            canvas[30:] = crop
            mark = " <<< GT" if off == 0.0 else ""
            cv2.putText(canvas, f"t={t + off:+.2f}s{mark}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(canvas)
        rows = [np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(2)]
        grid = np.vstack(rows)
        scale = 2400 / grid.shape[1]
        grid = cv2.resize(grid, (2400, int(grid.shape[0] * scale)))
        path = OUT / f"pool_t{int(t)}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", path)


if __name__ == "__main__":
    main()
