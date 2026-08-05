"""Dump video frames around a few ground-truth discard moments for visual check:
does the tile visibly land at the truth timestamp, or ~1s earlier?

    python tests/test1/exp_dump_frames.py
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "frames"
OUT.mkdir(exist_ok=True)

VIDEO = ROOT / "output" / "video_testset_pilot" / "clips_full" / "clip01_7507945925261200.mp4"

# (truth time, label) — pick events the diagnosis flagged as "appeared 1s early"
PROBES = [(18.0, "b7"), (29.0, "b7"), (12.0, "b3")]


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)
    offsets = [-1.5, -1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0]
    wanted = {}
    for t, label in PROBES:
        for off in offsets:
            idx = int(round((t + off) * fps))
            wanted.setdefault(idx, []).append((t, label, off))
    grabbed: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            grabbed[i] = frame
        i += 1
    for t, label in PROBES:
        crops = []
        for off in offsets:
            img = grabbed.get(int(round((t + off) * fps)))
            if img is None:
                img = np.zeros((720, 1280, 3), np.uint8)
            crops.append(img)
        h, w = crops[0].shape[:2]
        labelbar = 22
        rows = []
        for r in range(2):
            row = []
            for c in range(4):
                img = crops[r * 4 + c].copy()
                canvas = np.zeros((h + labelbar, w, 3), np.uint8)
                canvas[labelbar:] = img
                cv2.putText(canvas, f"t={t + offsets[r * 4 + c]:+.2f}s".replace("+", "+"), (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                row.append(canvas)
            rows.append(np.hstack(row))
        grid = np.vstack(rows)
        scale = 1600 / grid.shape[1]
        grid = cv2.resize(grid, (1600, int(grid.shape[0] * scale)))
        path = OUT / f"probe_t{int(t)}_{label}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("wrote", path)
    cap.release()


if __name__ == "__main__":
    main()
