"""Frame strips for re-anchoring all 14 truth events to the physical landing moment.

10 frames per event, 0.2s spacing, covering [t-2.1, t-0.3] (landing was observed
0.8-1.3s before the oral timestamp). Pick the frame where the tile leaves the
fingers and rests on the pile/table.

    python tests/test1/exp_fix_gt_strips.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "frames" / "fix"
OUT.mkdir(parents=True, exist_ok=True)

TESTSET = ROOT / "output" / "video_testset_pilot"
VIDEO = TESTSET / "clips_full" / "clip01_7507945925261200.mp4"
CROP = (330, 60, 850, 260)  # pool region, 1280x720 clip pixels
OFFSETS = [round(-2.1 + 0.2 * k, 2) for k in range(10)]  # -2.1 .. -0.3


def main():
    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wanted = set()
    for e in gt:
        for off in OFFSETS:
            wanted.add(int(round((float(e["t"]) + off) * fps)))
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
    for idx, e in enumerate(gt):
        t = float(e["t"])
        tiles = []
        for off in OFFSETS:
            img = grabbed.get(int(round((t + off) * fps)))
            if img is None:
                img = np.zeros((720, 1280, 3), np.uint8)
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, ((x2 - x1) * 2, (y2 - y1) * 2), interpolation=cv2.INTER_CUBIC)
            canvas = np.zeros((crop.shape[0] + 30, crop.shape[1], 3), np.uint8)
            canvas[30:] = crop
            cv2.putText(canvas, f"{t + off:+.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(canvas)
        rows = [np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(2)]
        grid = np.vstack(rows)
        scale = 2600 / grid.shape[1]
        grid = cv2.resize(grid, (2600, int(grid.shape[0] * scale)))
        path = OUT / f"{idx:02d}_t{t:04.1f}_{e['who']}_{e['tile']}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", path.name)


if __name__ == "__main__":
    main()
