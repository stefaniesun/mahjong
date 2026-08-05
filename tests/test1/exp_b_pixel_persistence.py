"""Experiment B: pixel-level "object left behind" timing, no models at all.

Hands move constantly (the doc's core obstacle), but a hand does not leave a
*permanent* change behind; a landed tile does. So: align frames with the recorded
per-frame homographies, diff frame i against frame i-1.0s inside the pool region,
and keep only pixels that are STILL changed 0.5s later. Peaks in that persistent-
change count are candidate discard moments.

Also answers, threshold-free: how far before each truth timestamp does the visible
change actually happen (annotation-lag measurement).

    python tests/test1/exp_b_pixel_persistence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mahjong-rt"))

from mahjong_rt.recording import Recording
from mahjong_rt.zones import ZoneConfig, assign_zones

TESTSET = ROOT / "output" / "video_testset_pilot"
OUT = Path(__file__).resolve().parent
VIDEO = TESTSET / "clips_full" / "clip01_7507945925261200.mp4"

W2, H2 = 640, 360          # working resolution
BACK = 30                  # diff against 1.0s earlier
AHEAD = 15                 # must still be changed 0.5s later
THRESH = 25                # grayscale diff threshold
TIME_TOL = 0.5


def to_world(box, w):
    corners = np.array([[box[0], box[1], 1.0], [box[2], box[1], 1.0],
                        [box[2], box[3], 1.0], [box[0], box[3], 1.0]])
    pts = (w @ corners.T).T
    pts = pts[:, :2] / pts[:, 2:3]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def main():
    rec = Recording.load(TESTSET / "recordings_full" / "clip01_7507945925261200.npz")
    cfg = yaml.safe_load((ROOT / "mahjong-rt" / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    zone_cfg = ZoneConfig(**cfg.get("zones", {}))

    # World transforms (frame -> frame0 coords) + pool mask from river detections.
    world = np.eye(3)
    worlds = []
    river_boxes = []
    for f in rec.frames:
        try:
            world = world @ np.linalg.inv(np.asarray(f.homography, dtype=np.float64))
        except np.linalg.LinAlgError:
            pass
        worlds.append(world.copy())
        labels = [rec.classes[int(x)] for x in f.labels]
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in f.boxes]
        zones = assign_zones(xywh, rec.frame_width, rec.frame_height, zone_cfg, labels) if len(f.boxes) else []
        for b, z in zip(f.boxes, zones):
            if z == "river":
                river_boxes.append(to_world(b, world))

    mask = np.zeros((H2, W2), np.uint8)
    for b in river_boxes:
        cv2.rectangle(mask, (int(b[0] / 2), int(b[1] / 2)), (int(b[2] / 2), int(b[3] / 2)), 255, -1)
    mask = cv2.dilate(mask, np.ones((21, 21), np.uint8))   # ~42px full-res margin
    cv2.imwrite(str(OUT / "exp_b_pool_mask.png"), mask)
    m = mask > 0

    # Read video, downscale, blur.
    cap = cv2.VideoCapture(str(VIDEO))
    grays = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (W2, H2)), cv2.COLOR_BGR2GRAY)
        grays.append(cv2.GaussianBlur(g, (5, 5), 0))
    cap.release()
    n = min(len(grays), len(worlds))
    fps = rec.fps

    # Warp every frame into world coords once.
    S = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0]])
    Sinv = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
    warped = []
    for i in range(n):
        M = S @ worlds[i] @ Sinv
        warped.append(cv2.warpPerspective(grays[i], M, (W2, H2), flags=cv2.WARP_INVERSE_MAP))
    # NOTE: W maps frame coords -> world coords; warpPerspective with WARP_INVERSE_MAP
    # samples dst(world) from src(frame), which is exactly the alignment we want.

    persist = np.zeros(n)
    settle = np.zeros(n)
    for i in range(BACK, n - AHEAD):
        a = warped[i - BACK].astype(np.int16)
        b = warped[i].astype(np.int16)
        c = warped[i + AHEAD].astype(np.int16)
        d1 = (np.abs(b - a) > THRESH) & m
        d2 = (np.abs(c - a) > THRESH) & m
        settle[i] = np.count_nonzero(d1)
        persist[i] = np.count_nonzero(d1 & d2)

    np.savez(OUT / "exp_b_curves.npz", persist=persist, settle=settle, fps=fps)

    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]
    times = [float(e["t"]) for e in gt]

    # Threshold-free lag measurement: peak of persist in [t-2.5, t+0.5] around each truth.
    print("== 每个真值时刻附近的持续变化峰值位置(相对真值) ==")
    lags = []
    for e in gt:
        t = float(e["t"])
        lo, hi = int((t - 2.5) * fps), int((t + 0.5) * fps)
        lo = max(lo, BACK); hi = min(hi, n - AHEAD)
        seg = persist[lo:hi]
        peak = int(np.argmax(seg)) + lo
        lag = peak / fps - t
        lags.append(lag)
        print(f"  t={t:5.1f} {e['tile']:3s}  峰值@{peak / fps:5.2f}s  偏移{lag:+.2f}s  峰高{int(seg.max())}")
    lags = np.array(lags)
    print(f"偏移: 中位 {np.median(lags):+.2f}s  均值 {lags.mean():+.2f}s  [{lags.min():+.2f}, {lags.max():+.2f}]")

    # Event extraction at a few thresholds; score each with random baseline.
    print("\n== 峰值事件提取(persist曲线, NMS 0.6s) ==")
    med = np.median(persist[BACK:n - AHEAD])
    for mult in (3, 5, 8):
        thr = med * mult
        cand = []
        i = BACK
        while i < n - AHEAD:
            if persist[i] >= thr:
                j = i
                while j + 1 < n - AHEAD and persist[j + 1] >= thr:
                    j += 1
                peak = i + int(np.argmax(persist[i:j + 1]))
                cand.append(peak / fps)
                i = j + int(0.6 * fps)
            else:
                i += 1
        used = set()
        hits = 0
        for p in sorted(cand):
            near = [(abs(p - t), k) for k, t in enumerate(times) if k not in used and abs(p - t) <= TIME_TOL]
            if near:
                used.add(min(near)[1]); hits += 1
        # random baseline for the same number of predictions
        rng = np.random.RandomState(0)
        rhits = []
        for _ in range(2000):
            u = set()
            for p in sorted(rng.uniform(0, max(times), len(cand))):
                near = [(abs(p - t), k) for k, t in enumerate(times) if k not in u and abs(p - t) <= TIME_TOL]
                if near:
                    u.add(min(near)[1])
            rhits.append(len(u))
        rhits = np.array(rhits)
        pval = (np.sum(rhits >= hits) + 1) / (len(rhits) + 1)
        # and against lag-corrected truth (annotation lag ~1s, see frames/)
        used2 = set(); hits2 = 0
        for p in sorted(cand):
            near = [(abs(p - (t - 1.0)), k) for k, t in enumerate(times) if k not in used2 and abs(p - (t - 1.0)) <= TIME_TOL]
            if near:
                used2.add(min(near)[1]); hits2 += 1
        print(f"  阈值{mult}x中位({int(thr)}): 检出{len(cand):2d}  命中{hits:2d}  随机期望{rhits.mean():.1f}  p={pval:.3f}"
              f"  | 对滞后1s真值: 命中{hits2:2d}")

    # Render the curve with truth markers for eyeballing.
    img = np.zeros((300, n, 3), np.uint8)
    scale = 250.0 / max(persist.max(), 1)
    for i in range(BACK, n - AHEAD - 1):
        cv2.line(img, (i, 300 - int(persist[i] * scale)), (i + 1, 300 - int(persist[i + 1] * scale)), (0, 255, 0))
        cv2.line(img, (i, 300 - int(settle[i] * scale)), (i + 1, 300 - int(settle[i + 1] * scale)), (0, 100, 255))
    for t in times:
        x = int(t * fps)
        cv2.line(img, (x, 0), (x, 300), (255, 255, 0), 1)
        cv2.line(img, (int((t - 1.0) * fps), 0), (int((t - 1.0) * fps), 300), (255, 0, 255), 1)
    img = cv2.resize(img, (n * 2, 600), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "exp_b_curve.png"), img)
    print("\nwrote exp_b_curves.npz / exp_b_curve.png / exp_b_pool_mask.png")
    print("曲线图: 绿=持续变化 橙=原始变化 青线=真值 品红线=真值-1s")


if __name__ == "__main__":
    main()
