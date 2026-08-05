"""Diagnosis for experiment A: at each ground-truth discard moment, what did the
raw detector actually see in the river zone? For the truth tile label, find the
first persistent river detection after the truth time and measure which gate of
the novelty detector (novelty IoU, persistence, zone, warmup) would reject it.

    python tests/test1/exp_a_diag.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mahjong-rt"))

from mahjong_rt.recording import Recording
from mahjong_rt.zones import ZoneConfig, assign_zones

TESTSET = ROOT / "output" / "video_testset_pilot"
OUT = Path(__file__).resolve().parent


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


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

    world = np.eye(3)
    frames = []
    for f in rec.frames:
        try:
            world = world @ np.linalg.inv(np.asarray(f.homography, dtype=np.float64))
        except np.linalg.LinAlgError:
            pass
        labels = [rec.classes[int(x)] for x in f.labels]
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in f.boxes]
        zones = assign_zones(xywh, rec.frame_width, rec.frame_height, zone_cfg, labels) if len(f.boxes) else []
        wboxes = np.array([to_world(b, world) for b in f.boxes]) if len(f.boxes) else np.zeros((0, 4))
        frames.append({"ts": f.timestamp, "wboxes": wboxes, "zones": zones, "labels": labels})

    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]
    ts_arr = np.array([fr["ts"] for fr in frames])

    rows = []
    for g in gt:
        t, tile = float(g["t"]), g["tile"]
        i0 = int(np.searchsorted(ts_arr, t - 1.0))
        i1 = int(np.searchsorted(ts_arr, t + 3.0))
        # First frame >= t where a river detection with this label persists
        # (seen again, IoU>=0.3, in >=5 of the next 10 frames).
        found = None
        for i in range(i0, min(i1, len(frames))):
            for k in range(len(frames[i]["wboxes"])):
                if frames[i]["labels"][k] != tile or frames[i]["zones"][k] != "river":
                    continue
                b = frames[i]["wboxes"][k]
                hits = 1
                for j in range(i + 1, min(len(frames), i + 11)):
                    if any(iou(b, x) >= 0.3 for x in frames[j]["wboxes"]):
                        hits += 1
                if hits < 5:
                    continue
                # Novelty at its first appearance: max IoU vs past 15 frames.
                past = [frames[j]["wboxes"] for j in range(max(0, i - 15), i)]
                past = [x for x in past if len(x)]
                nov = max((iou(b, pb) for pb in np.concatenate(past)), default=0.0) if past else 0.0
                # When did the detector FIRST see any detection overlapping b?
                first_seen = None
                for j in range(0, i):
                    if any(iou(b, x) >= 0.3 for x in frames[j]["wboxes"]):
                        first_seen = frames[j]["ts"]
                        break
                found = {"appear_ts": round(frames[i]["ts"], 2), "delay": round(frames[i]["ts"] - t, 2),
                         "novelty_max_iou": round(nov, 3),
                         "first_seen_any": first_seen, "hits": hits}
                break
            if found:
                break
        rows.append({"t": t, "tile": tile, "who": g["who"], "found": found})

    (OUT / "exp_a_diag.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        f = r["found"]
        if f is None:
            print(f"t={r['t']:5.1f} {r['tile']:3s} {r['who']:6s}  落地后3秒内没有持续的同类river检测")
        else:
            print(f"t={r['t']:5.1f} {r['tile']:3s} {r['who']:6s}  出现@{f['appear_ts']:5.2f}s (延迟{f['delay']:+.2f}s) "
                  f"新颖度IoU={f['novelty_max_iou']:.2f} 该位置首次被检到={f['first_seen_any']} hits={f['hits']}")


if __name__ == "__main__":
    main()
