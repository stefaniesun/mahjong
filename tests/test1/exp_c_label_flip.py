"""Experiment C: a discard is a LABEL FLIP at a fixed world location.

Experiment A's diagnosis showed why box-level novelty fails: a tile tossed onto the
pile lands where the detector already has a box (IoU ~1.0 with a pre-existing one).
But the box at that spot was showing a DIFFERENT tile face before the landing. The
per-frame classifier is 99.51% accurate, so a sustained label change at a fixed,
world-anchored location is a high-precision event signal that never looks at hands.

Also evaluated twice: against the raw truth timestamps, and against truth shifted
-1.0s, because frame strips (frames/pool_t*.jpg) show the tile visibly landing
~0.8-1.3s BEFORE the oral-annotation timestamp (speech reaction lag).

    python tests/test1/exp_c_label_flip.py
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

TIME_TOL = 0.5
LAG = 1.0  # annotation lag measured visually on pool_t12/18/29 strips

PARAMS = dict(
    warmup_frames=90,
    iou_site=0.5,       # detection -> same world-anchored site
    max_gap=10,         # carry a site's label across occluded frames up to this gap
    min_run=5,          # frames of stable label required on each side of a flip
    min_gap_s=0.6,      # dedup: two events cannot be closer than this
    name_frames=30,     # posterior averaging window after the flip
)


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


def runs_from_history(hist: dict[int, str], n_frames: int, max_gap: int, min_run: int):
    """Compress {frame: label} into (start, end, label) runs, bridging short gaps."""
    if not hist:
        return []
    frames = sorted(hist)
    runs = []
    start = frames[0]
    prev = frames[0]
    label = hist[frames[0]]
    for f in frames[1:]:
        if hist[f] == label and f - prev <= max_gap:
            prev = f
            continue
        runs.append([start, prev, label])
        start = prev = f
        label = hist[f]
    runs.append([start, prev, label])
    # Merge away runs shorter than min_run by absorbing them into the neighbour
    # with the longer flank; iterate until stable.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, r in enumerate(runs):
            if r[1] - r[0] + 1 < min_run:
                left_len = runs[k - 1][1] - runs[k - 1][0] if k > 0 else -1
                right_len = runs[k + 1][1] - runs[k + 1][0] if k < len(runs) - 1 else -1
                if left_len >= right_len and k > 0:
                    runs[k - 1][1] = r[1]
                    del runs[k]
                else:
                    runs[k + 1][0] = r[0]
                    del runs[k]
                changed = True
                break
    return [(s, e, l) for s, e, l in runs]


def random_baseline(n_preds, times, tol, trials=4000, seed=0):
    span = max(max(times), 1.0)
    rng = np.random.RandomState(seed)
    hits = []
    for _ in range(trials):
        used = set()
        for p in sorted(rng.uniform(0, span, n_preds)):
            near = [(abs(p - t), i) for i, t in enumerate(times) if i not in used and abs(p - t) <= tol]
            if near:
                used.add(min(near)[1])
        hits.append(len(used))
    return np.array(hits)


def score(events, times, tol):
    used = set()
    matched = []
    for ev in sorted(events, key=lambda e: e["ts"]):
        near = [(abs(ev["ts"] - t), k) for k, t in enumerate(times)
                if k not in used and abs(ev["ts"] - t) <= tol]
        if near:
            _, k = min(near)
            used.add(k)
            matched.append((ev, k))
    dist = random_baseline(len(events), times, tol)
    pval = float((np.sum(dist >= len(matched)) + 1) / (len(dist) + 1))
    return matched, float(dist.mean()), pval


def run(params):
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
        frames.append({"ts": f.timestamp, "wboxes": wboxes, "zones": zones,
                       "labels": labels, "probs": f.probs.astype(np.float64)})

    n = len(frames)
    sites: list[dict] = []   # {"box", "hist": {frame: label}, "dets": {frame: det_index}}
    for i, fr in enumerate(frames):
        claimed = set()
        for k in range(len(fr["wboxes"])):
            if fr["zones"][k] != "river":
                continue
            b = fr["wboxes"][k]
            best, best_iou = None, params["iou_site"]
            for s in sites:
                if s["last_frame"] < i - params["max_gap"]:
                    continue
                v = iou(b, s["box"])
                if v >= best_iou and s["id"] not in claimed:
                    best, best_iou = s, v
            if best is None:
                best = {"id": len(sites), "box": b.copy(), "hist": {}, "dets": {}, "born": i, "last_frame": i}
                sites.append(best)
            claimed.add(best["id"])
            # EMA-update the site box toward the observation (slow, so a landing
            # that shifts the box slightly does not spawn a new site).
            best["box"] = 0.9 * best["box"] + 0.1 * b
            best["hist"][i] = fr["labels"][k]
            best["dets"][i] = k
            best["last_frame"] = i

    # Extract flip / late-appearance events.
    events = []
    for s in sites:
        runs = runs_from_history(s["hist"], n, params["max_gap"], params["min_run"])
        if not runs:
            continue
        if s["born"] >= params["warmup_frames"] and runs[0][1] - runs[0][0] + 1 >= params["min_run"]:
            events.append({"kind": "appear", "frame": runs[0][0], "ts": frames[runs[0][0]]["ts"],
                           "tile": runs[0][2], "site": s})
        for (s0, e0, l0), (s1, e1, l1) in zip(runs, runs[1:]):
            if l0 == l1:
                continue
            if e0 < params["warmup_frames"] and s1 < params["warmup_frames"]:
                continue
            events.append({"kind": f"flip:{l0}->{l1}", "frame": s1, "ts": frames[s1]["ts"],
                           "tile": l1, "site": s})
    # Dedup by time (same landing can flip several neighbouring sites).
    events.sort(key=lambda e: e["ts"])
    kept: list[dict] = []
    for ev in events:
        if kept and ev["ts"] - kept[-1]["ts"] < params["min_gap_s"]:
            continue
        kept.append(ev)

    # Name by posterior averaging over the frames after the event.
    for ev in kept:
        s = ev["site"]
        votes = []
        for j in range(ev["frame"], min(n, ev["frame"] + params["name_frames"])):
            k = s["dets"].get(j)
            if k is not None:
                votes.append(frames[j]["probs"][k])
        if votes:
            mean_p = np.mean(votes, axis=0)
            ev["tile_named"] = rec.classes[int(np.argmax(mean_p))]
            ev["conf"] = float(mean_p.max())
        else:
            ev["tile_named"] = ev["tile"]
            ev["conf"] = 0.0

    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]
    times_raw = [float(e["t"]) for e in gt]
    times_lag = [t - LAG for t in times_raw]

    report = {"params": params, "n_sites": len(sites), "n_events": len(kept), "events": []}
    for tag, times in (("raw", times_raw), (f"lag-{LAG}s", times_lag)):
        matched, chance, pval = score(kept, times, TIME_TOL)
        tile_ok = sum(1 for ev, k in matched if ev["tile_named"] == gt[k]["tile"])
        report[tag] = {"matched": len(matched), "random_expected": round(chance, 2),
                       "above_random": round(len(matched) - chance, 2), "p_value": round(pval, 4),
                       "tile_ok": tile_ok, "tile_acc": round(tile_ok / max(len(matched), 1), 3)}
        for ev, k in matched:
            ev.setdefault("matched", {})[tag] = gt[k]["t"]

    for ev in kept:
        report["events"].append({
            "ts": round(ev["ts"], 2), "kind": ev["kind"], "tile": ev["tile_named"],
            "conf": round(ev["conf"], 3), "matched": ev.get("matched", {}),
        })
    return report


if __name__ == "__main__":
    rep = run(PARAMS)
    (OUT / "exp_c_result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"站点 {rep['n_sites']}  检出事件 {rep['n_events']}  真值 14")
    for tag in ("raw", "lag-1.0s"):
        r = rep[tag]
        print(f"[{tag}] 命中 {r['matched']}  随机期望 {r['random_expected']}  超出 {r['above_random']:+.2f}  "
              f"p={r['p_value']}  牌认对 {r['tile_ok']}/{r['matched']} = {r['tile_acc']:.1%}")
    for e in rep["events"]:
        m = e["matched"]
        tag = f"raw->{m.get('raw')}" if "raw" in m else (f"lag->{m.get('lag-1.0s')}" if "lag-1.0s" in m else "未命中")
        print(f"  @{e['ts']:5.2f}s  {e['kind']:14s} {str(e['tile']):4s} conf={e['conf']:.2f}  {tag}")
