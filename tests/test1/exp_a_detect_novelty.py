"""Experiment A: discard timing from detection-level novelty, tile identity from
post-landing classifier-posterior averaging.

Differs from game_events.py (cell occupancy on confirmed track labels) in two ways:

* Timing is keyed off RAW detections in world coordinates: a discard is a detection
  whose world box does not overlap anything the detector saw in the recent past
  (novelty) and which keeps being seen afterwards (persistence). No tracker, no voter,
  no 40px cells — a tile is 50-100px wide, so box-level IoU is the right granularity.
* Identity is read by averaging the stored 27-class posteriors of every detection that
  overlaps the landed box over the next ~1.5s. The per-frame classifier is 99.51%
  top-1, so averaging over ~30 clean views of a static tile should be near-perfect,
  independent of track fragmentation.

No model is trained or re-run; this consumes only recordings_full/*.npz.

    python tests/test1/exp_a_detect_novelty.py
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

TIME_TOL = 0.5  # same tolerance as scripts/eval_game_events.py

PARAMS = dict(
    warmup_frames=90,      # same warm-up as game_events.py
    lookback_frames=15,    # novelty: compare against detections of the past 0.5s
    iou_novel=0.10,        # max IoU vs recent past to count as "new object"
    iou_same=0.30,         # IoU to count as "same object" (persistence / occupied)
    persist_frames=10,     # confirm window: 0.33s
    persist_min=5,         # must be re-detected in at least this many of those frames
    min_gap_s=0.6,         # two discards cannot land in the same instant
    name_frames=45,        # average posteriors over 1.5s after landing
    iou_name=0.30,         # overlap with the event box to contribute a vote
)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: xyxy."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def to_world(box: np.ndarray, w: np.ndarray) -> np.ndarray:
    corners = np.array([
        [box[0], box[1], 1.0], [box[2], box[1], 1.0],
        [box[2], box[3], 1.0], [box[0], box[3], 1.0],
    ])
    pts = (w @ corners.T).T
    pts = pts[:, :2] / pts[:, 2:3]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def random_baseline(n_preds: int, times: list[float], trials: int = 4000, seed: int = 0):
    """Expected hits and empirical P(random >= observed) for the same number of
    randomly scattered events — copied from scripts/eval_game_events.py, extended
    with a p-value so 'above chance' claims carry their evidence."""
    span = max(max(times), 1.0)
    rng = np.random.RandomState(seed)
    hits = []
    for _ in range(trials):
        used: set[int] = set()
        for p in sorted(rng.uniform(0, span, n_preds)):
            near = [(abs(p - t), i) for i, t in enumerate(times) if i not in used and abs(p - t) <= TIME_TOL]
            if near:
                used.add(min(near)[1])
        hits.append(len(used))
    return float(np.mean(hits)), np.array(hits)


def run(params: dict) -> dict:
    rec = Recording.load(TESTSET / "recordings_full" / "clip01_7507945925261200.npz")
    cfg = yaml.safe_load((ROOT / "mahjong-rt" / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    zone_cfg = ZoneConfig(**cfg.get("zones", {}))

    # Per-frame: world transform, world boxes, river mask.
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
                       "probs": f.probs.astype(np.float64), "world": world.copy()})

    occupied: list[np.ndarray] = []   # confirmed arrivals (world boxes)
    pending: list[dict] = []          # candidates inside their confirmation window
    events: list[dict] = []
    last_ts = -99.0

    n = len(frames)
    for i, fr in enumerate(frames):
        river_idx = [k for k, z in enumerate(fr["zones"]) if z == "river"]
        # Resolve candidates whose confirmation window has closed.
        for cand in list(pending):
            if i - cand["start"] >= params["persist_frames"]:
                pending.remove(cand)
                if cand["hits"] >= params["persist_min"]:
                    if cand["ts"] - last_ts >= params["min_gap_s"]:
                        last_ts = cand["ts"]
                        events.append({"ts": cand["ts"], "frame": cand["start"], "box": cand["box"]})
                        occupied.append(cand["box"])
                    else:
                        occupied.append(cand["box"])   # one landing seen twice: still occupies
        warm = i < params["warmup_frames"]
        for k in river_idx:
            b = fr["wboxes"][k]
            if any(iou(b, ob) >= params["iou_same"] for ob in occupied):
                continue
            hit = None
            for cand in pending:
                if iou(b, cand["box"]) >= params["iou_same"]:
                    hit = cand
                    break
            if hit is not None:
                hit["hits"] += 1
                continue
            if warm:
                continue
            past = [frames[j]["wboxes"] for j in range(max(0, i - params["lookback_frames"]), i)]
            past = [x for x in past if len(x)]
            if past:
                past_all = np.concatenate(past)
                if max(iou(b, pb) for pb in past_all) >= params["iou_novel"]:
                    continue                     # the detector already saw this object
            pending.append({"start": i, "ts": fr["ts"], "box": b.copy(), "hits": 1})

    # Name each arrival by posterior averaging over the frames after it landed.
    for ev in events:
        votes = []
        for j in range(ev["frame"], min(n, ev["frame"] + params["name_frames"])):
            for k in range(len(frames[j]["wboxes"])):
                if iou(ev["box"], frames[j]["wboxes"][k]) >= params["iou_name"]:
                    votes.append(frames[j]["probs"][k])
        if votes:
            mean_p = np.mean(votes, axis=0)
            ev["tile"] = rec.classes[int(np.argmax(mean_p))]
            ev["conf"] = float(mean_p.max())
            ev["votes"] = len(votes)
        else:
            ev["tile"] = None
            ev["conf"] = 0.0
            ev["votes"] = 0

    # Score against the truth.
    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]
    times = [float(e["t"]) for e in gt]
    used: set[int] = set()
    matched = []
    for ev in sorted(events, key=lambda e: e["ts"]):
        near = [(abs(ev["ts"] - t), k) for k, t in enumerate(times) if k not in used and abs(ev["ts"] - t) <= TIME_TOL]
        if near:
            _, k = min(near)
            used.add(k)
            matched.append((ev, gt[k]))
    tile_ok = sum(1 for ev, g in matched if ev["tile"] == g["tile"])
    chance, dist = random_baseline(len(events), times)
    p_value = float((np.sum(dist >= len(matched)) + 1) / (len(dist) + 1))

    return {
        "params": params,
        "n_events": len(events),
        "n_truth": len(gt),
        "matched": len(matched),
        "missed": len(gt) - len(matched),
        "spurious": len(events) - len(matched),
        "random_expected": round(chance, 2),
        "above_random": round(len(matched) - chance, 2),
        "p_value": round(p_value, 4),
        "tile_ok": tile_ok,
        "tile_acc_on_matched": round(tile_ok / max(len(matched), 1), 3),
        "events": [{"ts": round(e["ts"], 2), "tile": e["tile"], "conf": round(e["conf"], 3),
                    "votes": e["votes"],
                    "matched_truth": next((g["t"] for ev, g in matched if ev is e), None)} for e in events],
        "truth_times": times,
    }


if __name__ == "__main__":
    result = run(PARAMS)
    out = OUT / "exp_a_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检出 {result['n_events']}  真值 {result['n_truth']}  命中 {result['matched']}  "
          f"随机期望 {result['random_expected']}  超出 {result['above_random']:+.2f}  p={result['p_value']}")
    print(f"牌认对(命中中) {result['tile_ok']}/{result['matched']} = {result['tile_acc_on_matched']:.1%}")
    for e in result["events"]:
        tag = f"-> 真值 {e['matched_truth']}s" if e["matched_truth"] is not None else "-> 误报/超时"
        print(f"  @{e['ts']:5.2f}s  {str(e['tile']):4s} conf={e['conf']:.2f} votes={e['votes']:3d}  {tag}")
