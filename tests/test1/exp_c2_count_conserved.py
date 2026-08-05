"""Experiment C2: count conservation — a discard is a NET +1 in live river sites.

Experiment C showed each landing fires a *cascade* of 2-3 site events: the new tile
lands on the pile, the detector re-segments the neighbourhood, old sites lose their
match (die) and new ones spawn (born). A rename is birth+death together; a real
discard is birth WITHOUT a compensating death. So: keep a new site only if no nearby
site died within the same window.

    python tests/test1/exp_c2_count_conserved.py
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
LAG = 1.0

PARAMS = dict(
    warmup_frames=90,
    iou_site=0.5,
    max_gap=10,          # site is declared dead after this many unmatched frames
    min_run=5,           # new site must persist this many frames to count at all
    death_window=15,     # a death within +/-this many frames cancels a birth
    death_iou=0.15,      # ...if the dead site overlapped the newborn this much
    min_gap_s=1.2,       # a landing cascade spans ~1s; the detectable truth set
                         # (t>=5s, earlier ones are inside warm-up) is >=2s apart
    keep="earlier",      # earlier = the landing itself; later = the re-segmentation echo
    name_lo=15,          # name from settled frames only: the first ~0.5s the tile is
    name_frames=45,      # still tumbling / finger-occluded and poisons the average
    iou_name=0.3,
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


def score(events, gt, times, tol):
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
    tile_ok = sum(1 for ev, k in matched if ev["tile"] == gt[k]["tile"])
    return matched, float(dist.mean()), pval, tile_ok


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

    sites: list[dict] = []
    for i, fr in enumerate(frames):
        claimed = set()
        for k in range(len(fr["wboxes"])):
            if fr["zones"][k] != "river":
                continue
            b = fr["wboxes"][k]
            best, best_iou = None, params["iou_site"]
            for s in sites:
                if s["last_frame"] < i - params["max_gap"] or s["id"] in claimed:
                    continue
                v = iou(b, s["box"])
                if v >= best_iou:
                    best, best_iou = s, v
            if best is None:
                best = {"id": len(sites), "box": b.copy(), "born": i, "last_frame": i,
                        "dets": {}, "hits": 0}
                sites.append(best)
            claimed.add(best["id"])
            best["box"] = 0.9 * best["box"] + 0.1 * b
            best["dets"][i] = k
            best["last_frame"] = i
            best["hits"] += 1

    # Births after warmup, requiring the site to persist.
    births = [s for s in sites if s["born"] >= params["warmup_frames"] and s["hits"] >= params["min_run"]]
    # Deaths: last frame a site was seen (only meaningful for sites seen enough).
    deaths = [{"frame": s["last_frame"], "box": s["box"]} for s in sites if s["hits"] >= params["min_run"]]

    events = []
    for s in births:
        renamed = any(abs(d["frame"] - s["born"]) <= params["death_window"]
                      and iou(d["box"], s["box"]) >= params["death_iou"]
                      and d["frame"] > params["warmup_frames"]
                      for d in deaths)
        if renamed:
            continue
        events.append({"frame": s["born"], "ts": frames[s["born"]]["ts"], "site": s})

    events.sort(key=lambda e: e["ts"])
    kept: list[dict] = []
    for ev in events:
        if kept and ev["ts"] - kept[-1]["ts"] < params["min_gap_s"]:
            if params["keep"] == "later":
                kept[-1] = ev
            continue
        kept.append(ev)

    # Name: posterior average over ALL river detections overlapping the site box,
    # but only over the settled window (skip the tumbling / finger-occluded start).
    for ev in kept:
        b = ev["site"]["box"]
        votes = []
        for j in range(ev["frame"] + params["name_lo"], min(n, ev["frame"] + params["name_frames"])):
            for k in range(len(frames[j]["wboxes"])):
                if frames[j]["zones"][k] == "river" and iou(b, frames[j]["wboxes"][k]) >= params["iou_name"]:
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

    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]
    times_raw = [float(e["t"]) for e in gt]
    times_lag = [t - LAG for t in times_raw]

    report = {"params": params, "n_sites": len(sites), "n_births": len(births),
              "n_events": len(kept), "events": []}
    for tag, times in (("raw", times_raw), (f"lag-{LAG}s", times_lag)):
        matched, chance, pval, tile_ok = score(kept, gt, times, TIME_TOL)
        report[tag] = {"matched": len(matched), "random_expected": round(chance, 2),
                       "above_random": round(len(matched) - chance, 2), "p_value": round(pval, 4),
                       "tile_ok": tile_ok, "tile_acc": round(tile_ok / max(len(matched), 1), 3)}
        for ev, k in matched:
            ev.setdefault("matched", {})[tag] = gt[k]["t"]
    for ev in kept:
        report["events"].append({"ts": round(ev["ts"], 2), "tile": ev["tile"],
                                 "conf": round(ev["conf"], 3), "votes": ev["votes"],
                                 "matched": ev.get("matched", {})})
    return report


if __name__ == "__main__":
    results = {}
    for keep in ("earlier", "later"):
        params = dict(PARAMS, keep=keep)
        rep = run(params)
        results[keep] = rep
        (OUT / f"exp_c2_result_{keep}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n== keep={keep} ==  站点 {rep['n_sites']}  出生 {rep['n_births']}  事件 {rep['n_events']}  真值 14")
        for tag in ("raw", "lag-1.0s"):
            r = rep[tag]
            print(f"[{tag}] 命中 {r['matched']}  随机期望 {r['random_expected']}  超出 {r['above_random']:+.2f}  "
                  f"p={r['p_value']}  牌认对 {r['tile_ok']}/{r['matched']} = {r['tile_acc']:.1%}")
        for e in rep["events"]:
            m = e["matched"]
            tag = f"raw->真值{m['raw']}" if "raw" in m else (f"lag->真值{m['lag-1.0s']}" if "lag-1.0s" in m else "未命中")
            print(f"  @{e['ts']:5.2f}s  {str(e['tile']):4s} conf={e['conf']:.2f} votes={e['votes']:3d}  {tag}")
