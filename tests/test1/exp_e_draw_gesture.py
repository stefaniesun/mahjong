"""Experiment E: the denominator test for the draw-gesture idea.

The user's hypothesis: the DRAW action (hand to the wall, grasp, retract) has a
distinctive gesture trajectory, unlike the discard. Before annotating any ground
truth, run the gate the doc's 12 failed observables all failed: count how many
times the signature fires in 30s. ~14 draws actually happened. If the signature
fires ~10-30 times it is sparse enough to be worth annotating against; if it
fires 60+ times the denominator kills it no matter the precision (doc section 5).

Signature: a tracked hand (a) enters the wall zone, (b) dwells there >= 3 frames
(speed drops), (c) leaves back toward its own side's home direction.

    python tests/test1/exp_e_draw_gesture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HANDS = ROOT / "output" / "video_testset_pilot" / "hands" / "clip01_7507945925261200.json"

# Wall zone in 1280x720 clip pixels, estimated from frames/fix strips:
# top strip = across wall, left strip = left wall, right edge = right wall,
# bottom strip = near wall (me). Generous on purpose; tightening only adds firings.
WALL_RECTS = [
    (0, 0, 1280, 130),      # top: across wall
    (0, 0, 460, 300),       # left: left wall
    (1060, 0, 1280, 420),   # right: right wall
    (0, 600, 1280, 720),    # bottom: near wall
]
DWELL_FRAMES = 3
MAX_MISS = 5          # frames a track may vanish before it is dropped
NMS_S = 1.0


def in_wall(x, y):
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in WALL_RECTS)


def main():
    frames = json.load(open(HANDS, encoding="utf-8"))

    # --- track hands across frames: greedy nearest-neighbour within 120px, same side
    tracks: list[dict] = []   # {"side", "pts": {frame: (x,y)}, "last"}
    events = []
    for f in frames:
        i, ts = f["frame"], f["ts"]
        used = set()
        for h in f["hands"]:
            x, y = h["cx"], h["cy"]
            best, best_d = None, 120.0
            for tr in tracks:
                if tr["side"] != h["side"] or i - tr["last"] > MAX_MISS or tr["id"] in used:
                    continue
                px, py = tr["pts"][tr["last"]]
                d = float(np.hypot(x - px, y - py))
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = {"id": len(tracks), "side": h["side"], "pts": {}, "last": i,
                        "in_wall": 0, "was_event": False}
                tracks.append(best)
            used.add(best["id"])
            best["pts"][i] = (x, y)
            best["last"] = i
            # --- signature state machine
            if in_wall(x, y):
                best["in_wall"] += 1
            else:
                if best["in_wall"] >= DWELL_FRAMES and not best["was_event"]:
                    events.append({"ts": ts, "frame": i, "side": best["side"],
                                   "dwell": best["in_wall"]})
                    best["was_event"] = True
                if best["in_wall"] > 0 and best["in_wall"] < DWELL_FRAMES:
                    pass  # brushed past the wall zone: not a grasp
                best["in_wall"] = 0
            # re-arm once the hand is clearly home again (left wall zone brushing
            # must not block the next draw)
            if best["was_event"] and best["in_wall"] == 0:
                best["was_event"] = False

    events.sort(key=lambda e: e["ts"])
    kept: list[dict] = []
    for e in events:
        if kept and e["ts"] - kept[-1]["ts"] < NMS_S:
            if e["dwell"] > kept[-1]["dwell"]:
                kept[-1] = e
            continue
        kept.append(e)

    (OUT / "exp_e_result.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"轨迹 {len(tracks)} 条  原始触发 {len(events)}  NMS后 {len(kept)}  (30秒内真实摸牌约14次)")
    for e in kept:
        print(f"  @{e['ts']:5.2f}s  side={e['side']:6s} dwell={e['dwell']}帧")


if __name__ == "__main__":
    main()
