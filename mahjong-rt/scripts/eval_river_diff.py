"""Recover discards by diffing the pool's contents between windows.

Thirteen earlier attempts asked "which frame did a tile land on" and all of them scored
at chance. This asks a different question — "what is in the pool now that was not in it
a second ago" — and it works: 11 of 14 on clip01, 6 of 8 on clip02, against a chance
level near 1, p < 0.0001 on both.

Three things changed, and each matters:

**The measurement is a difference between two settled states, not an instant.** A tile
landing is a fast, occluded, badly-detected event. A tile *sitting* in the pool is the
thing this system already reads at 99.5%. Within each window every track votes on its
own label, which removes most of the per-frame churn that swamped the earlier attempts —
the pool's confirmed count changes 103 times over clip01 against 14 real discards.

**The match key is the tile, not the time.** Twenty-seven classes make a coincidental
match unlikely, where a coincidental time match is nearly free at this event density.
The null here shuffles the *labels* of the detections while keeping their times, which
tests exactly the claim being made: that the identity carries information.

**A systematic lag is allowed.** Confirmation takes time, so detections trail the truth —
by about a second on clip01 and by almost nothing on clip02. Demanding +/-0.5s alignment,
as the earlier evaluation did, marked correct answers wrong.

What is not solved: precision. 65 detections carry the 11 true ones on clip01. Note the
figure is a lower bound where the truth is incomplete — clip02's truth has no discards
at all from the left seat, and the turn-order check flags three gaps.

    python scripts/eval_river_diff.py
    python scripts/eval_river_diff.py --clip clip02 --recordings recordings2 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.recording import Recording
from mahjong_rt.replay import replay

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "output" / "video_testset_pilot"
TILES = [f"{suit}{n}" for suit in "wtb" for n in range(1, 10)]


def detect(summaries, window: float, presence: float) -> list[tuple[float, str]]:
    """Tiles the pool gained, window by window.

    Each track votes on its own label within the window and must be present for a
    fraction of it, which is what turns a jittery per-frame state into a stable one.
    """
    previous: Counter | None = None
    found: list[tuple[float, str]] = []
    last = max((e["ts"] for e in summaries), default=0.0)
    for start in np.arange(0.0, last + window, window):
        votes: dict[int, Counter] = defaultdict(Counter)
        frames = 0
        for event in summaries:
            if not (start <= event["ts"] < start + window):
                continue
            frames += 1
            for tile in event["tiles"]:
                if tile["zone"] == "river" and tile.get("label"):
                    votes[tile["track_id"]][tile["label"]] += 1
        current: Counter = Counter()
        for counts in votes.values():
            label, seen = counts.most_common(1)[0]
            if seen >= presence * max(frames, 1):
                current[label] += 1
        if previous is not None:
            for label, gained in (current - previous).items():
                found.extend([(float(start), label)] * gained)
        previous = current
    return found


def match(found, truth, lag_lo: float, lag_hi: float):
    """Pair a truth event with a detection of the same tile inside the lag window."""
    used: set[int] = set()
    pairs = []
    for t_truth, tile in truth:
        for i, (t_found, label) in enumerate(found):
            if i in used or label != tile:
                continue
            if lag_lo <= t_found - t_truth <= lag_hi:
                used.add(i)
                pairs.append((t_truth, tile, t_found))
                break
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=TESTSET)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml")
    ap.add_argument("--recordings", default="recordings2")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--presence", type=float, default=0.15, help="Fraction of the window a track must appear in.")
    ap.add_argument("--lag", type=float, nargs=2, default=(0.0, 2.5), metavar=("LO", "HI"))
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    truth_all = json.loads((args.testset / "events_gt.json").read_text(encoding="utf-8"))
    rng = np.random.RandomState(0)

    print(f"窗口 {args.window}s  存在率 {args.presence}  延迟窗口 [{args.lag[0]:+.1f},{args.lag[1]:+.1f}]s\n")
    print(f"{'片段':>8} {'真值':>5} {'检出':>5} {'命中':>9} {'随机':>6} {'p':>9} {'精确率':>7}")
    for name, clip in truth_all.get("clips", {}).items():
        events = [e for e in clip.get("events", []) if e.get("type") == "discard" and e.get("tile")]
        if not events or (args.clip and args.clip not in name):
            continue
        path = args.testset / args.recordings / f"{name}.npz"
        if not path.exists():
            print(f"{name[:6]:>8}  缺录制 {path.name}")
            continue

        recording = Recording.load(path)
        result = replay(recording, tracker_cfg=cfg.get("tracker"), voter_cfg=cfg.get("voter"),
                        state_cfg=cfg.get("state"), zones_cfg=cfg.get("zones"), checkpoints={})
        summaries = [e for e in result["events"] if e.get("type") == "frame_summary"]

        found = detect(summaries, args.window, args.presence)
        truth = [(float(e["t"]), e["tile"]) for e in events]
        pairs = match(found, truth, *args.lag)

        # Null: keep every detection's time, replace its label at random. This asks
        # whether the identity carries information, which is the actual claim.
        null = [len(match([(t, TILES[rng.randint(27)]) for t, _ in found], truth, *args.lag))
                for _ in range(args.trials)]
        chance = float(np.mean(null))
        p = float(np.mean(np.asarray(null) >= len(pairs)))
        precision = len(pairs) / max(len(found), 1)
        print(f"{name[:6]:>8} {len(truth):>5} {len(found):>5} {len(pairs):>4}/{len(truth):<4} "
              f"{chance:>6.1f} {p:>9.4f} {precision:>6.0%}")
        if args.verbose:
            for t_truth, tile, t_found in pairs:
                print(f"         {t_truth:5.1f}s {tile}  ->  {t_found:5.1f}s  (延迟 {t_found - t_truth:+.1f}s)")
            missed = {t for t, _ in truth} - {t for t, _, _ in pairs}
            if missed:
                print(f"         漏检: " + "  ".join(f"{t:.0f}s" for t in sorted(missed)))

    print("\n注:精确率是下界 —— 真值不完整时,正确的检出会被算成误报。"
          "clip02 的真值里上家一次打牌都没有,轮次检查也标出 3 处缺口。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
