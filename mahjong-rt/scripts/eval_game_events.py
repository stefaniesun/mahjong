"""Score extracted game events against the hand-recorded truth.

The truth has no timestamps — recording them by scrubbing a video costs more than it
buys — so events are matched by sequence alignment rather than by time. Needleman-Wunsch
over the two lists, scoring a pair by how much of (tile, player, type) agrees. A missed
event then shifts nothing after it, which time-free matching by index would.

Reported separately, because they fail for different reasons:

* **tile sequence** — did the right tiles get discarded, in the right order? This is the
  vision question, and it is the one that has to work first.
* **attribution** — was each one credited to the right player? That is turn tracking,
  not vision; it can be perfect while the tiles are wrong, or the reverse.

    python scripts/eval_game_events.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.game_events import GameEventConfig, GameEventExtractor
from mahjong_rt.offline_game_events import (
    OfflineEventConfig,
    reconstruct_events,
    reconstruct_events_with_landings,
)
from mahjong_rt.raw_event_backtrack import BacktrackConfig, refine_events_with_raw_tracks
from mahjong_rt.recording import Recording
from mahjong_rt.replay import replay

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "output" / "video_testset_pilot"

GAP = -0.6       # cost of leaving an event unmatched; below the 0 a bad pair would score
# Seconds of slack before a timed truth event rejects a pair. It has to be well under
# the gap between events or the metric stops meaning anything: clip01's discards are
# 1-4s apart, and at 2.5s a random scatter of the same number of points scores 9 of 14.
# At 0.5s random scores 5.0, so there is room for a result to be above chance.
TIME_TOL = 0.5


def similarity(pred: dict, truth: dict) -> float:
    """Alignment score that never reads the tile or player being evaluated."""
    if pred["event_type"] != truth.get("type"):
        return -99.0
    truth_ts = truth.get("t")
    if not isinstance(truth_ts, (int, float)):
        return 1.0
    return 1.0 if abs(float(pred["ts"]) - float(truth_ts)) <= TIME_TOL else -99.0


def random_baseline(
    n_preds: int,
    truths: list[dict],
    duration_s: float,
    trials: int = 4000,
    seed: int = 0,
) -> float:
    """How many hits the same number of randomly scattered events would score.

    Without this the timing numbers flatter themselves: a handful of points thrown at a
    30-second clip lands near *something* most of the time. A detector has to beat this
    to have shown anything at all.
    """
    times = [float(t["t"]) for t in truths if isinstance(t.get("t"), (int, float))]
    if not times or not n_preds:
        return 0.0
    span = max(duration_s, 1.0)
    event_type = truths[0].get("type", "discard")
    rng = np.random.RandomState(seed)
    total = 0
    for _ in range(trials):
        random_preds = [
            {"event_type": event_type, "ts": float(ts), "tile": None, "player": "unknown"}
            for ts in sorted(rng.uniform(0, span, n_preds))
        ]
        total += sum(pi is not None and ti is not None for pi, ti in align(random_preds, truths))
    return total / trials


def align(preds: list[dict], truths: list[dict]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch. Returns (pred index or None, truth index or None) pairs."""
    n, m = len(preds), len(truths)
    table = np.zeros((n + 1, m + 1))
    table[:, 0] = np.arange(n + 1) * GAP
    table[0, :] = np.arange(m + 1) * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i, j] = max(
                table[i - 1, j - 1] + similarity(preds[i - 1], truths[j - 1]),
                table[i - 1, j] + GAP,
                table[i, j - 1] + GAP,
            )
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(table[i, j], table[i - 1, j - 1] + similarity(preds[i - 1], truths[j - 1])):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and np.isclose(table[i, j], table[i - 1, j] + GAP):
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return list(reversed(pairs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=TESTSET)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml")
    ap.add_argument("--recordings", type=str, default="recordings2", help="Recording folder under the testset.")
    ap.add_argument("--method", choices=("backtrack", "stable", "online"), default="stable",
                    help="Event reconstruction method (default: stable).")
    ap.add_argument("--clip", type=str, default=None, help="Only this clip.")
    ap.add_argument("--verbose", action="store_true", help="Print the alignment.")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    truth_all = json.loads((args.testset / "events_gt.json").read_text(encoding="utf-8"))

    grand = {"matched": 0, "tile_ok": 0, "player_ok": 0, "missed": 0, "spurious": 0, "truth": 0}
    for name, clip in truth_all.get("clips", {}).items():
        truths = clip.get("events", [])
        if args.clip and args.clip not in name:
            continue
        path = args.testset / args.recordings / f"{name}.npz"
        if not path.exists():
            print(f"{name}: 缺录制 {path.name}")
            continue

        recording = Recording.load(path)
        result = replay(recording, tracker_cfg=cfg.get("tracker"), voter_cfg=cfg.get("voter"),
                        state_cfg=cfg.get("state"), zones_cfg=cfg.get("zones"), checkpoints={})
        summaries = [e for e in result["events"] if e.get("type") == "frame_summary"]

        # Do not inject the truth's first player. Attribution remains unknown until an
        # observed claim or another non-oracle anchor is available.
        start_player = None
        if args.method == "backtrack":
            events, landings = reconstruct_events_with_landings(
                summaries,
                [frame.homography for frame in recording.frames],
                OfflineEventConfig(start_player=start_player),
            )
            events = refine_events_with_raw_tracks(
                recording,
                events,
                landings,
                BacktrackConfig(require_motion=False),
            )
        elif args.method == "stable":
            events = reconstruct_events(
                summaries,
                [frame.homography for frame in recording.frames],
                OfflineEventConfig(start_player=start_player),
            )
        else:
            extractor = GameEventExtractor(GameEventConfig(start_player=start_player))
            for summary, frame in zip(summaries, recording.frames):
                extractor.add_frame(summary, frame.homography, detections=frame.boxes)
            extractor.flush()
            events = extractor.events
        preds = [event.to_dict() for event in events]

        pairs = align(preds, truths)
        matched = tile_ok = player_ok = missed = spurious = 0
        rows = []
        for pi, ti in pairs:
            p = preds[pi] if pi is not None else None
            t = truths[ti] if ti is not None else None
            if p and t:
                matched += 1
                tile_ok += (t.get("tile") is None or p["tile"] == t["tile"])
                player_ok += (p["player"] == t.get("who"))
                mark = "OK " if p["tile"] == t.get("tile") and p["player"] == t.get("who") else "差异"
                rows.append(f"  {mark} 真值 {t.get('who'):6s} {t.get('type'):7s} {str(t.get('tile')):4s}"
                            f"   ->  识别 {p['player']:6s} {p['event_type']:7s} {str(p['tile']):4s}  @{p['ts']:.1f}s")
            elif t:
                missed += 1
                rows.append(f"  漏检 真值 {t.get('who'):6s} {t.get('type'):7s} {str(t.get('tile')):4s}")
            else:
                spurious += 1
                rows.append(f"  误报                              ->  识别 {p['player']:6s} {p['event_type']:7s} {str(p['tile']):4s}  @{p['ts']:.1f}s")

        print(f"\n=== {name} ===")
        print(f"真值 {len(truths)} 条,识别 {len(preds)} 条")
        print(f"  匹配上 {matched}   漏检 {missed}   误报 {spurious}")
        if any(isinstance(t.get("t"), (int, float)) for t in truths):
            duration_s = recording.frames[-1].timestamp if recording.frames else 0.0
            chance = random_baseline(len(preds), truths, duration_s)
            print(f"  时间对齐容差 ±{TIME_TOL}s;同样数量的随机事件平均能命中 {chance:.1f}"
                  f"  ->  超出随机 {matched - chance:+.1f}")
        if matched:
            print(f"  匹配事件里牌认对 {tile_ok}/{matched} = {tile_ok / matched:.1%}")
            print(f"  匹配事件里归属对 {player_ok}/{matched} = {player_ok / matched:.1%}")
        print(f"  端到端(牌+归属都对 / 真值总数) = {sum(1 for pi, ti in pairs if pi is not None and ti is not None and preds[pi]['tile'] == truths[ti].get('tile') and preds[pi]['player'] == truths[ti].get('who')) / max(len(truths), 1):.1%}")
        if args.verbose:
            print("\n".join(rows))

        for key, value in (("matched", matched), ("tile_ok", tile_ok), ("player_ok", player_ok),
                           ("missed", missed), ("spurious", spurious), ("truth", len(truths))):
            grand[key] += value

    if grand["truth"]:
        print(f"\n=== 合计 ===")
        print(f"真值 {grand['truth']} 条  匹配 {grand['matched']}  漏检 {grand['missed']}  误报 {grand['spurious']}")
        if grand["matched"]:
            print(f"牌认对 {grand['tile_ok'] / grand['matched']:.1%}   归属对 {grand['player_ok'] / grand['matched']:.1%}")
        print("\n注:评测不注入真值起始玩家；没有碰牌等视觉锚点时，归属输出为 unknown。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
