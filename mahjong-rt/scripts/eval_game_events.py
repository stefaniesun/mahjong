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
from mahjong_rt.recording import Recording
from mahjong_rt.replay import replay

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "output" / "video_testset_pilot"

GAP = -0.6      # cost of leaving an event unmatched; below the 0 a bad pair would score


def similarity(pred: dict, truth: dict) -> float:
    """How much of one event agrees with another. 1.0 is identical."""
    if pred["event_type"] != truth.get("type"):
        # Never pair different kinds of event. It has to be worse than two gaps, or the
        # aligner will happily marry a spurious pong to a missed discard and report both
        # as "matched but different".
        return -99.0
    score = 0.0
    if truth.get("tile") is None:
        score += 0.6                                  # unreadable in the video: no evidence either way
    elif pred["tile"] == truth["tile"]:
        score += 0.6
    if pred["player"] == truth.get("who"):
        score += 0.4
    return score


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
    ap.add_argument("--clip", type=str, default=None, help="Only this clip.")
    ap.add_argument("--verbose", action="store_true", help="Print the alignment.")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    truth_all = json.loads((args.testset / "events_gt.json").read_text(encoding="utf-8"))

    grand = {"matched": 0, "tile_ok": 0, "player_ok": 0, "missed": 0, "spurious": 0, "truth": 0}
    for name, clip in truth_all.get("clips", {}).items():
        truths = clip.get("events", [])
        if not truths or (args.clip and args.clip not in name):
            continue
        path = args.testset / args.recordings / f"{name}.npz"
        if not path.exists():
            print(f"{name}: 缺录制 {path.name}")
            continue

        recording = Recording.load(path)
        result = replay(recording, tracker_cfg=cfg.get("tracker"), voter_cfg=cfg.get("voter"),
                        state_cfg=cfg.get("state"), zones_cfg=cfg.get("zones"), checkpoints={})
        summaries = [e for e in result["events"] if e.get("type") == "frame_summary"]

        # The turn pointer has no anchor in a clip with no pong, so it is started from
        # the truth's first player and the tile sequence is scored independently. A live
        # system would anchor on the first claim instead; see the note printed below.
        extractor = GameEventExtractor(GameEventConfig(start_player=truths[0].get("who")))
        for summary, frame in zip(summaries, recording.frames):
            extractor.add_frame(summary, frame.homography, detections=frame.boxes)
        extractor.flush()
        preds = [e.to_dict() for e in extractor.events]

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
        print("\n注:归属是靠轮次推的,起点取自真值第一条。真实系统要靠第一次碰牌来定锚点——"
              "clip01 没有碰,所以它的归属分数偏乐观。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
