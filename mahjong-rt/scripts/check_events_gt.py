"""Validate a hand-written event ground truth file.

Ground truth is the one thing nothing else can check. A typo here does not fail loudly —
it silently marks correct predictions wrong, and every number downstream inherits it.
This project has already paid that bill once: zone accuracy was reported at 92.5% when
the honest figure was 88.9%, because the labels were never audited.

So: check what can be checked mechanically, report, and change nothing.

    python scripts/check_events_gt.py
    python scripts/check_events_gt.py --file ../output/video_testset_pilot/events_gt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "output" / "video_testset_pilot" / "events_gt.json"

SEATS = ["me", "right", "across", "left"]      # counter-clockwise turn order
TYPES = {"discard", "pong", "kong", "hu", "dingque"}
TILES = {f"{suit}{n}" for suit in "wtb" for n in range(1, 10)}
CLAIMS = {"pong", "kong", "hu"}                 # take a tile from someone else


def check_clip(name: str, clip: dict) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warns: list[str] = []
    events = clip.get("events", [])
    duration = float(clip.get("duration", 30.0))

    for i, e in enumerate(events):
        where = f"{name} 第{i + 1}条 (t={e.get('t')})"
        t = e.get("t")
        if not isinstance(t, (int, float)):
            errors.append(f"{where}: t 不是数字")
        elif not (0 <= t <= duration + 0.5):
            errors.append(f"{where}: t={t} 超出片段范围 0~{duration}")
        if e.get("who") not in SEATS:
            errors.append(f"{where}: who={e.get('who')!r} 不是 {'/'.join(SEATS)}")
        if e.get("type") not in TYPES:
            errors.append(f"{where}: type={e.get('type')!r} 不是 {'/'.join(sorted(TYPES))}")
        tile = e.get("tile")
        if tile is not None and tile not in TILES:
            errors.append(f"{where}: tile={tile!r} 不是合法牌名 (w/t/b + 1~9，t 是条 b 是筒)")
        frm = e.get("from")
        if e.get("type") in CLAIMS:
            if frm is None:
                warns.append(f"{where}: {e.get('type')} 没填 from，轮次链会少一个锚点")
            elif frm not in SEATS:
                errors.append(f"{where}: from={frm!r} 不是 {'/'.join(SEATS)}")
            elif frm == e.get("who"):
                errors.append(f"{where}: from 和 who 都是 {frm}，不能碰/胡自己打的牌")
        elif frm not in (None, ""):
            warns.append(f"{where}: {e.get('type')} 不该有 from={frm!r}")

    ordered = sorted(range(len(events)), key=lambda i: events[i].get("t", 0))
    if ordered != list(range(len(events))):
        warns.append(f"{name}: 事件没有按时间排序（不影响使用，但自己检查时容易漏）")

    # Turn order. Discards go counter-clockwise; a pong/kong jumps the turn to the
    # claimer. Anything else means a missed event — usually a discard that was not seen.
    turn = None
    seq = sorted(events, key=lambda e: e.get("t", 0))
    for e in seq:
        who, kind = e.get("who"), e.get("type")
        if who not in SEATS:
            continue
        if kind == "discard":
            if turn is not None and who != turn:
                gap = (SEATS.index(who) - SEATS.index(turn)) % 4
                warns.append(
                    f"{name} t={e.get('t')}: 轮到 {turn} 却是 {who} 打牌，中间可能漏了 {gap} 条"
                )
            turn = SEATS[(SEATS.index(who) + 1) % 4]
        elif kind in {"pong", "kong"}:
            # The claimer discards next, so the turn lands on them rather than moving on.
            # This is the whole reason a pong is worth recording: it is the only action
            # that breaks the counter-clockwise order, and the only one whose owner is
            # directly visible, so it re-anchors the turn pointer whenever it has drifted.
            turn = who

    counts = Counter()
    for e in events:
        tile, kind = e.get("tile"), e.get("type")
        if tile in TILES:
            counts[tile] += 4 if kind == "kong" else 3 if kind == "pong" else 1
    for tile, n in counts.items():
        if n > 4:
            errors.append(f"{name}: {tile} 累计出现 {n} 次，每种牌只有 4 张")

    stats = {
        "events": len(events),
        "by_type": dict(Counter(e.get("type") for e in events)),
        "unreadable": sum(1 for e in events if e.get("tile") is None),
    }
    return errors, warns, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT)
    args = ap.parse_args(argv)

    data = json.loads(args.file.read_text(encoding="utf-8"))
    clips = data.get("clips", {})
    all_errors, all_warns, filled = [], [], 0

    for name, clip in clips.items():
        errors, warns, stats = check_clip(name, clip)
        all_errors += errors
        all_warns += warns
        if stats["events"]:
            filled += 1
            detail = "  ".join(f"{k} {v}" for k, v in sorted(stats["by_type"].items()))
            note = f"  (其中 {stats['unreadable']} 条没认出牌)" if stats["unreadable"] else ""
            print(f"{name}: {stats['events']} 条   {detail}{note}")
        else:
            print(f"{name}: 未填写")

    print()
    for w in all_warns:
        print(f"  提醒  {w}")
    for e in all_errors:
        print(f"  错误  {e}")

    if not filled:
        print("还没有任何事件。填写说明见 output/video_testset_pilot/事件真值怎么填.md")
        return 1
    if all_errors:
        print(f"\n{len(all_errors)} 处需要修正。文件未被改动。")
        return 1
    print(f"\n检查通过（{filled} 段已填写）。" + (f"{len(all_warns)} 条提醒可以忽略。" if all_warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
