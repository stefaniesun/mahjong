"""Video-level metric computation (Phase 4 task 8).

Kept apart from the evaluation driver and free of IO so every number can be verified
against a hand-built example. Frame-level recall said nothing about whether the system
is usable over a video; these are the four things that do:

* **checkpoint accuracy** — at a frozen instant, does the confirmed world state match
  reality?
* **confirmation latency** — how long after a tile appears before the system commits?
* **flicker** — how often does a committed answer change? A display that keeps
  correcting itself is worse than a slightly less accurate one that holds still.
* **tracking continuity** — are ids stable, or is one physical tile being rediscovered?
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

SIZE_BUCKETS = ("lt20", "20to40", "gt40")


def size_bucket(width: float, height: float) -> str:
    short = min(width, height)
    if short < 20:
        return "lt20"
    if short <= 40:
        return "20to40"
    return "gt40"


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class CheckpointResult:
    matched: int = 0
    class_correct: int = 0
    missed: int = 0  # in GT, no confirmed tile
    spurious: int = 0  # confirmed, not in GT
    by_bucket: dict[str, dict[str, int]] = field(default_factory=lambda: {b: {"n": 0, "matched": 0, "class_correct": 0} for b in SIZE_BUCKETS})

    def merge(self, other: "CheckpointResult") -> None:
        self.matched += other.matched
        self.class_correct += other.class_correct
        self.missed += other.missed
        self.spurious += other.spurious
        for bucket, stats in other.by_bucket.items():
            for key, value in stats.items():
                self.by_bucket[bucket][key] += value

    def as_dict(self) -> dict[str, Any]:
        gt_total = self.matched + self.missed
        return {
            "gt_tiles": gt_total,
            "recall": round(self.matched / gt_total, 4) if gt_total else 0.0,
            "precision": round(self.matched / (self.matched + self.spurious), 4) if (self.matched + self.spurious) else 0.0,
            # The headline number: of everything really on the table, how much did the
            # system both find *and* name correctly?
            "class_accuracy": round(self.class_correct / gt_total, 4) if gt_total else 0.0,
            "matched": self.matched,
            "class_correct": self.class_correct,
            "missed": self.missed,
            "spurious": self.spurious,
            "by_bucket": {
                bucket: {
                    "n": stats["n"],
                    "recall": round(stats["matched"] / stats["n"], 4) if stats["n"] else 0.0,
                    "class_accuracy": round(stats["class_correct"] / stats["n"], 4) if stats["n"] else 0.0,
                }
                for bucket, stats in self.by_bucket.items()
            },
        }


def evaluate_checkpoint(
    gt: Sequence[dict[str, Any]],
    confirmed: Sequence[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> CheckpointResult:
    """gt / confirmed items: {"bbox": [x,y,w,h], "label": str}."""
    result = CheckpointResult()
    used: set[int] = set()
    for truth in gt:
        bucket = size_bucket(truth["bbox"][2], truth["bbox"][3])
        result.by_bucket[bucket]["n"] += 1
        best, best_iou = None, iou_threshold
        for index, tile in enumerate(confirmed):
            if index in used:
                continue
            value = iou_xywh(truth["bbox"], tile["bbox"])
            if value >= best_iou:
                best, best_iou = index, value
        if best is None:
            result.missed += 1
            continue
        used.add(best)
        result.matched += 1
        result.by_bucket[bucket]["matched"] += 1
        if confirmed[best].get("label") == truth.get("label"):
            result.class_correct += 1
            result.by_bucket[bucket]["class_correct"] += 1
    result.spurious = len(confirmed) - len(used)
    return result


def compute_latency(
    truth_events: Sequence[dict[str, Any]],
    confirmed_events: Sequence[dict[str, Any]],
    *,
    window: float = 5.0,
) -> dict[str, Any]:
    """Match each hand-timed appearance to the first confirmation of that tile after it.

    Only forward matches inside `window` count: a confirmation that fires *before* the
    tile was placed belongs to a different tile of the same kind, and one that fires
    much later is a miss rather than a slow success.
    """
    ordered = sorted(confirmed_events, key=lambda e: e["ts"])
    used: set[int] = set()
    latencies: list[float] = []
    unmatched: list[dict[str, Any]] = []
    for truth in sorted(truth_events, key=lambda e: e["t"]):
        found = None
        for index, event in enumerate(ordered):
            if index in used or event.get("label") != truth.get("tile"):
                continue
            delta = event["ts"] - truth["t"]
            if 0 <= delta <= window:
                found = (index, delta)
                break
        if found is None:
            unmatched.append(truth)
            continue
        used.add(found[0])
        latencies.append(found[1])

    latencies.sort()

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        position = (len(values) - 1) * q
        low, high = int(position), min(int(position) + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "n_events": len(truth_events),
        "n_matched": len(latencies),
        "n_missed": len(unmatched),
        "p50": round(percentile(latencies, 0.5), 3),
        "p95": round(percentile(latencies, 0.95), 3),
        "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "max": round(latencies[-1], 3) if latencies else 0.0,
        "unmatched": unmatched,
    }


def compute_flicker(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Class changes per confirmed track. The spec's "zero flicker" bar is <0.05/tile."""
    confirmed: set[int] = set()
    updates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        kind = event.get("type")
        if kind == "tile_confirmed":
            confirmed.add(int(event["track_id"]))
        elif kind == "tile_updated":
            updates[int(event["track_id"])].append(event)
    total_updates = sum(len(v) for v in updates.values())
    worst = sorted(updates.items(), key=lambda kv: -len(kv[1]))[:5]
    return {
        "confirmed_tracks": len(confirmed),
        "total_updates": total_updates,
        "updates_per_tile": round(total_updates / len(confirmed), 4) if confirmed else 0.0,
        "worst_tracks": [
            {
                "track_id": track_id,
                "changes": len(items),
                "sequence": [f"{i.get('previous_label')}->{i.get('label')}@{i.get('ts')}" for i in items],
            }
            for track_id, items in worst
        ],
    }


def estimate_track_continuity(checkpoints: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Approximate id churn from consecutive checkpoints.

    With only sparse ground truth we cannot compute true MOTA-style id switches, so this
    reports how often a tile at essentially the same place carries a new id — a proxy
    that catches the failure that matters (one physical tile counted repeatedly).
    """
    switches = 0
    comparisons = 0
    previous: list[dict[str, Any]] | None = None
    for checkpoint in checkpoints:
        current = checkpoint.get("confirmed", [])
        if previous is not None:
            for tile in current:
                partner = None
                best = 0.7
                for old in previous:
                    value = iou_xywh(tile["bbox"], old["bbox"])
                    if value >= best:
                        partner, best = old, value
                if partner is None:
                    continue
                comparisons += 1
                if partner.get("track_id") != tile.get("track_id"):
                    switches += 1
        previous = current
    return {
        "comparisons": comparisons,
        "id_switches": switches,
        "switch_rate": round(switches / comparisons, 4) if comparisons else 0.0,
    }


ACCEPTANCE = {
    "stable_accuracy": 0.995,
    "latency_p95": 0.5,
    "flicker_per_tile": 0.05,
    "fps": 30.0,
}


def check_acceptance(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The four red lines from the Phase 4 header, each as an explicit PASS/FAIL."""
    accuracy = summary.get("checkpoint", {}).get("class_accuracy", 0.0)
    latency = summary.get("latency", {}).get("p95", 999.0)
    flicker = summary.get("flicker", {}).get("updates_per_tile", 999.0)
    fps = summary.get("performance", {}).get("fps", 0.0)
    return [
        {"name": "稳定牌识别准确率", "value": accuracy, "target": f">={ACCEPTANCE['stable_accuracy']}", "pass": accuracy >= ACCEPTANCE["stable_accuracy"]},
        {"name": "确认延迟 p95 (秒)", "value": latency, "target": f"<={ACCEPTANCE['latency_p95']}", "pass": latency <= ACCEPTANCE["latency_p95"]},
        {"name": "闪烁率 (次/牌)", "value": flicker, "target": f"<{ACCEPTANCE['flicker_per_tile']}", "pass": flicker < ACCEPTANCE["flicker_per_tile"]},
        {"name": "吞吐 (fps)", "value": fps, "target": f">={ACCEPTANCE['fps']}", "pass": fps >= ACCEPTANCE["fps"]},
    ]
