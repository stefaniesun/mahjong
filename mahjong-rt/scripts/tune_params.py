"""Offline parameter search over recordings (Phase 4 task 9).

Sweeps the post-model knobs — voting window, majority ratio, hysteresis, track buffer,
association gates, detector threshold — by replaying stored model output. One
configuration costs seconds instead of the hour a full video pass would take.

Two disciplines the spec insists on, both implemented here:

* **tune / holdout split.** Search runs on half the clips; the recommendation is then
  scored on the half it never saw. Picking the best of a few hundred configurations on
  the same data it was chosen from reliably overstates the gain.
* **Constraints, not a single blended score.** The objective maximises checkpoint
  accuracy subject to flicker staying under its target, because a configuration that
  buys accuracy by letting the display churn is not an improvement.

Recommendations are written to a candidates file. They are never applied to
pipeline.yaml automatically — that stays a human decision.

Example:
    python scripts/tune_params.py --testset ../output/video_testset_pilot \
        --recordings ../output/video_testset_pilot/recordings --out ../output/tune_run1
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_rt.metrics import CheckpointResult, compute_flicker, estimate_track_continuity, evaluate_checkpoint
from mahjong_rt.recording import Recording
from mahjong_rt.replay import replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep pipeline parameters over recordings.")
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"), help="Baseline config; swept keys override it.")
    parser.add_argument("--trials", type=int, default=200, help="Random configurations to try (0 = full grid).")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--flicker-max", type=float, default=0.05)
    parser.add_argument("--id-switch-max", type=float, default=0.28, help="Cap on id churn. Without it the search buys a low flicker score by letting tracks break: a rebuilt track starts voting afresh, so its disagreements never count as a change of mind. Downstream that reads as the same tile being discarded twice.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


# The knobs that actually move the metrics, with ranges around the current defaults.
SEARCH_SPACE: dict[str, list[Any]] = {
    "det_conf": [0.05, 0.10, 0.20, 0.30],
    "voter.window": [5, 7, 9, 11],
    "voter.min_effective": [2, 3, 4],
    "voter.majority_ratio": [0.5, 0.6, 0.7],
    "voter.hysteresis": [3, 4, 6, 8],
    "voter.min_conf": [0.4, 0.5, 0.6],
    "tracker.track_buffer": [15, 30, 45],
    "tracker.match_thresh": [0.6, 0.7, 0.8],
    "tracker.match_thresh_low": [0.3, 0.4, 0.5],
    "state.lost_after": [20, 30, 45],
}


def load_gt(testset: Path, file_name: str) -> list[dict[str, Any]]:
    path = testset / "checkpoints" / Path(file_name).with_suffix(".json").name
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for shape in payload.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        out.append({"bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "label": str(shape.get("label", ""))})
    return out


def unpack(config: dict[str, Any]) -> tuple[float, dict, dict, dict]:
    tracker = {k.split(".", 1)[1]: v for k, v in config.items() if k.startswith("tracker.")}
    voter = {k.split(".", 1)[1]: v for k, v in config.items() if k.startswith("voter.")}
    state = {k.split(".", 1)[1]: v for k, v in config.items() if k.startswith("state.")}
    return float(config.get("det_conf", 0.25)), tracker, voter, state


def evaluate(config: dict[str, Any], clips: Sequence[tuple[Recording, dict[int, str], dict[int, list]]], zones_cfg: dict, warmup: int) -> dict[str, Any]:
    det_conf, tracker_cfg, voter_cfg, state_cfg = unpack(config)
    total = CheckpointResult()
    events: list[dict[str, Any]] = []
    continuity: list[dict[str, Any]] = []

    for recording, wanted, gts in clips:
        # Detector threshold is applied here rather than at record time, so one
        # recording covers the whole range of thresholds worth trying.
        filtered = Recording(
            clip=recording.clip,
            classes=recording.classes,
            frame_width=recording.frame_width,
            frame_height=recording.frame_height,
            fps=recording.fps,
            stride=recording.stride,
        )
        for frame in recording.frames:
            keep = frame.scores >= det_conf
            filtered.frames.append(
                type(frame)(
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                    boxes=frame.boxes[keep],
                    scores=frame.scores[keep],
                    labels=frame.labels[keep],
                    confidences=frame.confidences[keep],
                    homography=frame.homography,
                )
            )

        result = replay(
            filtered,
            tracker_cfg=tracker_cfg,
            voter_cfg=voter_cfg,
            state_cfg=state_cfg,
            zones_cfg=zones_cfg,
            checkpoints=wanted,
        )
        events.extend(result["events"])
        for snapshot in result["snapshots"]:
            if snapshot["clip_frame"] < warmup:
                continue
            gt = gts.get(snapshot["clip_frame"])
            if not gt:
                continue
            total.merge(evaluate_checkpoint(gt, snapshot["confirmed"], iou_threshold=0.5))
            continuity.append(snapshot)

    checkpoint = total.as_dict()
    flicker = compute_flicker(events)
    tracking = estimate_track_continuity(continuity)
    return {
        "class_accuracy": checkpoint["class_accuracy"],
        "recall": checkpoint["recall"],
        "precision": checkpoint["precision"],
        "flicker": flicker["updates_per_tile"],
        "id_switch_rate": tracking["switch_rate"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import yaml

    manifest = json.loads((args.testset / "MANIFEST.json").read_text(encoding="utf-8"))
    base = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    zones_cfg = base.get("zones", {})

    loaded: list[tuple[Recording, dict[int, str], dict[int, list]]] = []
    for clip in manifest.get("clips", []):
        path = args.recordings / f"{clip['name']}.npz"
        if not path.exists():
            print(f"  缺录制: {path.name}，先跑 record_clips.py")
            continue
        recording = Recording.load(path)
        wanted = {int(c["clip_frame"]): c["file"] for c in clip.get("checkpoints", [])}
        gts = {int(c["clip_frame"]): load_gt(args.testset, c["file"]) for c in clip.get("checkpoints", [])}
        loaded.append((recording, wanted, gts))
    if len(loaded) < 2:
        print("至少需要 2 段录制才能做 tune/holdout 划分")
        return 1

    rng = random.Random(args.seed)
    order = list(range(len(loaded)))
    rng.shuffle(order)
    half = max(1, len(order) // 2)
    tune = [loaded[i] for i in order[:half]]
    holdout = [loaded[i] for i in order[half:]]
    print(f"片段划分: tune {len(tune)} 段 / holdout {len(holdout)} 段")

    keys = list(SEARCH_SPACE)
    if args.trials:
        configs = [{k: rng.choice(SEARCH_SPACE[k]) for k in keys} for _ in range(args.trials)]
    else:
        configs = [dict(zip(keys, values)) for values in itertools.product(*(SEARCH_SPACE[k] for k in keys))]
    print(f"配置数: {len(configs)}")

    results: list[dict[str, Any]] = []
    for i, config in enumerate(configs, start=1):
        metrics = evaluate(config, tune, zones_cfg, args.warmup_frames)
        results.append({"config": config, **metrics})
        if i % 20 == 0 or i == len(configs):
            best_so_far = max(
                (r for r in results if r["flicker"] < args.flicker_max and r["id_switch_rate"] <= args.id_switch_max),
                key=lambda r: r["class_accuracy"],
                default=None,
            )
            note = f"最佳(满足闪烁约束) {best_so_far['class_accuracy']*100:.2f}%" if best_so_far else "尚无满足闪烁约束的配置"
            print(f"  {i}/{len(configs)}  {note}", flush=True)

    feasible = [r for r in results if r["flicker"] < args.flicker_max and r["id_switch_rate"] <= args.id_switch_max]
    ranked = sorted(feasible or results, key=lambda r: -r["class_accuracy"])
    best = ranked[0]

    baseline_cfg = {
        "det_conf": base.get("detector", {}).get("conf", 0.25),
        **{f"voter.{k}": v for k, v in base.get("voter", {}).items() if f"voter.{k}" in SEARCH_SPACE},
        **{f"tracker.{k}": v for k, v in base.get("tracker", {}).items() if f"tracker.{k}" in SEARCH_SPACE},
        **{f"state.{k}": v for k, v in base.get("state", {}).items() if f"state.{k}" in SEARCH_SPACE},
    }
    holdout_best = evaluate(best["config"], holdout, zones_cfg, args.warmup_frames)
    holdout_base = evaluate(baseline_cfg, holdout, zones_cfg, args.warmup_frames)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "trials": len(configs),
        "feasible": len(feasible),
        "flicker_max": args.flicker_max,
        "id_switch_max": args.id_switch_max,
        "best_on_tune": best,
        "best_on_holdout": holdout_best,
        "baseline_on_holdout": holdout_base,
        "pareto": [
            {"config": r["config"], "class_accuracy": r["class_accuracy"], "flicker": r["flicker"]}
            for r in sorted(results, key=lambda r: (-r["class_accuracy"], r["flicker"]))[:15]
        ],
    }
    (args.out / "tune_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "candidate_params.yaml").write_text(
        "# 扫参推荐值。不会自动写入 pipeline.yaml——确认后手动启用。\n"
        + yaml.safe_dump(best["config"], allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\n可行配置(闪烁 < {args.flicker_max}): {len(feasible)}/{len(configs)}")
    print(f"\n{'':16s}{'类别准确':>10s}{'召回':>9s}{'闪烁':>9s}{'ID切换':>9s}")
    print(f"{'tune 最佳':16s}{best['class_accuracy']*100:9.2f}%{best['recall']*100:8.2f}%{best['flicker']:9.4f}{best['id_switch_rate']*100:8.1f}%")
    print(f"{'holdout 验证':16s}{holdout_best['class_accuracy']*100:9.2f}%{holdout_best['recall']*100:8.2f}%{holdout_best['flicker']:9.4f}{holdout_best['id_switch_rate']*100:8.1f}%")
    print(f"{'holdout 基线':16s}{holdout_base['class_accuracy']*100:9.2f}%{holdout_base['recall']*100:8.2f}%{holdout_base['flicker']:9.4f}{holdout_base['id_switch_rate']*100:8.1f}%")
    gain = holdout_best["class_accuracy"] - holdout_base["class_accuracy"]
    print(f"\nholdout 上相对基线: {gain*100:+.2f} 个点 " + ("(推荐采用)" if gain > 0 else "(不优于基线，不建议采用)"))
    print(f"推荐参数: {args.out/'candidate_params.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
