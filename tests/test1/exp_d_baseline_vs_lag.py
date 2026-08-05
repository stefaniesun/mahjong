"""Experiment D: re-score the OFFICIAL baseline against lag-corrected truth.

Hypothesis: the truth timestamps carry ~1s of oral-annotation lag (visually
confirmed on pool_t12/18/29.jpg). The official extractor fires ~1s late itself
(settle frames + voter + track confirmation), so the two lags cancel and it
scores 7/13 within +/-0.5s. If the lag story is right, shifting the truth by
-1s should CHANGE the score in a diagnostic direction; and more importantly any
future fast detector must be evaluated against corrected truth.

    python tests/test1/exp_d_baseline_vs_lag.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mahjong-rt"))
sys.path.insert(0, str(ROOT / "mahjong-rt" / "scripts"))

from mahjong_rt.game_events import GameEventConfig, GameEventExtractor
from mahjong_rt.recording import Recording
from mahjong_rt.replay import replay

TESTSET = ROOT / "output" / "video_testset_pilot"
OUT = Path(__file__).resolve().parent
TIME_TOL = 0.5


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


def main():
    cfg = yaml.safe_load((ROOT / "mahjong-rt" / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    rec = Recording.load(TESTSET / "recordings_full" / "clip01_7507945925261200.npz")
    result = replay(rec, tracker_cfg=cfg.get("tracker"), voter_cfg=cfg.get("voter"),
                    state_cfg=cfg.get("state"), zones_cfg=cfg.get("zones"), checkpoints={})
    summaries = [e for e in result["events"] if e.get("type") == "frame_summary"]

    truth = json.loads((TESTSET / "events_gt.json").read_text(encoding="utf-8"))
    gt = truth["clips"]["clip01_7507945925261200"]["events"]

    extractor = GameEventExtractor(GameEventConfig(start_player=gt[0].get("who")))
    for summary, frame in zip(summaries, rec.frames):
        extractor.add_frame(summary, frame.homography, detections=frame.boxes)
    extractor.flush()
    preds = [e.to_dict() for e in extractor.events]

    times = [float(e["t"]) for e in gt]
    report = {"pred_ts": [round(p["ts"], 2) for p in preds], "truth_ts": times, "shifts": {}}
    print(f"基线检出 {len(preds)} 条: {[round(p['ts'],2) for p in preds]}")
    print(f"真值 {len(times)} 条: {times}")
    for shift in (0.0, -0.5, -1.0, -1.5):
        shifted = [t + shift for t in times]
        used = set()
        hits = 0
        tile_ok = 0
        for p in sorted(preds, key=lambda x: x["ts"]):
            near = [(abs(p["ts"] - t), k) for k, t in enumerate(shifted)
                    if k not in used and abs(p["ts"] - t) <= TIME_TOL]
            if near:
                _, k = min(near)
                used.add(k)
                hits += 1
                tile_ok += (p["tile"] == gt[k]["tile"])
        dist = random_baseline(len(preds), shifted, TIME_TOL)
        pval = float((np.sum(dist >= hits) + 1) / (len(dist) + 1))
        report["shifts"][str(shift)] = {"hits": hits, "random": round(float(dist.mean()), 2),
                                        "p": round(pval, 4), "tile_ok": tile_ok}
        print(f"真值平移 {shift:+.1f}s: 命中 {hits:2d}/{len(preds)}  随机期望 {dist.mean():.1f}  "
              f"超出 {hits - dist.mean():+.1f}  p={pval:.4f}  牌认对 {tile_ok}/{hits}")
    (OUT / "exp_d_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
