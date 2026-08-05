"""Apply the frame-anchored timestamp corrections to events_gt.json.

Original file backed up to events_gt.oral_t.json (oral-annotation timestamps).
Corrected t = the frame where the tile leaves the fingers and rests on the pile,
read off tests/test1/frames/fix/*.jpg strips (0.2s spacing, so ±0.2s accuracy).

    python tests/test1/exp_apply_gt_fix.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "output" / "video_testset_pilot" / "events_gt.json"
BAK = ROOT / "output" / "video_testset_pilot" / "events_gt.oral_t.json"

# (index, corrected t, confidence note)
FIX = [
    (0, 0.2,  "低置信:落桌发生在片段起点附近,条带只覆盖到 t=0.1 起"),
    (1, 1.2,  ""),
    (2, 2.4,  ""),
    (3, 4.2,  "中置信:3.1-3.3s 上家曾在牌堆上方持牌,4.0-4.3s 是清晰的放置动作"),
    (4, 7.1,  ""),
    (5, 8.1,  ""),
    (6, 10.5, ""),
    (7, 13.2, ""),
    (8, 17.1, ""),
    (9, 18.6, ""),
    (10, 21.1, ""),
    (11, 22.7, ""),
    (12, 26.2, ""),
    (13, 28.1, ""),
]


def main():
    data = json.loads(GT.read_text(encoding="utf-8"))
    if not BAK.exists():
        shutil.copy(GT, BAK)
        print("备份 ->", BAK.name)
    events = data["clips"]["clip01_7507945925261200"]["events"]
    assert len(events) == len(FIX)
    for idx, new_t, conf in FIX:
        e = events[idx]
        old_t = e["t"]
        e["t"] = new_t
        e["note"] = (f"t原{old_t}s(口述);2026-08-05按落桌帧修正" + (f";{conf}" if conf else "")).strip(";")
        print(f"  [{idx:02d}] {e['who']:6s} {e['tile']:3s}  {old_t:5.1f} -> {new_t:5.1f}  {conf}")
    data["_time_semantics"] = ("t = 牌离手落桌时刻(帧级复核,精度±0.2s,见 tests/test1/frames/fix/)。"
                               "2026-08-05 前的版本是口述标注时刻,系统性滞后约1s,备份于 events_gt.oral_t.json")
    GT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("已写入", GT)


if __name__ == "__main__":
    main()
