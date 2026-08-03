from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mahjong_rt.table_geometry import TableNormalizer
from mahjong_rt.zone_types import TableGeometry

CORNER_NAMES = ("left-bottom", "left-top", "right-top", "right-bottom")


def save_labels_atomic(path: Path, labels: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def annotate(labels_path: Path, images_dir: Path) -> None:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    window = "table corners: LB, LT, RT, RB | R reset | S save | Q quit"
    for item in labels:
        image = cv2.imread(str(images_dir / item["image"]))
        if image is None:
            raise ValueError(f"cannot decode image {item['image']}")
        height, width = image.shape[:2]
        existing = item.get("table_corners", [])
        points = [tuple(map(int, point)) for point in existing] if len(existing) == 4 else []

        def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            display = image.copy()
            for index, point in enumerate(points):
                cv2.circle(display, point, 7, (0, 0, 255), -1)
                cv2.putText(display, CORNER_NAMES[index], point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if len(points) == 4:
                cv2.polylines(display, [np.asarray(points, np.int32)], True, (0, 255, 0), 3)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                points.clear()
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return
            elif key == ord("s"):
                if len(points) != 4:
                    continue
                geometry = TableGeometry(np.asarray(points, dtype=np.float32))
                layout = TableNormalizer().normalize(item.get("boxes", []), geometry, width, height)
                if not layout.quality.valid:
                    raise ValueError(
                        f"invalid table corners for {item['image']}: {layout.quality.failures}"
                    )
                item["table_corners"] = [list(point) for point in points]
                save_labels_atomic(labels_path, labels)
                break
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate table corners in LB, LT, RT, RB order")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    annotate(args.labels, args.images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
