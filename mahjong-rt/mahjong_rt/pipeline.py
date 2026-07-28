"""Pipeline orchestration (Phase 4 task 7).

Deterministic single-threaded path first: every frame is processed, same input gives
byte-identical output. That is the mode evaluation and parameter sweeps run in, and it
is the one that has to be correct. The threaded realtime variant is a scheduling
optimisation layered on top later — it must never change the answers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .events import Event
from .state_machine import StateMachine
from .tracker import ByteTrackGMC
from .voter import Observation
from .zones import ZoneConfig, assign_zones


@dataclass
class PipelineConfig:
    det_weights: str
    cls_weights: str
    det_conf: float = 0.25
    det_iou: float = 0.6
    det_imgsz: int = 960
    crop_margin: float = 0.08
    device: str = "cpu"
    classify_every: int = 1  # classify on every Nth frame; tracking still runs each frame
    tracker: dict[str, Any] = field(default_factory=dict)
    voter: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    zones: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._load_models()
        self.tracker = ByteTrackGMC(**config.tracker)
        self.state_machine = StateMachine(voter_kwargs=config.voter, **config.state)
        self.zone_config = ZoneConfig(**config.zones)
        self.frame_idx = 0
        self.timings: dict[str, list[float]] = {"detect": [], "classify": [], "track": [], "total": []}

    def _load_models(self) -> None:
        """Prefer the ONNX classifier: Phase 4's constraint is "consume ONNX, no torch",
        so the runtime here matches what Phase 5 ships on device. It is also ~7x faster
        than torch on CPU for this model. A .pt path still works for quick experiments.
        """
        import json

        from ultralytics import YOLO

        self.detector = YOLO(self.config.det_weights)
        weights = Path(self.config.cls_weights)
        self.cls_backend = "onnx" if weights.suffix.lower() == ".onnx" else "torch"

        if self.cls_backend == "onnx":
            import onnxruntime as ort

            meta_path = weights.with_name("meta.json")
            if not meta_path.exists():
                raise FileNotFoundError(f"ONNX classifier needs its sidecar {meta_path}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.classes = list(meta["classes"])
            self.cls_imgsz = int(meta.get("imgsz", 96))
            self._mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32).reshape(3, 1, 1)
            self._std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32).reshape(3, 1, 1)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.config.device != "cpu" else ["CPUExecutionProvider"]
            available = set(ort.get_available_providers())
            self._session = ort.InferenceSession(str(weights), providers=[p for p in providers if p in available] or ["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            return

        import torch
        from torchvision import transforms

        import sys

        scripts = Path(__file__).resolve().parents[2] / "scripts"
        sys.path.insert(0, str(scripts))
        from train_classifier import build_model

        ckpt = torch.load(self.config.cls_weights, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.cls_imgsz = int(ckpt.get("imgsz", 96))
        self.classifier = build_model(ckpt.get("arch", "mobilenet_v3_small"), len(self.classes))
        self.classifier.load_state_dict(ckpt["model"])
        self.torch_device = torch.device(self.config.device)
        self.classifier.to(self.torch_device).eval()
        self._torch = torch
        self._tf = transforms.Compose(
            [
                transforms.Resize((self.cls_imgsz, self.cls_imgsz)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _crops(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        height, width = frame.shape[:2]
        patches: list[np.ndarray] = []
        shorts: list[float] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            bw, bh = x2 - x1, y2 - y1
            mx, my = bw * self.config.crop_margin, bh * self.config.crop_margin
            cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            cx2, cy2 = int(min(width, x2 + mx)), int(min(height, y2 + my))
            patch = frame[cy1:cy2, cx1:cx2]
            if patch.size == 0:
                patch = np.zeros((8, 8, 3), np.uint8)
            patches.append(patch)
            shorts.append(float(min(bw, bh)))
        return patches, shorts

    def _classify(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> list[Observation | None]:
        import cv2

        if len(boxes_xyxy) == 0:
            return []
        patches, shorts = self._crops(frame, boxes_xyxy)

        if self.cls_backend == "onnx":
            batch = np.empty((len(patches), 3, self.cls_imgsz, self.cls_imgsz), np.float32)
            for i, patch in enumerate(patches):
                resized = cv2.resize(patch, (self.cls_imgsz, self.cls_imgsz), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                batch[i] = (rgb.transpose(2, 0, 1) - self._mean) / self._std
            logits = self._session.run(None, {self._input_name: batch})[0]
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            prob = exp / exp.sum(axis=1, keepdims=True)
            pred = prob.argmax(axis=1)
            conf = prob.max(axis=1)
            return [
                Observation(label=self.classes[int(p)], confidence=float(c), short_side=s)
                for p, c, s in zip(pred.tolist(), conf.tolist(), shorts)
            ]

        from PIL import Image

        tensors = [self._tf(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))) for p in patches]
        with self._torch.no_grad():
            prob = self._torch.softmax(self.classifier(self._torch.stack(tensors).to(self.torch_device)), dim=1)
            conf, pred = prob.max(1)
        return [
            Observation(label=self.classes[int(p)], confidence=float(c), short_side=s)
            for p, c, s in zip(pred.cpu().tolist(), conf.cpu().tolist(), shorts)
        ]

    def process_frame(self, frame: np.ndarray, ts: float | None = None) -> list[Event]:
        import cv2

        start = time.perf_counter()
        timestamp = self.frame_idx / 30.0 if ts is None else ts

        t0 = time.perf_counter()
        result = self.detector.predict(
            source=frame,
            conf=self.config.det_conf,
            iou=self.config.det_iou,
            imgsz=self.config.det_imgsz,
            device=self.config.device,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = boxes.conf.cpu().numpy().astype(np.float32)
        else:
            xyxy = np.zeros((0, 4), np.float32)
            scores = np.zeros((0,), np.float32)
        self.timings["detect"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracks = self.tracker.update(xyxy, scores, gray)
        self.timings["track"].append(time.perf_counter() - t0)

        track_boxes = np.stack([t.bbox for t in tracks]) if tracks else np.zeros((0, 4), np.float32)
        t0 = time.perf_counter()
        do_classify = self.frame_idx % max(1, self.config.classify_every) == 0
        observations = self._classify(frame, track_boxes) if do_classify else [None] * len(tracks)
        self.timings["classify"].append(time.perf_counter() - t0)

        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in track_boxes]
        zones = assign_zones(xywh, frame.shape[1], frame.shape[0], self.zone_config)

        events = self.state_machine.update(
            [(t.track_id, xywh[i], observations[i], zones[i]) for i, t in enumerate(tracks)],
            frame_idx=self.frame_idx,
            ts=timestamp,
            stats={
                "detections": int(len(xyxy)),
                "tracks": len(tracks),
                "gmc_ok": self.tracker.stats["gmc_ok"],
                "gmc_fail": self.tracker.stats["gmc_fail"],
            },
        )
        self.timings["total"].append(time.perf_counter() - start)
        self.frame_idx += 1
        return events

    def perf_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for stage, values in self.timings.items():
            if not values:
                continue
            arr = np.asarray(values)
            out[stage] = {
                "mean_ms": round(float(arr.mean() * 1000), 2),
                "p50_ms": round(float(np.percentile(arr, 50) * 1000), 2),
                "p95_ms": round(float(np.percentile(arr, 95) * 1000), 2),
            }
        if self.timings["total"]:
            out["fps"] = round(1.0 / float(np.mean(self.timings["total"])), 2)
        out["tracker"] = dict(self.tracker.stats)
        return out
