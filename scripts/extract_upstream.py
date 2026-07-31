#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as nnf
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)
from tqdm import tqdm
from ultralytics import YOLO

from monocular_ttc.config import load_config
from monocular_ttc.features import make_frame_features
from monocular_ttc.geometry import (
    box_growth_ttc,
    corridor_overlap,
    estimate_foe,
    estimate_global_flow,
    ttc_from_radial_flow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen YOLO/ByteTrack/RAFT outputs")
    parser.add_argument("--input", type=Path, required=True, help="Video file or image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--fps", type=float, default=None, help="Required for image directories")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true", help="Use CUDA mixed precision")
    return parser.parse_args()


def frames_from_path(path: Path, fps_override: float | None) -> tuple[Iterator[np.ndarray], float]:
    if path.is_dir():
        files = sorted(
            item for item in path.iterdir() if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not files:
            raise ValueError(f"No images found in {path}")
        fps = fps_override or 10.0

        def iterator() -> Iterator[np.ndarray]:
            for filename in files:
                frame = cv2.imread(str(filename))
                if frame is None:
                    raise RuntimeError(f"Could not read {filename}")
                yield frame

        return iterator(), fps

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps_override or (detected_fps if detected_fps > 0 else 30.0)

    def iterator() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()

    return iterator(), fps


def load_flow_model(name: str, device: torch.device):
    if name == "raft_small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    elif name == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    else:
        raise ValueError(f"Unsupported flow model: {name}")
    model.requires_grad_(False).eval().to(device)
    return model, weights.transforms()


def frame_to_tensor(
    frame: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, tuple[int, int]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    height, width = rgb.shape[:2]
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8
    return nnf.pad(tensor, (0, pad_width, 0, pad_height)), (height, width)


@torch.inference_mode()
def infer_flow(model, transforms, previous, current, device, use_half: bool) -> np.ndarray:
    previous_tensor, original_size = frame_to_tensor(previous, device)
    current_tensor, _ = frame_to_tensor(current, device)
    previous_tensor, current_tensor = transforms(previous_tensor, current_tensor)
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=use_half and device.type == "cuda"
    ):
        prediction = model(previous_tensor, current_tensor)[-1]
    height, width = original_size
    return prediction[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy()


def parse_tracks(result, allowed_classes: set[int]) -> list[dict[str, object]]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    tracks = []
    for bbox, confidence, class_id, track_id in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.int().cpu().numpy(),
        boxes.id.int().cpu().numpy(),
        strict=True,
    ):
        if int(class_id) not in allowed_classes:
            continue
        tracks.append(
            {
                "bbox": tuple(float(value) for value in bbox),
                "confidence": float(confidence),
                "class_id": int(class_id),
                "track_id": int(track_id),
            }
        )
    return tracks


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    upstream = config["upstream"]
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    detector = YOLO(upstream["detector_model"])
    flow_model, flow_transforms = load_flow_model(upstream["flow_model"], device)
    frames, fps = frames_from_path(args.input, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous_frame = None
    previous_boxes: dict[int, tuple[float, float, float, float]] = {}
    previous_ttc: dict[int, float] = {}

    with args.output.open("w", encoding="utf-8") as output:
        for frame_id, frame in enumerate(tqdm(frames, desc=args.sequence_id)):
            result = detector.track(
                frame,
                persist=True,
                tracker=upstream["tracker"],
                conf=float(upstream["detector_confidence"]),
                iou=float(upstream["detector_iou"]),
                device=str(device),
                half=args.half and device.type == "cuda",
                verbose=False,
            )[0]
            tracks = parse_tracks(result, {int(item) for item in upstream["vehicle_classes"]})
            if previous_frame is None:
                previous_frame = frame
                previous_boxes = {int(track["track_id"]): track["bbox"] for track in tracks}
                continue

            flow = infer_flow(flow_model, flow_transforms, previous_frame, frame, device, args.half)
            excluded = [track["bbox"] for track in tracks]
            background_mask = np.ones(flow.shape[:2], dtype=bool)
            for bbox in excluded:
                x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                background_mask[
                    max(0, y1) : min(flow.shape[0], y2), max(0, x1) : min(flow.shape[1], x2)
                ] = False
            global_flow, _ = estimate_global_flow(
                flow, excluded, sample_stride=int(upstream["sample_stride"])
            )
            foe = estimate_foe(
                flow,
                mask=background_mask,
                sample_stride=int(upstream["sample_stride"]),
            )
            if foe is None:
                foe = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
            residual_flow = flow - global_flow

            current_boxes: dict[int, tuple[float, float, float, float]] = {}
            for track in tracks:
                track_id = int(track["track_id"])
                bbox = track["bbox"]
                current_boxes[track_id] = bbox
                overlap = corridor_overlap(
                    bbox,
                    foe[0],
                    frame.shape[1],
                    float(upstream["corridor_half_width_ratio"]),
                )
                if overlap < float(upstream["corridor_min_overlap"]):
                    continue
                raw_observation = ttc_from_radial_flow(
                    flow,
                    bbox,
                    foe,
                    fps,
                    max_ttc_seconds=float(upstream["max_ttc_seconds"]),
                    min_samples=int(upstream["min_radial_samples"]),
                )
                residual_observation = ttc_from_radial_flow(
                    residual_flow,
                    bbox,
                    foe,
                    fps,
                    max_ttc_seconds=float(upstream["max_ttc_seconds"]),
                    min_samples=int(upstream["min_radial_samples"]),
                )
                box_ttc, box_growth, box_valid = box_growth_ttc(
                    previous_boxes.get(track_id),
                    bbox,
                    fps,
                    max_ttc_seconds=float(upstream["max_ttc_seconds"]),
                )
                if raw_observation.valid:
                    ttc = raw_observation.ttc_seconds
                elif box_valid:
                    ttc = box_ttc
                else:
                    ttc = float(upstream["max_ttc_seconds"])
                consistency = raw_observation.consistency
                if residual_observation.valid:
                    agreement = np.exp(
                        -abs(raw_observation.ttc_seconds - residual_observation.ttc_seconds)
                        / max(float(upstream["max_ttc_seconds"]), 1.0)
                    )
                    consistency = float(0.5 * consistency + 0.5 * agreement)
                features = make_frame_features(
                    ttc=ttc,
                    previous_ttc=previous_ttc.get(track_id),
                    radial_expansion=raw_observation.radial_expansion_per_frame,
                    detector_confidence=float(track["confidence"]),
                    flow_consistency=consistency,
                    box_growth=box_growth,
                    max_ttc=float(upstream["max_ttc_seconds"]),
                )
                record = {
                    "sequence_id": args.sequence_id,
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "class_id": int(track["class_id"]),
                    "bbox": list(bbox),
                    "foe": list(foe),
                    "corridor_overlap": overlap,
                    "ttc_flow": raw_observation.ttc_seconds,
                    "ttc_residual": residual_observation.ttc_seconds,
                    "ttc_box": box_ttc,
                    **features.__dict__,
                }
                output.write(json.dumps(record) + "\n")
                previous_ttc[track_id] = ttc
            previous_frame = frame
            previous_boxes = current_boxes


if __name__ == "__main__":
    main()
