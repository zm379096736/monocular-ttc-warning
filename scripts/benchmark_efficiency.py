#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from itertools import cycle, islice
from pathlib import Path

import cv2
import numpy as np
import torch
from extract_upstream import infer_flow, load_flow_model
from ultralytics import YOLO

from monocular_ttc.config import load_config
from monocular_ttc.geometry import estimate_foe, estimate_global_flow, ttc_from_radial_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark frozen upstream modules")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def result(elapsed: float, runs: int, device: torch.device) -> dict[str, float]:
    return {
        "runs": runs,
        "latency_ms": 1000.0 * elapsed / runs,
        "fps": runs / elapsed,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    upstream = config["upstream"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    files = sorted(args.input.glob("*.png"))
    if len(files) < 2:
        raise ValueError(f"Need at least two images in {args.input}")
    frames = [cv2.imread(str(path)) for path in files]
    if any(frame is None for frame in frames):
        raise RuntimeError("One or more benchmark images could not be read")
    stream = list(islice(cycle(frames), args.warmup + args.runs + 1))

    detector = YOLO(upstream["detector_model"])
    for frame in stream[: args.warmup]:
        detector.track(
            frame,
            persist=True,
            tracker=upstream["tracker"],
            device=str(device),
            half=device.type == "cuda",
            verbose=False,
        )
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for frame in stream[args.warmup : args.warmup + args.runs]:
        detector.track(
            frame,
            persist=True,
            tracker=upstream["tracker"],
            device=str(device),
            half=device.type == "cuda",
            verbose=False,
        )
    synchronize(device)
    detector_result = result(time.perf_counter() - started, args.runs, device)
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    flow_model, transforms = load_flow_model(upstream["flow_model"], device)
    pairs = list(zip(stream[:-1], stream[1:], strict=True))
    for previous, current in pairs[: args.warmup]:
        infer_flow(flow_model, transforms, previous, current, device, device.type == "cuda")
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    last_flow = None
    for previous, current in pairs[args.warmup : args.warmup + args.runs]:
        last_flow = infer_flow(
            flow_model, transforms, previous, current, device, device.type == "cuda"
        )
    synchronize(device)
    raft_result = result(time.perf_counter() - started, args.runs, device)
    assert last_flow is not None

    height, width = last_flow.shape[:2]
    bbox = (0.35 * width, 0.35 * height, 0.65 * width, 0.75 * height)
    mask = np.ones((height, width), dtype=bool)
    geometry_runs = []
    for _ in range(args.warmup + args.runs):
        started = time.perf_counter()
        global_flow, _ = estimate_global_flow(last_flow, [bbox], int(upstream["sample_stride"]))
        foe = estimate_foe(last_flow, mask=mask, sample_stride=int(upstream["sample_stride"]))
        if foe is None:
            foe = (width / 2.0, height / 2.0)
        ttc_from_radial_flow(last_flow - global_flow, bbox, foe, 10.0)
        geometry_runs.append(time.perf_counter() - started)
    geometry_elapsed = sum(geometry_runs[args.warmup :])
    geometry_result = {
        "runs": args.runs,
        "latency_ms": 1000.0 * geometry_elapsed / args.runs,
        "fps": args.runs / geometry_elapsed,
        "peak_gpu_memory_mb": 0.0,
    }
    total_latency = (
        detector_result["latency_ms"] + raft_result["latency_ms"] + geometry_result["latency_ms"]
    )
    results = {
        "yolo_bytetrack": detector_result,
        "raft_large": raft_result,
        "geometry": geometry_result,
        "estimated_upstream_total": {
            "latency_ms": total_latency,
            "fps": 1000.0 / total_latency,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
