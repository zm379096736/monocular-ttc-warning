#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate target-level TTC labels from KITTI tracking labels"
    )
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-ttc", type=float, default=20.0)
    parser.add_argument("--types", nargs="+", default=["Car", "Van", "Truck"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks: dict[int, list[tuple[int, float, list[float]]]] = defaultdict(list)
    with args.label_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 17 or fields[2] not in args.types:
                continue
            frame_id, track_id = int(fields[0]), int(fields[1])
            bbox = [float(value) for value in fields[6:10]]
            z_camera = float(fields[15])
            tracks[track_id].append((frame_id, z_camera, bbox))

    labels = []
    for track_id, observations in tracks.items():
        observations.sort()
        previous = None
        for frame_id, distance, bbox in observations:
            if previous is None or frame_id - previous[0] != 1:
                previous = (frame_id, distance, bbox)
                continue
            relative_speed = (distance - previous[1]) * args.fps
            if relative_speed < -1e-3:
                ttc = distance / -relative_speed
            else:
                ttc = args.max_ttc
            labels.append(
                {
                    "sequence_id": args.sequence_id,
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "bbox": bbox,
                    "distance_z": distance,
                    "relative_speed_z": relative_speed,
                    "ttc": float(np.clip(ttc, 0.0, args.max_ttc)),
                }
            )
            previous = (frame_id, distance, bbox)
    labels.sort(key=lambda item: (item["frame_id"], item["track_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label) + "\n")
    print(f"Wrote {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    main()
