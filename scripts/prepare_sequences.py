#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from monocular_ttc.config import load_config
from monocular_ttc.data import split_by_sequence
from monocular_ttc.features import FeatureNormalizer

FEATURE_NAMES = [
    "ttc",
    "delta_ttc",
    "radial_expansion",
    "detector_confidence",
    "flow_consistency",
    "box_growth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-length TTC sequences from cached records"
    )
    parser.add_argument(
        "--records", type=Path, required=True, help="JSONL from extract_upstream.py"
    )
    parser.add_argument("--labels", type=Path, required=True, help="JSONL TTC ground truth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-6)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sequence_length = int(config["sequence"]["length"])
    max_frame_gap = int(config["sequence"]["max_frame_gap"])
    max_ttc = float(config["model"]["max_ttc_seconds"])

    labels = read_jsonl(args.labels)
    label_lookup = {
        (str(item["sequence_id"]), int(item["frame_id"]), int(item["track_id"])): item
        for item in labels
    }
    labels_by_frame: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for label in labels:
        labels_by_frame[(str(label["sequence_id"]), int(label["frame_id"]))].append(label)
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in read_jsonl(args.records):
        key = (str(record["sequence_id"]), int(record["track_id"]))
        grouped[key].append(record)

    candidates: list[dict[str, object]] = []
    for (sequence_id, track_id), track in grouped.items():
        track.sort(key=lambda item: int(item["frame_id"]))
        segment: list[dict[str, object]] = []
        for record in track:
            if segment and int(record["frame_id"]) - int(segment[-1]["frame_id"]) > max_frame_gap:
                segment = []
            segment.append(record)
            segment = segment[-sequence_length:]
            frame_labels = labels_by_frame.get((sequence_id, int(record["frame_id"])), [])
            if not frame_labels or len(segment) < 2:
                continue
            matched = max(frame_labels, key=lambda item: bbox_iou(record["bbox"], item["bbox"]))
            if bbox_iou(record["bbox"], matched["bbox"]) < 0.30:
                continue
            label = float(matched["ttc"])
            previous_key = (
                sequence_id,
                int(record["frame_id"]) - 1,
                int(matched["track_id"]),
            )
            previous_label = float(label_lookup.get(previous_key, {"ttc": label})["ttc"])
            candidates.append(
                {
                    "sequence_id": sequence_id,
                    "track_id": track_id,
                    "ground_truth_track_id": int(matched["track_id"]),
                    "frame_id": int(record["frame_id"]),
                    "frames": list(segment),
                    "target_ttc": float(np.clip(label, 0.0, max_ttc)),
                    "previous_target_ttc": float(np.clip(previous_label, 0.0, max_ttc)),
                }
            )

    splits = split_by_sequence(candidates, seed=int(config["seed"]))
    train_features = np.concatenate(
        [
            np.asarray([[float(frame[name]) for name in FEATURE_NAMES] for frame in item["frames"]])
            for item in splits["train"]
        ],
        axis=0,
    )
    normalizer = FeatureNormalizer().fit(train_features)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "normalizer.json").open("w", encoding="utf-8") as handle:
        json.dump(normalizer.state_dict(), handle, indent=2)

    for split_name, split_records in splits.items():
        split_dir = args.output / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_entries = []
        for index, item in enumerate(split_records):
            frames = item["frames"]
            valid_length = len(frames)
            raw = np.zeros((sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
            ttc_candidates = np.full((sequence_length,), max_ttc, dtype=np.float32)
            mask = np.zeros((sequence_length,), dtype=bool)
            offset = sequence_length - valid_length
            raw[offset:] = np.asarray(
                [[float(frame[name]) for name in FEATURE_NAMES] for frame in frames],
                dtype=np.float32,
            )
            ttc_candidates[offset:] = np.asarray(
                [float(frame["ttc"]) for frame in frames], dtype=np.float32
            )
            mask[offset:] = True
            normalized = normalizer.transform(raw)
            normalized[~mask] = 0.0
            filename = f"{index:08d}.npz"
            np.savez_compressed(
                split_dir / filename,
                features=normalized,
                ttc_candidates=ttc_candidates,
                mask=mask,
                target_ttc=np.float32(item["target_ttc"]),
                previous_target_ttc=np.float32(item["previous_target_ttc"]),
            )
            manifest_entries.append(
                {
                    "path": f"{split_name}/{filename}",
                    "sequence_id": item["sequence_id"],
                    "track_id": item["track_id"],
                    "frame_id": item["frame_id"],
                }
            )
        with (args.output / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for entry in manifest_entries:
                handle.write(json.dumps(entry) + "\n")
    print({name: len(records) for name, records in splits.items()})


if __name__ == "__main__":
    main()
