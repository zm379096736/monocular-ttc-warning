#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

COLORS = {
    "safe": (60, 200, 60),
    "caution": (0, 210, 255),
    "danger": (30, 30, 230),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render TTC predictions over a KITTI sequence")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = parse_args()
    boxes = {
        (int(item["frame_id"]), int(item["track_id"])): item["bbox"]
        for item in read_jsonl(args.upstream)
    }
    predictions: dict[int, list[dict]] = defaultdict(list)
    for item in read_jsonl(args.predictions):
        predictions[int(item["frame_id"])].append(item)
    images = sorted(args.images.glob("*.png"))
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {args.output}")
    try:
        for frame_id, image_path in enumerate(images):
            frame = cv2.imread(str(image_path))
            cv2.rectangle(frame, (0, 0), (width, 48), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"YOLO11m + ByteTrack + RAFT-Large + GRU | KITTI 0000 | frame {frame_id:04d}",
                (14, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            for item in predictions.get(frame_id, []):
                key = (frame_id, int(item["track_id"]))
                if key not in boxes:
                    continue
                x1, y1, x2, y2 = [int(round(value)) for value in boxes[key]]
                risk = str(item["risk_level"])
                color = COLORS.get(risk, (220, 220, 220))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"ID {item['track_id']} | TTC {float(item['ttc']):.1f}s | {risk.upper()}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
                top = max(50, y1 - text_size[1] - 10)
                label_x = max(0, min(x1, width - text_size[0] - 8))
                cv2.rectangle(
                    frame,
                    (label_x, top),
                    (label_x + text_size[0] + 8, top + text_size[1] + 8),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (label_x + 4, top + text_size[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (10, 10, 10),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()
    print(args.output)


if __name__ == "__main__":
    main()
