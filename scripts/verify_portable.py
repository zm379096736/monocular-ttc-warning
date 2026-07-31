#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from monocular_ttc.features import FeatureNormalizer
from monocular_ttc.model import build_temporal_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify portable TTC artifacts and KITTI layout")
    parser.add_argument("--kitti-root", type=Path, default=None)
    return parser.parse_args()


def verify_checkpoint(path: Path, normalizer: FeatureNormalizer) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_type = checkpoint.get("model_type", "mlp")
    model = build_temporal_model(config, model_type).eval()
    model.load_state_dict(checkpoint["model"])

    length = int(config["sequence"]["length"])
    raw = np.tile(normalizer.mean, (length, 1)).astype(np.float32)
    features = torch.from_numpy(normalizer.transform(raw)).unsqueeze(0)
    candidates = torch.full((1, length), 8.0, dtype=torch.float32)
    with torch.inference_mode():
        prediction, weights = model(features, candidates)
    if prediction.shape != (1,) or weights.shape != (1, length):
        raise RuntimeError(f"Unexpected output shapes for {path}")
    print(f"OK checkpoint: {path.name} ({model_type}, window={length})")


def verify_kitti(root: Path) -> None:
    image_root = root / "image_02"
    label_root = root / "label_02"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            "KITTI 目录应包含 image_02/ 和 label_02/；详见 PORTABLE_RUN.md"
        )
    sequences = sorted(path for path in image_root.iterdir() if path.is_dir())
    labels = sorted(label_root.glob("*.txt"))
    if not sequences or not labels:
        raise RuntimeError("KITTI image_02 或 label_02 为空")
    print(f"OK KITTI: {len(sequences)} image sequences, {len(labels)} label files")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "artifacts" / "normalizer.json").open("r", encoding="utf-8") as handle:
        normalizer = FeatureNormalizer.from_state_dict(json.load(handle))

    checkpoint_root = root / "artifacts" / "checkpoints"
    verify_checkpoint(checkpoint_root / "gru_warning_best_seed43.pt", normalizer)
    verify_checkpoint(checkpoint_root / "lstm_ttc_best_seed43.pt", normalizer)
    if args.kitti_root is not None:
        verify_kitti(args.kitti_root)
    print("Portable package verification passed")


if __name__ == "__main__":
    main()
