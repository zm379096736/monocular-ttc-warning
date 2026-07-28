#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from monocular_ttc.features import FeatureNormalizer
from monocular_ttc.model import TemporalWeightMLP
from monocular_ttc.risk import RiskPolicy

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
        description="Run streaming TTC fusion on cached upstream records"
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    model = TemporalWeightMLP(
        feature_dim=len(FEATURE_NAMES),
        hidden_dims=model_config["hidden_dims"],
        dropout=float(model_config["dropout"]),
        min_ttc=float(model_config["min_ttc_seconds"]),
        max_ttc=float(model_config["max_ttc_seconds"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with args.normalizer.open("r", encoding="utf-8") as handle:
        normalizer = FeatureNormalizer.from_state_dict(json.load(handle))
    sequence_length = int(config["sequence"]["length"])
    warmup = int(config["sequence"]["warmup_frames"])
    reset_jump = float(config["sequence"]["reset_ttc_jump_seconds"])
    histories: dict[int, deque[dict[str, object]]] = defaultdict(
        lambda: deque(maxlen=sequence_length)
    )
    last_frame: dict[int, int] = {}
    policy = RiskPolicy(
        **{
            "danger_ttc_seconds": float(config["risk"]["danger_ttc_seconds"]),
            "caution_ttc_seconds": float(config["risk"]["caution_ttc_seconds"]),
            "trend_upgrade_threshold": float(config["risk"]["trend_upgrade_threshold"]),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.records.open("r", encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            record = json.loads(line)
            track_id, frame_id = int(record["track_id"]), int(record["frame_id"])
            history = histories[track_id]
            if last_frame.get(track_id, frame_id - 1) != frame_id - 1:
                history.clear()
            if history and abs(float(record["ttc"]) - float(history[-1]["ttc"])) > reset_jump:
                history.clear()
            history.append(record)
            last_frame[track_id] = frame_id
            if len(history) < warmup:
                prediction = float(record["ttc"])
                weights = [1.0]
                mode = "single_frame_warmup"
            else:
                raw = np.asarray(
                    [[float(item[name]) for name in FEATURE_NAMES] for item in history],
                    dtype=np.float32,
                )
                features = torch.from_numpy(normalizer.transform(raw)).unsqueeze(0).to(device)
                candidates = torch.tensor(
                    [[float(item["ttc"]) for item in history]], dtype=torch.float32, device=device
                )
                with torch.inference_mode():
                    fused, temporal_weights = model(features, candidates)
                prediction = float(fused.item())
                weights = temporal_weights[0].cpu().tolist()
                mode = "mlp_temporal"
            decision = policy.classify(prediction, float(record["delta_ttc"]))
            target.write(
                json.dumps(
                    {
                        "sequence_id": record["sequence_id"],
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "ttc": prediction,
                        "risk_level": decision.level.name.lower(),
                        "upgraded_by_trend": decision.upgraded_by_trend,
                        "mode": mode,
                        "temporal_weights": weights,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
