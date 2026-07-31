#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from monocular_ttc.data import TTCSequenceDataset
from monocular_ttc.metrics import regression_metrics, risk_metrics
from monocular_ttc.model import build_temporal_model
from monocular_ttc.risk import RiskPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold, scenario, failure and latency analyses"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def metrics(prediction: np.ndarray, target: np.ndarray, danger=3.0, caution=5.0) -> dict:
    policy = RiskPolicy(danger_ttc_seconds=danger, caution_ttc_seconds=caution)
    predicted_risk = np.asarray([int(policy.classify(value).level) for value in prediction])
    target_risk = np.asarray([int(policy.classify(value).level) for value in target])
    return {
        **regression_metrics(prediction, target),
        **risk_metrics(predicted_risk, target_risk),
        "samples": int(len(target)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_temporal_model(checkpoint["config"], checkpoint.get("model_type", "mlp"))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    dataset = TTCSequenceDataset(args.data / "test.jsonl")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    predictions, targets, features = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch_features = batch["features"].to(device)
            prediction, _ = model(
                batch_features,
                batch["ttc_candidates"].to(device),
                batch["mask"].to(device),
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(batch["target_ttc"].numpy())
            features.append(batch["features"].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    feature = np.concatenate(features)

    threshold_rows = []
    for danger in (2.0, 3.0, 4.0):
        for caution in (4.0, 5.0, 6.0):
            if caution <= danger:
                continue
            threshold_rows.append(
                {
                    "danger_seconds": danger,
                    "caution_seconds": caution,
                    **metrics(prediction, target, danger, caution),
                }
            )

    valid_last = feature[:, -1, :]
    scenario_masks = {
        "low_detector_confidence": valid_last[:, 3] <= np.quantile(valid_last[:, 3], 0.25),
        "low_flow_consistency": valid_last[:, 4] <= np.quantile(valid_last[:, 4], 0.25),
        "rapid_ttc_change": np.abs(valid_last[:, 1]) >= np.quantile(np.abs(valid_last[:, 1]), 0.75),
        "danger_ground_truth": target <= 3.0,
        "all_test_samples": np.ones(len(target), dtype=bool),
    }
    scenarios = {
        name: metrics(prediction[mask], target[mask])
        for name, mask in scenario_masks.items()
        if mask.any()
    }

    manifest = [
        json.loads(line)
        for line in (args.data / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    errors = np.abs(prediction - target)
    failure_indices = np.argsort(errors)[-50:][::-1]
    failures = [
        {
            **manifest[int(index)],
            "prediction_ttc": float(prediction[index]),
            "target_ttc": float(target[index]),
            "absolute_error": float(errors[index]),
        }
        for index in failure_indices
    ]

    sample = dataset[0]
    model_inputs = (
        sample["features"].unsqueeze(0).to(device),
        sample["ttc_candidates"].unsqueeze(0).to(device),
        sample["mask"].unsqueeze(0).to(device),
    )
    with torch.inference_mode():
        for _ in range(100):
            model(*model_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(1000):
            model(*model_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    latency_ms = elapsed
    efficiency = {
        "runs": 1000,
        "batch_size": 1,
        "total_seconds": elapsed,
        "latency_ms": latency_ms,
        "fps": 1000.0 / elapsed,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "threshold_sensitivity.json").write_text(
        json.dumps(threshold_rows, indent=2), encoding="utf-8"
    )
    (args.output / "scenario_analysis.json").write_text(
        json.dumps(scenarios, indent=2), encoding="utf-8"
    )
    (args.output / "failure_cases.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    (args.output / "downstream_efficiency.json").write_text(
        json.dumps(efficiency, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"thresholds": threshold_rows, "scenarios": scenarios, "efficiency": efficiency},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
