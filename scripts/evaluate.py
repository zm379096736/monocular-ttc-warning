#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        description="Evaluate learned fusion and traditional baselines"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/base/test_metrics.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_type = checkpoint.get("model_type", "mlp")
    active_features = checkpoint.get("active_features")
    model = build_temporal_model(config, model_type).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = TTCSequenceDataset(args.data / "test.jsonl")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    methods: dict[str, list[np.ndarray]] = {
        "single_frame": [],
        "moving_average": [],
        "confidence_weighted": [],
        "mlp_temporal": [],
    }
    targets = []
    weight_rows = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            if active_features is not None:
                inactive = [i for i in range(features.shape[-1]) if i not in active_features]
                features[:, :, inactive] = 0.0
            candidates = batch["ttc_candidates"].to(device)
            mask = batch["mask"].to(device)
            prediction, weights = model(features, candidates, mask)
            count = mask.sum(dim=1)
            last_indices = count - 1 + (mask.shape[1] - count)
            last = candidates.gather(1, last_indices[:, None]).squeeze(1)
            average = (candidates * mask).sum(dim=1) / count
            # Normalized features 3 and 4 still preserve ordering; exponentiate a bounded score.
            score = torch.exp(torch.clamp(features[:, :, 3] + features[:, :, 4], -8.0, 8.0)) * mask
            confidence = (score * candidates).sum(dim=1) / score.sum(dim=1).clamp_min(1e-6)
            methods["single_frame"].append(last.cpu().numpy())
            methods["moving_average"].append(average.cpu().numpy())
            methods["confidence_weighted"].append(confidence.cpu().numpy())
            methods["mlp_temporal"].append(prediction.cpu().numpy())
            targets.append(batch["target_ttc"].numpy())
            weight_rows.append(weights.cpu().numpy())

    target = np.concatenate(targets)
    risk_config = config["risk"]
    policy = RiskPolicy(
        danger_ttc_seconds=float(risk_config["danger_ttc_seconds"]),
        caution_ttc_seconds=float(risk_config["caution_ttc_seconds"]),
        trend_upgrade_threshold=float(risk_config["trend_upgrade_threshold"]),
    )
    target_risk = np.asarray([int(policy.classify(value).level) for value in target])
    results = {}
    for name, chunks in methods.items():
        prediction = np.concatenate(chunks)
        prediction_risk = np.asarray([int(policy.classify(value).level) for value in prediction])
        results[name] = {
            **regression_metrics(prediction, target),
            **risk_metrics(prediction_risk, target_risk),
        }
    results["model_type"] = model_type
    results["active_features"] = active_features
    results["interpretability"] = {
        "mean_temporal_weights": np.concatenate(weight_rows).mean(axis=0).tolist()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
