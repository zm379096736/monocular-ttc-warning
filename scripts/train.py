#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from monocular_ttc.config import load_config
from monocular_ttc.data import TTCSequenceDataset
from monocular_ttc.model import build_temporal_model, trend_consistency_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal TTC weighting MLP")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/base"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", choices=["mlp", "gru", "lstm"], default="mlp")
    parser.add_argument("--active-features", nargs="*", type=int, default=None)
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer, trend_weight: float, active_features=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if active_features is not None:
                inactive = [
                    i for i in range(batch["features"].shape[-1]) if i not in active_features
                ]
                batch["features"][:, :, inactive] = 0.0
            prediction, _ = model(batch["features"], batch["ttc_candidates"], batch["mask"])
            primary = nn.functional.smooth_l1_loss(prediction, batch["target_ttc"])
            trend = trend_consistency_loss(
                prediction, batch["target_ttc"], batch["previous_target_ttc"]
            )
            loss = primary + trend_weight * trend
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += float(loss.detach()) * len(prediction)
            count += len(prediction)
    return total_loss / max(count, 1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    training = config["training"]
    train_set = TTCSequenceDataset(args.data / "train.jsonl")
    validation_set = TTCSequenceDataset(args.data / "validation.jsonl")
    train_loader = DataLoader(
        train_set,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set, batch_size=int(training["batch_size"]), shuffle=False, num_workers=0
    )
    model = build_temporal_model(config, args.model_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            float(training["trend_loss_weight"]),
            args.active_features,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            device,
            None,
            float(training["trend_loss_weight"]),
            args.active_features,
        )
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        history.append(row)
        print(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "best_validation_loss": best_loss,
                    "model_type": args.model_type,
                    "active_features": args.active_features,
                },
                args.output / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                break
    with (args.output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
