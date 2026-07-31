#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs" / "experiments"
FIGURES = ROOT / "figures"


def plot_family(rows: list[dict], family: str, filename: str, title: str) -> None:
    selected = [row for row in rows if row["family"] == family]
    labels = [str(row["variant"]) for row in selected]
    mae = [row["mae_mean"] for row in selected]
    mae_error = [row["mae_std"] for row in selected]
    f1 = [row["danger_f1_mean"] for row in selected]
    f1_error = [row["danger_f1_std"] for row in selected]
    positions = np.arange(len(labels))
    figure, first = plt.subplots(figsize=(9, 5))
    first.bar(positions - 0.18, mae, 0.36, yerr=mae_error, label="MAE (s)", color="#3973ac")
    first.set_ylabel("MAE (s)")
    first.set_xticks(positions, labels, rotation=20, ha="right")
    second = first.twinx()
    second.bar(positions + 0.18, f1, 0.36, yerr=f1_error, label="Danger F1", color="#e07a3f")
    second.set_ylabel("Danger F1")
    second.set_ylim(0, 1)
    first.set_title(title)
    handles = first.containers[:1] + second.containers[:1]
    first.legend(handles, ["MAE (s)", "Danger F1"], loc="upper center")
    figure.tight_layout()
    figure.savefig(FIGURES / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    plot_family(rows, "models", "model_comparison.png", "Temporal model comparison")
    plot_family(rows, "ablations", "feature_ablation.png", "Cumulative feature ablation")
    plot_family(rows, "windows", "window_sensitivity.png", "Temporal window sensitivity")

    thresholds = json.loads(
        (ROOT / "analysis" / "threshold_sensitivity.json").read_text(encoding="utf-8")
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for danger in sorted({row["danger_seconds"] for row in thresholds}):
        subset = sorted(
            [row for row in thresholds if row["danger_seconds"] == danger],
            key=lambda row: row["caution_seconds"],
        )
        axis.plot(
            [row["caution_seconds"] for row in subset],
            [row["danger_f1"] for row in subset],
            marker="o",
            label=f"Danger={danger:g}s",
        )
    axis.set_xlabel("Caution threshold (s)")
    axis.set_ylabel("Danger F1")
    axis.set_ylim(0, 1)
    axis.set_title("Risk-threshold sensitivity")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "threshold_sensitivity.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
