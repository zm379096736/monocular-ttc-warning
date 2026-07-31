#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

SEEDS = (42, 43, 44)
WINDOWS = (5, 8, 12, 15)
ABLATIONS = {
    "box_ttc": [0, 5],
    "plus_radial_motion": [0, 2, 5],
    "plus_flow_reliability": [0, 2, 4, 5],
    "plus_trend": [0, 1, 2, 4, 5],
    "all_features": [0, 1, 2, 3, 4, 5],
}
METRIC_KEYS = ("mae", "rmse", "relative_error", "risk_accuracy", "danger_recall", "danger_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the remaining TTC experiment matrix")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(command: list[str], cwd: Path, log) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {command}")


def write_config(base: dict, output: Path, seed: int, window: int | None = None) -> Path:
    config = deepcopy(base)
    config["seed"] = seed
    if window is not None:
        config["sequence"]["length"] = window
        config["sequence"]["warmup_frames"] = min(int(config["sequence"]["warmup_frames"]), window)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output


def train_evaluate(
    python: str,
    project: Path,
    data: Path,
    config: Path,
    output: Path,
    device: str,
    model_type: str = "mlp",
    active_features: list[int] | None = None,
    log=None,
) -> None:
    checkpoint = output / "best.pt"
    metrics = output / "test_metrics.json"
    if not checkpoint.exists():
        command = [
            python,
            "scripts/train.py",
            "--data",
            str(data),
            "--config",
            str(config),
            "--output",
            str(output),
            "--device",
            device,
            "--model-type",
            model_type,
        ]
        if active_features is not None:
            command.extend(["--active-features", *map(str, active_features)])
        run(command, project, log)
    if not metrics.exists():
        run(
            [
                python,
                "scripts/evaluate.py",
                "--data",
                str(data),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(metrics),
                "--device",
                device,
            ],
            project,
            log,
        )


def aggregate(experiment_root: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for path in experiment_root.glob("**/seed_*/test_metrics.json"):
        relative = path.relative_to(experiment_root)
        family, variant = relative.parts[0], relative.parts[1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault((family, variant), []).append(payload["mlp_temporal"])
    rows = []
    for (family, variant), values in sorted(grouped.items()):
        row: dict[str, object] = {"family": family, "variant": variant, "seeds": len(values)}
        for key in METRIC_KEYS:
            samples = np.asarray([float(item[key]) for item in values])
            row[f"{key}_mean"] = float(samples.mean())
            row[f"{key}_std"] = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
        rows.append(row)
    return rows


def write_summary(rows: list[dict[str, object]], root: Path) -> None:
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Experiment matrix summary",
        "",
        "| Family | Variant | Seeds | MAE | RMSE | Risk accuracy | Danger F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['variant']} | {row['seeds']} | "
            f"{row['mae_mean']:.3f} ± {row['mae_std']:.3f} | "
            f"{row['rmse_mean']:.3f} ± {row['rmse_std']:.3f} | "
            f"{row['risk_accuracy_mean']:.3f} ± {row['risk_accuracy_std']:.3f} | "
            f"{row['danger_f1_mean']:.3f} ± {row['danger_f1_std']:.3f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    python = str(project / ".venv" / "Scripts" / "python.exe")
    base_config_path = project / "configs" / "base.yaml"
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    root = project / "outputs" / "experiments"
    configs = root / "configs"
    root.mkdir(parents=True, exist_ok=True)
    base_data = project / "data" / "full" / "sequences"
    records = project / "data" / "full" / "all_upstream.jsonl"
    labels = project / "data" / "full" / "all_labels.jsonl"
    with (root / "matrix.log").open("a", encoding="utf-8", buffering=1) as log:
        for model_type in ("mlp", "gru", "lstm"):
            for seed in SEEDS:
                config = write_config(base, configs / f"seed_{seed}.yaml", seed)
                train_evaluate(
                    python,
                    project,
                    base_data,
                    config,
                    root / "models" / model_type / f"seed_{seed}",
                    args.device,
                    model_type=model_type,
                    log=log,
                )

        for name, active in ABLATIONS.items():
            for seed in SEEDS:
                config = write_config(base, configs / f"seed_{seed}.yaml", seed)
                train_evaluate(
                    python,
                    project,
                    base_data,
                    config,
                    root / "ablations" / name / f"seed_{seed}",
                    args.device,
                    active_features=active,
                    log=log,
                )

        for window in WINDOWS:
            window_data = project / "data" / "experiments" / f"window_{window}" / "sequences"
            prepare_config = write_config(base, configs / f"window_{window}.yaml", 42, window)
            if not (window_data / "test.jsonl").exists():
                run(
                    [
                        python,
                        "scripts/prepare_sequences.py",
                        "--records",
                        str(records),
                        "--labels",
                        str(labels),
                        "--output",
                        str(window_data),
                        "--config",
                        str(prepare_config),
                    ],
                    project,
                    log,
                )
            for seed in SEEDS:
                config = write_config(
                    base,
                    configs / f"window_{window}_seed_{seed}.yaml",
                    seed,
                    window,
                )
                train_evaluate(
                    python,
                    project,
                    window_data,
                    config,
                    root / "windows" / str(window) / f"seed_{seed}",
                    args.device,
                    log=log,
                )

        prediction_root = project / "outputs" / "base" / "predictions"
        prediction_root.mkdir(parents=True, exist_ok=True)
        for cache in sorted((project / "data" / "cache" / "upstream").glob("*.jsonl")):
            output = prediction_root / cache.name
            if output.exists():
                continue
            run(
                [
                    python,
                    "scripts/predict_cache.py",
                    "--records",
                    str(cache),
                    "--checkpoint",
                    str(project / "outputs" / "base" / "best.pt"),
                    "--normalizer",
                    str(base_data / "normalizer.json"),
                    "--output",
                    str(output),
                    "--device",
                    args.device,
                ],
                project,
                log,
            )

    write_summary(aggregate(root), root)
    print(f"Completed experiment matrix: {root}")


if __name__ == "__main__":
    main()
