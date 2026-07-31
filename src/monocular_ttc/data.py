from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TTCSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Read fixed-length sequence samples stored as compressed NumPy arrays."""

    REQUIRED_KEYS = {"features", "ttc_candidates", "mask", "target_ttc", "previous_target_ttc"}

    def __init__(self, manifest_path: str | Path):
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        self.root = manifest.parent
        with manifest.open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]
        if not self.samples:
            raise ValueError(f"Manifest contains no samples: {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.samples[index]
        sample_path = self.root / record["path"]
        with np.load(sample_path) as sample:
            missing = self.REQUIRED_KEYS - set(sample.files)
            if missing:
                raise KeyError(f"{sample_path} is missing: {sorted(missing)}")
            return {
                "features": torch.from_numpy(sample["features"].astype(np.float32)),
                "ttc_candidates": torch.from_numpy(sample["ttc_candidates"].astype(np.float32)),
                "mask": torch.from_numpy(sample["mask"].astype(bool)),
                "target_ttc": torch.tensor(float(sample["target_ttc"]), dtype=torch.float32),
                "previous_target_ttc": torch.tensor(
                    float(sample["previous_target_ttc"]), dtype=torch.float32
                ),
            }


def split_by_sequence(
    records: list[dict[str, object]],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, object]]]:
    """Split by source sequence to prevent adjacent-frame leakage."""
    sequence_ids = sorted({str(record["sequence_id"]) for record in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(sequence_ids)
    train_end = max(1, int(len(sequence_ids) * train_ratio))
    validation_end = max(train_end + 1, int(len(sequence_ids) * (train_ratio + validation_ratio)))
    assignments: dict[str, str] = {}
    for position, sequence_id in enumerate(sequence_ids):
        split = (
            "train"
            if position < train_end
            else "validation"
            if position < validation_end
            else "test"
        )
        assignments[sequence_id] = split
    result = {"train": [], "validation": [], "test": []}
    for record in records:
        result[assignments[str(record["sequence_id"])]].append(record)
    return result
