from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FrameFeatures:
    ttc: float
    delta_ttc: float
    radial_expansion: float
    detector_confidence: float
    flow_consistency: float
    box_growth: float

    def as_array(self) -> np.ndarray:
        return np.asarray(list(asdict(self).values()), dtype=np.float32)


def make_frame_features(
    ttc: float,
    previous_ttc: float | None,
    radial_expansion: float,
    detector_confidence: float,
    flow_consistency: float,
    box_growth: float,
    max_ttc: float,
) -> FrameFeatures:
    current = float(np.clip(ttc, 0.0, max_ttc))
    delta = 0.0 if previous_ttc is None else current - float(previous_ttc)
    return FrameFeatures(
        ttc=current,
        delta_ttc=float(np.clip(delta, -max_ttc, max_ttc)),
        radial_expansion=float(radial_expansion),
        detector_confidence=float(np.clip(detector_confidence, 0.0, 1.0)),
        flow_consistency=float(np.clip(flow_consistency, 0.0, 1.0)),
        box_growth=float(box_growth),
    )


class FeatureNormalizer:
    """Serializable z-score normalization fitted on the training split only."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, features: np.ndarray, mask: np.ndarray | None = None) -> FeatureNormalizer:
        values = features if mask is None else features[mask.astype(bool)]
        if values.size == 0:
            raise ValueError("Cannot fit normalizer on an empty feature array")
        self.mean = values.mean(axis=0).astype(np.float32)
        self.std = values.std(axis=0).astype(np.float32)
        self.std = np.maximum(self.std, 1e-6)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")
        return ((features - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer has no fitted state")
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> FeatureNormalizer:
        return cls(
            np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32)
        )
