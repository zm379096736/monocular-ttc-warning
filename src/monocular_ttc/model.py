from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class TemporalWeightMLP(nn.Module):
    """Learn interpretable per-frame weights, then fuse physical TTC candidates."""

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dims: Sequence[int] = (32, 16),
        dropout: float = 0.1,
        min_ttc: float = 0.05,
        max_ttc: float = 20.0,
    ) -> None:
        super().__init__()
        dimensions = [feature_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 2):
            layers.extend(
                [
                    nn.Linear(dimensions[index], dimensions[index + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.scorer = nn.Sequential(*layers)
        self.min_ttc = min_ttc
        self.max_ttc = max_ttc

    def forward(
        self,
        features: Tensor,
        ttc_candidates: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return fused TTC and temporal weights.

        Args:
            features: [batch, time, feature_dim] normalized features.
            ttc_candidates: [batch, time] physical per-frame TTC estimates.
            mask: [batch, time], True for valid frames.
        """
        if features.ndim != 3 or ttc_candidates.ndim != 2:
            raise ValueError("Expected features [B,T,F] and candidates [B,T]")
        if features.shape[:2] != ttc_candidates.shape:
            raise ValueError("Feature and TTC time dimensions must match")
        scores = self.scorer(features).squeeze(-1)
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()
        if (~mask).all(dim=1).any():
            raise ValueError("Every sample must contain at least one valid frame")
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        candidates = ttc_candidates.clamp(self.min_ttc, self.max_ttc)
        fused = torch.sum(weights * candidates, dim=1)
        return fused, weights


def trend_consistency_loss(prediction: Tensor, target: Tensor, previous_target: Tensor) -> Tensor:
    """Penalize a predicted TTC change whose sign disagrees with ground truth."""
    true_delta = target - previous_target
    predicted_delta = prediction - previous_target
    return torch.relu(-(true_delta * predicted_delta)).mean()
