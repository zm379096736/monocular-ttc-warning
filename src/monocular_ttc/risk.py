from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class RiskLevel(IntEnum):
    SAFE = 0
    CAUTION = 1
    DANGER = 2


@dataclass(frozen=True)
class RiskDecision:
    level: RiskLevel
    upgraded_by_trend: bool


@dataclass(frozen=True)
class RiskPolicy:
    danger_ttc_seconds: float = 3.0
    caution_ttc_seconds: float = 5.0
    trend_upgrade_threshold: float = -0.30

    def classify(self, ttc_seconds: float, delta_ttc: float = 0.0) -> RiskDecision:
        if ttc_seconds <= self.danger_ttc_seconds:
            base = RiskLevel.DANGER
        elif ttc_seconds <= self.caution_ttc_seconds:
            base = RiskLevel.CAUTION
        else:
            base = RiskLevel.SAFE

        upgraded = delta_ttc <= self.trend_upgrade_threshold and base < RiskLevel.DANGER
        level = RiskLevel(base + 1) if upgraded else base
        return RiskDecision(level=level, upgraded_by_trend=upgraded)
