from monocular_ttc.risk import RiskLevel, RiskPolicy


def test_thresholds_and_trend_upgrade() -> None:
    policy = RiskPolicy()
    assert policy.classify(6.0).level == RiskLevel.SAFE
    assert policy.classify(4.0).level == RiskLevel.CAUTION
    assert policy.classify(2.0).level == RiskLevel.DANGER
    decision = policy.classify(6.0, delta_ttc=-0.5)
    assert decision.level == RiskLevel.CAUTION
    assert decision.upgraded_by_trend
