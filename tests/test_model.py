import pytest
import torch

from monocular_ttc.model import TemporalRecurrentRegressor, TemporalWeightMLP


def test_temporal_weights_are_normalized_and_masked() -> None:
    model = TemporalWeightMLP(feature_dim=6, dropout=0.0)
    features = torch.zeros((2, 4, 6))
    candidates = torch.tensor([[20.0, 20.0, 5.0, 3.0], [20.0, 7.0, 6.0, 5.0]])
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    fused, weights = model(features, candidates, mask)
    assert fused.shape == (2,)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.all(weights[~mask] == 0)


def test_empty_sequence_is_rejected() -> None:
    model = TemporalWeightMLP(feature_dim=6)
    with pytest.raises(ValueError):
        model(torch.zeros((1, 2, 6)), torch.ones((1, 2)), torch.zeros((1, 2), dtype=torch.bool))


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_recurrent_baseline_outputs_bounded_ttc(cell: str) -> None:
    model = TemporalRecurrentRegressor(feature_dim=6, cell=cell)
    features = torch.zeros((2, 4, 6))
    candidates = torch.ones((2, 4))
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    prediction, weights = model(features, candidates, mask)
    assert prediction.shape == (2,)
    assert torch.all((prediction >= 0.05) & (prediction <= 20.0))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
