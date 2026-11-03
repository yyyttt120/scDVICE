import numpy as np
import pytest
import torch
from torch import nn

from scdvice.training import train_fixed, evaluate_predictions


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))


def test_fixed_training_returns_last_state_not_best_metric():
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    seen = []

    def loss_fn(current, epoch):
        seen.append(epoch)
        return (current.weight - 1.0) ** 2

    train_fixed(model, optimizer, 3, loss_fn)
    assert seen == [1, 2, 3]
    assert model.weight.item() == pytest.approx(2.0)


def test_nonfinite_loss_fails_instead_of_returning_partial_model():
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    with pytest.raises(FloatingPointError):
        train_fixed(model, optimizer, 3, lambda m, epoch: m.weight * torch.tensor(float('nan')))


def test_cluster_metrics_match_permuted_labels():
    metrics = evaluate_predictions(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
    assert metrics == pytest.approx({'acc': 1.0, 'nmi': 1.0, 'ari': 1.0})


def test_reject_output_inside_input_directories(tmp_path):
    from train import check_output_path
    data_dir = tmp_path / 'datasets'
    weights_dir = tmp_path / 'pretrained'
    data_dir.mkdir()
    weights_dir.mkdir()
    with pytest.raises(ValueError, match='output'):
        check_output_path(weights_dir / 'results', data_dir, weights_dir)
    with pytest.raises(ValueError, match='output'):
        check_output_path(tmp_path, data_dir, weights_dir)
    check_output_path(tmp_path / 'results', data_dir, weights_dir)


def test_training_loss_has_finite_gradients():
    from scdvice.model import AttrSingleHeadStructRetentionV2
    from scdvice.losses import edge_index_to_dense_adj
    from scdvice.training import training_loss
    torch.manual_seed(29)
    model = AttrSingleHeadStructRetentionV2(12, 8, 3, 10, 6, 4, inter_dim=5, num_layers=1)
    x = torch.randn(12, 8)
    cell_edges = torch.tensor([list(range(12)), [(i + 1) % 12 for i in range(12)]])
    gene_edges = torch.tensor([list(range(8)), [(i + 1) % 8 for i in range(8)]])
    adjacency = edge_index_to_dense_adj(cell_edges, 12, 'cpu')
    loss = training_loss(model, x, cell_edges, gene_edges, adjacency, epoch=6)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_labels_do_not_change_training_updates():
    from scdvice.model import AttrSingleHeadStructRetentionV2
    from scdvice.losses import edge_index_to_dense_adj
    from scdvice.training import training_loss
    torch.manual_seed(22)
    first = AttrSingleHeadStructRetentionV2(12, 8, 3, 10, 6, 4, inter_dim=5, num_layers=1)
    second = AttrSingleHeadStructRetentionV2(12, 8, 3, 10, 6, 4, inter_dim=5, num_layers=1)
    second.load_state_dict(first.state_dict())
    x = torch.randn(12, 8)
    cells = torch.tensor([list(range(12)), [(i + 1) % 12 for i in range(12)]])
    genes = torch.tensor([list(range(8)), [(i + 1) % 8 for i in range(8)]])
    adjacency = edge_index_to_dense_adj(cells, 12, 'cpu')
    for model, labels in ((first, np.arange(12) % 3), (second, np.arange(12) % 2)):
        torch.manual_seed(29)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_fixed(model, optimizer, 2,
                    lambda current, epoch: training_loss(current, x, cells, genes, adjacency, epoch))
        evaluate_predictions(labels, np.arange(12) % 3)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
