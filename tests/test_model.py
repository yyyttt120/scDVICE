import pytest
import torch

from scdvice.model import AttrSingleHeadStructRetentionV2, mix_z
from scdvice.losses import compute_recon_loss, compute_struct_loss, compute_sep_loss
from scdvice.checkpoint import load_pretrained


def test_model_uses_the_same_mixed_representation_for_head_and_decoder():
    torch.manual_seed(7)
    model = AttrSingleHeadStructRetentionV2(12, 8, 3, 10, 6, 4, inter_dim=5, num_layers=1).eval()
    x = torch.randn(12, 8)
    cell_edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]])
    gene_edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
    with torch.no_grad():
        zg, z, mixed, recon, logits, p = model(x, cell_edges, gene_edges)
    torch.testing.assert_close(mixed, mix_z(z, zg))
    torch.testing.assert_close(logits, model.clustering_head(mixed))
    torch.testing.assert_close(recon, model.feature_decoder(mixed))
    assert torch.isfinite(p).all()
    assert torch.isfinite(compute_recon_loss(recon, x))
    assert torch.isfinite(compute_struct_loss(p, mixed, 3, 0.2))
    assert torch.isfinite(compute_sep_loss(p))


def test_checkpoint_rejects_incompatible_cell_dimensions(tmp_path):
    model = AttrSingleHeadStructRetentionV2(12, 8, 3, 10, 6, 4, inter_dim=5, num_layers=1)
    other = AttrSingleHeadStructRetentionV2(13, 8, 3, 10, 6, 4, inter_dim=5, num_layers=1)
    path = tmp_path / 'weights.pt'
    state = {k: v for k, v in other.state_dict().items() if not k.startswith('clustering_head.')}
    torch.save({'model_state_dict': state, 'Z': torch.zeros(13, 3), 'Zg': torch.zeros(13, 3)}, path)
    with pytest.raises(ValueError, match='incompatible'):
        load_pretrained(model, path, 12, 3)


def test_model_is_independent_of_pyg_sequential_class_template(monkeypatch):
    from torch_geometric.nn import sequential

    def incompatible_template(self, x, edge_index):
        raise RuntimeError('another graph sequential changed its class template')

    monkeypatch.setattr(sequential.Sequential, 'forward', incompatible_template)
    model = AttrSingleHeadStructRetentionV2(12, 8, 3, 10, 6, 4, inter_dim=5, num_layers=1).eval()
    x = torch.randn(12, 8)
    cell_edges = torch.tensor([[0, 1], [1, 0]])
    gene_edges = torch.tensor([[0, 1], [1, 0]])
    with torch.no_grad():
        assert model(x, cell_edges, gene_edges)[2].shape == (12, 3)
