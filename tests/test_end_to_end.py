import hashlib
import json
import sys

import h5py
import numpy as np
from scipy import sparse
import torch

import train
from scdvice.model import AttrSingleHeadStructRetentionV2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_small_end_to_end_run(tmp_path, monkeypatch):
    name = 'Example'
    data_path = tmp_path / 'datasets' / name / 'data.h5'
    weights_path = tmp_path / 'pretrained' / f'{name}.pt'
    data_path.parent.mkdir(parents=True)
    weights_path.parent.mkdir()
    rng = np.random.default_rng(123)
    matrix = sparse.csr_matrix(rng.poisson(rng.uniform(0.5, 4, size=(1, 180)), size=(48, 180)))
    with h5py.File(data_path, 'w') as f:
        group = f.create_group('exprs')
        for key in ('data', 'indices', 'indptr'):
            group[key] = getattr(matrix, key)
        group['shape'] = matrix.shape
        f['obs_names'] = np.array([f'cell{i}'.encode() for i in range(48)])
        f['obs/cell_type1'] = np.array([f'type{i % 3}'.encode() for i in range(48)])
    model = AttrSingleHeadStructRetentionV2(48, 100, 3, 16, 12, 6, inter_dim=6, num_layers=1)
    state = {key: value for key, value in model.state_dict().items() if not key.startswith('clustering_head.')}
    z = torch.tensor(rng.normal(size=(48, 3)), dtype=torch.float32)
    torch.save({'model_state_dict': state, 'Z': z, 'Zg': z}, weights_path)
    manifest = {name: {'data_sha256': digest(data_path), 'checkpoint_sha256': digest(weights_path)}}
    (weights_path.parent / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    monkeypatch.setattr(train, 'EPOCHS', 2)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--dataset', name, '--device', 'cpu'])
    train.main()
    result = np.load(tmp_path / 'outputs' / name / 'results.npz')
    metrics = json.loads((tmp_path / 'outputs' / name / 'metrics.json').read_text())
    assert result['embedding'].shape == (48, 3)
    assert result['probabilities'].shape == (48, 3)
    assert list(result['cell_ids']) == [f'cell{i}' for i in range(48)]
    assert metrics['epoch'] == 2
    assert all(np.isfinite(metrics[key]) for key in ('nmi', 'ari', 'acc'))
