import h5py
import numpy as np
import pytest
from scipy import sparse

from scdvice.data import read_data, verify_sha256


def test_read_sparse_hdf5_preserves_cell_order(tmp_path):
    path = tmp_path / 'data.h5'
    x = sparse.csr_matrix(np.array([[1, 0, 3], [0, 2, 4]], dtype=np.int32))
    with h5py.File(path, 'w') as f:
        group = f.create_group('exprs')
        for key in ('data', 'indices', 'indptr'):
            group[key] = getattr(x, key)
        group['shape'] = x.shape
        f['obs_names'] = np.array([b'cell_b', b'cell_a'])
        f['obs/cell_type1'] = np.array([b'B', b'A'])
    counts, labels, cells = read_data(path)
    np.testing.assert_array_equal(counts, x.toarray())
    assert list(cells) == ['cell_b', 'cell_a']
    assert list(labels) == ['B', 'A']


def test_checksum_rejects_different_dataset(tmp_path):
    path = tmp_path / 'data.h5'
    path.write_bytes(b'another dataset')
    with pytest.raises(ValueError, match='checksum'):
        verify_sha256(path, '0' * 64)
