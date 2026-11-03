"""Data loading and preprocessing for the bundled datasets."""

import hashlib

import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from .graph import compute_knn_edge_index, compute_ppi_edge_index, getAdj, laplacianFliter


def verify_sha256(path, expected):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Data checksum does not match the pretrained model: {path}")


def read_data(path):
    with h5py.File(path, 'r') as handle:
        values = handle['exprs']
        if isinstance(values, h5py.Group):
            counts = sp.csr_matrix((values['data'][...], values['indices'][...], values['indptr'][...]), shape=tuple(values['shape'][...])).toarray()
        else:
            counts = values[...].astype(np.float32)
        labels = handle['obs/cell_type1'].asstr()[...]
        cells = handle['obs_names'].asstr()[...]
    if counts.ndim != 2 or len(labels) != len(counts) or len(cells) != len(counts):
        raise ValueError('Data matrix, labels and cell identifiers do not agree')
    return counts, labels, cells


def prepare_data(path, n_input, n_clusters, device):
    counts, labels, cells = read_data(path)
    adata = sc.AnnData(np.ceil(counts).astype(int))
    adata.obs_names = cells
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    if len(adata.obs_names) != len(cells) or not np.array_equal(adata.obs_names.to_numpy(), cells):
        raise ValueError('Preprocessing changed the input cell order or removed cells')
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_input, subset=True)
    sc.pp.scale(adata)
    if adata.n_vars != n_input:
        raise ValueError(f'Expected {n_input} input genes, got {adata.n_vars}')
    k = int(len(cells) / 2 / n_clusters) + 10
    if k >= len(cells):
        raise ValueError('Dataset is too small for the cell graph')
    filtered = laplacianFliter(getAdj(adata.X, k=k), adata.X, 1)
    x_cpu = torch.tensor(filtered, dtype=torch.float32)
    if not torch.isfinite(x_cpu).all():
        raise ValueError('Preprocessed data contains nonfinite values')
    cell_edges = compute_knn_edge_index(x_cpu, k=k, device='cpu')
    gene_edges = compute_ppi_edge_index(x_cpu, k=min(20, adata.n_vars // 100), threshold=0.5, device='cpu')
    return x_cpu.to(device), cell_edges.to(device), gene_edges.to(device), labels, cells
