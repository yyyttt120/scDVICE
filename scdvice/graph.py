"""Cell and gene graph construction."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

EPS = 1e-8


def getAdj(x, k=10):
    n_samples = x.shape[0]
    x_normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    similarities = np.dot(x_normalized, x_normalized.T)
    similarities = (similarities + 1) / 2
    k = min(k, n_samples - 1)
    indices = np.argsort(-similarities, axis=1)[:, 1:k + 1]
    adjacency = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        adjacency[i, indices[i]] = similarities[i, indices[i]]
    return 0.5 * (adjacency + adjacency.T)


def laplacianFliter(adjacency, features, n):
    if sp.issparse(adjacency):
        adjacency = adjacency.toarray()
    a_hat = adjacency + np.eye(adjacency.shape[0])
    degree = np.sum(a_hat, axis=1).reshape((a_hat.shape[0],))
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    d_inv_sqrt = np.diag(degree_inv_sqrt)
    a_norm = np.dot(np.dot(d_inv_sqrt, a_hat), d_inv_sqrt)
    filtered_features = features.copy()
    for _ in range(n):
        filtered_features = np.dot(a_norm, filtered_features)
    return filtered_features


def compute_knn_edge_index(x, k=10, device='cuda'):
    x_normalized = F.normalize(x, p=2, dim=1)
    similarities = torch.mm(x_normalized, x_normalized.t())
    similarities.fill_diagonal_(-float('inf'))
    _, topk_indices = torch.topk(similarities, k=k, dim=1)
    src_nodes = torch.arange(x.shape[0], device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst_nodes = topk_indices.reshape(-1)
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
    edge_index, _ = add_self_loops(edge_index)
    return edge_index


def compute_ppi_edge_index(x, k=20, threshold=0.5, device='cuda'):
    x_centered = x.T - x.T.mean(dim=1, keepdim=True)
    x_std = x_centered / (x_centered.std(dim=1, keepdim=True) + EPS)
    corr_matrix = torch.mm(x_std, x_std.t()) / max(x.shape[0] - 1, 1)
    corr_matrix = (corr_matrix + corr_matrix.t()) / 2
    corr_matrix.fill_diagonal_(-float('inf'))
    topk_values, topk_indices = torch.topk(corr_matrix, k=k, dim=1)
    mask = topk_values >= threshold
    src_nodes = torch.arange(x.shape[1], device=device).unsqueeze(1).expand(-1, k)[mask]
    dst_nodes = topk_indices[mask]
    return torch.stack([src_nodes, dst_nodes], dim=0)
