"""Training loss functions."""

import torch
import torch.nn.functional as F

EPS = 1e-8


def edge_index_to_dense_adj(edge_index, n_samples, device, include_self=True, symmetric=True):
    adjacency = torch.zeros((n_samples, n_samples), dtype=torch.float32, device=device)
    if edge_index is not None and edge_index.numel() > 0:
        adjacency[edge_index[0], edge_index[1]] = 1.0
    if symmetric:
        adjacency = torch.maximum(adjacency, adjacency.t())
    if include_self:
        adjacency.fill_diagonal_(1.0)
    return adjacency


def compute_nonnegative_similarity(features):
    features = F.normalize(features, p=2, dim=1)
    similarity = torch.mm(features, features.t())
    similarity = F.relu(similarity)
    return torch.clamp(similarity, min=0.0, max=1.0)


def topk_truncate_similarity(similarity, k, include_self=True, symmetric=True):
    n_samples = similarity.shape[0]
    if k is None or k <= 0 or k >= n_samples:
        truncated = similarity.clone()
    else:
        working = similarity.clone()
        if not include_self:
            working.fill_diagonal_(0.0)
        topk_values, topk_indices = torch.topk(working, k=k, dim=1)
        truncated = torch.zeros_like(similarity)
        truncated.scatter_(1, topk_indices, topk_values)
    if symmetric:
        truncated = torch.maximum(truncated, truncated.t())
    if include_self:
        truncated.fill_diagonal_(1.0)
    return torch.clamp(truncated, min=0.0, max=1.0)


def compute_contrastive_loss(zg, z, target_topk, alpha, adjacency_target):
    zg_norm = F.normalize(zg, p=2, dim=1)
    z_norm = F.normalize(z, p=2, dim=1)
    similarity = F.relu(torch.mm(zg_norm, z_norm.t()))
    similarity = torch.clamp(similarity, min=0.0, max=1.0)
    with torch.no_grad():
        feature_similarity = compute_nonnegative_similarity(z.detach())
        feature_similarity = topk_truncate_similarity(feature_similarity, k=target_topk, include_self=True, symmetric=True)
        target = alpha * adjacency_target + (1.0 - alpha) * feature_similarity
        target = torch.clamp(target, min=0.0, max=1.0)
    diff = similarity - target
    return torch.sum(diff.pow(2)) / diff.numel()


def compute_recon_loss(recon_x_graph: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
    recon_x_graph_normalized = (recon_x_graph - recon_x_graph.mean(dim=0)) / (recon_x_graph.std(dim=0) + EPS)
    return F.mse_loss(recon_x_graph_normalized, x_raw)


def compute_struct_loss(p: torch.Tensor, cluster_repr: torch.Tensor, k: int, tau_g: float, eps: float=EPS):
    cluster_repr_norm = F.normalize(cluster_repr.detach(), p=2, dim=1)
    sim = torch.mm(cluster_repr_norm, cluster_repr_norm.t())
    n = sim.size(0)
    effective_k = min(int(k), max(n - 1, 0))
    if effective_k <= 0:
        return p.new_tensor(0.0)
    sim_no_self = sim.clone()
    sim_no_self.fill_diagonal_(-float('inf'))
    _, idx = torch.topk(sim_no_self, k=effective_k, dim=1)
    mask = torch.zeros_like(sim)
    mask.scatter_(1, idx, 1.0)
    tau_g = max(float(tau_g), eps)
    a_tilde = mask * torch.exp(torch.clamp(sim / tau_g, min=-50.0, max=50.0))
    a_tilde.fill_diagonal_(0.0)
    a = 0.5 * (a_tilde + a_tilde.t())
    a_sum = a.sum().clamp(min=eps)
    a_star = a / a_sum
    p_sq = torch.sum(p * p, dim=1, keepdim=True)
    dist = torch.clamp(p_sq + p_sq.t() - 2.0 * torch.mm(p, p.t()), min=0.0)
    return torch.sum(a_star * dist)


def compute_sep_loss(p: torch.Tensor):
    g = torch.mm(p.t(), p) / max(float(p.shape[0]), 1.0)
    offdiag = g - torch.diag_embed(torch.diag(g))
    return torch.sum(offdiag.pow(2))
