"""Training loop and clustering evaluation."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .losses import compute_contrastive_loss, compute_recon_loss, compute_sep_loss, compute_struct_loss


def training_loss(model, x, cell_edges, gene_edges, adjacency, epoch):
    zg, z, mixed, recon, _, p = model(x, cell_edges, gene_edges)
    reconstruction = compute_recon_loss(recon, x)
    contrastive = compute_contrastive_loss(zg, z, target_topk=10,
                                           alpha=0.7548748789553268, adjacency_target=adjacency)
    structure = compute_struct_loss(p, mixed, k=8, tau_g=0.21421580356818337)
    separation = compute_sep_loss(p)
    warmup = min(1.0, max(epoch - 1, 0) / 10)
    return (0.3068239364069008 * reconstruction + 0.04977902548672514 * contrastive
            + 0.10485693692474955 * warmup * structure + 0.0032463750755358806 * separation)


def train_fixed(model, optimizer, epochs, loss_fn):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model, epoch)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
        optimizer.step()
        print(f"Epoch {epoch:03d}/{epochs}: loss={loss.item():.6f}", flush=True)


def evaluate_predictions(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("labels and predictions must be nonempty vectors of equal length")
    _, true = np.unique(labels, return_inverse=True)
    _, pred = np.unique(predictions, return_inverse=True)
    contingency = np.zeros((true.max() + 1, pred.max() + 1), dtype=np.int64)
    np.add.at(contingency, (true, pred), 1)
    row, col = linear_sum_assignment(-contingency)
    return {
        "acc": float(contingency[row, col].sum() / len(true)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "ari": float(adjusted_rand_score(true, pred)),
    }
