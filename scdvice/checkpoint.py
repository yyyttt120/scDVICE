"""Load the pretrained model and initialize its clustering head."""

from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model_state_dict", "Z", "Zg"} <= checkpoint.keys():
        raise ValueError(f"Invalid pretrained weights: {path}")
    return checkpoint


def model_dimensions(checkpoint):
    state = checkpoint["model_state_dict"]
    try:
        layers = {int(k.split('.')[2]) for k in state if k.startswith('mutual_encoder.rows_layers.')}
        return {
            "n_cells": int(state['mutual_encoder.cols_layers.0.module_0.lin_l.weight'].shape[0]),
            "n_genes": int(state['attribute_encoder.fc1.weight'].shape[1]),
            "n_clusters": int(state['attribute_encoder.fc4.weight'].shape[0]),
            "n_enc_1": int(state['attribute_encoder.fc1.weight'].shape[0]),
            "n_enc_2": int(state['attribute_encoder.fc2.weight'].shape[0]),
            "n_enc_3": int(state['attribute_encoder.fc3.weight'].shape[0]),
            "inter_dim": int(state['feature_decoder.decoder.0.weight'].shape[0]),
            "num_layers": max(layers) + 1,
        }
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("Incomplete pretrained model structure") from exc


def load_pretrained(model, path, n_cells, n_clusters):
    checkpoint = read_checkpoint(path)
    z = torch.as_tensor(checkpoint['Z'])
    zg = torch.as_tensor(checkpoint['Zg'])
    if z.shape != (n_cells, n_clusters) or zg.shape != z.shape:
        raise ValueError("Pretrained embeddings are incompatible with the dataset")
    state = checkpoint['model_state_dict']
    current = model.state_dict()
    missing = set(current) - set(state) - {'clustering_head.weight', 'clustering_head.bias'}
    unexpected = set(state) - set(current)
    mismatched = [key for key in state.keys() & current.keys() if state[key].shape != current[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f"Pretrained weights are incompatible: missing={sorted(missing)}, unexpected={sorted(unexpected)}, mismatched={sorted(mismatched)}")
    model.load_state_dict(state, strict=False)
    return checkpoint


def initialize_head(model, features, checkpoint):
    saved_z = torch.as_tensor(checkpoint['Z']).cpu().numpy()
    saved_zg = torch.as_tensor(checkpoint['Zg']).cpu().numpy()
    saved_mixed = 0.9 * saved_z + 0.1 * saved_zg
    clusters = model.clustering_head.out_features
    pseudo = SpectralClustering(n_clusters=clusters, affinity='nearest_neighbors', n_neighbors=min(10, len(saved_mixed) - 1), assign_labels='kmeans', random_state=42).fit_predict(saved_mixed)
    probe = LogisticRegression(max_iter=5000, random_state=42, C=100.0)
    probe.fit(features.detach().cpu().numpy(), pseudo)
    with torch.no_grad():
        model.clustering_head.weight.copy_(torch.as_tensor(probe.coef_, dtype=model.clustering_head.weight.dtype, device=model.clustering_head.weight.device))
        model.clustering_head.bias.copy_(torch.as_tensor(probe.intercept_, dtype=model.clustering_head.bias.dtype, device=model.clustering_head.bias.device))
