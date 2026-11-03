"""Train scDVICE on a bundled dataset."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from scdvice.checkpoint import initialize_head, load_pretrained, model_dimensions, read_checkpoint
from scdvice.data import prepare_data, verify_sha256
from scdvice.losses import edge_index_to_dense_adj
from scdvice.model import AttrSingleHeadStructRetentionV2
from scdvice.training import evaluate_predictions, train_fixed, training_loss


ROOT = Path(__file__).resolve().parent
EPOCHS = 200


def check_output_path(output, data_dir, weights_dir):
    output = Path(output).resolve()
    for protected in (Path(data_dir).resolve(), Path(weights_dir).resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('The output directory must be separate from the input directories')


def main():
    manifest = json.loads((ROOT / 'pretrained' / 'manifest.json').read_text())
    parser = argparse.ArgumentParser(description='Train scDVICE on a bundled dataset')
    parser.add_argument('--dataset', required=True, choices=sorted(manifest))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--pretrained-dir', type=Path, default=ROOT / 'pretrained')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    check_output_path(args.output_dir, args.data_dir, args.pretrained_dir)
    data_path = args.data_dir / args.dataset / 'data.h5'
    weights_path = args.pretrained_dir / f'{args.dataset}.pt'
    verify_sha256(data_path, manifest[args.dataset]['data_sha256'])
    verify_sha256(weights_path, manifest[args.dataset]['checkpoint_sha256'])
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(29)
    device = torch.device(args.device)
    dimensions = model_dimensions(read_checkpoint(weights_path))
    x, cell_edges, gene_edges, labels, cells = prepare_data(data_path, dimensions['n_genes'], dimensions['n_clusters'], device)
    if len(x) != dimensions['n_cells']:
        raise ValueError('Dataset cell count is incompatible with the pretrained model')
    model = AttrSingleHeadStructRetentionV2(**dimensions).to(device)
    checkpoint = load_pretrained(model, weights_path, len(x), dimensions['n_clusters'])
    model.eval()
    with torch.no_grad():
        _, _, mixed, _, _, _ = model(x, cell_edges, gene_edges)
    initialize_head(model, mixed, checkpoint)
    del checkpoint
    head = list(model.clustering_head.parameters())
    backbone = [p for name, p in model.named_parameters() if not name.startswith('clustering_head')]
    optimizer = torch.optim.Adam([{'params': backbone, 'lr': 8.18990591981491e-07},
                                  {'params': head, 'lr': 0.00040949529599074554}], weight_decay=1e-5)
    adjacency = edge_index_to_dense_adj(cell_edges, len(x), device)

    train_fixed(model, optimizer, EPOCHS,
                lambda current, epoch: training_loss(current, x, cell_edges, gene_edges, adjacency, epoch))
    model.eval()
    with torch.no_grad():
        _, _, mixed, _, _, probabilities = model(x, cell_edges, gene_edges)
    if not torch.isfinite(mixed).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite final predictions')
    predictions = probabilities.argmax(dim=1).cpu().numpy()
    scores = evaluate_predictions(labels, predictions)
    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'results.npz', embedding=mixed.cpu().numpy(),
                        probabilities=probabilities.cpu().numpy(), labels=predictions,
                        cell_ids=np.asarray(cells, dtype=str))
    (output / 'metrics.json').write_text(json.dumps({'dataset': args.dataset, 'epoch': EPOCHS, **scores}, indent=2) + '\n')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
