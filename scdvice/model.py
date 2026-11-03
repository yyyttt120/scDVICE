"""scDVICE model architecture."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import softmax


class GraphSequential(nn.Module):
    def __init__(self, graph_layer, *layers):
        super().__init__()
        for i, layer in enumerate((graph_layer, *layers)):
            self.add_module(f'module_{i}', layer)

    def forward(self, x, edge_index):
        x = self.module_0(x, edge_index)
        for layer in list(self.children())[1:]:
            x = layer(x)
        return x


class FeatureDecoder(nn.Module):

    def __init__(self, feature_dim: int, embd_dim: int, inter_dim: int, drop_p: float=0.0):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embd_dim, inter_dim), nn.Dropout(drop_p), nn.ReLU(),
            nn.Linear(inter_dim, inter_dim), nn.Dropout(drop_p), nn.ReLU(),
            nn.Linear(inter_dim, feature_dim), nn.Dropout(drop_p),
        )

    def forward(self, z):
        return self.decoder(z)


class MutualEncoder(nn.Module):

    def __init__(self, col_dim: int, row_dim: int, num_layers: int=4, drop_p: float=0.25):
        super().__init__()
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.num_layers = num_layers
        self.rows_layers = nn.ModuleList([
            GraphSequential(SAGEConv(row_dim, row_dim), nn.Dropout(drop_p), nn.LeakyReLU())
            for _ in range(num_layers)
        ])
        self.cols_layers = nn.ModuleList([
            GraphSequential(SAGEConv(col_dim, col_dim), nn.LeakyReLU(), nn.Dropout(drop_p))
            for _ in range(num_layers)
        ])

    def forward(self, x, knn_edge_index, ppi_edge_index):
        embedded = x.clone()
        for i in range(self.num_layers):
            embedded = self.cols_layers[i](embedded.T, ppi_edge_index).T
            embedded = self.rows_layers[i](embedded, knn_edge_index)
        return embedded


class TransformerConvReducerLayer(TransformerConv):

    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, add_self_loops=True, scale_param=2, **kwargs):
        super().__init__(in_channels, out_channels, heads, dropout, add_self_loops, **kwargs)
        self.scale_param = scale_param

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j += edge_attr
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if self.scale_param is not None:
            alpha = alpha - alpha.mean()
            alpha = alpha / (1 / self.scale_param * alpha.std())
            alpha = torch.sigmoid(alpha)
        else:
            alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j
        if edge_attr is not None:
            out += edge_attr
        out *= alpha.view(-1, self.heads, 1)
        return out


class DimEncoder(nn.Module):

    def __init__(self, feature_dim: int, inter_dim: int, embd_dim: int,
                 reducer: bool = False, drop_p: float = 0.2,
                 scale_param: int | None = 3, attr_dim: int = 0):
        super().__init__()
        self.reducer = reducer
        self.attr_dim = attr_dim
        self.encoder = GraphSequential(
            GCNConv(feature_dim, inter_dim), nn.LayerNorm(inter_dim),
            nn.LeakyReLU(inplace=True), nn.Dropout(drop_p),
        )
        atten_in = inter_dim + attr_dim
        if reducer:
            self.atten_layer = TransformerConvReducerLayer(
                atten_in, embd_dim, dropout=drop_p, add_self_loops=False,
                heads=1, scale_param=scale_param,
            )
        else:
            self.atten_layer = TransformerConv(atten_in, embd_dim, dropout=drop_p)

    def forward(self, x, edge_index, infrance=False, attr_feat=None):
        embedded = self.encoder(x.clone(), edge_index)
        if attr_feat is not None:
            embedded = torch.cat([embedded, attr_feat], dim=1)
        embedded, _ = self.atten_layer(embedded, edge_index, return_attention_weights=True)
        return embedded


class AEEncoder(nn.Module):

    def __init__(self, n_enc_1: int, n_enc_2: int, n_enc_3: int, n_input: int, n_z: int):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_enc_1)
        self.bn1 = nn.BatchNorm1d(num_features=n_enc_1, affine=False)
        self.fc2 = nn.Linear(n_enc_1, n_enc_2)
        self.bn2 = nn.BatchNorm1d(num_features=n_enc_2, affine=False)
        self.fc3 = nn.Linear(n_enc_2, n_enc_3)
        self.bn3 = nn.BatchNorm1d(num_features=n_enc_3, affine=False)
        self.fc4 = nn.Linear(n_enc_3, n_z)
        self.fc5 = nn.Linear(n_z, n_enc_3)
        self.bn5 = nn.BatchNorm1d(num_features=n_enc_3, affine=False)
        self.fc6 = nn.Linear(n_enc_3, n_enc_2)
        self.bn6 = nn.BatchNorm1d(num_features=n_enc_2, affine=False)
        self.fc7 = nn.Linear(n_enc_2, n_enc_1)
        self.bn7 = nn.BatchNorm1d(num_features=n_enc_1, affine=False)
        self.fc8 = nn.Linear(n_enc_1, n_input)

    def encoder(self, x):
        l1 = F.relu(self.bn1(self.fc1(x)))
        l2 = F.relu(self.bn2(self.fc2(l1)))
        l3 = F.relu(self.bn3(self.fc3(l2)))
        return self.fc4(l3)

    def decoder(self, z):
        l5 = F.relu(self.bn5(self.fc5(z)))
        l6 = F.relu(self.bn6(self.fc6(l5)))
        l7 = F.relu(self.bn7(self.fc7(l6)))
        return self.fc8(l7)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return (z, recon_x)


class AttrSingleHeadStructRetentionV2(nn.Module):

    def __init__(self, n_cells: int, n_genes: int, n_clusters: int,
                 n_enc_1: int, n_enc_2: int, n_enc_3: int,
                 inter_dim: int = 250, num_layers: int = 3):
        super().__init__()
        self.mutual_encoder = MutualEncoder(col_dim=n_cells, row_dim=n_genes, num_layers=num_layers)
        self.graph_encoder = DimEncoder(
            feature_dim=n_genes, inter_dim=inter_dim, embd_dim=n_clusters,
            scale_param=None, reducer=True, attr_dim=n_clusters,
        )
        self.feature_decoder = FeatureDecoder(feature_dim=n_genes, embd_dim=n_clusters, inter_dim=inter_dim, drop_p=0)
        self.attribute_encoder = AEEncoder(n_enc_1=n_enc_1, n_enc_2=n_enc_2, n_enc_3=n_enc_3, n_input=n_genes, n_z=n_clusters)
        self.clustering_head = nn.Linear(n_clusters, n_clusters)

    def forward(self, x, knn_edge_index, ppi_edge_index, collect_attention: bool=False):
        z, _ = self.attribute_encoder(x)
        embedded = self.mutual_encoder(x, knn_edge_index, ppi_edge_index)
        zg = self.graph_encoder(embedded, knn_edge_index, infrance=not collect_attention, attr_feat=z)
        cluster_repr = mix_z(z, zg)
        recon_x_graph = self.feature_decoder(cluster_repr)
        logits = self.clustering_head(cluster_repr)
        probs = F.softmax(logits, dim=1)
        return (zg, z, cluster_repr, recon_x_graph, logits, probs)


def mix_z(z1, z2):
    return z1 * 0.9 + z2 * 0.1
