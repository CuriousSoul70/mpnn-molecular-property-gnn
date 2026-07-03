"""
model.py
--------
A Message Passing Neural Network (MPNN) for molecular property prediction.

This follows the architecture described in the project doc, Section 3.2:
  - Atoms are nodes, bonds are edges
  - Atoms exchange "messages" using an edge-conditioned convolution
    (edge features -> per-edge weight matrix -> message)
  - A readout (global pooling) turns per-atom features into one
    molecule-level vector
  - A small MLP maps that vector to the predicted property

This is intentionally the "MPNN" tier from the doc's GNN progression
(GCN -> GAT -> MPNN -> SchNet -> DimeNet -> GemNet). It is a good, honest
starting point: more expressive than plain GCN/GAT, much simpler to get
running correctly than SchNet/DimeNet, and it is exactly what the original
QM9 benchmark papers used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool


class MPNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64,
                 num_mp_steps=3, readout="mean", dropout=0.2):
        """
        node_feat_dim : size of each atom's input feature vector
        edge_feat_dim : size of each bond's input feature vector
                        (e.g. bond type, or 3D distance for QM9)
        hidden_dim    : size of the hidden atom representation
        num_mp_steps  : how many rounds of message passing (like MPNN's
                        "message passing" stage in the doc)
        readout       : "mean" or "sum" pooling over atoms -> molecule vector
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mp_steps = num_mp_steps

        # Project raw atom features into hidden space
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # Edge network: maps edge features -> a (hidden_dim x hidden_dim)
        # matrix used to transform neighbor messages. This is what makes
        # it an MPNN rather than a plain GCN: bonds/distances directly
        # shape how information flows, not just adjacency.
        edge_net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, edge_net, aggr="mean")
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        readout_fn = {"mean": global_mean_pool, "sum": global_add_pool}
        self.readout = readout_fn[readout]
        self.dropout = nn.Dropout(dropout)

        # Final MLP: molecule vector -> scalar property prediction
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        h = self.node_encoder(x)

        for _ in range(self.num_mp_steps):
            m = F.relu(self.conv(h, edge_index, edge_attr))
            m = self.dropout(m)
            h = self.gru(m, h)

        mol_vec = self.readout(h, batch)
        out = self.head(mol_vec)
        return out.view(-1)
