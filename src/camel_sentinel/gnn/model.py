"""
model.py — HeteroGNN fraud detection model.

Architecture
------------
Input features (per node type)
    ↓
Linear projection to shared embedding dim (per node type)
    ↓
HeteroConv layer 1 — SAGEConv per edge type, then mean aggregation
    ↓  ReLU + Dropout
HeteroConv layer 2 — same structure
    ↓
Transaction node embeddings only
    ↓
MLP classification head  (Linear → ReLU → Dropout → Linear)
    ↓
Fraud risk score  ∈ ℝ  (raw logit — apply sigmoid for [0, 1] score)

Why SAGEConv?
  SAGEConv (Hamilton et al., 2017) aggregates a node's own features
  with a mean of its neighbours' features.  For this graph topology —
  where a transaction touches one account, one merchant, and (via the
  account) one device — most transactions have sparse, well-defined
  neighbourhoods.  SAGEConv handles these sparse, heterogeneous
  neighbourhoods better than attention-based convolutions, which need
  many neighbours to compute meaningful weights.

Why 2 layers?
  With 2 message-passing layers, a transaction node can aggregate
  information from:
    Layer 1: direct neighbours (account, merchant)
    Layer 2: neighbours of neighbours (customer, device — via account)
  A third layer would reach accounts and transactions of the same
  customer, which starts pulling in global structure and risks
  over-smoothing embeddings across unrelated nodes.

Output convention
  The model returns raw logits (pre-sigmoid).  Training uses
  BCEWithLogitsLoss for numerical stability.  Fraud risk scores for
  display are produced by sigmoid(logit).  We deliberately do NOT call
  this output a "probability" — it is an uncalibrated model score.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Node feature dimensions
# (must match the feature column counts in graph_builder.XXXXX_FEATURES)
# ─────────────────────────────────────────────────────────────────────────────

NODE_IN_DIMS: dict[str, int] = {
    "customer": 4,
    "account": 4,
    "transaction": 8,
    "merchant": 3,
    "device": 3,
}

# Edge types present in the HeteroData object
EDGE_TYPES = [
    ("customer", "owns", "account"),
    ("account", "rev_owns", "customer"),
    ("account", "makes", "transaction"),
    ("transaction", "rev_makes", "account"),
    ("transaction", "paid_to", "merchant"),
    ("merchant", "rev_paid_to", "transaction"),
    ("customer", "uses", "device"),
    ("device", "rev_uses_c", "customer"),
    ("account", "uses", "device"),
    ("device", "rev_uses_a", "account"),
]


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for transaction-level fraud scoring.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the shared node embedding space.
    num_layers : int
        Number of HeteroConv message-passing layers (default 2).
    dropout : float
        Dropout probability applied after each message-passing layer
        and between the MLP head layers.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # ── Linear projections: map each node type's raw features → hidden_dim
        # Separate projection per node type because they have different feature
        # dimensions and different semantics.
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_dim)
            for node_type, in_dim in NODE_IN_DIMS.items()
        })

        # ── HeteroConv layers
        # Each layer contains one SAGEConv per edge type.  The convolution
        # reads `hidden_dim`-dimensional features from both source and target
        # nodes and writes `hidden_dim`-dimensional output back to the target.
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=hidden_dim,
                    aggr="mean",
                    bias=True,
                )
                for edge_type in EDGE_TYPES
            }
            self.conv_layers.append(HeteroConv(conv_dict, aggr="sum"))

        # ── Layer normalisations (one per node type per layer)
        # Stabilises training without requiring careful learning-rate tuning.
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in NODE_IN_DIMS
            })
            for _ in range(num_layers)
        ])

        # ── MLP classification head (applied to transaction embeddings only)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_dict : dict
            {node_type: feature_tensor}  — raw input features.
        edge_index_dict : dict
            {edge_type: edge_index}  — graph connectivity.

        Returns
        -------
        logits : Tensor of shape [num_transactions]
            Raw pre-sigmoid scores.  Apply sigmoid(logits) to get
            fraud risk scores ∈ [0, 1].
        """
        # Project each node type's raw features to the shared embedding space.
        h = {
            node_type: F.relu(self.input_proj[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Message-passing layers.
        for i, conv in enumerate(self.conv_layers):
            h_new = conv(h, edge_index_dict)

            # Residual connection: add previous embedding to new one.
            # Prevents over-smoothing and helps gradient flow.
            for node_type in h_new:
                h_new[node_type] = self.layer_norms[i][node_type](
                    h_new[node_type] + h[node_type]
                )
                h_new[node_type] = F.relu(h_new[node_type])
                h_new[node_type] = F.dropout(
                    h_new[node_type], p=self.dropout, training=self.training
                )
            h = h_new

        # Classification head on transaction embeddings only.
        txn_emb = h["transaction"]                    # [N_transactions, hidden_dim]
        logits = self.head(txn_emb).squeeze(-1)       # [N_transactions]
        return logits

    def predict_scores(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Return sigmoid fraud risk scores ∈ [0, 1] (inference only)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
