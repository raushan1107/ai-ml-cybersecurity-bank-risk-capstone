"""
detector.py — Inference wrapper and configurable fraud-decision policy.

The model produces:
    fraud_risk_score ∈ [0, 1]   (uncalibrated — not a probability)

A separate configurable policy converts that score into a decision:
    APPROVE   score < low_threshold
    REVIEW    low_threshold ≤ score < high_threshold
    BLOCK     score ≥ high_threshold

Thresholds are configurable because the right operating point depends on
the cost of a false positive vs. a missed fraud, which varies by context.
Do not treat these defaults as universal fraud-detection rules.

This module is used by the FastAPI router.  It loads the pre-trained model
once at module import time (lazy, on first call) and serves subsequent
requests from the cached objects.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FraudPolicy:
    """
    Converts a fraud risk score into a discrete decision.

    Thresholds are configurable — defaults are illustrative, not universal.
    """
    low_threshold: float = 0.30
    high_threshold: float = 0.70

    def decide(self, score: float) -> str:
        if score >= self.high_threshold:
            return "BLOCK"
        elif score >= self.low_threshold:
            return "REVIEW"
        return "APPROVE"

    def to_dict(self) -> dict:
        return {
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "approve_range": f"score < {self.low_threshold}",
            "review_range": f"{self.low_threshold} ≤ score < {self.high_threshold}",
            "block_range": f"score ≥ {self.high_threshold}",
        }


DEFAULT_POLICY = FraudPolicy()


# ─────────────────────────────────────────────────────────────────────────────
# Model cache (lazy load on first call)
# ─────────────────────────────────────────────────────────────────────────────

_CACHE: dict[str, Any] = {}


def _ensure_loaded(models_dir: Path | str | None = None) -> None:
    """Load the trained model + meta into _CACHE if not already loaded."""
    if "model" in _CACHE:
        return
    from camel_sentinel.gnn.trainer import load_model
    try:
        model, meta, baselines = load_model(models_dir)
        _CACHE["model"] = model
        _CACHE["meta"] = meta
        _CACHE["baselines"] = baselines
        _CACHE["loaded"] = True
    except FileNotFoundError:
        _CACHE["loaded"] = False
        _CACHE["error"] = (
            "GNN model not found.  Run notebooks/04_gnn_fraud_detection.ipynb "
            "to train and save the model first."
        )


def is_model_ready(models_dir: Path | str | None = None) -> bool:
    _ensure_loaded(models_dir)
    return _CACHE.get("loaded", False)


def get_load_error() -> str:
    return _CACHE.get("error", "Model not loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def score_transaction(
    transaction_features: dict,
    account_features: dict,
    customer_features: dict,
    merchant_features: dict,
    device_features: dict,
    policy: FraudPolicy | None = None,
    models_dir: Path | str | None = None,
) -> dict:
    """
    Score a single transaction through the GNN and return a structured result.

    Because the GNN needs neighbourhood context to compute its score, this
    function constructs a minimal single-transaction subgraph:
      customer (1) → owns → account (1) → makes → transaction (1)
      transaction (1) → paid_to → merchant (1)
      account (1) → uses → device (1)
      + all reverse edges

    This is an approximation of the full graph inference — the model was
    trained on a densely connected graph, so a single-transaction subgraph
    will not capture shared-device patterns unless those patterns are
    encoded in the input features (e.g., device.num_accounts).

    For full graph inference (including ring detection), use
    score_transaction_in_graph() with the pre-built HeteroData.

    Parameters
    ----------
    transaction_features : dict
        Keys matching TRANSACTION_FEATURE_NAMES (after standardisation).
    account_features : dict
        Keys matching ACCOUNT_FEATURE_NAMES.
    customer_features : dict
        Keys matching CUSTOMER_FEATURE_NAMES.
    merchant_features : dict
        Keys matching MERCHANT_FEATURE_NAMES.
    device_features : dict
        Keys matching DEVICE_FEATURE_NAMES.
    policy : FraudPolicy, optional
        Decision thresholds.  Defaults to DEFAULT_POLICY.
    """
    from camel_sentinel.gnn.graph_builder import (
        TRANSACTION_FEATURES, ACCOUNT_FEATURES, CUSTOMER_FEATURES,
        MERCHANT_FEATURES, DEVICE_FEATURES,
    )
    from torch_geometric.data import HeteroData
    from camel_sentinel.gnn.explainer import explain_transaction

    _ensure_loaded(models_dir)
    if not _CACHE.get("loaded"):
        return {"error": _CACHE.get("error", "Model not loaded.")}

    model = _CACHE["model"]
    if policy is None:
        policy = DEFAULT_POLICY

    # Build a tiny 5-node subgraph (1 of each type)
    def _feat_tensor(feat_dict: dict, cols: list[str]) -> torch.Tensor:
        row = [float(feat_dict.get(c, 0.0)) for c in cols]
        return torch.tensor([row], dtype=torch.float32)

    mini = HeteroData()
    mini["customer"].x = _feat_tensor(customer_features, CUSTOMER_FEATURES)
    mini["account"].x = _feat_tensor(account_features, ACCOUNT_FEATURES)
    mini["transaction"].x = _feat_tensor(transaction_features, TRANSACTION_FEATURES)
    mini["merchant"].x = _feat_tensor(merchant_features, MERCHANT_FEATURES)
    mini["device"].x = _feat_tensor(device_features, DEVICE_FEATURES)

    zero = torch.tensor([[0], [0]], dtype=torch.long)
    mini["customer", "owns", "account"].edge_index = zero.clone()
    mini["account", "rev_owns", "customer"].edge_index = zero.clone()
    mini["account", "makes", "transaction"].edge_index = zero.clone()
    mini["transaction", "rev_makes", "account"].edge_index = zero.clone()
    mini["transaction", "paid_to", "merchant"].edge_index = zero.clone()
    mini["merchant", "rev_paid_to", "transaction"].edge_index = zero.clone()
    mini["customer", "uses", "device"].edge_index = zero.clone()
    mini["device", "rev_uses_c", "customer"].edge_index = zero.clone()
    mini["account", "uses", "device"].edge_index = zero.clone()
    mini["device", "rev_uses_a", "account"].edge_index = zero.clone()

    x_dict = {nt: mini[nt].x for nt in ["customer", "account", "transaction", "merchant", "device"]}
    edge_index_dict = {et: mini[et].edge_index for et in mini.edge_types}

    model.eval()
    with torch.no_grad():
        logit = model(x_dict, edge_index_dict)
        score = torch.sigmoid(logit[0]).item()

    decision = policy.decide(score)
    explanation = explain_transaction(model, mini, txn_node_idx=0)

    return {
        "fraud_risk_score": round(score, 4),
        "decision": decision,
        "policy": policy.to_dict(),
        "important_nodes": explanation["important_nodes"],
        "important_edges": explanation["important_edges"],
        "narrative": explanation["narrative"],
        "caveats": explanation["caveats"],
        "warning": (
            "SYNTHETIC DEMONSTRATION — not trained on real customer transaction data. "
            "Single-transaction inference does not capture shared-device ring patterns "
            "from the full graph."
        ),
    }


def get_demo_results(
    data: Any,
    model: Any,
    policy: FraudPolicy | None = None,
) -> list[dict]:
    """
    Run the three pre-defined demo transactions through the trained model.

    These are the controlled examples from the notebook:
      demo_1: normal features, ring account (suspicious neighbourhood)
      demo_2: suspicious features, moderate neighbourhood
      demo_3: Raushan transfers $2,000 (normal)

    Each result includes the GNN score AND a note about what the tabular
    baseline would likely predict, for educational comparison.
    """
    from camel_sentinel.gnn.explainer import explain_transaction

    if policy is None:
        policy = DEFAULT_POLICY

    if "demo" not in data.node_types or not hasattr(data["demo"], "transaction_ids"):
        return []

    demo_ids = data["demo"].transaction_ids
    demo_patterns = data["demo"].fraud_pattern

    results = []
    for i, (tid, pattern) in enumerate(zip(demo_ids, demo_patterns)):
        demo_x = data["demo"].x[i]
        # Build a mini 5-node graph using demo transaction features
        # We can't do full graph inference for demo rows since they're not
        # in the main graph. Use the mini-graph approach + note the limitation.
        amount = data["demo"].amounts[i] if hasattr(data["demo"], "amounts") else 0.0

        from torch_geometric.data import HeteroData as HD
        mini = HD()
        # Use the demo transaction features as the transaction node
        mini["transaction"].x = demo_x.unsqueeze(0)
        # Use average account/customer/merchant/device features from the full graph
        mini["customer"].x = data["customer"].x.mean(dim=0, keepdim=True)
        mini["account"].x = data["account"].x.mean(dim=0, keepdim=True)
        mini["merchant"].x = data["merchant"].x.mean(dim=0, keepdim=True)
        mini["device"].x = data["device"].x.mean(dim=0, keepdim=True)

        zero = torch.tensor([[0], [0]], dtype=torch.long)
        for et in [("customer","owns","account"), ("account","rev_owns","customer"),
                   ("account","makes","transaction"), ("transaction","rev_makes","account"),
                   ("transaction","paid_to","merchant"), ("merchant","rev_paid_to","transaction"),
                   ("customer","uses","device"), ("device","rev_uses_c","customer"),
                   ("account","uses","device"), ("device","rev_uses_a","account")]:
            mini[et].edge_index = zero.clone()

        x_d = {nt: mini[nt].x for nt in ["customer","account","transaction","merchant","device"]}
        ei_d = {et: mini[et].edge_index for et in mini.edge_types}
        model.eval()
        with torch.no_grad():
            logit_d = model(x_d, ei_d)
            score = torch.sigmoid(logit_d[0]).item()

        decision = policy.decide(score)
        results.append({
            "transaction_id": tid,
            "fraud_pattern": pattern,
            "amount": amount,
            "fraud_risk_score": round(score, 4),
            "decision": decision,
            "note": _demo_note(pattern),
        })

    return results


def _demo_note(pattern: str) -> str:
    notes = {
        "demo_1_ring_neighbourhood": (
            "Demo 1: This transaction's features look normal, but its account "
            "belongs to a shared-device ring. In the full graph, the GNN can detect "
            "the ring pattern; in single-transaction inference, only static node "
            "features are available."
        ),
        "demo_2_suspicious_features": (
            "Demo 2: This transaction has suspicious features (high amount, late hour, "
            "cross-border), but its account history and device context are clean. "
            "Contextual adjustment is visible when the full graph is available."
        ),
        "demo_3_raushan_normal": (
            "Demo 3: Raushan transfers $2,000 — normal amount, normal hour, domestic, "
            "clean account and device. Both baseline and GNN should assign low risk."
        ),
    }
    return notes.get(pattern, "")
