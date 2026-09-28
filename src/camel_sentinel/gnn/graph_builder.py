"""
graph_builder.py — Convert five DataFrames into a PyTorch Geometric HeteroData object.

The graph has five node types and five edge types (plus reverses):

  Node types:
    customer    — the individual who owns accounts
    account     — a bank account owned by a customer
    transaction — a payment / transfer (the nodes being classified)
    merchant    — the recipient of a transaction
    device      — the device fingerprint from which accounts are accessed

  Edge types (forward):
    customer  → owns      → account
    account   → makes     → transaction
    transaction → paid_to → merchant
    customer  → uses      → device
    account   → uses      → device

  Edge types (reverse, for bidirectional message passing):
    account     → rev_owns     → customer
    transaction → rev_makes    → account
    merchant    → rev_paid_to  → transaction
    device      → rev_uses_c   → customer
    device      → rev_uses_a   → account

Labels (transaction nodes only):
  y = 0 (genuine), 1 (fraud)
  train_mask / val_mask / test_mask  — temporal split

Split strategy (temporal):
  Sort transactions by created_at.
  Train: earliest 70 %, Val: next 15 %, Test: last 15 %.
  Transactions with is_fraud == -1 (demo rows) are excluded from all splits.

Feature scaling:
  All node features are standardised (zero mean, unit variance) using stats
  computed on the training set only, to avoid look-ahead leakage.
  Encoders are saved as a dict and returned alongside the HeteroData object
  so they can be applied at inference time.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Feature column definitions
# ─────────────────────────────────────────────────────────────────────────────

CUSTOMER_FEATURES = [
    "age",
    "tenure_days",
    "country_risk_score",
    "num_accounts",
]

ACCOUNT_FEATURES = [
    "balance",
    "avg_transaction_amount",
    "transaction_velocity",
    "account_age_days",
]

TRANSACTION_FEATURES = [
    "amount",
    "amount_zscore",
    "hour_sin",          # cyclic encoding
    "hour_cos",
    "is_cross_border",
    "time_since_last_txn",
    "merchant_risk_score",
    "merchant_category_enc",
]

MERCHANT_FEATURES = [
    "merchant_risk_score",
    "transaction_count",
    "country_risk_score",
]

DEVICE_FEATURES = [
    "num_accounts",
    "num_customers",
    "device_risk_score",
]

# Tabular features used by the sklearn baseline (no graph context)
TABULAR_BASELINE_FEATURES = [
    "amount",
    "amount_zscore",
    "hour_sin",
    "hour_cos",
    "is_cross_border",
    "time_since_last_txn",
    "merchant_risk_score",
    "merchant_category_enc",
    "customer_country_risk",    # joined from customer table
    "account_velocity",         # joined from account table
    "balance_ratio",            # amount / account balance
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_categories(series: pd.Series) -> tuple[pd.Series, dict]:
    cats = sorted(series.dropna().unique())
    mapping = {c: i for i, c in enumerate(cats)}
    return series.map(mapping).fillna(0).astype(int), mapping


def _standardise(
    df: pd.DataFrame,
    cols: list[str],
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Standardise columns to zero mean / unit variance.

    If `stats` is None, compute them from `df` (training phase).
    Otherwise apply pre-computed stats (inference / val / test phase).
    Returns the modified DataFrame and the stats dict.
    """
    out = df.copy()
    computed: dict[str, tuple[float, float]] = {}
    for c in cols:
        if stats is not None:
            mean, std = stats[c]
        else:
            mean = float(out[c].mean())
            std = max(float(out[c].std()), 1e-6)
        out[c] = ((out[c] - mean) / std).astype(np.float32)
        computed[c] = (mean, std)
    return out, computed


def _to_tensor(df: pd.DataFrame, cols: list[str]) -> torch.Tensor:
    # Convert via Python list to avoid torch-numpy interop issues with numpy 2.x.
    # torch.tensor() accepts nested Python lists without requiring the numpy bridge.
    return torch.tensor(df[cols].values.tolist(), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_hetero_data(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
) -> tuple[HeteroData, dict[str, Any]]:
    """
    Build a PyG HeteroData graph from the five DataFrames.

    Returns
    -------
    data : HeteroData
        The graph with node features, edge indices, labels, and split masks.
    meta : dict
        Index mappings and feature scalers needed to build new inference graphs.
    """
    # ── 0. Separate labelled training data from demo rows ────────────────────
    demo_mask = transactions["is_fraud"] == -1
    txns_labelled = transactions[~demo_mask].copy().reset_index(drop=True)
    txns_demo = transactions[demo_mask].copy().reset_index(drop=True)

    # ── 1. Categorical encodings ─────────────────────────────────────────────
    txns_labelled["merchant_category_enc"], cat_mapping = _encode_categories(
        txns_labelled["merchant_category"]
    )
    if len(txns_demo) > 0:
        txns_demo["merchant_category_enc"] = (
            txns_demo["merchant_category"].map(cat_mapping).fillna(0).astype(int)
        )

    # ── 2. Cyclic hour encoding ──────────────────────────────────────────────
    for df in [txns_labelled, txns_demo]:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24).astype(np.float32)

    # ── 3. Integer indices for every entity ─────────────────────────────────
    cust_idx = {cid: i for i, cid in enumerate(customers["customer_id"])}
    acc_idx = {aid: i for i, aid in enumerate(accounts["account_id"])}
    txn_idx = {tid: i for i, tid in enumerate(txns_labelled["transaction_id"])}
    demo_txn_idx = {tid: i for i, tid in enumerate(txns_demo["transaction_id"])}
    merch_idx = {mid: i for i, mid in enumerate(merchants["merchant_id"])}
    dev_idx = {did: i for i, did in enumerate(devices["device_id"])}

    # ── 4. Temporal train / val / test split on labelled transactions ────────
    n = len(txns_labelled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_ids = set(txns_labelled.iloc[:n_train]["transaction_id"])
    val_ids = set(txns_labelled.iloc[n_train: n_train + n_val]["transaction_id"])
    # test = remaining

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for i, tid in enumerate(txns_labelled["transaction_id"]):
        if tid in train_ids:
            train_mask[i] = True
        elif tid in val_ids:
            val_mask[i] = True
        else:
            test_mask[i] = True

    # ── 5. Feature standardisation (fit on train only) ───────────────────────
    train_txns = txns_labelled[train_mask.tolist()]

    _, txn_stats = _standardise(train_txns, TRANSACTION_FEATURES)
    txns_labelled_scaled, _ = _standardise(txns_labelled, TRANSACTION_FEATURES, txn_stats)

    _, cust_stats = _standardise(customers, CUSTOMER_FEATURES)
    customers_scaled, _ = _standardise(customers, CUSTOMER_FEATURES, cust_stats)

    _, acc_stats = _standardise(accounts, ACCOUNT_FEATURES)
    accounts_scaled, _ = _standardise(accounts, ACCOUNT_FEATURES, acc_stats)

    _, merch_stats = _standardise(merchants, MERCHANT_FEATURES)
    merchants_scaled, _ = _standardise(merchants, MERCHANT_FEATURES, merch_stats)

    _, dev_stats = _standardise(devices, DEVICE_FEATURES)
    devices_scaled, _ = _standardise(devices, DEVICE_FEATURES, dev_stats)

    # ── 6. Build HeteroData ──────────────────────────────────────────────────
    data = HeteroData()

    # Node features
    data["customer"].x = _to_tensor(customers_scaled, CUSTOMER_FEATURES)
    data["account"].x = _to_tensor(accounts_scaled, ACCOUNT_FEATURES)
    data["transaction"].x = _to_tensor(txns_labelled_scaled, TRANSACTION_FEATURES)
    data["merchant"].x = _to_tensor(merchants_scaled, MERCHANT_FEATURES)
    data["device"].x = _to_tensor(devices_scaled, DEVICE_FEATURES)

    # Labels and split masks (transaction nodes only)
    data["transaction"].y = torch.tensor(
        txns_labelled["is_fraud"].values, dtype=torch.float32
    )
    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask = val_mask
    data["transaction"].test_mask = test_mask

    # Store transaction IDs for lookup in the explainer / detector
    data["transaction"].transaction_ids = list(txns_labelled["transaction_id"])
    data["transaction"].fraud_pattern = list(txns_labelled["fraud_pattern"])

    # ── 7. Edges  (vectorised — no iterrows over large frames) ──────────────

    def _edge(src_list: list, dst_list: list) -> torch.Tensor:
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    # customer → owns → account  (one edge per account)
    co_src = accounts["customer_id"].map(cust_idx).tolist()
    co_dst = list(range(len(accounts)))
    data["customer", "owns", "account"].edge_index = _edge(co_src, co_dst)
    data["account", "rev_owns", "customer"].edge_index = _edge(co_dst, co_src)

    # account → makes → transaction  (one edge per transaction)
    am_src = txns_labelled["account_id"].map(acc_idx).tolist()
    am_dst = list(range(len(txns_labelled)))
    data["account", "makes", "transaction"].edge_index = _edge(am_src, am_dst)
    data["transaction", "rev_makes", "account"].edge_index = _edge(am_dst, am_src)

    # transaction → paid_to → merchant  (one edge per transaction)
    tp_src = list(range(len(txns_labelled)))
    tp_dst = txns_labelled["merchant_id"].map(merch_idx).tolist()
    data["transaction", "paid_to", "merchant"].edge_index = _edge(tp_src, tp_dst)
    data["merchant", "rev_paid_to", "transaction"].edge_index = _edge(tp_dst, tp_src)

    # customer → uses → device  (one edge per unique customer_id in accounts)
    uniq_cust_dev = accounts.drop_duplicates("customer_id")[["customer_id", "device_id"]]
    cd_src = uniq_cust_dev["customer_id"].map(cust_idx).tolist()
    cd_dst = uniq_cust_dev["device_id"].map(dev_idx).tolist()
    data["customer", "uses", "device"].edge_index = _edge(cd_src, cd_dst)
    data["device", "rev_uses_c", "customer"].edge_index = _edge(cd_dst, cd_src)

    # account → uses → device  (one edge per account)
    ad_src = list(range(len(accounts)))
    ad_dst = accounts["device_id"].map(dev_idx).tolist()
    data["account", "uses", "device"].edge_index = _edge(ad_src, ad_dst)
    data["device", "rev_uses_a", "account"].edge_index = _edge(ad_dst, ad_src)

    # ── 8. Tabular baseline feature matrix (for sklearn, no graph) ───────────
    # Build a flat feature table that joins account and customer context into
    # each transaction row.  This is what the tabular baseline (LR / LightGBM)
    # will see — transaction features + simple per-account/per-customer stats,
    # but NO graph message-passing information.
    acc_to_cust = accounts.set_index("account_id")["customer_id"].to_dict()
    txns_for_baseline = txns_labelled_scaled.copy()
    txns_for_baseline["customer_id"] = txns_for_baseline["account_id"].map(acc_to_cust)

    txns_for_baseline = txns_for_baseline.merge(
        accounts[["account_id", "transaction_velocity", "balance"]],
        on="account_id", how="left"
    ).merge(
        customers[["customer_id", "country_risk_score"]].rename(
            columns={"country_risk_score": "customer_country_risk"}
        ),
        on="customer_id", how="left"
    )
    txns_for_baseline["account_velocity"] = txns_for_baseline["transaction_velocity"].fillna(1.0)
    txns_for_baseline["balance_ratio"] = (
        txns_for_baseline["amount"] / txns_for_baseline["balance"].clip(lower=1)
    ).fillna(0.0)
    txns_for_baseline["customer_country_risk"] = txns_for_baseline["customer_country_risk"].fillna(1.0)
    data["transaction"].tabular_features = _to_tensor(
        txns_for_baseline, TABULAR_BASELINE_FEATURES
    )

    # ── 9. Demo transaction subgraph context (stored separately) ─────────────
    # Store demo transaction features for the detector's inference path.
    if len(txns_demo) > 0:
        txns_demo_scaled, _ = _standardise(txns_demo, TRANSACTION_FEATURES, txn_stats)
        data["demo"].x = _to_tensor(txns_demo_scaled, TRANSACTION_FEATURES)
        data["demo"].transaction_ids = list(txns_demo["transaction_id"])
        data["demo"].fraud_pattern = list(txns_demo["fraud_pattern"])
        data["demo"].amounts = txns_demo["amount"].tolist()

    # ── 10. Meta dict ─────────────────────────────────────────────────────────
    meta = {
        "cust_idx": cust_idx,
        "acc_idx": acc_idx,
        "txn_idx": txn_idx,
        "demo_txn_idx": demo_txn_idx,
        "merch_idx": merch_idx,
        "dev_idx": dev_idx,
        "cat_mapping": cat_mapping,
        "stats": {
            "transaction": txn_stats,
            "customer": cust_stats,
            "account": acc_stats,
            "merchant": merch_stats,
            "device": dev_stats,
        },
        "feature_cols": {
            "transaction": TRANSACTION_FEATURES,
            "customer": CUSTOMER_FEATURES,
            "account": ACCOUNT_FEATURES,
            "merchant": MERCHANT_FEATURES,
            "device": DEVICE_FEATURES,
            "tabular_baseline": TABULAR_BASELINE_FEATURES,
        },
        "split_sizes": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
    }

    return data, meta
