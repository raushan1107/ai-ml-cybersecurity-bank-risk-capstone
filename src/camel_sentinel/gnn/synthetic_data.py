"""
synthetic_data.py — Generate the five-entity synthetic fraud dataset.

Produces five DataFrames (customers, accounts, transactions, merchants,
devices) with deliberately planted fraud patterns.  The dataset is split
into two broad categories so the notebook can compare:

  • Tabular fraud (Pattern A): detectable from transaction features alone.
  • Relational fraud (Patterns B-D): requires graph neighbourhood context.

⚠ SYNTHETIC DEMONSTRATION — not trained on real customer transaction data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── reproducibility ─────────────────────────────────────────────────────────
SEED = 42


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── public constants ─────────────────────────────────────────────────────────
N_CUSTOMERS = 1_000
N_MERCHANTS = 350
N_RING_DEVICES = 12          # devices shared across fraud-ring accounts
N_NORMAL_DEVICES = 200       # one device per ordinary customer (approx)
TARGET_FRAUD_RATE = 0.031    # ~3.1 %


# ─────────────────────────────────────────────────────────────────────────────
# 1. Merchants
# ─────────────────────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "retail", "grocery", "restaurant", "travel", "electronics",
    "healthcare", "utilities", "entertainment", "wholesale", "fuel",
]


def _make_merchants(rng: np.random.Generator) -> pd.DataFrame:
    n = N_MERCHANTS
    # 15 merchants are high-risk fraud attractors (Pattern C)
    is_fraud_merchant = np.zeros(n, dtype=bool)
    is_fraud_merchant[:15] = True
    rng.shuffle(is_fraud_merchant)

    merchant_risk = np.where(
        is_fraud_merchant,
        rng.uniform(0.70, 0.95, n),
        rng.uniform(0.01, 0.15, n),
    )
    df = pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in range(n)],
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, n),
        "merchant_risk_score": merchant_risk.round(4),
        "country_risk_score": rng.choice([1, 1, 1, 2, 2, 3, 4], n).astype(float),
        "is_fraud_merchant": is_fraud_merchant,
        # transaction_count updated after transactions are generated
        "transaction_count": 0,
    })
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Devices
# ─────────────────────────────────────────────────────────────────────────────

def _make_devices(rng: np.random.Generator) -> pd.DataFrame:
    n_ring = N_RING_DEVICES
    n_normal = N_NORMAL_DEVICES
    n = n_ring + n_normal

    is_ring_device = np.array([True] * n_ring + [False] * n_normal)
    device_risk = np.where(
        is_ring_device,
        rng.uniform(0.65, 0.95, n),
        rng.uniform(0.01, 0.12, n),
    )
    df = pd.DataFrame({
        "device_id": [f"D{i:04d}" for i in range(n)],
        "device_risk_score": device_risk.round(4),
        "is_ring_device": is_ring_device,
        # num_accounts / num_customers filled in after account assignment
        "num_accounts": 0,
        "num_customers": 0,
    })
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Customers
# ─────────────────────────────────────────────────────────────────────────────

def _make_customers(rng: np.random.Generator) -> pd.DataFrame:
    n = N_CUSTOMERS
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "age": rng.integers(22, 72, n).astype(float),
        "tenure_days": rng.integers(90, 5_000, n).astype(float),
        "country_risk_score": rng.choice([1, 1, 1, 2, 2, 3, 4], n).astype(float),
        "num_accounts": 0,  # filled after account assignment
        # flag set if this customer is part of a fraud ring (Pattern B)
        "is_ring_customer": False,
    })
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Accounts + device assignment
# ─────────────────────────────────────────────────────────────────────────────

def _make_accounts(
    customers: pd.DataFrame,
    devices: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    account_idx = 0
    ring_device_ids = devices.loc[devices["is_ring_device"], "device_id"].tolist()
    normal_device_ids = devices.loc[~devices["is_ring_device"], "device_id"].tolist()

    # First 120 customers are ring customers; each is assigned a ring device.
    # Their accounts will carry elevated fraud rates (Pattern B).
    ring_customer_ids = customers["customer_id"].iloc[:120].tolist()
    customers.loc[customers["customer_id"].isin(ring_customer_ids), "is_ring_customer"] = True

    # Distribute ring customers across ring devices (8–14 accounts per device)
    ring_assignments: dict[str, str] = {}  # customer_id → device_id
    device_pool = ring_device_ids * 20  # repeat so we have enough to assign
    rng.shuffle(device_pool)
    for i, cid in enumerate(ring_customer_ids):
        ring_assignments[cid] = device_pool[i % len(ring_device_ids)]

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        n_accs = int(rng.choice([1, 1, 2, 2, 2, 3, 4, 5]))
        for _ in range(n_accs):
            if cid in ring_assignments:
                device_id = ring_assignments[cid]
                # ring accounts: lower balance, tighter typical range
                balance = float(rng.uniform(500, 8_000))
                avg_txn = float(rng.uniform(100, 600))
            else:
                device_id = rng.choice(normal_device_ids)
                balance = float(rng.uniform(1_000, 80_000))
                avg_txn = float(rng.uniform(200, 3_000))

            rows.append({
                "account_id": f"A{account_idx:05d}",
                "customer_id": cid,
                "device_id": device_id,
                "balance": round(balance, 2),
                "avg_transaction_amount": round(avg_txn, 2),
                "account_age_days": float(rng.integers(30, 3_650)),
                "transaction_velocity": 0.0,  # filled after txns generated
                "is_ring_account": cid in ring_assignments,
            })
            account_idx += 1

    df = pd.DataFrame(rows).reset_index(drop=True)
    # back-fill customer account counts
    counts = df.groupby("customer_id").size()
    customers["num_accounts"] = customers["customer_id"].map(counts).fillna(0).astype(int)
    # update device account/customer counts
    acc_per_dev = df.groupby("device_id")["account_id"].nunique()
    cust_per_dev = df.groupby("device_id")["customer_id"].nunique()
    devices["num_accounts"] = devices["device_id"].map(acc_per_dev).fillna(0).astype(int)
    devices["num_customers"] = devices["device_id"].map(cust_per_dev).fillna(0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transactions
# ─────────────────────────────────────────────────────────────────────────────

def _make_transactions(
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate ~52 k transactions with four fraud patterns:

      A  Tabular fraud      (35 %) — high amount + unusual hour + cross-border
      B  Shared-device ring (30 %) — normal-looking txn, ring account
      C  High-risk merchant (20 %) — txn paid to fraud-attracting merchant
      D  Velocity burst     (15 %) — rapid succession of transactions
    """
    fraud_merchant_ids = merchants.loc[
        merchants["is_fraud_merchant"], "merchant_id"
    ].tolist()
    normal_merchant_ids = merchants.loc[
        ~merchants["is_fraud_merchant"], "merchant_id"
    ].tolist()

    rows = []
    txn_idx = 0
    base_ts = pd.Timestamp("2023-01-01")

    for _, acc in accounts.iterrows():
        acc_id = acc["account_id"]
        avg = acc["avg_transaction_amount"]
        std = avg * 0.4
        is_ring = acc["is_ring_account"]

        # ~25 transactions per account, more for ring accounts
        n_txns = int(rng.integers(15, 30 if not is_ring else 40))

        for j in range(n_txns):
            # base timestamp — spread transactions across the year
            ts = base_ts + pd.Timedelta(days=int(rng.integers(0, 360)),
                                         hours=int(rng.integers(0, 24)))

            amount = max(1.0, float(rng.normal(avg, std)))
            merchant_id = rng.choice(normal_merchant_ids)
            hour = ts.hour
            is_cross_border = False
            is_fraud = False
            fraud_pattern = "none"

            # ── Pattern A: tabular fraud (35 % of fraudulent txns) ──────────
            # Planted on ~0.9 % of all transactions, regardless of account type
            if rng.random() < 0.009:
                amount = avg * float(rng.uniform(5, 15))   # 5-15× average
                hour = int(rng.integers(1, 5))              # 01:00–05:00
                is_cross_border = True
                is_fraud = True
                fraud_pattern = "A_tabular"

            # ── Pattern B: ring account (30 % of fraudulent txns) ───────────
            # Planted on ring accounts with NORMAL-LOOKING individual features.
            # The fraud signal lives entirely in the shared-device neighbourhood,
            # not in the transaction amount, hour, or merchant.
            elif is_ring and rng.random() < 0.18:
                # amount stays normal — the signal is in the graph, not here
                is_fraud = True
                fraud_pattern = "B_ring"

            # ── Pattern C: fraud merchant (20 % of fraudulent txns) ─────────
            elif rng.random() < 0.009:
                merchant_id = rng.choice(fraud_merchant_ids)
                is_fraud = True
                fraud_pattern = "C_merchant"

            # ── Pattern D: velocity burst (15 % of fraudulent txns) ─────────
            elif j > 0 and rows and rows[-1]["account_id"] == acc_id:
                last_ts = rows[-1]["created_at"]
                gap_minutes = (ts - last_ts).total_seconds() / 60
                if gap_minutes < 30 and rng.random() < 0.035:
                    amount = avg * float(rng.uniform(0.8, 1.5))
                    is_fraud = True
                    fraud_pattern = "D_velocity"

            merchant_row = merchants.loc[
                merchants["merchant_id"] == merchant_id
            ].iloc[0]

            rows.append({
                "transaction_id": f"T{txn_idx:06d}",
                "account_id": acc_id,
                "merchant_id": merchant_id,
                "amount": round(amount, 2),
                "hour_of_day": hour,
                "is_cross_border": int(is_cross_border),
                "merchant_category": merchant_row["merchant_category"],
                "merchant_risk_score": float(merchant_row["merchant_risk_score"]),
                "created_at": ts,
                "is_fraud": int(is_fraud),
                "fraud_pattern": fraud_pattern,
            })
            txn_idx += 1

    df = pd.DataFrame(rows).reset_index(drop=True)
    df = df.sort_values("created_at").reset_index(drop=True)

    # ── Derived features ─────────────────────────────────────────────────────
    # amount_zscore: how many std devs above the account's mean amount
    acc_stats = (
        df.groupby("account_id")["amount"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
    )
    df = df.merge(acc_stats, on="account_id")
    df["acc_std"] = df["acc_std"].fillna(1.0).clip(lower=1.0)
    df["amount_zscore"] = ((df["amount"] - df["acc_mean"]) / df["acc_std"]).round(4)
    df.drop(columns=["acc_mean", "acc_std"], inplace=True)

    # time_since_last_transaction (seconds); 0 for first txn of each account
    df = df.sort_values(["account_id", "created_at"]).reset_index(drop=True)
    df["time_since_last_txn"] = (
        df.groupby("account_id")["created_at"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
        .round(1)
    )

    # account_velocity_24h: count of transactions from same account in prior 24h
    df = df.sort_values("created_at").reset_index(drop=True)

    # update merchant transaction counts
    txn_counts = df.groupby("merchant_id").size()
    merchants["transaction_count"] = merchants["merchant_id"].map(txn_counts).fillna(0).astype(int)

    # update account velocity (avg transactions per day)
    days = df.groupby("account_id")["created_at"].apply(
        lambda s: max(1, (s.max() - s.min()).days)
    )
    txn_per_acc = df.groupby("account_id").size()
    velocity = (txn_per_acc / days).round(4)
    accounts["transaction_velocity"] = (
        accounts["account_id"].map(velocity).fillna(1.0)
    )

    return df.sort_values("created_at").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Controlled demo transactions
# ─────────────────────────────────────────────────────────────────────────────

def _inject_demo_transactions(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append three hand-crafted transactions used in the notebook teaching section.

    Demo 1 — Normal-looking, suspicious neighbourhood (ring account)
    Demo 2 — Suspicious-looking features, moderately clean neighbourhood
    Demo 3 — Raushan transfers $2,000 (normal in every dimension)

    Expected score direction (not a hard constraint — the trained model decides):
      Demo 1: GNN should assign higher risk than the tabular baseline.
      Demo 2: GNN may adjust risk relative to tabular baseline based on context.
      Demo 3: Both models should assign low risk.
    """
    ring_accs = accounts.loc[accounts["is_ring_account"], "account_id"].tolist()
    normal_accs = accounts.loc[~accounts["is_ring_account"], "account_id"].tolist()
    normal_merchants = merchants.loc[~merchants["is_fraud_merchant"], "merchant_id"].tolist()

    acc1 = ring_accs[0] if ring_accs else normal_accs[0]
    acc_row1 = accounts.loc[accounts["account_id"] == acc1].iloc[0]
    avg1 = acc_row1["avg_transaction_amount"]
    cat = "retail"

    acc2 = normal_accs[5]
    acc_row2 = accounts.loc[accounts["account_id"] == acc2].iloc[0]
    avg2 = acc_row2["avg_transaction_amount"]

    acc3 = normal_accs[10]
    acc_row3 = accounts.loc[accounts["account_id"] == acc3].iloc[0]
    avg3 = acc_row3["avg_transaction_amount"]

    merch_normal = normal_merchants[0]
    merch_row = merchants.loc[merchants["merchant_id"] == merch_normal].iloc[0]

    demo_ts = pd.Timestamp("2023-12-20 14:00:00")
    last_txn_id = transactions["transaction_id"].str[1:].astype(int).max() + 1

    demos = [
        {   # Demo 1: normal features, ring account (suspicious neighbourhood)
            "transaction_id": f"T{last_txn_id:06d}",
            "account_id": acc1,
            "merchant_id": merch_normal,
            "amount": round(avg1 * 1.1, 2),
            "hour_of_day": 14,
            "is_cross_border": 0,
            "merchant_category": cat,
            "merchant_risk_score": float(merch_row["merchant_risk_score"]),
            "created_at": demo_ts,
            "is_fraud": -1,      # -1 = demo; excluded from training
            "fraud_pattern": "demo_1_ring_neighbourhood",
            "amount_zscore": 0.25,
            "time_since_last_txn": 3_600.0,
        },
        {   # Demo 2: high amount, late hour, cross-border; moderate neighbourhood
            "transaction_id": f"T{last_txn_id+1:06d}",
            "account_id": acc2,
            "merchant_id": merch_normal,
            "amount": round(avg2 * 12.0, 2),
            "hour_of_day": 2,
            "is_cross_border": 1,
            "merchant_category": "wholesale",
            "merchant_risk_score": float(merch_row["merchant_risk_score"]),
            "created_at": demo_ts + pd.Timedelta(hours=1),
            "is_fraud": -1,
            "fraud_pattern": "demo_2_suspicious_features",
            "amount_zscore": 11.5,
            "time_since_last_txn": 86_400.0,
        },
        {   # Demo 3: Raushan transfers $2,000 — normal in every dimension
            "transaction_id": f"T{last_txn_id+2:06d}",
            "account_id": acc3,
            "merchant_id": merch_normal,
            "amount": min(2_000.0, round(avg3 * 1.2, 2)),
            "hour_of_day": 15,
            "is_cross_border": 0,
            "merchant_category": "retail",
            "merchant_risk_score": float(merch_row["merchant_risk_score"]),
            "created_at": demo_ts + pd.Timedelta(hours=2),
            "is_fraud": -1,
            "fraud_pattern": "demo_3_raushan_normal",
            "amount_zscore": 0.4,
            "time_since_last_txn": 7_200.0,
        },
    ]
    demo_df = pd.DataFrame(demos)
    return pd.concat([transactions, demo_df], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    seed: int = SEED,
    save_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and return (customers, accounts, transactions, merchants, devices).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    save_dir : path-like, optional
        If provided, saves each DataFrame as a parquet file in that directory.

    Returns
    -------
    customers, accounts, transactions, merchants, devices — all DataFrames.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    rng = _rng(seed)

    merchants = _make_merchants(rng)
    devices = _make_devices(rng)
    customers = _make_customers(rng)
    accounts = _make_accounts(customers, devices, rng)
    transactions = _make_transactions(accounts, merchants, rng)
    transactions = _inject_demo_transactions(transactions, accounts, merchants)

    if save_dir is not None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        customers.to_parquet(p / "customers.parquet", index=False)
        accounts.to_parquet(p / "accounts.parquet", index=False)
        transactions.to_parquet(p / "transactions.parquet", index=False)
        merchants.to_parquet(p / "merchants.parquet", index=False)
        devices.to_parquet(p / "devices.parquet", index=False)

    return customers, accounts, transactions, merchants, devices


def summary(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
) -> dict:
    """Return a concise summary dict for notebook display."""
    labelled = transactions[transactions["is_fraud"] >= 0]
    fraud = labelled[labelled["is_fraud"] == 1]
    by_pattern = fraud["fraud_pattern"].value_counts().to_dict()
    return {
        "customers": len(customers),
        "ring_customers": int(customers["is_ring_customer"].sum()),
        "accounts": len(accounts),
        "ring_accounts": int(accounts["is_ring_account"].sum()),
        "transactions_total": len(labelled),
        "transactions_fraud": len(fraud),
        "fraud_rate_pct": round(len(fraud) / len(labelled) * 100, 2),
        "by_pattern": by_pattern,
        "merchants": len(merchants),
        "fraud_merchants": int(merchants["is_fraud_merchant"].sum()),
        "devices": len(devices),
        "ring_devices": int(devices["is_ring_device"].sum()),
        "demo_transactions": int((transactions["is_fraud"] == -1).sum()),
    }
