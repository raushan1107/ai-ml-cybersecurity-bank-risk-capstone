"""
trainer.py — Train the tabular baselines and the HeteroGNN; evaluate and save.

Training flow
-------------
1. generate() synthetic data
2. build_hetero_data() → HeteroData + meta
3. train_baselines()   → LogisticRegression + LightGBM on tabular features
4. train_gnn()         → HeteroGNN with BCEWithLogitsLoss + early stopping
5. evaluate_all()      → compare models on overall test set AND on the
                          relational-only fraud subset (Pattern B)
6. save_model()        → writes models/gnn_fraud_model.pt

The relational-fraud comparison is the educational centrepiece:
  baseline recall on Pattern B ≈ low   (features look normal)
  GNN recall on Pattern B     ≈ higher (graph neighbourhood reveals ring)
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# sklearn / lightgbm imports are deferred to avoid import-time cost
# when this module is loaded as part of the backend.

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: list, y_score: list, threshold: float = 0.5) -> dict:
    """Compute classification metrics without numpy arrays (avoids torch-numpy bridge)."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    n_fraud = sum(y_true)
    prevalence = n_fraud / max(len(y_true), 1)
    return {
        "n_total": len(y_true),
        "n_fraud": n_fraud,
        "prevalence_pct": round(prevalence * 100, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm,
    }


def _tensor_to_list(t: torch.Tensor) -> list:
    """Convert tensor to Python list without numpy bridge."""
    return t.detach().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Tabular baselines
# ─────────────────────────────────────────────────────────────────────────────

def train_baselines(
    data: HeteroData,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train Logistic Regression and LightGBM on transaction-level tabular features.

    These models see:
      - transaction's own features (amount, zscore, hour encoding, cross-border…)
      - simple account stats (velocity, balance ratio)
      - customer country risk
    They do NOT see:
      - which device the account is associated with
      - how many other accounts share that device
      - the device's risk score
      - the merchant's historical fraud rate aggregated across graph neighbours

    Returns a dict with the trained model objects and their test metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    txn = data["transaction"]
    X_all = _tensor_to_list(txn.tabular_features)
    y_all = _tensor_to_list(txn.y)

    train_idx = [i for i, m in enumerate(_tensor_to_list(txn.train_mask)) if m]
    val_idx = [i for i, m in enumerate(_tensor_to_list(txn.val_mask)) if m]
    test_idx = [i for i, m in enumerate(_tensor_to_list(txn.test_mask)) if m]

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    X_val = [X_all[i] for i in val_idx]
    y_val = [y_all[i] for i in val_idx]
    X_test = [X_all[i] for i in test_idx]
    y_test = [y_all[i] for i in test_idx]

    # ── Logistic Regression ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_train_s, y_train)

    lr_val_scores = lr.predict_proba(X_val_s)[:, 1].tolist()
    lr_test_scores = lr.predict_proba(X_test_s)[:, 1].tolist()
    lr_val_metrics = _metrics(y_val, lr_val_scores)
    lr_test_metrics = _metrics(y_test, lr_test_scores)

    if verbose:
        print(f"[LR]  val PR-AUC={lr_val_metrics['pr_auc']:.4f}  "
              f"test PR-AUC={lr_test_metrics['pr_auc']:.4f}")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    # LightGBM requires ndarray, not plain Python lists.
    import numpy as np
    lgb_train = lgb.Dataset(np.array(X_train, dtype=np.float32), label=np.array(y_train, dtype=np.float32))
    lgb_val = lgb.Dataset(np.array(X_val, dtype=np.float32), label=np.array(y_val, dtype=np.float32), reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "is_unbalance": True,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
    }
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False),
                 lgb.log_evaluation(period=-1)]
    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    lgb_val_scores = lgb_model.predict(np.array(X_val, dtype=np.float32)).tolist()
    lgb_test_scores = lgb_model.predict(np.array(X_test, dtype=np.float32)).tolist()
    lgb_val_metrics = _metrics(y_val, lgb_val_scores)
    lgb_test_metrics = _metrics(y_test, lgb_test_scores)

    if verbose:
        print(f"[LGB] val PR-AUC={lgb_val_metrics['pr_auc']:.4f}  "
              f"test PR-AUC={lgb_test_metrics['pr_auc']:.4f}")

    return {
        "lr": {
            "model": lr,
            "scaler": scaler,
            "val_metrics": lr_val_metrics,
            "test_metrics": lr_test_metrics,
            "test_scores": lr_test_scores,
            "test_indices": test_idx,
        },
        "lgb": {
            "model": lgb_model,
            "val_metrics": lgb_val_metrics,
            "test_metrics": lgb_test_metrics,
            "test_scores": lgb_test_scores,
            "test_indices": test_idx,
        },
        "y_test": y_test,
        "test_indices": test_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GNN training
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(
    data: HeteroData,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    epochs: int = 80,
    patience: int = 10,
    verbose: bool = True,
) -> tuple[Any, dict]:
    """
    Train the HeteroGNN on the labelled transactions.

    Loss: BCEWithLogitsLoss with pos_weight to handle class imbalance.
    Early stopping on validation PR-AUC.

    Returns (trained_model, training_history).
    """
    from sklearn.metrics import average_precision_score
    from camel_sentinel.gnn.model import HeteroGNN

    model = HeteroGNN(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if verbose:
        print(f"[GNN] Parameters: {model.count_parameters():,}")

    # pos_weight: weight fraud class inversely to its prevalence so the loss
    # treats a missed fraud as more costly than a false alarm.
    y_all = _tensor_to_list(data["transaction"].y)
    train_mask = data["transaction"].train_mask
    y_train = [y_all[i] for i, m in enumerate(_tensor_to_list(train_mask)) if m]
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=5
    )

    x_dict = {nt: data[nt].x for nt in ["customer", "account", "transaction", "merchant", "device"]}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    y_tensor = data["transaction"].y
    val_mask = data["transaction"].val_mask
    test_mask = data["transaction"].test_mask

    history: dict[str, list] = {
        "train_loss": [], "val_pr_auc": [], "test_pr_auc": []
    }
    best_val_pr = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        optimiser.zero_grad()
        logits = model(x_dict, edge_index_dict)
        loss = criterion(logits[train_mask], y_tensor[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        # ── Evaluate ─────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(logits).detach()

        val_scores = [scores[i].item() for i, m in enumerate(_tensor_to_list(val_mask)) if m]
        val_labels = [y_all[i] for i, m in enumerate(_tensor_to_list(val_mask)) if m]
        test_scores = [scores[i].item() for i, m in enumerate(_tensor_to_list(test_mask)) if m]
        test_labels = [y_all[i] for i, m in enumerate(_tensor_to_list(test_mask)) if m]

        try:
            val_pr = average_precision_score(val_labels, val_scores)
            test_pr = average_precision_score(test_labels, test_scores)
        except Exception:
            val_pr = 0.0
            test_pr = 0.0

        history["train_loss"].append(float(loss))
        history["val_pr_auc"].append(val_pr)
        history["test_pr_auc"].append(test_pr)

        scheduler.step(val_pr)

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  loss={loss:.4f}  "
                  f"val_PR-AUC={val_pr:.4f}  test_PR-AUC={test_pr:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (best val_PR-AUC={best_val_pr:.4f})")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    data: HeteroData,
    gnn_model: Any,
    baseline_results: dict,
    verbose: bool = True,
) -> dict:
    """
    Compute full metrics for all models on the test set.

    Also isolates Pattern B (relational fraud — shared-device ring) to show
    whether graph context improves recall on transactions that look normal
    but live in a suspicious neighbourhood.

    This is the key educational comparison:
      "Traditional ML: is this transaction suspicious on its own features?"
      "GNN: is this transaction suspicious given its features AND its neighbourhood?"
    """
    from camel_sentinel.gnn.model import HeteroGNN

    test_mask = data["transaction"].test_mask
    test_idx_list = [i for i, m in enumerate(_tensor_to_list(test_mask)) if m]

    y_all = _tensor_to_list(data["transaction"].y)
    patterns_all = data["transaction"].fraud_pattern
    y_test = [y_all[i] for i in test_idx_list]

    # ── GNN scores on test set ───────────────────────────────────────────────
    x_dict = {nt: data[nt].x for nt in ["customer", "account", "transaction", "merchant", "device"]}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    gnn_model.eval()
    with torch.no_grad():
        logits = gnn_model(x_dict, edge_index_dict)
        all_scores = torch.sigmoid(logits).detach().tolist()
    gnn_test_scores = [all_scores[i] for i in test_idx_list]

    # ── Overall metrics ──────────────────────────────────────────────────────
    gnn_metrics = _metrics(y_test, gnn_test_scores)
    lr_metrics = baseline_results["lr"]["test_metrics"]
    lgb_metrics = baseline_results["lgb"]["test_metrics"]

    if verbose:
        print("\n=== Test-set metrics (overall) ===")
        for name, m in [("Logistic Regression", lr_metrics),
                         ("LightGBM", lgb_metrics),
                         ("HeteroGNN", gnn_metrics)]:
            print(f"  {name:25s}  "
                  f"PR-AUC={m['pr_auc']:.4f}  "
                  f"ROC-AUC={m['roc_auc']:.4f}  "
                  f"F1={m['f1']:.4f}  "
                  f"Recall={m['recall']:.4f}")

    # ── Relational fraud subset (Pattern B only) ─────────────────────────────
    b_idx = [i for i, idx in enumerate(test_idx_list)
             if patterns_all[idx] == "B_ring"]
    if b_idx:
        y_b = [y_test[i] for i in b_idx]
        gnn_b_scores = [gnn_test_scores[i] for i in b_idx]

        # Re-index baseline scores to the same positions
        lr_b_scores = [baseline_results["lr"]["test_scores"][i] for i in b_idx]
        lgb_b_scores = [baseline_results["lgb"]["test_scores"][i] for i in b_idx]

        gnn_b_metrics = _metrics(y_b, gnn_b_scores)
        lr_b_metrics = _metrics(y_b, lr_b_scores)
        lgb_b_metrics = _metrics(y_b, lgb_b_scores)

        if verbose:
            print("\n=== Pattern B (relational ring fraud) — test subset ===")
            print(f"  {'n_fraud':8s}: {sum(y_b)} of {len(y_b)}")
            for name, m in [("Logistic Regression", lr_b_metrics),
                             ("LightGBM", lgb_b_metrics),
                             ("HeteroGNN", gnn_b_metrics)]:
                print(f"  {name:25s}  "
                      f"Recall={m['recall']:.4f}  "
                      f"Precision={m['precision']:.4f}  "
                      f"PR-AUC={m['pr_auc']:.4f}")
    else:
        gnn_b_metrics = lr_b_metrics = lgb_b_metrics = {}

    return {
        "gnn": {"overall": gnn_metrics, "pattern_b": gnn_b_metrics},
        "lr": {"overall": lr_metrics, "pattern_b": lr_b_metrics},
        "lgb": {"overall": lgb_metrics, "pattern_b": lgb_b_metrics},
        "gnn_test_scores": gnn_test_scores,
        "y_test": y_test,
        "test_indices": test_idx_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    gnn_model: Any,
    meta: dict,
    baseline_results: dict,
    save_dir: Path | str | None = None,
) -> Path:
    """
    Save the trained GNN model, meta dict, and baseline models.

    Saves:
      models/gnn_fraud_model.pt          — GNN state dict + architecture config
      models/gnn_fraud_meta.pkl          — index mappings, scalers, feature cols
      models/gnn_baselines.pkl           — LR model + scaler, LGB model
    """
    from camel_sentinel.gnn.model import HeteroGNN

    out = Path(save_dir) if save_dir else MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": gnn_model.state_dict(),
            "hidden_dim": gnn_model.hidden_dim,
            "num_layers": gnn_model.num_layers,
            "dropout": gnn_model.dropout,
        },
        out / "gnn_fraud_model.pt",
    )

    with open(out / "gnn_fraud_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    with open(out / "gnn_baselines.pkl", "wb") as f:
        pickle.dump(
            {
                "lr": {"model": baseline_results["lr"]["model"],
                       "scaler": baseline_results["lr"]["scaler"]},
                "lgb": {"model": baseline_results["lgb"]["model"]},
            },
            f,
        )

    print(f"[save] Models written to {out}/")
    return out


def load_model(save_dir: Path | str | None = None) -> tuple[Any, dict, dict]:
    """
    Load the trained GNN model, meta dict, and baseline models.

    Returns (gnn_model, meta, baselines).
    """
    from camel_sentinel.gnn.model import HeteroGNN

    src = Path(save_dir) if save_dir else MODELS_DIR
    checkpoint = torch.load(src / "gnn_fraud_model.pt", map_location="cpu",
                            weights_only=True)

    model = HeteroGNN(
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(src / "gnn_fraud_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    with open(src / "gnn_baselines.pkl", "rb") as f:
        baselines = pickle.load(f)

    return model, meta, baselines
