"""Phase 7: small, explainable FedAvg + noisy-update simulation.

This is a teaching simulation. It partitions public panel rows by state,
trains a linear classifier locally, clips each client's update, adds Gaussian
noise, and averages updates. Raw client rows never enter the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS


@dataclass
class FederatedResult:
    model: SGDClassifier
    history: list[dict[str, float]]
    clients: list[str]
    privacy: dict[str, float]


def _prepare(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = panel[DEFAULT_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    scale = X.std().replace(0, 1.0)
    return (X - X.mean()) / scale, panel["failed"].astype(int)


def _new_model() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="constant",
        eta0=0.01,
        max_iter=1,
        tol=None,
        random_state=42,
    )


def _fit_local(global_model: SGDClassifier, X: pd.DataFrame, y: pd.Series, epochs: int) -> SGDClassifier:
    local = _new_model()
    local.partial_fit(X, y, classes=np.array([0, 1]))
    local.coef_ = global_model.coef_.copy()
    local.intercept_ = global_model.intercept_.copy()
    for _ in range(epochs):
        local.partial_fit(X, y)
    return local


def train_federated(
    panel: pd.DataFrame,
    rounds: int = 5,
    local_epochs: int = 1,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.05,
    min_client_rows: int = 500,
) -> FederatedResult:
    """Train a logistic model by noisy FedAvg across state-based clients.

    ``clip_norm`` bounds each client's model update before aggregation.
    ``noise_multiplier`` controls Gaussian noise as a fraction of that bound;
    it is a mechanism parameter, not a formal end-to-end epsilon guarantee.
    """
    X, y = _prepare(panel)
    client_values = panel["STALP"].fillna("Unknown").astype(str)
    client_masks = {
        client: client_values == client
        for client in sorted(client_values.unique())
        if int((client_values == client).sum()) >= min_client_rows
    }
    if not client_masks:
        raise ValueError("No clients meet min_client_rows")

    global_model = _new_model()
    global_model.partial_fit(X.iloc[:1], y.iloc[:1], classes=np.array([0, 1]))
    history: list[dict[str, float]] = []

    for round_number in range(1, rounds + 1):
        updates = []
        weights = []
        for client, mask in client_masks.items():
            local = _fit_local(global_model, X[mask], y[mask], local_epochs)
            delta = np.concatenate(
                [
                    local.coef_ - global_model.coef_,
                    (local.intercept_ - global_model.intercept_).reshape(-1, 1),
                ],
                axis=1,
            )
            norm = float(np.linalg.norm(delta))
            scale = min(1.0, clip_norm / max(norm, 1e-12))
            updates.append(delta * scale)
            weights.append(int(mask.sum()))

        weighted = sum(update * weight for update, weight in zip(updates, weights)) / sum(weights)
        noise = np.random.default_rng(42 + round_number).normal(
            0.0, noise_multiplier * clip_norm, size=weighted.shape
        )
        aggregate = weighted + noise
        global_model.coef_ += aggregate[:, :-1]
        global_model.intercept_ += aggregate[:, -1]
        probability = global_model.predict_proba(X)[:, 1]
        history.append({
            "round": float(round_number),
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc": float(average_precision_score(y, probability)),
        })

    return FederatedResult(
        model=global_model,
        history=history,
        clients=list(client_masks),
        privacy={"clip_norm": clip_norm, "noise_multiplier": noise_multiplier, "formal_epsilon": float("nan")},
    )
