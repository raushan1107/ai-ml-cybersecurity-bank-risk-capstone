"""Phase 6: subgroup performance and fairness audit for CAMEL Sentinel."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS

REGION_MAP = {
    "Northeast": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"},
    "Midwest": {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"},
    "South": {"DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"},
    "West": {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"},
}


def add_audit_groups(panel: pd.DataFrame) -> pd.DataFrame:
    """Add auditable size and region groups without changing model features."""
    audited = panel.copy()
    audited["size_band"] = pd.qcut(
        audited["ASSET"].rank(method="first"),
        q=3,
        labels=["small", "medium", "large"],
    )
    state_to_region = {
        state: region for region, states in REGION_MAP.items() for state in states
    }
    audited["region"] = audited["STALP"].map(state_to_region).fillna("Unknown")
    return audited


def _group_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate threshold and ranking metrics for one subgroup."""
    y_actual = np.asarray(y_true, dtype=int)
    y_pred = (probability >= threshold).astype(int)
    positives = int(y_actual.sum())
    negatives = int(len(y_actual) - positives)
    true_positive = int(((y_actual == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_actual == 0) & (y_pred == 1)).sum())
    return {
        "support": int(len(y_actual)),
        "positive_rate": float(y_actual.mean()),
        "selection_rate": float(y_pred.mean()),
        "recall": float(recall_score(y_actual, y_pred, zero_division=0)),
        "precision": float(precision_score(y_actual, y_pred, zero_division=0)),
        "false_positive_rate": float(false_positive / negatives) if negatives else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_actual, y_pred)) if positives and negatives else 0.0,
    }


def audit_model(
    model: Any,
    panel: pd.DataFrame,
    threshold: float = 0.60,
    min_support: int = 100,
) -> dict[str, Any]:
    """Audit a fitted classifier across supported size and geographic groups.

    The audit uses the model's six CAMEL inputs and a threshold matching the
    deployed medium-risk tier. It reports absolute subgroup metrics and ratios
    against the strongest-supported subgroup, not a legal fairness conclusion.
    """
    missing = [column for column in [*DEFAULT_FEATURE_COLS, "failed", "ASSET", "STALP"] if column not in panel]
    if missing:
        raise ValueError(f"Panel is missing audit columns: {missing}")

    X = panel[DEFAULT_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = panel["failed"].astype(int)
    probabilities = model.predict_proba(X)[:, 1]
    audited = add_audit_groups(panel)

    groups: dict[str, dict[str, dict[str, float]]] = {}
    for dimension in ("size_band", "region"):
        dimension_rows: dict[str, dict[str, float]] = {}
        for value in sorted(audited[dimension].dropna().unique()):
            mask = audited[dimension] == value
            if int(mask.sum()) < min_support:
                continue
            dimension_rows[str(value)] = _group_metrics(y[mask], probabilities[mask], threshold)

        reference = max(dimension_rows.values(), key=lambda row: row["support"], default=None)
        for row in dimension_rows.values():
            if reference:
                row["selection_rate_ratio"] = float(row["selection_rate"] / reference["selection_rate"]) if reference["selection_rate"] else 0.0
                row["recall_ratio"] = float(row["recall"] / reference["recall"]) if reference["recall"] else 0.0
                row["precision_ratio"] = float(row["precision"] / reference["precision"]) if reference["precision"] else 0.0
        groups[dimension] = dimension_rows

    return {
        "threshold": threshold,
        "minimum_group_support": min_support,
        "overall": _group_metrics(y, probabilities, threshold),
        "groups": groups,
        "unavailable_dimensions": [
            "charter_type (not present in the cached panel)",
            "protected attributes such as race and gender (not appropriate or available for this bank-level dataset)",
        ],
        "interpretation": (
            "Ratios are descriptive disparity indicators against the largest-supported subgroup. "
            "They are not proof of discrimination or proof of fairness; investigate data quality, "
            "base-rate differences, calibration, and operational thresholds before acting."
        ),
    }
