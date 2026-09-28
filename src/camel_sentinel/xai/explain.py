"""
explain.py — Phase 5: turn a prediction into "why".

WHAT this file does: wraps `shap.TreeExplainer` so a single bank's
prediction comes back as a base rate plus a per-ratio contribution (how
much each CAMEL ratio pushed the failure-risk score up or down from that
base rate) — exactly what the backend's `/score` endpoint and the
Streamlit UI need to answer "why did the model say this?" for one bank,
not just "the model says X%".

WHY `shap.TreeExplainer` specifically: it computes *exact* Shapley values
for tree ensembles (as opposed to the slower, sampling-based approximation
a model-agnostic explainer would need), which is why Phase 3 chose a
RandomForest in the first place — see the "WHY RandomForestClassifier"
note in `camel_sentinel.models.baseline`.

HOW it connects: `build_explainer()` is called once, at backend startup,
on the model loaded via `camel_sentinel.models.registry.load_model()`;
`explain_instance()` is then called per request, on one bank's feature
row, from `backend/main.py`'s `/score` endpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator


def build_lime_explainer(X_background: pd.DataFrame):
    """Build a LIME tabular explainer when the optional dependency is installed."""
    try:
        from lime.lime_tabular import LimeTabularExplainer

        return LimeTabularExplainer(
            X_background.to_numpy().astype(float),
            feature_names=list(X_background.columns),
            class_names=["healthy", "failed"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
    except ImportError:
        return None
    except Exception:
        return None


def explain_with_lime(explainer, model: BaseEstimator, X_row: pd.DataFrame) -> dict:
    """Return a local LIME surrogate explanation for one bank."""
    if explainer is None:
        return {"available": False, "reason": "LIME not available — run notebooks to generate the training panel or install the lime package."}

    feature_names = list(X_row.columns)

    # LIME perturbs features and calls predict_fn with bare numpy arrays.
    # Wrap in a DataFrame so sklearn sees the named columns it was trained on;
    # without this, HistGradientBoostingClassifier emits a UserWarning (and in
    # strict environments raises it as an error) on every perturbed batch.
    def _predict_fn(X_numpy: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(X_numpy, columns=feature_names))

    explanation = explainer.explain_instance(
        X_row.iloc[0].to_numpy(), _predict_fn, num_features=len(feature_names)
    )
    return {
        "available": True,
        "label": "failed",
        "score": float(model.predict_proba(X_row)[0][1]),
        "features": [
            {"feature": feature, "weight": float(weight)}
            for feature, weight in explanation.as_list(label=1)
        ],
        "method": "local linear surrogate around this bank's ratios",
    }


def build_explainer(model: BaseEstimator) -> shap.TreeExplainer:
    """
    Build a SHAP explainer for a fitted tree-ensemble model.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A fitted tree-based model (e.g. the `RandomForestClassifier` from
        `baseline.train_baseline_model()`). `TreeExplainer` inspects the
        fitted tree structure directly, so this must already be trained.

    Returns
    -------
    shap.TreeExplainer
        Reusable across many calls to `explain_instance()` — building it
        is the relatively expensive step (it walks every tree once), so
        callers should build it once (e.g. at backend startup) rather than
        per request.
    """
    return shap.TreeExplainer(model)


def explain_instance(explainer: shap.TreeExplainer, X_row: pd.DataFrame) -> dict:
    """
    Explain one bank's prediction: how much did each ratio push the
    failure-risk score up or down from the model's average prediction?

    Parameters
    ----------
    explainer : shap.TreeExplainer
        From `build_explainer()`.
    X_row : pandas.DataFrame
        Exactly one row, with the same feature columns (and column order)
        the model was trained on — see
        `camel_sentinel.models.baseline.DEFAULT_FEATURE_COLS`.

    Returns
    -------
    dict
        `{"base_value": float, "contributions": {feature_name: float, ...}}`.
        `base_value` is the model's average predicted failure-risk across
        its training data (before looking at this specific bank);
        `contributions[feature]` is how many percentage points that one
        feature's value moved the prediction away from `base_value` for
        *this* bank — a positive number pushes failure-risk up, negative
        pushes it down. `base_value + sum(contributions.values())`
        reconstructs this bank's predicted failure probability (SHAP's
        additivity property) — the UI can show that as "the math checks out".

    Notes
    -----
    `TreeExplainer.shap_values()`'s return shape genuinely differs across
    SHAP versions and model types — this was hit and fixed while wiring up
    the backend (SHAP 0.49 returns something different from what older
    tutorials assume), so it's handled explicitly for three shapes rather
    than trusting one:
      1. `list` of one array per class — `[array_for_class_0, array_for_class_1]`
         (older API).
      2. `ndarray` of shape `(n_rows, n_features, n_classes)` — the current
         SHAP behavior for a binary `RandomForestClassifier`; index
         `[:, :, 1]` for the "failed" class.
      3. `ndarray` of shape `(n_rows, n_features)` — already scoped to a
         single (positive) class.
    Printing `np.asarray(shap_values).shape` is the fastest way to check
    which case a given SHAP version/model combination hits.
    """
    raw_shap_values = explainer.shap_values(X_row)

    if isinstance(raw_shap_values, list):
        # Case 1 — one array per class. Index 1 is the "failed" class (see
        # baseline.py — 1 = failed, 0 = healthy); [0] takes the first (and
        # only) row, since X_row is one bank.
        values = np.asarray(raw_shap_values[1])[0]
        base_value = explainer.expected_value[1]
    else:
        arr = np.asarray(raw_shap_values)
        if arr.ndim == 3:
            # Case 2 — (n_rows, n_features, n_classes).
            values = arr[0, :, 1]
        else:
            # Case 3 — (n_rows, n_features), already the positive class.
            values = arr[0]
        base_value = (
            explainer.expected_value
            if np.isscalar(explainer.expected_value)
            else np.asarray(explainer.expected_value)[-1]
        )

    contributions = {
        col: float(val) for col, val in zip(X_row.columns, values)
    }
    return {"base_value": float(base_value), "contributions": contributions}
