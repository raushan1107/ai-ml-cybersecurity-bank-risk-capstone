"""
baseline.py — Phase 3: the first real (non-rule-based) model.

WHAT this file does: trains a `RandomForestClassifier` to predict whether a
bank fails, using the six CAMEL ratios as features — this is the model
Phase 5's XAI layer (SHAP) will explain, and the one Phase 6's RAI layer
will audit for fairness.

WHY train on `failed` and not `proxy_rating_1_5`: the composite rating is
built *deterministically* from these same six ratios (see
`camel_sentinel.features.camel_ratios.score_composite`) — a model "trained"
to reproduce it would just be re-deriving a formula it already has direct
access to, not learning anything. `failed` is a real, independent outcome
pulled from the FDIC's failure history, so predicting it is a genuine
supervised-learning problem. The composite score's job (see Phase 1) is to
be validated *against* this same `failed` flag, not to be the model's
prediction target itself.

WHY RandomForestClassifier as the first model: tabular data, handles
non-linear ratio interactions without manual feature crosses, gives
`feature_importances_` for free, and — most importantly for Phase 5 — pairs
directly with `shap.TreeExplainer`, which is exact and fast for tree
ensembles (as opposed to the slower, approximate model-agnostic SHAP
explainers a neural net would require).

HOW it connects to the rest of the pipeline: takes the `camel` DataFrame
produced by `camel_sentinel.features.camel_ratios` as input; its output
(a fitted model) is what `camel_sentinel.xai` (Phase 5) and
`camel_sentinel.rai` (Phase 6) will both take as input in later phases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# The six CAMEL ratio columns built by camel_ratios.build_camel_ratios() —
# kept as one shared constant so the notebook, this module, and later the
# XAI/RAI modules all agree on exactly which columns are "the features"
# without retyping the list in three places.
DEFAULT_FEATURE_COLS = [
    "tier1_ratio",
    "npl_ratio",
    "efficiency_ratio",
    "roa",
    "loan_deposit_ratio",  # Liquidity input — NOT liquid_asset_ratio, which
                            # is 100% missing on real FDIC data (see the
                            # comment on it in camel_ratios.build_camel_ratios())
    "loan_concentration",
]


def prepare_training_data(
    camel: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "failed",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Slice a scored CAMEL table into a feature matrix and target vector.

    Parameters
    ----------
    camel : pandas.DataFrame
        Output of `camel_ratios.score_composite()` (or at least
        `build_camel_ratios()`) — must contain `feature_cols` and
        `target_col`.
    feature_cols : list[str], optional
        Which columns to use as model inputs. Defaults to
        `DEFAULT_FEATURE_COLS` (the six CAMEL ratios) if not given.
    target_col : str, default "failed"
        Which column is the prediction target. Defaults to the real,
        independent failure flag — see the module docstring for why this
        is deliberate and not `proxy_rating_1_5`.

    Returns
    -------
    tuple(X, y)
        `X`: pandas.DataFrame of just the feature columns, with missing
        values filled using each column's median (a simple, defensible
        default for a small number of continuous ratio columns — revisit
        if a feature turns out to have structurally-missing-not-at-random
        gaps).
        `y`: pandas.Series, the target column, unchanged.
    """
    cols = feature_cols or DEFAULT_FEATURE_COLS
    X = camel[cols].copy()
    # A ratio like loan_deposit_ratio divides by a real-world field (DEP)
    # that is occasionally exactly 0 for a handful of banks, producing
    # +/-inf rather than NaN — replace those with NaN first so the median
    # fill below (and the model fit after it) never sees an infinity.
    X = X.replace([np.inf, -np.inf], np.nan)
    # DataFrame.fillna(DataFrame.median()) — median is used (rather than
    # mean) because ratio columns like npl_ratio are right-skewed, so the
    # median is a more representative "typical bank" fill value.
    X = X.fillna(X.median())
    y = camel[target_col]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split — wraps `sklearn.model_selection.train_test_split`.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix, from `prepare_training_data()`.
    y : pandas.Series
        Target vector, from `prepare_training_data()`.
    test_size : float, default 0.25
        Fraction of rows held out for testing. Optional.
    seed : int, default 42
        Passed to `train_test_split(random_state=...)` for a reproducible
        split. Optional.

    Returns
    -------
    tuple(X_train, X_test, y_train, y_test)

    Notes
    -----
    `stratify=y` is passed deliberately: because `failed` is rare
    (typically a few percent of rows), a plain random split could easily
    put almost none of the minority class in the test set by chance —
    `stratify` forces the same failure rate in both splits.
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    seed: int = 42,
) -> RandomForestClassifier:
    """
    Fit the baseline RandomForestClassifier.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / Series
        Training split from `split_train_test()`.
    n_estimators : int, default 300
        Number of trees in the forest — passed straight to
        `RandomForestClassifier`. Optional; 300 is a reasonable default for
        a dataset of a few thousand rows (more trees costs training time,
        not accuracy, past a point).
    max_depth : int, default 6
        Max depth per tree — kept shallow deliberately. Optional; a
        shallower forest is both less prone to overfitting a few hundred
        failure examples and produces simpler, more explainable SHAP
        attributions in Phase 5.
    seed : int, default 42
        `random_state` for reproducibility. Optional.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Already `.fit()` on the given training data.

    Notes
    -----
    `class_weight="balanced"` is the key parameter here: it automatically
    re-weights the loss so the rare `failed` class counts for as much as
    the common `healthy` class during training, instead of the model
    learning "always predict healthy" as a shortcut to high accuracy.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: tuple[str, str] = ("healthy", "failed"),
) -> dict:
    """
    Score the model on held-out data and print an imbalance-aware report.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted model, from `train_baseline_model()`.
    X_test, y_test : pandas.DataFrame / Series
        Held-out split from `split_train_test()`.
    target_names : tuple[str, str], default ("healthy", "failed")
        Human-readable labels for class 0 and class 1, used only for the
        printed report. Optional.

    Returns
    -------
    dict
        `{"report": str, "confusion_matrix": numpy.ndarray, "y_pred": numpy.ndarray}`
        — returned (not just printed) so a notebook or the RAI module can
        reuse `y_pred` (e.g. for the fairness breakdown in Phase 6) without
        re-predicting.

    Notes
    -----
    Report precision/recall/F1 on the minority ("failed") class, not raw
    accuracy — with failures at a few percent of banks, a model that
    predicts "healthy" for everyone would still score >95% accuracy while
    catching zero real distress (see docs/01_glossary.md, "Class imbalance").
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=list(target_names))
    matrix = confusion_matrix(y_test, y_pred)

    print(report)
    print("Confusion matrix:\n", matrix)

    return {"report": report, "confusion_matrix": matrix, "y_pred": y_pred}
