"""
registry.py — save/load a trained model as a reusable artifact.

WHAT this file does: two thin functions around `joblib` (scikit-learn's
own recommended serialization tool — more efficient than plain `pickle`
for objects holding large NumPy arrays, which a fitted RandomForest's
trees are).

WHY this exists as its own file instead of inline `pickle.dump` calls in a
notebook: this is the one seam every later consumer of the trained model
shares — the FastAPI backend loads the model through `load_model()`, not
by re-running training; the agent's "score this bank" tool does the same.
One tested save/load path instead of three copies of `joblib.load(...)`.

HOW it connects: `models/` at the project root (gitignored — see
.gitignore) is where artifacts land by default; `02_baseline_modeling.ipynb`
calls `save_model()` at the end of Phase 3.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

# Default location for saved model artifacts — the project-root `models/`
# directory (gitignored; see .gitignore). Resolved relative to this file
# (not the current working directory) so it works the same whether called
# from a notebook in notebooks/, a test, or the backend.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def save_model(model: BaseEstimator, filename: str = "camel_baseline_model.joblib") -> Path:
    """
    Save a fitted model to the project's `models/` directory.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Any fitted scikit-learn-compatible model (e.g. the
        `RandomForestClassifier` from `baseline.train_baseline_model()`).
    filename : str, default "camel_baseline_model.joblib"
        Name for the saved file. Optional — override if saving more than
        one model version (e.g. `"camel_tuned_model.joblib"` in Phase 4).

    Returns
    -------
    pathlib.Path
        The full path the model was written to, so the caller can log or
        print it.
    """
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEFAULT_MODELS_DIR / filename
    # joblib.dump(obj, path) — like pickle.dump, but stores large NumPy
    # arrays (a RandomForest's many trees) more compactly and loads them
    # faster; this is scikit-learn's own documented recommendation for
    # persisting its models.
    joblib.dump(model, out_path)
    return out_path


def load_model(filename: str = "camel_baseline_model.joblib") -> BaseEstimator:
    """
    Load a previously saved model back into memory.

    Parameters
    ----------
    filename : str, default "camel_baseline_model.joblib"
        Which file under the project's `models/` directory to load.
        Optional.

    Returns
    -------
    sklearn.base.BaseEstimator
        The fitted model, ready to call `.predict()` / `.predict_proba()`
        on new data with the same feature columns it was trained on
        (see `baseline.DEFAULT_FEATURE_COLS`).
    """
    return joblib.load(DEFAULT_MODELS_DIR / filename)
