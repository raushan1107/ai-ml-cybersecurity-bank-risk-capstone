"""Local counterfactual search for the six CAMEL model inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS

FEATURE_LIMITS = {
    "tier1_ratio": (3.5, 50.0),
    "npl_ratio": (0.0, 15.0),
    "efficiency_ratio": (30.0, 110.0),
    "roa": (-2.0, 4.0),
    "loan_deposit_ratio": (20.0, 150.0),
    "loan_concentration": (0.0, 100.0),
}

HEALTHY_DIRECTION = {
    "tier1_ratio": "higher",
    "npl_ratio": "lower",
    "efficiency_ratio": "lower",
    "roa": "higher",
    "loan_deposit_ratio": "lower",
    "loan_concentration": "lower",
}


def _probability(model: Any, values: dict[str, float]) -> float:
    row = pd.DataFrame([values])[DEFAULT_FEATURE_COLS]
    return float(model.predict_proba(row)[0][1])


def find_counterfactual(
    model: Any,
    X_row: pd.DataFrame,
    target_threshold: float = 0.60,
    steps: int = 24,
) -> dict[str, Any]:
    """Find a simple one-feature-at-a-time path below the target risk threshold.

    This is a transparent search, not a globally optimal counterfactual solver:
    it changes only CAMEL ratios, respects the UI bounds, and chooses the
    smallest normalized change that lowers the model score at each step.
    """
    current = {column: float(X_row.iloc[0][column]) for column in DEFAULT_FEATURE_COLS}
    starting_probability = _probability(model, current)
    if starting_probability < target_threshold:
        return {
            "found": True,
            "already_below_threshold": True,
            "target_threshold": target_threshold,
            "starting_probability": starting_probability,
            "counterfactual_probability": starting_probability,
            "changes": {},
            "values": current,
            "method": "greedy bounded one-feature search",
        }

    working = current.copy()
    changes: dict[str, float] = {}
    for _ in range(len(DEFAULT_FEATURE_COLS) * 2):
        candidates = []
        for feature in DEFAULT_FEATURE_COLS:
            low, high = FEATURE_LIMITS[feature]
            direction = HEALTHY_DIRECTION[feature]
            endpoint = high if direction == "higher" else low
            values = np.linspace(working[feature], endpoint, steps + 1)[1:]
            for value in values:
                trial = working.copy()
                trial[feature] = float(value)
                probability = _probability(model, trial)
                normalized_change = abs(value - current[feature]) / (high - low)
                candidates.append((probability, normalized_change, feature, float(value)))

        if not candidates:
            break
        best = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
        probability, _, feature, value = best
        if probability >= _probability(model, working) and len(changes) >= len(DEFAULT_FEATURE_COLS):
            break
        working[feature] = value
        changes[feature] = value - current[feature]
        if probability < target_threshold:
            return {
                "found": True,
                "already_below_threshold": False,
                "target_threshold": target_threshold,
                "starting_probability": starting_probability,
                "counterfactual_probability": probability,
                "changes": changes,
                "values": working,
                "method": "greedy bounded one-feature search",
            }

    final_probability = _probability(model, working)
    return {
        "found": final_probability < target_threshold,
        "already_below_threshold": False,
        "target_threshold": target_threshold,
        "starting_probability": starting_probability,
        "counterfactual_probability": final_probability,
        "changes": changes,
        "values": working,
        "method": "greedy bounded one-feature search",
    }
