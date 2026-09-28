"""Adversarial evasion ("window dressing") against the real CAMEL model.

Attack: a bank the model rates high-risk nudges its reported ratios, a little at
a time, in whichever direction lowers the model's score most per unit of
change, until it drops below the 0.60 "low risk" line. This is the tabular
version of the gradient-sign idea behind FGSM (perturb in the direction
that moves the loss most), done by black-box search because trees have no
gradient.

Defence: the model is not the only gate. A plausibility layer checks (1) hard
ranges, (2) whether the profile looks like any real bank (IsolationForest
trained on the FDIC panel) and (3) quarter-over-quarter jumps versus what
the same bank reported last time. Any flag routes the case to a human.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS
from camel_sentinel.xai.counterfactual import FEATURE_LIMITS, HEALTHY_DIRECTION

# Largest believable one-quarter move per ratio (percentage points), rule-of-thumb
# values for the demo — a supervisor would calibrate these from history.
MAX_QOQ_CHANGE = {
    "tier1_ratio": 3.0,
    "npl_ratio": 1.5,
    "efficiency_ratio": 10.0,
    "roa": 0.75,
    "loan_deposit_ratio": 12.0,
    "loan_concentration": 8.0,
}


def _prob(model: Any, values: dict[str, float]) -> float:
    return float(model.predict_proba(pd.DataFrame([values])[DEFAULT_FEATURE_COLS])[0][1])


def evade(model: Any, ratios: dict[str, float], target: float = 0.60,
          step_frac: float = 0.02, max_steps: int = 80) -> dict[str, Any]:
    """Greedy black-box evasion: best probability drop per normalised step."""
    current = {c: float(ratios[c]) for c in DEFAULT_FEATURE_COLS}
    start_p = _prob(model, current)
    path = [{"step": 0, "feature": None, "value": None, "probability": start_p}]
    p = start_p
    for step in range(1, max_steps + 1):
        if p < target:
            break
        best = None
        for feat in DEFAULT_FEATURE_COLS:
            low, high = FEATURE_LIMITS[feat]
            delta = (high - low) * step_frac * (1 if HEALTHY_DIRECTION[feat] == "higher" else -1)
            new_val = float(np.clip(current[feat] + delta, low, high))
            if new_val == current[feat]:
                continue
            trial = {**current, feat: new_val}
            tp = _prob(model, trial)
            if best is None or tp < best[0]:
                best = (tp, feat, new_val)
        if best is None:
            break
        # Trees are piecewise-flat, so a step may not lower p yet; keep walking
        # (bounded by max_steps) — the next split threshold may be one step away.
        p, feat, new_val = best
        current[feat] = new_val
        path.append({"step": step, "feature": feat, "value": round(new_val, 3), "probability": p})
    changes = {c: round(current[c] - float(ratios[c]), 3) for c in DEFAULT_FEATURE_COLS
               if abs(current[c] - float(ratios[c])) > 1e-9}
    l1 = sum(abs(v) / (FEATURE_LIMITS[c][1] - FEATURE_LIMITS[c][0]) for c, v in changes.items())
    return {"original_probability": start_p, "adversarial_probability": p, "evaded": p < target,
            "adversarial_ratios": {c: round(v, 3) for c, v in current.items()},
            "changes": changes, "normalized_l1": round(l1, 4), "path": path, "target": target}


def build_plausibility_model(panel: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Learn what real bank profiles look like (1st-99th percentile + IsolationForest)."""
    X = panel[DEFAULT_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    X = X.sample(min(20000, len(X)), random_state=seed)
    forest = IsolationForest(n_estimators=150, contamination=0.02, random_state=seed).fit(X.values)
    return {"forest": forest, "p01": X.quantile(0.01).to_dict(), "p99": X.quantile(0.99).to_dict()}


def plausibility_check(pm: dict[str, Any], previous: dict[str, float], reported: dict[str, float]) -> dict[str, Any]:
    """Flags that would send this filing to a human instead of auto-accepting the score."""
    flags = []
    for c in DEFAULT_FEATURE_COLS:
        v = float(reported[c])
        low, high = FEATURE_LIMITS[c]
        if not low <= v <= high:
            flags.append({"check": "hard range", "feature": c, "detail": f"{v} outside [{low}, {high}]"})
        elif not pm["p01"][c] <= v <= pm["p99"][c]:
            flags.append({"check": "peer range", "feature": c,
                          "detail": f"{v:.2f} outside the 1st–99th percentile of real banks "
                                    f"[{pm['p01'][c]:.2f}, {pm['p99'][c]:.2f}]"})
        jump = abs(v - float(previous[c]))
        if jump > MAX_QOQ_CHANGE[c]:
            flags.append({"check": "quarter-over-quarter", "feature": c,
                          "detail": f"moved {jump:.2f} pts in one quarter (limit {MAX_QOQ_CHANGE[c]})"})
    row = np.array([[float(reported[c]) for c in DEFAULT_FEATURE_COLS]])
    anomaly = float(-pm["forest"].score_samples(row)[0])
    is_outlier = bool(pm["forest"].predict(row)[0] == -1)
    if is_outlier:
        flags.append({"check": "profile anomaly", "feature": "all",
                      "detail": f"IsolationForest: this combination of ratios is unlike real banks (score {anomaly:.3f})"})
    return {"flags": flags, "blocked": bool(flags), "anomaly_score": round(anomaly, 4),
            "decision": "Route to human review" if flags else "Auto-accept model score"}
