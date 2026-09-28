"""Model extraction ("stealing") through the scoring API, and output hardening.

Attacker model (Tramèr et al., 2016, "Stealing Machine Learning Models via
Prediction APIs"): no training data, no weights — only the ability to send
inputs to /score and read what comes back. They sample plausible-looking
ratio vectors, record the victim's answers, and fit their own copy.

Fidelity is measured on *real* banks the attacker never queried:
  * rank agreement (Spearman ρ) between surrogate and victim scores, and
  * overlap of the 100 banks each model ranks riskiest.

Defences compared:
  * `probability` — the API returns the full probability (what /score does today),
  * `rounded`     — probability rounded to one decimal,
  * `tier_only`   — only low / medium / high.
Plus two operational controls that don't change fidelity per query but change
the attacker's cost and visibility: a per-key rate limit and a detector that
flags query streams that don't look like real bank filings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest

from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS
from camel_sentinel.xai.counterfactual import FEATURE_LIMITS

QUERY_BUDGETS = [50, 100, 250, 500, 1000, 2500, 5000]
TIER_EDGES = (0.60, 0.98)


def _tier(p: np.ndarray) -> np.ndarray:
    return np.digitize(p, TIER_EDGES)  # 0 low, 1 medium, 2 high


def attacker_queries(n: int, seed: int = 3) -> pd.DataFrame:
    """Uniform samples inside the public UI limits — all an outsider knows."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.uniform(lo, hi, n) for c, (lo, hi) in FEATURE_LIMITS.items()})[DEFAULT_FEATURE_COLS]


def _api_response(victim: Any, X: pd.DataFrame, mode: str) -> np.ndarray:
    p = victim.predict_proba(X)[:, 1]
    if mode == "rounded":
        return np.round(p, 1)
    if mode == "tier_only":
        return _tier(p).astype(float)
    return p


def _fit_surrogate(X: pd.DataFrame, y: np.ndarray, mode: str) -> Any:
    if mode == "tier_only":
        if len(np.unique(y)) < 2:
            return None
        return HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y.astype(int))
    return HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X, y)


def _surrogate_score(model: Any, X: pd.DataFrame, mode: str) -> np.ndarray:
    if model is None:
        return np.zeros(len(X))
    if mode == "tier_only":
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        return sum(proba[:, classes.index(k)] * k for k in classes)  # expected tier as a risk score
    return model.predict(X)


def run_model_stealing_lab(victim: Any, holdout: pd.DataFrame, output_mode: str = "probability",
                           max_queries: int = 5000, rate_limit_per_hour: int = 100,
                           seed: int = 3) -> dict[str, Any]:
    """Fidelity-vs-queries curve for one output mode, plus the operational defences."""
    if output_mode not in {"probability", "rounded", "tier_only"}:
        raise ValueError("output_mode must be probability, rounded or tier_only")
    real = holdout[DEFAULT_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    real = real.sample(min(4000, len(real)), random_state=seed)
    victim_real = victim.predict_proba(real)[:, 1]
    top_victim = set(np.argsort(-victim_real)[:100])

    pool = attacker_queries(max(QUERY_BUDGETS), seed)
    answers = _api_response(victim, pool, output_mode)
    curve = []
    for n in [b for b in QUERY_BUDGETS if b <= max_queries]:
        surrogate = _fit_surrogate(pool.iloc[:n], answers[:n], output_mode)
        s = _surrogate_score(surrogate, real, output_mode)
        rho = float(spearmanr(s, victim_real).correlation) if np.std(s) > 0 else 0.0
        top_s = set(np.argsort(-s)[:100])
        curve.append({"queries": n, "spearman": round(rho, 4),
                      "top100_overlap": len(top_victim & top_s) / 100,
                      "hours_at_rate_limit": round(n / rate_limit_per_hour, 1)})

    # Query-pattern detection: does the attacker's stream look like real filings?
    forest = IsolationForest(n_estimators=150, contamination=0.02, random_state=seed).fit(real)
    attacker_flags = forest.predict(pool.iloc[:1000]) == -1
    normal_flags = forest.predict(real.iloc[:1000]) == -1
    attacker_flagged, normal_flagged = float(attacker_flags.mean()), float(normal_flags.mean())
    # One odd query proves nothing; a *stream* from one key that keeps looking unlike real
    # filings does. Alert when >= 8% of a key's last 100 queries are anomalous.
    def first_alert(flags: np.ndarray, window: int = 100, share: float = 0.08) -> int | None:
        rolling = np.convolve(flags.astype(float), np.ones(window), "valid") / window
        hits = np.where(rolling >= share)[0]
        return int(hits[0] + window) if len(hits) else None
    return {
        "output_mode": output_mode, "curve": curve,
        "final": curve[-1] if curve else None,
        "detection": {"attacker_queries_flagged": round(attacker_flagged, 3),
                      "normal_queries_flagged": round(normal_flagged, 3),
                      "attacker_alert_after_queries": first_alert(attacker_flags),
                      "normal_user_alert_after_queries": first_alert(normal_flags),
                      "rule": "alert when >= 8% of a key's last 100 queries look unlike real banks"},
        "rate_limit_per_hour": rate_limit_per_hour,
        "sample_exchange": [
            {**{c: round(float(pool.iloc[i][c]), 2) for c in DEFAULT_FEATURE_COLS},
             "api_returned": (["low", "medium", "high"][int(answers[i])] if output_mode == "tier_only"
                              else round(float(answers[i]), 4))}
            for i in range(5)
        ],
    }
