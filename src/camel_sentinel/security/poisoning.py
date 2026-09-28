"""Training-data poisoning on a sample of the real FDIC panel, and its defence.

Two attacks an adversary with write access to the data feed could run:

* **label_flip** — relabel a share of genuinely failed banks as "survived", so
  the retrained model learns that failing profiles look safe (a targeted
  integrity attack: it hurts recall on exactly the banks that matter).
* **injection** — append fabricated rows that look like failing banks but are
  labelled "survived" (plus some sloppy, out-of-range junk rows).

Defence pipeline before (re)training:
  1. schema/range validation (drops impossible rows);
  2. label audit (confident-learning idea): a model trained only on a small,
     independently audited slice of history scores every incoming row, and rows
     whose label strongly contradicts it are held back for human review.

(An IsolationForest outlier filter was tried first and rejected: failed banks
are rare, so an outlier filter throws away exactly the examples that matter.)

All three are measured on the same untouched test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import train_test_split

from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS
from camel_sentinel.xai.counterfactual import FEATURE_LIMITS


def prepare_sample(panel: pd.DataFrame, n: int = 24000, seed: int = 7) -> pd.DataFrame:
    """Balanced-enough working sample: every failure (up to n/4) plus survivors."""
    df = panel[[*DEFAULT_FEATURE_COLS, "failed"]].replace([np.inf, -np.inf], np.nan).dropna()
    for c, (low, high) in FEATURE_LIMITS.items():
        df = df[df[c].between(low, high)]
    pos = df[df["failed"] == 1]
    pos = pos.sample(min(len(pos), n // 4), random_state=seed)
    neg = df[df["failed"] == 0].sample(n - len(pos), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _fit_eval(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0)
    model.fit(train[DEFAULT_FEATURE_COLS], train["failed"])
    proba = model.predict_proba(test[DEFAULT_FEATURE_COLS])[:, 1]
    return {
        "pr_auc": round(float(average_precision_score(test["failed"], proba)), 4),
        "recall_at_0_5": round(float(recall_score(test["failed"], proba >= 0.5)), 4),
        "train_rows": int(len(train)),
    }


def _poison(train: pd.DataFrame, attack: str, rate: float, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Return poisoned training data and a boolean mask of which rows are poisoned."""
    train = train.copy()
    if attack == "label_flip":
        pos_idx = train.index[train["failed"] == 1].to_numpy()
        k = int(round(rate * len(pos_idx)))
        flip = rng.choice(pos_idx, size=k, replace=False) if k else np.array([], dtype=int)
        train.loc[flip, "failed"] = 0
        return train, train.index.isin(flip)
    # injection: fabricated rows that *look like* failing banks but are labelled "survived".
    # 80% are plausible (real failed profiles + small noise); 20% are sloppy, out-of-range junk.
    k = int(round(rate * len(train)))
    k_junk = k // 5
    failed = train[train["failed"] == 1][DEFAULT_FEATURE_COLS]
    base = failed.sample(k - k_junk, replace=True, random_state=int(rng.integers(1e9))).to_numpy()
    plausible = base * rng.normal(1.0, 0.03, base.shape)
    junk = np.column_stack([rng.uniform(0.5, 3.0, k_junk), rng.uniform(16, 40, k_junk), rng.uniform(115, 180, k_junk),
                            rng.uniform(-6, -2.5, k_junk), rng.uniform(155, 220, k_junk), rng.uniform(85, 100, k_junk)])
    fake = pd.DataFrame(np.vstack([plausible, junk]), columns=DEFAULT_FEATURE_COLS)
    fake["failed"] = 0
    out = pd.concat([train, fake], ignore_index=True)
    return out, np.r_[np.zeros(len(train), bool), np.ones(k, bool)]


def _defend(train: pd.DataFrame, trusted: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Range validation, then a label audit against a model trained only on trusted data."""
    removed = {"range_validation": 0, "label_audit": 0}
    keep = np.ones(len(train), bool)
    for c, (low, high) in FEATURE_LIMITS.items():
        keep &= train[c].between(low, high).to_numpy()
    removed["range_validation"] = int((~keep).sum())
    t = train[keep]

    # Confident-learning style audit: a model fit on the trusted, audited slice scores every
    # incoming row; rows whose label strongly contradicts it are held back for review.
    auditor = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    auditor.fit(trusted[DEFAULT_FEATURE_COLS], trusted["failed"])
    p_fail = auditor.predict_proba(t[DEFAULT_FEATURE_COLS])[:, 1]
    labels = t["failed"].to_numpy()
    suspicious = ((labels == 0) & (p_fail >= 0.5)) | ((labels == 1) & (p_fail <= 0.01))
    removed["label_audit"] = int(suspicious.sum())
    return t[~suspicious], removed


def run_poisoning_lab(sample: pd.DataFrame, attack: str = "label_flip", rate: float = 0.3,
                      seed: int = 11) -> dict[str, Any]:
    """Clean vs poisoned vs poisoned-then-defended, on one untouched test set."""
    if attack not in {"label_flip", "injection"}:
        raise ValueError("attack must be 'label_flip' or 'injection'")
    rng = np.random.default_rng(seed)
    train, test = train_test_split(sample, test_size=0.3, stratify=sample["failed"], random_state=seed)
    # A small, independently verified slice of history the defender trusts (e.g. audited quarters).
    train, trusted = train_test_split(train, test_size=0.25, stratify=train["failed"], random_state=seed)
    train, trusted = train.reset_index(drop=True), trusted.reset_index(drop=True)

    clean = _fit_eval(pd.concat([train, trusted]), test)
    poisoned_train, mask = _poison(train, attack, rate, rng)
    poisoned = _fit_eval(pd.concat([poisoned_train, trusted]), test)
    defended_train, removed = _defend(poisoned_train, trusted, seed)
    defended = _fit_eval(pd.concat([defended_train, trusted]), test)

    kept_poison = int(mask[defended_train.index.to_numpy()].sum()) if len(defended_train) else 0
    total_poison = int(mask.sum())
    return {
        "attack": attack, "rate": rate, "poisoned_rows": total_poison,
        "clean": clean, "poisoned": poisoned, "defended": defended,
        "removed": removed, "poison_caught": total_poison - kept_poison,
        "clean_rows_removed": int(sum(removed.values()) - (total_poison - kept_poison)),
        "test_rows": int(len(test)), "test_failures": int(test["failed"].sum()),
    }
