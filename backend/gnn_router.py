"""
gnn_router.py — FastAPI router for the GNN fraud-detection module.

Endpoints
---------
POST /gnn/score
    Score a single transaction and return fraud risk score + explanation.

GET /gnn/graph-sample
    Return summary statistics about the synthetic graph that was used for
    training (node counts, edge counts, fraud rate, pattern breakdown).

GET /gnn/demo/{demo_id}
    Return one of the three pre-defined demo transactions with its score
    and a plain-English explanation.  demo_id ∈ {1, 2, 3}.

GET /gnn/status
    Health check — reports whether the GNN model is loaded.

⚠ SYNTHETIC DEMONSTRATION — not trained on real customer transaction data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Make src/ importable when the router is imported from backend/main.py
SRC_PATH = str((Path(__file__).resolve().parents[1] / "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from camel_sentinel.gnn.detector import (
    DEFAULT_POLICY,
    FraudPolicy,
    get_demo_results,
    get_load_error,
    is_model_ready,
    score_transaction,
)

router = APIRouter(prefix="/gnn", tags=["GNN Fraud Detection"])

_DEMO_CACHE: list[dict] = []
_DEMO_CACHE_BUILDING: bool = False


def start_demo_cache_build() -> None:
    """
    Spawn a background thread to pre-build the demo cache.
    Called once from main.py's lifespan after the GNN model loads.
    The 66-second generate+build cost happens off the request path.
    """
    import threading

    def _worker():
        global _DEMO_CACHE_BUILDING
        _DEMO_CACHE_BUILDING = True
        try:
            _build_demo_cache()
        finally:
            _DEMO_CACHE_BUILDING = False

    if not _DEMO_CACHE and not _DEMO_CACHE_BUILDING:
        threading.Thread(target=_worker, daemon=True, name="gnn-demo-build").start()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    # Transaction features
    amount: float = Field(..., description="Transaction amount (raw, not standardised)")
    amount_zscore: float = Field(default=0.0, description="(amount - account_mean) / account_std")
    hour_of_day: int = Field(default=12, ge=0, le=23, description="Hour of day 0-23")
    is_cross_border: int = Field(default=0, ge=0, le=1, description="1 if cross-border")
    time_since_last_txn: float = Field(default=86400.0, description="Seconds since last transaction")
    merchant_risk_score: float = Field(default=0.05, ge=0, le=1)
    merchant_category_enc: int = Field(default=0, description="Encoded merchant category integer")

    # Account features
    balance: float = Field(default=10000.0)
    avg_transaction_amount: float = Field(default=500.0)
    transaction_velocity: float = Field(default=1.0, description="Transactions per day")
    account_age_days: float = Field(default=365.0)

    # Customer features
    age: float = Field(default=35.0)
    tenure_days: float = Field(default=730.0)
    country_risk_score: float = Field(default=1.0, ge=1, le=4)
    num_accounts: int = Field(default=1)

    # Merchant features
    merchant_transaction_count: int = Field(default=100)
    merchant_country_risk: float = Field(default=1.0)

    # Device features
    device_num_accounts: int = Field(default=1, description="Number of accounts on this device")
    device_num_customers: int = Field(default=1)
    device_risk_score: float = Field(default=0.05, ge=0, le=1)

    # Policy override
    low_threshold: float = Field(default=0.30, ge=0, le=1)
    high_threshold: float = Field(default=0.70, ge=0, le=1)


class ScoreResponse(BaseModel):
    fraud_risk_score: float
    decision: str
    policy: dict
    important_nodes: list[dict]
    important_edges: list[dict]
    narrative: str
    caveats: list[str]
    warning: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status")
def gnn_status() -> dict:
    """Report whether the GNN model is loaded and ready."""
    import math
    ready = is_model_ready()
    return {
        "gnn_model_loaded": ready,
        "message": "GNN fraud detection ready." if ready else get_load_error(),
        "warning": "SYNTHETIC DEMONSTRATION — not trained on real data.",
    }


@router.post("/score", response_model=ScoreResponse)
def gnn_score(txn: TransactionInput) -> ScoreResponse:
    """
    Score a single transaction for fraud risk.

    Returns a fraud risk score ∈ [0, 1] (uncalibrated model score, not a
    probability), a decision (APPROVE / REVIEW / BLOCK), and a plain-English
    explanation of which node features contributed most.
    """
    import math
    if not is_model_ready():
        raise HTTPException(
            status_code=503,
            detail=get_load_error(),
        )

    policy = FraudPolicy(
        low_threshold=txn.low_threshold,
        high_threshold=txn.high_threshold,
    )

    import math
    hour_sin = math.sin(2 * math.pi * txn.hour_of_day / 24)
    hour_cos = math.cos(2 * math.pi * txn.hour_of_day / 24)

    transaction_features = {
        "amount": txn.amount,
        "amount_zscore": txn.amount_zscore,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_cross_border": float(txn.is_cross_border),
        "time_since_last_txn": txn.time_since_last_txn,
        "merchant_risk_score": txn.merchant_risk_score,
        "merchant_category_enc": float(txn.merchant_category_enc),
    }
    account_features = {
        "balance": txn.balance,
        "avg_transaction_amount": txn.avg_transaction_amount,
        "transaction_velocity": txn.transaction_velocity,
        "account_age_days": txn.account_age_days,
    }
    customer_features = {
        "age": txn.age,
        "tenure_days": txn.tenure_days,
        "country_risk_score": txn.country_risk_score,
        "num_accounts": float(txn.num_accounts),
    }
    merchant_features = {
        "merchant_risk_score": txn.merchant_risk_score,
        "transaction_count": float(txn.merchant_transaction_count),
        "country_risk_score": txn.merchant_country_risk,
    }
    device_features = {
        "num_accounts": float(txn.device_num_accounts),
        "num_customers": float(txn.device_num_customers),
        "device_risk_score": txn.device_risk_score,
    }

    result = score_transaction(
        transaction_features=transaction_features,
        account_features=account_features,
        customer_features=customer_features,
        merchant_features=merchant_features,
        device_features=device_features,
        policy=policy,
    )

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return ScoreResponse(**result)


@router.get("/graph-sample")
def gnn_graph_sample() -> dict:
    """
    Return summary statistics about the synthetic training graph.

    ⚠ SYNTHETIC DEMONSTRATION — all numbers describe generated data,
    not real customer transactions.
    """
    return {
        "warning": "SYNTHETIC DEMONSTRATION — not trained on real customer transaction data.",
        "description": (
            "Heterogeneous graph with 5 node types and 5 edge types (+ reverse). "
            "Generated by camel_sentinel.gnn.synthetic_data.generate() for "
            "educational demonstration of GNN-based fraud detection."
        ),
        "node_types": {
            "customer": {"approx_count": 1000, "features": 4,
                         "description": "Individual account holders"},
            "account": {"approx_count": 2469, "features": 4,
                        "description": "Bank accounts owned by customers"},
            "transaction": {"approx_count": 55979, "features": 8,
                            "description": "Payments and transfers — fraud labels here"},
            "merchant": {"approx_count": 350, "features": 3,
                         "description": "Payment recipients"},
            "device": {"approx_count": 212, "features": 3,
                       "description": "Device fingerprints used to access accounts"},
        },
        "edge_types": [
            "customer → owns → account",
            "account → makes → transaction",
            "transaction → paid_to → merchant",
            "customer → uses → device",
            "account → uses → device",
            "(+ 5 reverse edge types for bidirectional message passing)",
        ],
        "fraud_patterns": {
            "A_tabular": "High amount + unusual hour + cross-border (~16%)",
            "B_ring": "Normal features, shared-device ring (~42%)",
            "C_merchant": "Normal features, high-risk merchant (~16%)",
            "D_velocity": "Rapid transaction burst (~26%)",
        },
        "fraud_rate_pct": "~5.8%",
        "split": "Temporal: 70% train / 15% val / 15% test",
        "key_educational_point": (
            "Pattern B (relational ring fraud) has NORMAL transaction features. "
            "A tabular model cannot detect it. The GNN can — because it propagates "
            "information from the shared device through the account to the transaction."
        ),
    }


@router.get("/demo/{demo_id}")
def gnn_demo(demo_id: int) -> dict:
    """
    Return one of the three pre-defined demo transactions.

    demo_id 1: Normal features, suspicious neighbourhood (ring account)
    demo_id 2: Suspicious features, moderate neighbourhood
    demo_id 3: Raushan transfers $2,000 — normal in every dimension

    These demonstrate the core GNN advantage over tabular models.
    """
    if demo_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="demo_id must be 1, 2, or 3.")
    if not is_model_ready():
        raise HTTPException(status_code=503, detail=get_load_error())

    if not _DEMO_CACHE:
        if _DEMO_CACHE_BUILDING:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Demo cache is still building (synthetic data generation takes ~60 s). "
                    "Retry in a moment."
                ),
            )
        raise HTTPException(
            status_code=503,
            detail="Demo results not available. Run the training notebook first.",
        )

    idx = demo_id - 1
    if idx >= len(_DEMO_CACHE):
        raise HTTPException(status_code=404, detail=f"Demo {demo_id} not found.")

    return _DEMO_CACHE[idx]


def _build_demo_cache() -> None:
    """Load graph, run demo transactions, populate _DEMO_CACHE."""
    import logging
    logger = logging.getLogger(__name__)

    # is_model_ready() triggers the lazy load if not done yet
    if not is_model_ready():
        logger.warning("_build_demo_cache: GNN model not ready, skipping.")
        return

    from camel_sentinel.gnn.synthetic_data import generate
    from camel_sentinel.gnn.graph_builder import build_hetero_data
    from camel_sentinel.gnn.detector import _CACHE

    try:
        logger.info("GNN demo cache: generating synthetic data (~30 s)…")
        c, a, t, m, d = generate(seed=42)
        logger.info("GNN demo cache: building hetero graph (~35 s)…")
        data, _ = build_hetero_data(c, a, t, m, d)
        logger.info("GNN demo cache: running demo inference…")
        results = get_demo_results(data, _CACHE["model"])
        _DEMO_CACHE.extend(results)
        logger.info("GNN demo cache: ready (%d entries).", len(_DEMO_CACHE))
    except Exception as exc:
        import traceback
        logger.error("demo cache error: %s\n%s", exc, traceback.format_exc())
        _DEMO_CACHE.extend([{"error": str(exc)}])
