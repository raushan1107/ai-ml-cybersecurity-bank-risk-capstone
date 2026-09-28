"""
backend/main.py — the served model: a FastAPI wrapper around the Phase 4
tuned model and the Phase 5 SHAP explainer.

WHAT this file does: on startup, loads the trained model
(`models/camel_tuned_model.joblib` — the Phase 4 HistGradientBoosting
model, tuned and validated in `notebooks/03_model_tuning.ipynb` on a
265k-row, 2005-2023 multi-quarter panel; see that notebook for why it
replaced Phase 3's single-quarter `camel_baseline_model.joblib`), builds a
SHAP explainer for it once, and pulls one quarter of real bank data to
serve as a browsable sample list. Exposes three endpoints: `GET /banks`
(sample banks to pick from), `POST /score` (predict + explain one bank's
ratios), and `GET /model-card` (the caveats every prediction should be
read with).

WHY a backend instead of the UI calling the model directly: this is the
seam between "a model that only runs inside a notebook" and "a model
something else can query" — the Streamlit UI in `frontend/app.py` is one
caller today; a future agent tool-call (Phase 8) would be another, against
the same endpoints, without duplicating the loading/explaining logic.

HOW to run it: `python -m uvicorn main:app --app-dir backend --reload
--port 8000` from the project root (or see README.md). Requires
`notebooks/03_model_tuning.ipynb` to have been run at least once, so
`models/camel_tuned_model.joblib` exists.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import asyncio

import openai
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Load .env if python-dotenv is installed — no-op otherwise.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Make src/camel_sentinel importable the same way the notebooks do — see
# the comment in notebooks/00_dataset_research.ipynb's setup cell.
SRC_PATH = str((Path(__file__).resolve().parents[1] / "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from camel_sentinel.data import fdic_client
from camel_sentinel.agent.tools import dispatch_tool, tool_catalog
from camel_sentinel.chat.agent import ChatAgent
from camel_sentinel.chat.history import ConversationHistory
from camel_sentinel.chat.speech import synthesize_text, transcribe_audio
from camel_sentinel.features import camel_ratios
from camel_sentinel.models import registry
from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS
from camel_sentinel.rai.audit import audit_model
from camel_sentinel.xai.counterfactual import find_counterfactual
from camel_sentinel.xai import explain as xai_explain

# Phase 10 — GNN fraud detection (self-contained; no existing endpoints changed)
from gnn_router import router as gnn_router, start_demo_cache_build

# AI Security Lab — attacks and defences against this system (docs/07_ai_security_lab.md)
from security_router import router as security_router, bind_state as bind_security_state, prewarm as prewarm_security
from camel_sentinel.security.pii import redact as redact_pii
from camel_sentinel.security.prompt_guard import detect_injection

# Populated once at startup by `lifespan()` below — kept as module-level
# state (rather than re-loaded per request) because loading the model and
# building the SHAP explainer are the two relatively expensive steps in
# this whole service; every request should be cheap by comparison.
STATE: dict = {}

MODEL_CATALOG = {
    "tuned": {
        "label": "HistGradientBoosting (tuned) — recommended",
        "filename": "camel_tuned_model.joblib",
        "description": "Best PR-AUC on the untouched future holdout; optimized for early warning.",
        "available_for_scoring": True,
    },
    "baseline": {
        "label": "RandomForest — baseline",
        "filename": "camel_baseline_model.joblib",
        "description": "Original Phase 3 reference model; useful for comparing predictions.",
        "available_for_scoring": True,
    },
}
MODEL_CARD = {
    "model": "CAMEL Sentinel tuned (HistGradientBoostingClassifier, Phase 4)",
    "predicts": "Probability of failure within 2 years, from 6 CAMEL ratios",
    "training_data": (
        "FDIC BankFind Suite (Institutions + Financials + Failures), a pooled "
        "multi-quarter panel: 265,114 bank-quarters, 2005-2023 (quarterly through "
        "2005-2013 to cover the 2008-2012 crisis, annual afterward), 4,165 "
        "failed-within-2-years positive examples. Replaces Phase 3's single-quarter "
        "baseline (~15 real failures), which was too few examples for any model "
        "comparison to mean anything - see notebooks/03_model_tuning.ipynb."
    ),
    "validation": (
        "Hyperparameters were tuned using ONLY 2005-2011 data, then evaluated on "
        "the untouched 2012-2023 period the model never saw during training or "
        "tuning (a stricter, no-look-ahead check, not just cross-validated folds). "
        "On that true future holdout: ROC-AUC 0.973, PR-AUC 0.549 - versus 0.963 / "
        "0.495 for a RandomForest trained the same way, confirming the improvement "
        "is real and not an artifact of same-era cross-validation."
    ),
    "proxy_label_caveat": (
        "Real CAMELS ratings are confidential; the composite score and the "
        "'failed' label this model was trained on are transparent proxies, "
        "not regulator-assigned ratings."
    ),
    "management_caveat": (
        "The efficiency ratio approximates 'Management' - the weakest of the "
        "six components, since it has no direct financial-statement equivalent."
    ),
    "calibration_caveat": (
        "The model was trained with class_weight='balanced' to counter severe "
        "class imbalance, so predicted probabilities are a *relative risk ranking*, "
        "not calibrated real-world failure odds - a 60% score does not mean a 60% "
        "chance of failure. Worse, the failure rate itself shifted 5x between the "
        "training-era regimes seen (2.1% in 2005-2011 vs 0.43% in 2012-2023): a "
        "model calibrated to one regime over-flags under a calmer one, so the "
        "'risk_tier' cutoffs below (not a flat 0.5) are the actual decision "
        "boundaries, calibrated on the calmer 2012-2023 period, and should be "
        "revisited if the real-world failure rate shifts again (e.g. a new crisis)."
    ),
    "risk_tier_thresholds": (
        "'medium' at probability >= 0.60: on the 2012-2023 holdout this catches "
        "~90% of real failures-within-2-years at ~11% precision - a deliberately "
        "broad early-warning net. 'high' at probability >= 0.98: ~53% recall at "
        "~57% precision - a much smaller, higher-confidence watchlist. Everything "
        "below 0.60 is 'low'. Pick which tier to act on based on whether missing a "
        "real failure (use 'medium') or minimizing false alarms (use 'high') "
        "matters more for the intended use."
    ),
    "intended_use": (
        "Decision-support / early-warning signal - not a replacement for "
        "supervisory judgment."
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI's lifespan hook — code before `yield` runs once at startup,
    # code after would run once at shutdown (nothing needed there here).
    STATE["models"] = {
        key: registry.load_model(filename=details["filename"])
        for key, details in MODEL_CATALOG.items()
    }
    STATE["explainers"] = {
        key: xai_explain.build_explainer(model)
        for key, model in STATE["models"].items()
    }
    panel_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "camel_panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        # Use float("nan") not pd.NA so columns stay float64 dtype — pd.NA
        # would convert them to object, making LIME's np.percentile raise
        # "TypeError: boolean value of NA is ambiguous".
        X_background = panel[DEFAULT_FEATURE_COLS].replace([float("inf"), float("-inf")], float("nan"))
        col_medians = X_background.median(numeric_only=True)
        X_background = X_background.fillna(col_medians).sample(min(5000, len(panel)), random_state=42)
        lime_exp = xai_explain.build_lime_explainer(X_background)
        STATE["lime_explainers"] = {key: lime_exp for key in STATE["models"]}
    else:
        panel = None
        STATE["lime_explainers"] = {key: None for key in STATE["models"]}
    # Keep the default aliases for health checks and any existing callers.
    STATE["model"] = STATE["models"]["tuned"]
    STATE["explainer"] = STATE["explainers"]["tuned"]

    if panel is not None:
        STATE["rai_audit"] = audit_model(STATE["models"]["tuned"], panel)
    else:
        STATE["rai_audit"] = {"available": False, "reason": "Cached training panel is unavailable."}

    institutions, financials, failures, _ = fdic_client.try_live_pull()
    if institutions is None:
        institutions, financials, failures = fdic_client.build_synthetic_sample()
    camel = camel_ratios.score_composite(
        camel_ratios.build_camel_ratios(institutions, financials, failures)
    )
    # Keep only banks with a full set of feature values, so /banks never
    # hands the UI a row that /score would then reject as incomplete.
    STATE["sample_banks"] = camel.dropna(subset=DEFAULT_FEATURE_COLS).reset_index(drop=True)

    print(f"Model loaded. {len(STATE['sample_banks'])} sample banks cached.")
    start_demo_cache_build()  # pre-build GNN demo cache in background (~60 s)
    prewarm_security()        # build Security Lab caches in background so first clicks are fast
    yield
    STATE.clear()


app = FastAPI(title="CAMEL Sentinel API", lifespan=lifespan)
app.include_router(gnn_router)
app.include_router(security_router)
bind_security_state(STATE)

# Security (docs/09 F7): the Streamlit UI calls this API server-side, so browsers never
# need cross-origin access. Only explicitly configured origins are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CAMEL_CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc: RequestValidationError):
    """422 without echoing the submitted values back.

    Echoing input can leak data into logs/clients and crashes on non-JSON values
    such as Infinity (found by the Security Posture self-test).
    """
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg", "invalid"), "type": e.get("type", "")} for e in exc.errors()
    ]})


# Input limits (docs/09 F5) — generous for real use, small enough to stop abuse.
MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_TURNS = 40
MAX_TTS_CHARS = 3000
MAX_AUDIO_BYTES = 10 * 1024 * 1024
RATIO_BOUNDS = {"ge": -1000.0, "le": 1000.0, "allow_inf_nan": False}


class BankRatios(BaseModel):
    """
    Request body for `POST /score` — the six CAMEL ratios for one bank.
    Field order matches `camel_sentinel.models.baseline.DEFAULT_FEATURE_COLS`
    so the model always sees columns in the order it was trained on.
    """

    name: str = Field(default="Custom bank", max_length=120, description="Display name only — not used by the model.")
    tier1_ratio: float = Field(..., **RATIO_BOUNDS, description="Tier 1 risk-based capital ratio (%). Higher = healthier.")
    npl_ratio: float = Field(..., **RATIO_BOUNDS, description="Non-performing loans / total loans (%). Lower = healthier.")
    efficiency_ratio: float = Field(..., **RATIO_BOUNDS, description="Non-interest expense / revenue (%). Lower = healthier.")
    roa: float = Field(..., **RATIO_BOUNDS, description="Return on assets (%). Higher = healthier.")
    loan_deposit_ratio: float = Field(..., **RATIO_BOUNDS, description="Loans / deposits (%). Lower = more liquid = healthier.")
    loan_concentration: float = Field(..., **RATIO_BOUNDS, description="Loans / total assets (%). Lower = less rate-sensitive.")


class ScoreResponse(BaseModel):
    name: str
    model: str
    predicted_failure_probability: float
    risk_tier: str
    base_value: float
    contributions: dict[str, float]
    lime: dict
    counterfactual: dict


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[ChatMessage] = Field(default=[], max_length=MAX_HISTORY_TURNS)
    mode: str = Field(default="text", pattern="^(text|voice)$")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TTS_CHARS)
    voice: str = Field(default="en-US-AriaNeural", pattern=r"^[a-z]{2,3}-[A-Z]{2}-[A-Za-z]+Neural$")


@app.get("/health")
def health() -> dict:
    """Liveness check — also reports whether the model actually loaded."""
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/model-card")
def model_card() -> dict:
    """The caveats every prediction from this service should be read with."""
    return MODEL_CARD


@app.get("/models")
def available_models() -> list[dict]:
    """Return the persisted models that the scorer can actually run."""
    return [
        {"key": key, **{field: value for field, value in details.items() if field != "filename"}}
        for key, details in MODEL_CATALOG.items()
    ]


@app.get("/rai-audit")
def rai_audit() -> dict:
    """Return the cached subgroup performance and disparity audit."""
    if "rai_audit" not in STATE:
        raise HTTPException(status_code=503, detail="RAI audit is not ready yet.")
    return STATE["rai_audit"]


@app.get("/agent/tools")
def agent_tools() -> list[dict]:
    """Return the explicit tools available to an agent client."""
    return tool_catalog()


@app.get("/banks")
def list_banks(limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    """
    A sample of real (or synthetic-fallback) banks with their CAMEL ratios
    already computed — lets the UI offer "pick a real bank" instead of
    only manual ratio entry.

    Parameters
    ----------
    limit : int, default 100
        Max number of banks to return, taken from the front of the cached
        sample (not randomly sampled — fine for a demo UI; revisit if a
        representative sample matters later).
    """
    cols = ["CERT", "NAME", "STALP", *DEFAULT_FEATURE_COLS, "composite_score_0_100", "proxy_rating_1_5", "failed"]
    return STATE["sample_banks"][cols].head(limit).to_dict(orient="records")


@app.post("/score", response_model=ScoreResponse)
def score(
    bank: BankRatios,
    model: str = Query("tuned", description="Persisted model key returned by GET /models."),
) -> ScoreResponse:
    """Predict failure risk and explain it with the selected persisted model."""
    if "models" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if model not in MODEL_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Use GET /models.")

    row = pd.DataFrame([bank.model_dump()])[DEFAULT_FEATURE_COLS]
    probability = float(STATE["models"][model].predict_proba(row)[0][1])
    if probability < 0.60:
        risk_tier = "low"
    elif probability < 0.98:
        risk_tier = "medium"
    else:
        risk_tier = "high"

    explanation = xai_explain.explain_instance(STATE["explainers"][model], row)
    lime = xai_explain.explain_with_lime(STATE["lime_explainers"][model], STATE["models"][model], row)
    counterfactual = find_counterfactual(STATE["models"][model], row)

    return ScoreResponse(
        name=bank.name,
        model=MODEL_CATALOG[model]["label"],
        predicted_failure_probability=probability,
        risk_tier=risk_tier,
        base_value=explanation["base_value"],
        contributions=explanation["contributions"],
        lime=lime,
        counterfactual=counterfactual,
    )


@app.post("/agent/score")
def agent_score(
    bank: BankRatios,
    model: str = Query("tuned", description="Persisted model key returned by GET /models."),
) -> dict:
    """Invoke the allow-listed score_bank tool through an agent-shaped API."""
    if "models" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if model not in MODEL_CATALOG:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Use GET /models.")

    ratios = bank.model_dump()
    ratios.pop("name", None)
    result = dispatch_tool(
        "score_bank",
        model=STATE["models"][model],
        explainer=STATE["explainers"][model],
        lime_explainer=STATE["lime_explainers"][model],
        ratios=ratios,
    )
    return {"tool": "score_bank", "model": MODEL_CATALOG[model]["label"], "name": bank.name, **result}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """
    One conversational turn with the CAMEL Sentinel GPT agent.

    The agent may call score_bank internally when the user asks about a
    specific bank's risk; the raw ML result is returned alongside the
    natural-language reply so the frontend can optionally render the SHAP
    chart inline (Slice B).

    Requires AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in the environment (see .env.example).
    Voice mode ("mode": "voice") is accepted here; TTS synthesis is a
    separate POST /speech/synthesize call added in Slice D.
    """
    if "models" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    # Security (docs/09 F4): obvious injection attempts are refused before the LLM is
    # called (saves cost, and the Security Lab's defence now protects the real app),
    # and PII is redacted so it never reaches the model provider or its logs.
    detection = detect_injection(request.message)
    if detection["verdict"] == "blocked":
        return {
            "reply": ("I can't follow instructions that try to change how I work. I only explain "
                      "scores from the CAMEL model — share a bank's six ratios and I'll score it."),
            "tool_called": False, "score_result": None, "mode": request.mode,
            "guardrail": {"blocked": True, "rules": [m["rule"] for m in detection["matches"]]},
        }
    if not os.environ.get("AZURE_OPENAI_KEY") or not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        raise HTTPException(
            status_code=503,
            detail="AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT are not configured. See .env.example.",
        )

    try:
        history = ConversationHistory([m.model_dump() for m in request.history])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    redaction = redact_pii(request.message)
    safe_message = redaction["redacted_text"]
    safe_history = [{**m, "content": redact_pii(m["content"])["redacted_text"]} for m in history.to_list()]

    agent = ChatAgent(
        model=STATE["models"]["tuned"],
        explainer=STATE["explainers"]["tuned"],
        lime_explainer=STATE["lime_explainers"]["tuned"],
    )

    try:
        result = agent.run(safe_message, safe_history)
    except openai.OpenAIError as exc:
        # Security (docs/09 F8): log details server-side, return a generic message.
        print(f"[chat] upstream LLM error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="The language model service is unavailable. Please try again.")

    return jsonable_encoder(
        {
            "reply": result["reply"],
            "tool_called": result["tool_called"],
            "score_result": result.get("score_result"),
            "mode": request.mode,
            "guardrail": {"blocked": False, "pii_redacted": redaction["counts"]},
        }
    )


@app.post("/speech/transcribe")
async def speech_transcribe(audio: UploadFile = File(...)) -> dict:
    """
    Convert uploaded WAV audio to text using Azure Speech-to-Text.

    Expects multipart/form-data with an 'audio' field containing WAV bytes
    (st.audio_input in Streamlit 1.37+ records 16 kHz mono WAV).
    Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (see .env.example).

    Returns {"text": "transcribed text"} — empty string if nothing heard.
    TTS synthesis is a separate POST /speech/synthesize added in Slice D.
    """
    if not os.environ.get("AZURE_SPEECH_KEY") or not os.environ.get("AZURE_SPEECH_REGION"):
        raise HTTPException(
            status_code=503,
            detail="AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are not configured. See .env.example.",
        )

    audio_bytes = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 10 MB).")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, transcribe_audio, audio_bytes)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"text": text}


@app.post("/speech/synthesize")
async def speech_synthesize(request: SynthesizeRequest) -> Response:
    """
    Convert text to WAV audio bytes using Azure Text-to-Speech.

    Request body: {"text": "...", "voice": "en-US-AriaNeural"}
    Response: audio/wav bytes.

    Called by the frontend after /chat returns in voice mode — the reply
    text is synthesised here and played back with st.audio(autoplay=True).
    Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (see .env.example).
    """
    if not os.environ.get("AZURE_SPEECH_KEY") or not os.environ.get("AZURE_SPEECH_REGION"):
        raise HTTPException(
            status_code=503,
            detail="AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are not configured. See .env.example.",
        )

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    loop = asyncio.get_event_loop()
    try:
        audio_bytes = await loop.run_in_executor(
            None, synthesize_text, request.text, request.voice
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return Response(content=audio_bytes, media_type="audio/wav")
