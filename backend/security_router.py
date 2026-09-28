"""
security_router.py — FastAPI router for the AI Security Lab.

Every lab attacks and defends CAMEL Sentinel itself. Design and endpoint
contract: docs/07_ai_security_lab.md. Nothing here calls an external service;
labs marked synthetic generate their own data.

GET  /security/threat-model       STRIDE + MITRE ATLAS for this system
POST /security/prompt-injection   detector + simulated vulnerable vs defended assistant
POST /security/pii                PII detection + redaction
POST /security/tool-call          naive vs allow-listed tool dispatch
POST /security/adversarial        evasion against the real tuned model + plausibility defence
POST /security/poisoning          clean vs poisoned vs defended retraining
POST /security/model-stealing     extraction attack fidelity by output mode
POST /security/deepfake           synthetic voice signal + detector
POST /security/soc                synthetic API log triage
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

SRC_PATH = str((Path(__file__).resolve().parents[1] / "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from camel_sentinel.security import (  # noqa: E402
    adversarial,
    deepfake,
    model_stealing,
    pii,
    poisoning,
    prompt_guard,
    soc,
    threat_model,
    tool_guard,
)

router = APIRouter(prefix="/security", tags=["AI Security Lab"])

PANEL_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "camel_panel.parquet"
_STATE: dict[str, Any] = {}


def bind_state(state: dict[str, Any]) -> None:
    """main.py hands over its STATE dict so labs reuse the already-loaded model."""
    _STATE["app"] = state


def _victim_model() -> Any:
    app_state = _STATE.get("app") or {}
    model = (app_state.get("models") or {}).get("tuned")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return model


@lru_cache(maxsize=1)
def _panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Training panel not available for this lab.")
    return pd.read_parquet(PANEL_PATH)


@lru_cache(maxsize=1)
def _plausibility_model() -> dict[str, Any]:
    return adversarial.build_plausibility_model(_panel())


@lru_cache(maxsize=1)
def _poison_sample() -> pd.DataFrame:
    return poisoning.prepare_sample(_panel())


def prewarm() -> None:
    """Build the slow caches in a background thread at startup (called from main.py)."""
    import threading

    def _work() -> None:
        try:
            _plausibility_model()
            _poison_sample()
            deepfake.detector()
            _soc(5)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[security] prewarm skipped: {exc}")

    threading.Thread(target=_work, daemon=True).start()


# ── request models ────────────────────────────────────────────────────────────

class InjectionRequest(BaseModel):
    text: str = Field(..., max_length=4000)
    untrusted_data: str = Field(default="", max_length=8000)


class TextRequest(BaseModel):
    text: str = Field(..., max_length=8000)


class ToolCallRequest(BaseModel):
    text: str = Field(default="", max_length=4000)
    tool_name: str = ""
    arguments: dict[str, Any] = {}


class Ratios(BaseModel):
    tier1_ratio: float
    npl_ratio: float
    efficiency_ratio: float
    roa: float
    loan_deposit_ratio: float
    loan_concentration: float


class AdversarialRequest(Ratios):
    max_steps: int = Field(default=80, ge=5, le=200)
    step_frac: float = Field(default=0.02, gt=0.0, le=0.2)


class PoisoningRequest(BaseModel):
    attack: Literal["label_flip", "injection"] = "label_flip"
    rate: float = Field(default=0.3, ge=0.0, le=0.6)


class StealingRequest(BaseModel):
    output_mode: Literal["probability", "rounded", "tier_only"] = "probability"
    rate_limit_per_hour: int = Field(default=100, ge=1, le=100000)


class DeepfakeRequest(BaseModel):
    kind: Literal["human", "clone_basic", "clone_advanced"] = "human"
    seed: int = Field(default=42, ge=0, le=10_000)


class SocRequest(BaseModel):
    seed: int = Field(default=5, ge=0, le=10_000)


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/threat-model")
def get_threat_model() -> dict:
    return threat_model.threat_model()


@router.post("/prompt-injection")
def prompt_injection(req: InjectionRequest) -> dict:
    return {**prompt_guard.run_prompt_injection_lab(req.text, req.untrusted_data),
            "atlas": threat_model.atlas_for("prompt_injection") + threat_model.atlas_for("indirect_injection")}


@router.post("/pii")
def pii_redaction(req: TextRequest) -> dict:
    return {**pii.redact(req.text), "atlas": threat_model.atlas_for("pii")}


@router.post("/tool-call")
def tool_call(req: ToolCallRequest) -> dict:
    result = tool_guard.run_tool_hijack_lab(req.text, req.tool_name, req.arguments)
    guarded = result.get("guarded")
    if guarded and guarded["allowed"]:
        # Allowed calls go through the project's real dispatcher to show the happy path.
        from camel_sentinel.agent.tools import dispatch_tool
        state = _STATE.get("app") or {}
        args = {k: float(v) for k, v in result["request"]["arguments"].items() if k != "model"}
        out = dispatch_tool("score_bank", model=_victim_model(), explainer=state["explainers"]["tuned"],
                            lime_explainer=None, ratios=args)
        guarded["result"] = {"predicted_failure_probability": out["predicted_failure_probability"],
                             "risk_tier": out["risk_tier"]}
    return {**result, "atlas": threat_model.atlas_for("tool_hijack")}


@router.post("/adversarial")
def adversarial_evasion(req: AdversarialRequest) -> dict:
    ratios = req.model_dump(exclude={"max_steps", "step_frac"})
    attack = adversarial.evade(_victim_model(), ratios, step_frac=req.step_frac, max_steps=req.max_steps)
    check_attack = adversarial.plausibility_check(_plausibility_model(), ratios, attack["adversarial_ratios"])
    check_honest = adversarial.plausibility_check(_plausibility_model(), ratios, ratios)
    return {**attack, "plausibility": check_attack, "plausibility_if_unchanged": check_honest,
            "atlas": threat_model.atlas_for("adversarial")}


@lru_cache(maxsize=32)
def _poisoning(attack: str, rate: float) -> dict:
    return poisoning.run_poisoning_lab(_poison_sample(), attack, rate)


@router.post("/poisoning")
def data_poisoning(req: PoisoningRequest) -> dict:
    # Rate rounded to 0.05 so results are cacheable: retraining on every click would let
    # anyone with API access burn CPU (docs/09 F10).
    return {**_poisoning(req.attack, round(round(req.rate / 0.05) * 0.05, 2)),
            "atlas": threat_model.atlas_for("poisoning")}


@lru_cache(maxsize=8)
def _stealing(mode: str, rate_limit: int) -> dict:
    return model_stealing.run_model_stealing_lab(_victim_model(), _panel(), mode, rate_limit_per_hour=rate_limit)


@router.post("/model-stealing")
def model_extraction(req: StealingRequest) -> dict:
    return {**_stealing(req.output_mode, req.rate_limit_per_hour), "atlas": threat_model.atlas_for("stealing")}


@router.post("/deepfake")
def deepfake_voice(req: DeepfakeRequest) -> dict:
    return {**deepfake.run_deepfake_lab(req.kind, req.seed), "atlas": threat_model.atlas_for("deepfake")}


@lru_cache(maxsize=8)
def _soc(seed: int) -> dict:
    return soc.run_soc_lab(seed)


@router.post("/soc")
def soc_triage(req: SocRequest) -> dict:
    return _soc(req.seed)
