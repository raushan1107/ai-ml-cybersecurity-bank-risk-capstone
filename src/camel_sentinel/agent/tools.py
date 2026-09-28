"""Explicit, allow-listed tools for an agent or another API client."""

from __future__ import annotations

from typing import Any, Callable

from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS
from camel_sentinel.xai import explain as xai_explain
from camel_sentinel.xai.counterfactual import find_counterfactual


def tool_catalog() -> list[dict[str, Any]]:
    """Describe the tools an agent is allowed to call."""
    return [
        {
            "name": "score_bank",
            "description": "Score one bank from six CAMEL ratios and return explanations.",
            "input_fields": [*DEFAULT_FEATURE_COLS, "model"],
            "side_effects": "none",
        }
    ]


def score_bank_tool(
    model: Any,
    explainer: Any,
    lime_explainer: Any,
    ratios: dict[str, float],
) -> dict[str, Any]:
    """Run the same prediction pipeline as the UI, as an explicit tool."""
    import pandas as pd

    row = pd.DataFrame([ratios])[DEFAULT_FEATURE_COLS]
    probability = float(model.predict_proba(row)[0][1])
    tier = "low" if probability < 0.60 else "medium" if probability < 0.98 else "high"
    shap_result = xai_explain.explain_instance(explainer, row)
    return {
        "predicted_failure_probability": probability,
        "risk_tier": tier,
        "shap": shap_result,
        "lime": xai_explain.explain_with_lime(lime_explainer, model, row),
        "counterfactual": find_counterfactual(model, row),
    }


TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {"score_bank": score_bank_tool}


def dispatch_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch only an allow-listed tool; unknown calls fail closed."""
    if name not in TOOL_HANDLERS:
        raise ValueError(f"Tool '{name}' is not allow-listed")
    return TOOL_HANDLERS[name](**kwargs)
