"""
chat/agent.py — ChatAgent: one conversational turn with tool-calling.

WHAT this does: wraps the OpenAI client, defines score_bank in OpenAI
function-calling format, runs a two-step exchange (first call may return
a tool_call; if it does, the tool result is fed back for a second call
that produces the final natural-language answer).

WHY two calls instead of one: OpenAI function calling works as
request → tool_call decision → tool execution → request again with result
→ final text. The LLM never sees the raw numpy floats from the ML model;
it sees a JSON string of those values, which it then rephrases.

WHY the tool schema mirrors DEFAULT_FEATURE_COLS exactly: the LLM must
pass the same six column names that the trained model was fit on. A name
mismatch would silently produce a wrong prediction (the ML layer would
fill the missing column with median, shifting the score). The schema
names here are the ground truth; do not rename them.
"""

from __future__ import annotations

import json
import os
from typing import Any

import openai

from camel_sentinel.agent.tools import TOOL_HANDLERS, score_bank_tool
from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS
from camel_sentinel.security.tool_guard import validate_score_args
from camel_sentinel.chat.prompts import SYSTEM_PROMPT

# OpenAI function schema — names match DEFAULT_FEATURE_COLS in
# camel_sentinel.models.baseline exactly.
_TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "score_bank",
            "description": (
                "Score one bank's failure risk from its six CAMEL ratios. "
                "Returns the predicted probability, risk tier (low/medium/high), "
                "and the SHAP-driven contribution of each ratio to that score. "
                "Call this whenever the user asks about a specific bank's risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tier1_ratio": {
                        "type": "number",
                        "description": "Tier 1 risk-based capital ratio (%). Higher is healthier.",
                    },
                    "npl_ratio": {
                        "type": "number",
                        "description": "Non-performing loans / total loans (%). Lower is healthier.",
                    },
                    "efficiency_ratio": {
                        "type": "number",
                        "description": "Non-interest expense / revenue (%). Lower is healthier.",
                    },
                    "roa": {
                        "type": "number",
                        "description": "Return on assets (%). Higher is healthier.",
                    },
                    "loan_deposit_ratio": {
                        "type": "number",
                        "description": "Loans / deposits (%). Lower means more liquid, healthier.",
                    },
                    "loan_concentration": {
                        "type": "number",
                        "description": "Loans / total assets (%). Lower means less rate-sensitive.",
                    },
                },
                "required": [
                    "tier1_ratio",
                    "npl_ratio",
                    "efficiency_ratio",
                    "roa",
                    "loan_deposit_ratio",
                    "loan_concentration",
                ],
                "additionalProperties": False,
            },
        },
    }
]


def _to_json(obj: Any) -> str:
    """Serialize a score_bank result, converting numpy scalars to Python types."""
    def _default(o: Any) -> Any:
        try:
            import numpy as np
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return str(o)

    return json.dumps(obj, default=_default)


class ChatAgent:
    """
    Run one conversational turn.

    Parameters
    ----------
    model, explainer, lime_explainer
        The already-loaded ML model objects from backend STATE — passed in
        so the agent never reloads them and the expensive SHAP explainer
        construction only happens once at backend startup.
    """

    def __init__(self, model: Any, explainer: Any, lime_explainer: Any) -> None:
        self._ml_model = model
        self._explainer = explainer
        self._lime_explainer = lime_explainer

        api_key = os.environ.get("AZURE_OPENAI_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set. "
                "See .env.example."
            )
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-08-07-preview"),
        )
        self._openai_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

    def run(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Process one user message and return the agent's reply.

        Returns
        -------
        dict
            reply        : str        — natural-language answer
            tool_called  : bool       — True if score_bank was invoked
            score_result : dict|None  — raw ML output if tool was called
        """
        messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        first = self._client.chat.completions.create(
            model=self._openai_model,
            messages=messages,
            tools=_TOOL_SCHEMA,
            tool_choice="auto",
        )
        choice = first.choices[0]

        if choice.finish_reason != "tool_calls":
            return {
                "reply": choice.message.content or "",
                "tool_called": False,
                "score_result": None,
            }

        tool_call = choice.message.tool_calls[0]
        # Security (docs/09 F2): the model's tool request is untrusted output. Only the
        # allow-listed tool, with valid JSON arguments inside the ratio limits, may run.
        # Anything else fails closed with a safe reply instead of an exception.
        refusal = {
            "reply": ("I couldn't run the scoring model with that request — I need the six CAMEL "
                      "ratios as plain numbers within their normal ranges."),
            "tool_called": False,
            "score_result": None,
        }
        if tool_call.function.name not in TOOL_HANDLERS:
            return refusal
        try:
            ratios = json.loads(tool_call.function.arguments)
        except (TypeError, ValueError):
            return refusal
        if not isinstance(ratios, dict) or validate_score_args(ratios):
            return refusal
        ratios = {k: float(v) for k, v in ratios.items() if k in DEFAULT_FEATURE_COLS}

        score_result = score_bank_tool(
            model=self._ml_model,
            explainer=self._explainer,
            lime_explainer=self._lime_explainer,
            ratios=ratios,
        )

        # Re-build the assistant turn as a plain dict so the second request
        # is not tied to the OpenAI SDK's internal Pydantic types.
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": _to_json(score_result),
            }
        )

        second = self._client.chat.completions.create(
            model=self._openai_model,
            messages=messages,
        )

        return {
            "reply": second.choices[0].message.content or "",
            "tool_called": True,
            "score_result": score_result,
        }
