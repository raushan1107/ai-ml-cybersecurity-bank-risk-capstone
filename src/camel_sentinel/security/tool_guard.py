"""Tool-call hijacking: a naive agent dispatcher vs the project's allow-listed one.

When an LLM can call tools, a prompt injection becomes an *action*, not just
bad text. The naive dispatcher below looks functions up by whatever name the
model produced and passes arguments through unchecked. The guarded path uses
the real allow-list in `camel_sentinel.agent.tools` plus argument validation.

Nothing here performs a real side effect: "dangerous" tools only describe what
they *would* have done.
"""

from __future__ import annotations

import re
from typing import Any

from camel_sentinel.agent.tools import TOOL_HANDLERS
from camel_sentinel.models.baseline import DEFAULT_FEATURE_COLS
from camel_sentinel.xai.counterfactual import FEATURE_LIMITS

# What a careless developer might register "for convenience". Descriptions only.
NAIVE_REGISTRY: dict[str, str] = {
    "score_bank": "Scored the bank with the CAMEL model.",
    "export_all_customers": "Exported 12,480 customer records (names, accounts, balances) to the caller.",
    "delete_records": "Deleted the requested records from the risk database.",
    "send_email": "Sent an email with the conversation contents to the given address.",
    "run_shell": "Ran an operating-system command on the API server.",
    "update_rating": "Overwrote the stored risk rating for the bank.",
}

TOOL_CALL = re.compile(r"\b(?:call|invoke|run|execute|use)\s+(?:the\s+)?(?:tool\s+)?`?([a-z_][a-z0-9_]*)`?\s*(?:\((.*?)\))?", re.I)


def parse_tool_request(text: str) -> dict[str, Any] | None:
    """What a naive agent would 'decide' after reading injected text."""
    for m in TOOL_CALL.finditer(text or ""):
        name = m.group(1).lower()
        if name in NAIVE_REGISTRY or "_" in name:
            args: dict[str, Any] = {}
            for part in (m.group(2) or "").split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    args[k.strip()] = v.strip().strip("'\"")
            return {"tool_name": name, "arguments": args}
    return None


def naive_dispatch(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Anything the model asks for, it gets."""
    effect = NAIVE_REGISTRY.get(tool_name, f"Dynamically resolved and ran '{tool_name}' via getattr().")
    return {"executed": True, "tool_name": tool_name, "arguments": arguments, "effect": effect,
            "dangerous": tool_name != "score_bank"}


def validate_score_args(arguments: dict[str, Any]) -> list[str]:
    """Schema check for score_bank: exactly the six ratios, numeric, inside limits."""
    problems = []
    extra = set(arguments) - set(DEFAULT_FEATURE_COLS) - {"model"}
    missing = [c for c in DEFAULT_FEATURE_COLS if c not in arguments]
    if extra:
        problems.append(f"Unexpected argument(s): {', '.join(sorted(extra))}")
    if missing:
        problems.append(f"Missing ratio(s): {', '.join(missing)}")
    for col in DEFAULT_FEATURE_COLS:
        if col not in arguments:
            continue
        try:
            value = float(arguments[col])
        except (TypeError, ValueError):
            problems.append(f"{col} is not a number: {arguments[col]!r}")
            continue
        low, high = FEATURE_LIMITS[col]
        if not low <= value <= high:
            problems.append(f"{col}={value} outside allowed range [{low}, {high}]")
    return problems


def guarded_dispatch(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The project's rule: allow-list first (fail closed), then validate arguments."""
    checks = []
    if tool_name not in TOOL_HANDLERS:
        checks.append({"check": "allow-list", "passed": False,
                       "detail": f"'{tool_name}' is not in the allow-list {sorted(TOOL_HANDLERS)}"})
        return {"allowed": False, "reason": checks[-1]["detail"], "checks": checks}
    checks.append({"check": "allow-list", "passed": True, "detail": f"'{tool_name}' is allow-listed"})
    problems = validate_score_args(arguments)
    checks.append({"check": "argument schema + ranges", "passed": not problems,
                   "detail": "; ".join(problems) if problems else "six numeric ratios within limits"})
    checks.append({"check": "side effects", "passed": True, "detail": "score_bank is read-only"})
    return {"allowed": not problems, "reason": "; ".join(problems) or "Allowed", "checks": checks}


def run_tool_hijack_lab(text: str = "", tool_name: str = "", arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the tool request (from injected text or explicit), then run both dispatchers."""
    request = {"tool_name": tool_name, "arguments": arguments or {}} if tool_name else parse_tool_request(text)
    if request is None:
        return {"request": None, "naive": None, "guarded": None,
                "note": "No tool call found in the text — the agent would just reply."}
    return {"request": request,
            "naive": naive_dispatch(request["tool_name"], request["arguments"]),
            "guarded": guarded_dispatch(request["tool_name"], request["arguments"])}
