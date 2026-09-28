"""STRIDE threat model and MITRE ATLAS mapping for CAMEL Sentinel.

STRIDE (Microsoft SDL) asks six questions of every component: can someone
Spoof identity, Tamper with data, Repudiate actions, cause Information
disclosure, Deny service, or Elevate privilege? MITRE ATLAS is the
ATT&CK-style knowledge base of real adversary techniques against AI systems.
This module states both for this project, so the Security Lab page can render
them and every lab can point back to the threat it demonstrates.
"""

from __future__ import annotations

from typing import Any

STRIDE_CATEGORIES = {
    "S": {"name": "Spoofing", "question": "Can someone pretend to be a user, a bank, or a system component?", "property": "Authentication"},
    "T": {"name": "Tampering", "question": "Can someone change data, models or prompts they shouldn't?", "property": "Integrity"},
    "R": {"name": "Repudiation", "question": "Can someone act and later deny it because nothing was logged?", "property": "Non-repudiation"},
    "I": {"name": "Information disclosure", "question": "Can someone read data, prompts or model internals they shouldn't?", "property": "Confidentiality"},
    "D": {"name": "Denial of service", "question": "Can someone make the system unavailable or too slow?", "property": "Availability"},
    "E": {"name": "Elevation of privilege", "question": "Can someone make the system do more than their role allows?", "property": "Authorization"},
}

COMPONENTS: list[dict[str, Any]] = [
    {"id": "user", "name": "Analyst / browser", "zone": "untrusted", "desc": "A person using the Streamlit UI or voice mode."},
    {"id": "ui", "name": "Streamlit UI", "zone": "app", "desc": "frontend/app.py + pages — talks to the API over HTTP only."},
    {"id": "api", "name": "FastAPI backend", "zone": "app", "desc": "backend/main.py — /score, /chat, /speech/*, /gnn/*, /security/*."},
    {"id": "llm", "name": "LLM (Azure OpenAI)", "zone": "external", "desc": "gpt-5-mini — rephrases model output, may call score_bank."},
    {"id": "tools", "name": "Agent tools", "zone": "app", "desc": "src/camel_sentinel/agent/tools.py — allow-listed score_bank only."},
    {"id": "model", "name": "CAMEL model + XAI", "zone": "app", "desc": "Tuned HistGradientBoosting + SHAP/LIME/counterfactual."},
    {"id": "data", "name": "Training data feed", "zone": "external", "desc": "FDIC BankFind API → camel_panel.parquet."},
    {"id": "container", "name": "Container image + registry", "zone": "supply-chain", "desc": "Dockerfile → Docker Hub / ACR → cloud runtime."},
]

# (component, STRIDE letter, threat, mitigation in THIS project, lab key or None)
STRIDE_THREATS: list[dict[str, Any]] = [
    {"component": "user", "stride": "S", "threat": "Deepfake voice of an executive requests an urgent action via voice mode", "mitigation": "Voice is never an authorisation factor; out-of-band callback for any action", "lab": "deepfake"},
    {"component": "user", "stride": "R", "threat": "User denies having submitted manipulated ratios", "mitigation": "Structured request logs with API key, timestamp and payload hash", "lab": "soc"},
    {"component": "ui", "stride": "I", "threat": "Customer PII pasted into chat is sent to the LLM provider and logs", "mitigation": "Server-side PII redaction before any prompt leaves the API", "lab": "pii"},
    {"component": "api", "stride": "I", "threat": "Model extraction by scripting thousands of /score calls", "mitigation": "Per-key rate limits, coarse outputs for untrusted callers, query-pattern detection", "lab": "stealing"},
    {"component": "api", "stride": "D", "threat": "Flood of expensive /score (SHAP+LIME) or /speech calls", "mitigation": "Rate limiting, request size limits, autoscaling with max replicas", "lab": "soc"},
    {"component": "api", "stride": "S", "threat": "Stolen API key used from a new location", "mitigation": "Keys in a secrets manager, rotation, anomaly alerts on key usage", "lab": "soc"},
    {"component": "llm", "stride": "T", "threat": "Direct prompt injection makes the assistant invent a 'safe' rating", "mitigation": "Numbers only from score_bank; injection detector; output check", "lab": "prompt_injection"},
    {"component": "llm", "stride": "T", "threat": "Indirect injection hidden in pasted analyst notes or retrieved text", "mitigation": "Spotlighting (delimit + mark untrusted data), detector on data channel", "lab": "indirect_injection"},
    {"component": "llm", "stride": "I", "threat": "System prompt extraction ('repeat your instructions')", "mitigation": "No secrets in prompts; canary token + output filter", "lab": "prompt_injection"},
    {"component": "tools", "stride": "E", "threat": "Injected text makes the agent call an unlisted, side-effecting tool", "mitigation": "Allow-list (fail closed), argument schema + range validation, no write tools", "lab": "tool_hijack"},
    {"component": "model", "stride": "T", "threat": "Adversarial 'window dressing' — small ratio edits flip high risk to low", "mitigation": "Plausibility and quarter-over-quarter consistency checks; human review", "lab": "adversarial"},
    {"component": "model", "stride": "I", "threat": "Explanations (SHAP/counterfactual) leak decision boundary faster", "mitigation": "Full explanations only for authenticated analysts", "lab": "stealing"},
    {"component": "data", "stride": "T", "threat": "Poisoned or mislabelled rows in the training feed", "mitigation": "Schema/range validation, outlier filter, label-consistency check, data versioning", "lab": "poisoning"},
    {"component": "container", "stride": "I", "threat": "API keys baked into a public image", "mitigation": ".dockerignore excludes .env; secrets injected at run time", "lab": "supply_chain"},
    {"component": "container", "stride": "E", "threat": "Container runs as root; compromise = root in container", "mitigation": "Non-root USER, runAsNonRoot, no privilege escalation", "lab": "supply_chain"},
    {"component": "container", "stride": "T", "threat": "Unpinned / malicious dependency or pickled model swapped in", "mitigation": "Pinned versions, Trivy scan, trusted registry, image digests", "lab": "supply_chain"},
]

# MITRE ATLAS techniques demonstrated in the lab (IDs from atlas.mitre.org).
ATLAS_TECHNIQUES: list[dict[str, Any]] = [
    {"id": "AML.T0051.000", "name": "LLM Prompt Injection: Direct", "tactic": "Initial Access / Execution", "lab": "prompt_injection"},
    {"id": "AML.T0051.001", "name": "LLM Prompt Injection: Indirect", "tactic": "Initial Access / Execution", "lab": "indirect_injection"},
    {"id": "AML.T0054", "name": "LLM Jailbreak", "tactic": "Privilege Escalation / Defense Evasion", "lab": "prompt_injection"},
    {"id": "AML.T0056", "name": "LLM Meta Prompt Extraction", "tactic": "Discovery", "lab": "prompt_injection"},
    {"id": "AML.T0057", "name": "LLM Data Leakage", "tactic": "Exfiltration", "lab": "pii"},
    {"id": "AML.T0053", "name": "LLM Plugin Compromise", "tactic": "Execution / Privilege Escalation", "lab": "tool_hijack"},
    {"id": "AML.T0043", "name": "Craft Adversarial Data", "tactic": "ML Attack Staging", "lab": "adversarial"},
    {"id": "AML.T0015", "name": "Evade ML Model", "tactic": "Defense Evasion / Impact", "lab": "adversarial"},
    {"id": "AML.T0020", "name": "Poison Training Data", "tactic": "Resource Development / Persistence", "lab": "poisoning"},
    {"id": "AML.T0040", "name": "ML Model Inference API Access", "tactic": "ML Model Access", "lab": "stealing"},
    {"id": "AML.T0024.002", "name": "Exfiltration via ML Inference API: Extract ML Model", "tactic": "Exfiltration", "lab": "stealing"},
    {"id": "AML.T0029", "name": "Denial of ML Service", "tactic": "Impact", "lab": "soc"},
    {"id": "AML.T0052", "name": "Phishing (incl. deepfake-enabled social engineering)", "tactic": "Initial Access", "lab": "deepfake"},
    {"id": "AML.T0010", "name": "ML Supply Chain Compromise", "tactic": "Initial Access", "lab": "supply_chain"},
]


def threat_model() -> dict[str, Any]:
    """Everything the Threat Model tab renders, in one payload."""
    return {
        "stride_categories": STRIDE_CATEGORIES,
        "components": COMPONENTS,
        "stride": STRIDE_THREATS,
        "atlas": ATLAS_TECHNIQUES,
    }


def atlas_for(lab: str) -> list[dict[str, Any]]:
    """ATLAS techniques demonstrated by one lab (used to badge each lab section)."""
    return [t for t in ATLAS_TECHNIQUES if t["lab"] == lab]
