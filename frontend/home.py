"""
frontend/home.py (was app.py) — the UI: pick a bank (or type in ratios), see the
prediction and why the model made it.

WHAT this file does: a Streamlit page that calls the FastAPI backend
(`backend/main.py`) over plain HTTP — it never imports `camel_sentinel`
or loads the model itself. That separation is deliberate (see WHY below).

WHY the UI only talks HTTP to the backend: this is the same shape a
future agent tool-call (Phase 8) or any other caller would use — the UI
isn't special-cased with direct model access, so "does the API work" and
"does the UI work" are two separately testable questions, and the backend
can be swapped, redeployed, or scaled without touching this file.

HOW to run it: `streamlit run frontend/app.py` from the project root (the
backend must already be running — see README.md). Opens at
http://localhost:8501 by default.
"""

from __future__ import annotations

import os
import math

import matplotlib
matplotlib.use("Agg")  # force non-interactive backend before any other matplotlib import
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from ui_kit import backend_url_setting

st.set_page_config(page_title="CAMEL Sentinel", page_icon="\U0001F42B", layout="centered")

FEATURE_FIELDS = [
    ("tier1_ratio", "Tier 1 capital ratio (%)", 3.5, 50.0, 14.0, "Higher = more capital cushion = healthier."),
    ("npl_ratio", "Non-performing loans ratio (%)", 0.0, 15.0, 1.0, "Lower = healthier."),
    ("efficiency_ratio", "Efficiency ratio (%)", 30.0, 110.0, 62.0, "Non-interest expense / revenue. Lower = healthier."),
    ("roa", "Return on assets (%)", -2.0, 4.0, 1.0, "Higher = healthier."),
    ("loan_deposit_ratio", "Loan / deposit ratio (%)", 20.0, 150.0, 85.0, "Lower = more liquid = healthier."),
    ("loan_concentration", "Loan concentration (%)", 0.0, 100.0, 60.0, "Share of assets in loans. Lower = less rate-sensitive."),
]

RISK_COLOR = {"low": "#2E7D5C", "medium": "#B4741F", "high": "#B23A2E"}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_banks(backend_url: str, limit: int = 150) -> list[dict]:
    """GET /banks — cached for 60s so switching the dropdown doesn't re-hit the API every rerun."""
    resp = requests.get(f"{backend_url}/banks", params={"limit": limit}, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_card(backend_url: str) -> dict:
    """GET /model-card — cached longer since this never changes while the backend is running."""
    resp = requests.get(f"{backend_url}/model-card", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models(backend_url: str) -> list[dict]:
    """GET /models — only models with persisted artifacts are offered for live scoring."""
    resp = requests.get(f"{backend_url}/models", timeout=15)
    resp.raise_for_status()
    return resp.json()


def score(backend_url: str, ratios: dict, model: str = "tuned") -> dict:
    """POST /score — runs the selected model plus its SHAP explanation."""
    resp = requests.post(f"{backend_url}/score", params={"model": model}, json=ratios, timeout=15)
    resp.raise_for_status()
    return resp.json()


def render_contributions_chart(contributions: dict[str, float]) -> None:
    """Horizontal bar chart of SHAP contributions, sorted by magnitude, red = pushes risk up / green = pushes it down."""
    series = pd.Series(contributions).sort_values(key=abs)
    colors = [RISK_COLOR["high"] if v > 0 else RISK_COLOR["low"] for v in series]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(series.index, series.values, color=colors)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to failure-risk score")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


st.title("\U0001F42B CAMEL Sentinel")
st.caption("Pick a bank or enter ratios, score it, and see exactly which ratios drove the prediction.")

with st.sidebar:
    st.header("Settings")
    backend_url = backend_url_setting()
    try:
        health = requests.get(f"{backend_url}/health", timeout=5).json()
        st.success("Backend connected" if health.get("model_loaded") else "Backend up, model not loaded")
    except requests.RequestException:
        st.error("Can't reach the backend — start it with:\n\npython -m uvicorn main:app --app-dir backend --port 8000")
        st.stop()

    st.divider()
    mode = st.radio("Bank source", ["Pick a real bank", "Enter ratios manually"])

    models = fetch_models(backend_url)
    model_options = {item["label"]: item["key"] for item in models}
    selected_model_label = st.selectbox(
        "Scoring model",
        options=list(model_options),
        index=0,
        help="Both choices are trained artifacts. Compare their theory and holdout metrics on the Documentation page.",
    )
    selected_model = model_options[selected_model_label]

    st.divider()
    st.caption("\U0001F4DA See the **Documentation** page (sidebar above) for how this project works, the folder structure, and which file does what.")
    st.caption("Developer: **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**")

ratios: dict[str, float] = {}
display_name = "Custom bank"

if mode == "Pick a real bank":
    banks = fetch_banks(backend_url)
    if not banks:
        st.warning("No sample banks came back from the backend (data pull may have failed). Switch to manual entry.")
        st.stop()

    labels = [f"{b['NAME']} ({b['STALP']}) — our rating {b['proxy_rating_1_5']}" for b in banks]
    choice = st.selectbox("Choose a bank", options=range(len(banks)), format_func=lambda i: labels[i])
    bank = banks[choice]
    display_name = bank["NAME"]

    st.caption(
        f"Loaded {bank['NAME']}'s real Q4-2023 ratios. Our own composite score already rated it "
        f"**{bank['proxy_rating_1_5']}/5** ({bank['composite_score_0_100']}/100); FDIC records show it "
        f"{'DID' if bank['failed'] else 'did NOT'} fail. Adjust any ratio below to explore a what-if."
    )
    cols = st.columns(2)
    for i, (key, label, lo, hi, _default, help_text) in enumerate(FEATURE_FIELDS):
        with cols[i % 2]:
            value = float(bank.get(key) or _default)
            ratios[key] = st.slider(label, min_value=float(lo), max_value=float(hi), value=min(max(value, lo), hi), help=help_text)
else:
    st.caption("Enter a hypothetical bank's ratios and see what the model predicts.")
    cols = st.columns(2)
    for i, (key, label, lo, hi, default, help_text) in enumerate(FEATURE_FIELDS):
        with cols[i % 2]:
            ratios[key] = st.slider(label, min_value=float(lo), max_value=float(hi), value=float(default), help=help_text)

if st.button("Score this bank", type="primary"):
    with st.spinner("Scoring..."):
        result = score(backend_url, {"name": display_name, **ratios}, selected_model)

    tier = result["risk_tier"]
    st.markdown(
        f"### Predicted risk: "
        f"<span style='color:{RISK_COLOR[tier]}; font-weight:700; text-transform:uppercase;'>{tier}</span>"
        f" &nbsp; ({result['predicted_failure_probability']:.1%} failure-risk score)",
        unsafe_allow_html=True,
    )
    st.caption(f"Scored with: **{result['model']}**")

    st.subheader("Why the model said this")
    render_contributions_chart(result["contributions"])
    # SHAP for gradient-boosted trees works in log-odds; show the base value as a probability.
    base = result["base_value"]
    base_prob = base if 0.0 <= base <= 1.0 else 1.0 / (1.0 + math.exp(-base))
    st.caption(
        f"Base rate (the model's average risk): {base_prob:.1%}. Each bar shows how far that one ratio moved this "
        f"bank's score away from the base rate, in the model's log-odds units — bars pointing right push risk up, "
        f"left pulls it down."
    )

    with st.expander("LIME cross-check"):
        lime = result["lime"]
        if lime.get("available"):
            st.caption("LIME fits a simple local model around this bank. Its weights are a cross-check, not a replacement for SHAP.")
            lime_table = pd.DataFrame(lime["features"])
            lime_table["direction"] = lime_table["weight"].map(lambda value: "raises risk" if value > 0 else "lowers risk")
            st.dataframe(lime_table, use_container_width=True, hide_index=True)
        else:
            st.info(lime["reason"])

    with st.expander("Counterfactual: what could lower this risk score?"):
        counterfactual = result["counterfactual"]
        if counterfactual.get("already_below_threshold"):
            st.success("This bank is already below the medium-risk threshold.")
        elif counterfactual.get("found"):
            st.success(
                f"A bounded what-if path lowers the score from {counterfactual['starting_probability']:.1%} "
                f"to {counterfactual['counterfactual_probability']:.1%}."
            )
            changes = pd.DataFrame(
                [{"ratio": key, "change": value, "new value": counterfactual["values"][key]} for key, value in counterfactual["changes"].items()]
            )
            st.dataframe(changes, use_container_width=True, hide_index=True)
        else:
            st.warning("No bounded ratio-only path reached the medium-risk threshold.")
        st.caption("This is a model what-if, not a causal recommendation. Ratios may be correlated, constrained, or impossible to change independently.")

    with st.expander("Model card — read this before trusting the number above"):
        card = fetch_model_card(backend_url)
        for key, value in card.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

st.divider()
st.caption("CAMEL Sentinel · Developer: **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**")
