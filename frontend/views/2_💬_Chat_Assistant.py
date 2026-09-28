"""
frontend/views/2_💬_Chat_Assistant.py — Phase 9 + UX improvements.

Additions over Slice D:
  - Quick-start prompt buttons shown in empty state; click to submit instantly.
  - ? help popover (top-right) with project overview and example questions.
  - Out-of-scope redirect enforced via system prompt (prompts.py); generic
    questions receive a redirect message, no special UI handling needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_kit import backend_url, backend_url_setting, safe_markdown  # noqa: E402

st.set_page_config(
    page_title="CAMEL Sentinel — Chat",
    page_icon="\U0001F42B",
    layout="centered",
)

BACKEND_DEFAULT = backend_url()  # set in the sidebar 🔌 panel
RISK_COLOR = {"low": "#2E7D5C", "medium": "#B4741F", "high": "#B23A2E"}

# Quick-start prompts shown as clickable buttons in the empty state.
# (label shown on button, full prompt sent to the agent)
QUICK_PROMPTS: list[tuple[str, str]] = [
    (
        "What are CAMEL ratios?",
        "What CAMEL ratios does this model use and what does each one measure?",
    ),
    (
        "Score a sample bank",
        "Score a bank with: Tier-1 12%, NPL 2%, efficiency 65%, ROA 0.8%, "
        "loan-deposit 80%, loan concentration 55%",
    ),
    (
        "What is a risk tier?",
        "What does each risk tier (low, medium, high) mean in this model?",
    ),
    (
        "What drives risk up?",
        "Which financial ratios push a bank's risk score higher, and why?",
    ),
    (
        "How to lower risk?",
        "What steps could a medium-risk bank take to lower its CAMEL Sentinel risk score?",
    ),
    (
        "Model limitations",
        "What are the key limitations and caveats of this CAMEL Sentinel model?",
    ),
]

_HELP_TEXT = """\
### About CAMEL Sentinel Chat

**CAMEL Sentinel** is a bank-health scoring tool trained on **265,000+ FDIC \
bank-quarters** (2005–2023). This assistant explains the model's findings in \
plain English — it does not originate risk numbers from its own knowledge; \
it calls the ML model and rephrases what it returns.

**Built by:** [Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)

---

#### What you can ask
- What each CAMEL ratio means and why it matters
- Score a specific bank by supplying its six ratios
- Understand which ratios are driving a bank's risk tier (SHAP breakdown)
- Ask "what if?" questions — counterfactual improvements
- Model limitations, calibration caveats, and data-lag notes

#### Six ratios required for scoring
| Ratio | Healthy direction |
|---|---|
| Tier-1 capital ratio (%) | Higher ↑ |
| NPL ratio (%) | Lower ↓ |
| Efficiency ratio (%) | Lower ↓ |
| Return on assets — ROA (%) | Higher ↑ |
| Loan-to-deposit ratio (%) | Lower ↓ |
| Loan concentration (%) | Lower ↓ |

---

#### What this tool does NOT answer
Questions outside CAMEL Sentinel bank scoring (geography, politics, general \
technology, news, etc.) are out of scope. For those, please use \
**Microsoft Copilot**, **Google Gemini**, or **ChatGPT**.
"""


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def call_chat(backend_url: str, message: str, history: list[dict], mode: str = "text") -> dict:
    resp = requests.post(
        f"{backend_url}/chat",
        json={"message": message, "history": history, "mode": mode},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def call_transcribe(backend_url: str, audio_bytes: bytes) -> str:
    """POST /speech/transcribe — returns recognised text or empty string."""
    resp = requests.post(
        f"{backend_url}/speech/transcribe",
        files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("text", "")


def call_synthesize(backend_url: str, text: str, voice: str = "en-US-AriaNeural") -> bytes:
    """POST /speech/synthesize — returns WAV bytes for st.audio()."""
    resp = requests.post(
        f"{backend_url}/speech/synthesize",
        json={"text": text, "voice": voice},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _as_probability(value: float) -> float:
    """SHAP base values for boosted trees are log-odds; show them as a probability."""
    import math
    return value if 0.0 <= value <= 1.0 else 1.0 / (1.0 + math.exp(-value))


def _render_score_block(score_result: dict) -> None:
    """Render risk tier badge + SHAP chart inside a chat bubble."""
    tier = score_result.get("risk_tier", "")
    prob = score_result.get("predicted_failure_probability", 0.0)
    color = RISK_COLOR.get(tier, "#888888")
    st.markdown(
        f"**Model output** &mdash; risk tier: "
        f"<span style='color:{color}; font-weight:700; text-transform:uppercase;'>"
        f"{tier}</span>&nbsp;&nbsp;({prob:.1%} failure-risk score)",
        unsafe_allow_html=True,
    )
    shap = score_result.get("shap", {})
    contributions = shap.get("contributions")
    if not contributions:
        return
    with st.expander("SHAP breakdown — which ratios drove this score"):
        series = pd.Series(contributions).sort_values(key=abs)
        bar_colors = [RISK_COLOR["high"] if v > 0 else RISK_COLOR["low"] for v in series]
        fig, ax = plt.subplots(figsize=(6, 3.0))
        ax.barh(series.index, series.values, color=bar_colors)
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_xlabel("SHAP contribution to failure-risk score")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(
            f"Base rate (model's average risk): {_as_probability(shap.get('base_value', 0.0)):.1%}. "
            "Red bars push risk up, green bars pull it down. "
            "Scores are relative rankings, not calibrated real-world failure odds."
        )


# ---------------------------------------------------------------------------
# Shared message processor
# ---------------------------------------------------------------------------

def _process_message(backend_url: str, prompt: str, mode: str) -> None:
    """
    Full turn handler for both text and voice:
      1. Display + store user message.
      2. Call /chat → natural-language reply.
      3. If voice: call /speech/synthesize → autoplay audio.
      4. Display SHAP breakdown if the ML model was invoked.
      5. Store assistant message (with audio_bytes so history can replay it).
    """
    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # User bubble — mic prefix for voice turns so origin is visible in history
    display_text = f"\U0001F3A4 {prompt}" if mode == "voice" else prompt
    with st.chat_message("user"):
        st.markdown(display_text)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # /chat
    result: dict | None = None
    chat_error: str | None = None
    with st.spinner("Thinking..."):
        try:
            result = call_chat(backend_url, prompt, api_history, mode=mode)
        except requests.HTTPError as exc:
            try:
                chat_error = exc.response.json().get("detail", str(exc))
            except Exception:
                chat_error = str(exc)
        except requests.RequestException as exc:
            chat_error = str(exc)

    if chat_error:
        with st.chat_message("assistant"):
            st.error(f"Backend error: {chat_error}")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"_(Error: {chat_error})_"}
        )
        return

    reply = result.get("reply", "")
    score = result.get("score_result") if result.get("tool_called") else None

    # TTS — only for voice mode; text mode stays silent
    audio_bytes: bytes | None = None
    if mode == "voice" and reply:
        with st.spinner("Generating audio reply..."):
            try:
                audio_bytes = call_synthesize(backend_url, reply)
            except requests.HTTPError as exc:
                try:
                    synth_err = exc.response.json().get("detail", str(exc))
                except Exception:
                    synth_err = str(exc)
                st.toast(f"Audio generation failed: {synth_err}", icon="⚠️")
            except requests.RequestException as exc:
                st.toast(f"Audio generation failed: {exc}", icon="⚠️")

    with st.chat_message("assistant"):
        st.markdown(safe_markdown(reply))
        # Audio immediately after text so the user hears the reply as they read it
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav", autoplay=True)
        if score:
            _render_score_block(score)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "score_result": score,
            "audio_bytes": audio_bytes,  # stored so history can replay it
        }
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    backend_url = backend_url_setting()

    backend_ok = False
    try:
        health = requests.get(f"{backend_url}/health", timeout=5).json()
        if health.get("model_loaded"):
            st.success("Backend connected")
            backend_ok = True
        else:
            st.warning("Backend up, model not loaded yet")
    except requests.RequestException:
        st.error(
            "Can't reach the backend — start it with:\n\n"
            "python -m uvicorn main:app --app-dir backend --port 8000"
        )

    st.divider()
    input_mode = st.radio(
        "Input mode",
        ["Text", "Voice"],
        horizontal=True,
        help="Voice records from your microphone, transcribes with Azure STT, "
             "and reads the reply aloud with Azure TTS.",
    )

    st.divider()
    if st.button("Clear conversation", disabled=not st.session_state.get("messages")):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Developer: **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**")

# ---------------------------------------------------------------------------
# Page header — title on the left, ? help popover on the right
# ---------------------------------------------------------------------------
hdr_left, hdr_right = st.columns([11, 1])
with hdr_left:
    st.title("\U0001F4AC CAMEL Sentinel Chat")
    st.caption(
        "Ask questions in plain English. Provide the six CAMEL ratios to score a "
        "specific bank — the model will predict failure risk and explain which "
        "ratios drove the result."
    )
with hdr_right:
    st.write("")  # vertical spacing to align with title
    with st.popover("?", help="About this assistant"):
        st.markdown(_HELP_TEXT)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# ---------------------------------------------------------------------------
# Welcome hint + quick-start buttons (empty state only)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.info(
        "Ask about CAMEL ratios, score a bank by providing its six financial "
        "ratios, or explore what drives risk up or down. "
        "Click a quick-start button below to get going instantly.",
        icon="\U0001F4A1",
    )

    if backend_ok:
        st.markdown("**Quick start — click any prompt to run it:**")
        # Three buttons per row
        cols = st.columns(3)
        for idx, (label, prompt_text) in enumerate(QUICK_PROMPTS):
            with cols[idx % 3]:
                if st.button(label, use_container_width=True, key=f"qs_{idx}"):
                    _process_message(backend_url, prompt_text, mode="text")
                    st.rerun()
    else:
        st.markdown(
            "_Connect to the backend to enable quick-start prompts._"
        )

# ---------------------------------------------------------------------------
# Render conversation history
# Each turn is re-drawn from session state on every rerun.
# Voice turns include a replay button (autoplay=False — don't auto-replay history).
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Model output (and echoed user text) is sanitised before rendering — docs/09 F3.
        st.markdown(safe_markdown(msg["content"]))
        if msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/wav", autoplay=False)
        if msg.get("score_result"):
            _render_score_block(msg["score_result"])

# ---------------------------------------------------------------------------
# Input — text or voice
# ---------------------------------------------------------------------------
if input_mode == "Voice":
    if not backend_ok:
        st.info("Connect to the backend to use voice input.")
    else:
        st.caption(
            "\U0001F3A4 Record your question — it will be transcribed, answered, "
            "and read back aloud."
        )
        # Key tied to message count so the widget resets after each processed turn.
        audio_value = st.audio_input(
            "Record your question",
            key=f"voice_{len(st.session_state.messages)}",
        )

        if audio_value is not None:
            audio_bytes = audio_value.read()

            transcript: str = ""
            transcribe_error: str | None = None
            with st.spinner("Transcribing..."):
                try:
                    transcript = call_transcribe(backend_url, audio_bytes)
                except requests.HTTPError as exc:
                    try:
                        transcribe_error = exc.response.json().get("detail", str(exc))
                    except Exception:
                        transcribe_error = str(exc)
                except requests.RequestException as exc:
                    transcribe_error = str(exc)

            if transcribe_error:
                st.error(f"Transcription error: {transcribe_error}")
            elif not transcript:
                st.warning(
                    "Nothing was recognised in the recording. "
                    "Try speaking more clearly or check your microphone."
                )
            else:
                _process_message(backend_url, transcript, mode="voice")
                st.rerun()  # reset audio widget via new key

else:  # Text mode
    prompt = st.chat_input(
        "Ask about a bank's risk, or provide the six CAMEL ratios to score one...",
        disabled=not backend_ok,
    )
    if prompt:
        _process_message(backend_url, prompt, mode="text")

# ---------------------------------------------------------------------------
st.divider()
st.caption("CAMEL Sentinel · Developer: **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**")
