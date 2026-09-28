# Phase 9 — Conversational Chat Interface

> This document is the design reference for Phase 9. It answers three
> questions before any code is written: *what* we are building, *why* each
> design decision was made, and *how* the pieces fit together. Everything
> in `src/camel_sentinel/chat/`, the new backend endpoints, and the new
> frontend page should trace back to a decision recorded here.

---

## 1. What we are building

A **chat assistant page** layered on top of the existing CAMEL Sentinel
scoring engine. A user can ask questions in plain English (or by speaking)
and receive natural-language answers that incorporate the ML model's actual
findings — risk tier, SHAP-driven drivers, counterfactuals — rather than
having to interpret raw numbers themselves.

Two input modes, one underlying model:

| Input mode | Path |
|---|---|
| **Text** | User types → GPT-5-mini agent → text response in chat bubble |
| **Voice** | User speaks → Azure STT → same agent → text → Azure TTS → audio |

The LLM is the *translator*, not the decision-maker. It rephrases what the
ML model returns. It is not allowed to produce a risk rating from its own
knowledge. This is the same guardrail recorded in
`docs/03_module_mapping.md` under Generative AI, LLMs & RLHF.

---

## 2. Design decisions

### 2.1 GPT-5-mini via OpenAI API, configurable via `.env`

The model name lives in an environment variable (`OPENAI_MODEL`, default
`gpt-5-mini`). Swapping to a different model requires only a `.env` change,
not a code change. All API calls go through a single `agent.py` wrapper,
so a future provider swap is isolated to one file.

### 2.2 Tool-calling, not free generation

The agent is given exactly one tool in OpenAI function-calling format:
`score_bank`. It maps directly to the existing `score_bank_tool()` in
`src/camel_sentinel/agent/tools.py`. The LLM decides *when* to call it
(i.e., when the user is asking about a specific bank's risk), then
formats the structured result it receives into prose. It never generates
a risk number from its own weights.

### 2.3 Mirror-input voice mode

Voice responses are generated only when the user's input was also voice.
Text questions get text answers. This keeps voice mode purposeful and
avoids autoplay surprises when someone is typing.

### 2.4 Conversation history in session state

The Streamlit page holds the conversation history in `st.session_state`.
On each turn it sends the full history to `/chat` so the agent can answer
follow-up questions correctly ("what if that ratio were higher?"). History
is capped at a rolling window (configurable, default 20 turns) to stay
within the model's context budget.

### 2.5 Caveats baked into the system prompt

The same caveats from `docs/model_card.md` are embedded in the system
prompt (proxy label, calibration, Management proxy, reporting lag). The
LLM cannot strip them out and cannot be instructed by a user to ignore
them. Any response that includes a risk score must attach the caveat that
probabilities are relative-risk rankings, not real odds.

### 2.6 Speech runs server-side (backend endpoints)

Azure Speech SDK calls happen in the backend, not in the Streamlit
process. This keeps API keys out of the browser and away from Streamlit's
multi-thread model, and makes the speech layer testable independently of
the UI.

---

## 3. Architecture

```
  STREAMLIT PAGE (2_Chat_Assistant.py)
  ┌──────────────────────────────────────────┐
  │  chat history display (st.chat_message)  │
  │  text input  OR  audio recorder widget   │
  │  (voice mode) audio player (st.audio)    │
  └─────────────────┬────────────────────────┘
                    │  HTTP POST /chat
                    │  {message, history, mode}
                    ▼
  FASTAPI BACKEND (main.py)
  ┌──────────────────────────────────────────┐
  │  POST /chat                              │
  │    └─► chat.agent.ChatAgent.run()        │
  │         ├─ GPT-5-mini with tool schema   │
  │         │   (score_bank tool)            │
  │         └─ if tool called:               │
  │              agent/tools.score_bank_tool │
  │              → ML model + SHAP + LIME    │
  │  POST /speech/transcribe                 │
  │    └─► chat.speech.transcribe_audio()    │
  │  POST /speech/synthesize                 │
  │    └─► chat.speech.synthesize_text()     │
  └──────────────────────────────────────────┘
  ┌──────────────────────────────────────────┐
  │  CAMEL ML MODEL (already loaded)         │
  │  SHAP + LIME explainers (already loaded) │
  └──────────────────────────────────────────┘
```

The `/chat` endpoint is **stateless from the backend's perspective** — all
history is sent with each request from the frontend. The backend loads the
ML model once at startup (same as today) and the `ChatAgent` is
instantiated per-request, receiving the loaded model objects via
dependency injection.

---

## 4. New files

```
src/camel_sentinel/chat/
├── __init__.py
├── agent.py          # OpenAI client, tool schema, run() method
├── speech.py         # Azure Speech STT + TTS wrappers
├── history.py        # ConversationHistory: rolling-window message list
└── prompts.py        # SYSTEM_PROMPT constant (role + domain + caveats)

backend/main.py       # add: POST /chat, POST /speech/transcribe,
                      #           POST /speech/synthesize

frontend/views/
└── 2_💬_Chat_Assistant.py   # new Streamlit chat page

.env.example          # new file: documents all required env vars
```

No existing files are deleted. `src/camel_sentinel/agent/tools.py` is
reused as-is; `chat/agent.py` imports from it.

---

## 5. New environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API authentication |
| `OPENAI_MODEL` | `gpt-5-mini` | Model name, swap without code change |
| `AZURE_SPEECH_KEY` | *(required for voice)* | Azure Cognitive Services key |
| `AZURE_SPEECH_REGION` | *(required for voice)* | e.g. `eastus` |

Voice mode gracefully degrades to text-only if the Azure credentials are
not set. Text mode requires only `OPENAI_API_KEY`.

---

## 6. New dependencies

```
openai>=1.50.0          # GPT-5-mini API + function calling
python-dotenv>=1.0      # load .env in backend at startup
streamlit-audiorecorder # in-browser mic recording for voice mode
```

`azure-cognitiveservices-speech` is already in the environment (installed
but unused). No version bump required.

---

## 7. Backend endpoint contracts

### POST `/chat`

**Request body:**
```json
{
  "message": "How risky is First National Bank with a Tier-1 ratio of 8%?",
  "history": [
    {"role": "user",    "content": "..."},
    {"role": "assistant","content": "..."}
  ],
  "mode": "text"
}
```

**Response:**
```json
{
  "reply": "Based on the CAMEL scoring model...",
  "tool_called": true,
  "score_result": { ... },
  "mode": "text"
}
```

`score_result` is included in the response so the frontend can optionally
render the SHAP chart alongside the natural-language reply.

### POST `/speech/transcribe`

**Request:** `multipart/form-data`, field `audio` (WAV or WebM bytes)
**Response:** `{"text": "transcribed text"}`

### POST `/speech/synthesize`

**Request:** `{"text": "...", "voice": "en-US-AriaNeural"}`
**Response:** `audio/wav` bytes (streamed)

---

## 8. System prompt contract

The system prompt in `chat/prompts.py` establishes four constraints that
the LLM cannot override at runtime:

1. **Role**: "You are a financial-risk assistant for CAMEL Sentinel. You
   explain the outputs of the CAMEL scoring model in plain English."

2. **Tool discipline**: "To answer questions about a specific bank's risk,
   you MUST call the `score_bank` tool and use the numbers it returns.
   Do not estimate, guess, or recall risk scores from your own knowledge."

3. **Calibration caveat**: "Whenever you report a failure probability,
   state that it is a relative-risk ranking from the model, not a real
   odds estimate. This is required, not optional."

4. **Scope boundary**: "You are not a replacement for supervisory judgment.
   If asked to make a regulatory decision, explain the tool's scope and
   decline."

---

## 9. Implementation phases

Work is split into four sequential slices so each slice is testable
independently before the next begins.

| Slice | Status | What shipped |
|---|---|---|
| **A — LLM core** | **Done** | `chat/prompts.py`, `chat/history.py`, `chat/agent.py` (Azure OpenAI tool-calling), `POST /chat` |
| **B — Chat UI** | **Done** | `frontend/views/2_💬_Chat_Assistant.py` (text mode), `.env.example`, SHAP inline |
| **C — Voice input** | **Done** | `chat/speech.py` STT, `POST /speech/transcribe`, `st.audio_input` in frontend (native Streamlit 1.37+, no extra package) |
| **D — Voice output** | **Done** | `chat/speech.py` TTS, `POST /speech/synthesize`, `st.audio(autoplay=True)` for new replies, `autoplay=False` for history replay |

---

## 10. What this is not

- **Not a replacement for the existing scoring UI** — the slider-based
  scoring page (`frontend/app.py`) stays in place. The chat page is a
  second way to reach the same model.
- **Not a fine-tuned model** — GPT-5-mini is used off the shelf, prompted
  to act as a translator. No training data, no RLHF, no model artifact.
- **Not a general financial chatbot** — the system prompt scopes it to the
  CAMEL Sentinel scoring context. Questions outside that scope receive a
  brief explanation of the tool's scope and no fabricated answer.
- **Not real-time** — same reporting-lag caveat as the rest of the system:
  FDIC Call Report data is 45–75 days behind quarter-end.

---

## 11. Security notes

- Azure and OpenAI API keys live in `.env`, gitignored, never in frontend
  code or Streamlit session state.
- The `/chat` endpoint validates that `history` entries contain only
  `role` and `content` fields; extra fields are stripped before they
  reach the OpenAI API.
- Prompt injection: user messages are passed as `role: "user"` content,
  never concatenated into the system prompt. The system prompt is a
  server-side constant.
- The `score_bank` tool is the same allow-listed, read-only tool already
  in use via `/agent/score`. No new tool surface is introduced.
