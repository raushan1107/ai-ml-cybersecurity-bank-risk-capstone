# Learning Path Mapping

> How this project uses each stage of the
> [AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)
> ([GitHub](https://github.com/raushan1107/AI-Machine-Learning-Handbook)), written by
> **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**, the developer of this project. The handbook teaches the fundamentals
> and concepts behind every stage; this project applies them to one real problem.
>
> Stages are named by what they teach, with a plain number only to show the sequence.
> The rule for this project: **solve the actual problem, and use a stage's techniques
> where they genuinely serve that problem**. We don't force every stage in for coverage's
> sake. Where a stage isn't used, that's stated explicitly below, with the reasoning.

| # | Learning path stage | Used in this project? | Where / how |
|---|---|---|---|
| 1 | Data Science + Financial Data Analysis (EDA, feature engineering, regression, classification, clustering, anomaly detection, data-poisoning defence) | **Yes — the backbone.** | `docs/00_problem_statement.md` (problem framing), `src/camel_sentinel/data/` (acquisition), `src/camel_sentinel/features/camel_ratios.py` (feature engineering), and the EDA/baseline-modeling notebooks built in later phases. This project *is* a financial-data-analysis problem: the CAMEL ratios are the feature engineering, the composite score/rating is the regression+classification target, peer clustering and anomaly detection are planned for Phase 3 (see the blueprint's Phase 3 row). Basic input validation on the FDIC pull (schema/range checks before ratios are built) is this stage's data-poisoning-defence idea applied to an external data feed. |
| 2 | Deep Neural Networks (perceptrons, ANNs, backprop, optimizers, CNNs, adversarial examples, adversarial training) | **Partially — only the pieces that fit tabular data.** | The CAMEL problem is tabular (structured ratios per bank-quarter), which gradient-boosted trees handle at least as well as a deep net, and more interpretably — so this project does **not** force a CNN in artificially (there's no image/spatial data here to justify one). What *does* apply: an ANN/MLP baseline is planned alongside the tree-based model in Phase 4 as a second model family to compare, and a lightweight adversarial-robustness check (small, deliberate perturbations to a bank's ratios — "how much would someone need to fudge their numbers to flip the model's rating?") is planned as a security-relevant extension of Phase 4, directly reusing the adversarial-examples idea from Deep Neural Networks in a tabular setting. |
| 3 | NLP + Financial Text AI (text representations, embeddings, NER, sentiment analysis, prompt injection, PII redaction) | **Partially — only in the agent layer.** | The core CAMEL pipeline is numeric, not text, so full NLP modeling (embeddings/NER/sentiment) isn't part of the MVP. What *is* used: once the tool-calling agent (RAG, LangChain & AI Agents) is built in Phase 8, any free-text user input to it needs the same PII-redaction and prompt-injection-defence thinking this stage teaches, since the agent is a natural-language interface over financial data — flagged as a real, small integration at that phase, not a placeholder. |
| 4 | Generative AI, LLMs & RLHF (transformers, fine-tuning, alignment, LLM red-teaming) | **Partially — only for explanation generation.** | No fine-tuning or RLHF is needed for a tabular scoring problem. What's built (Phase 9): gpt-5-mini via Azure OpenAI turns SHAP contributions into plain-language answers in `src/camel_sentinel/chat/agent.py`. The guardrail discipline this stage teaches is enforced in `src/camel_sentinel/chat/prompts.py`: the server-side system prompt prevents the LLM from originating risk ratings; it may only rephrase numbers returned by the `score_bank` tool. User messages are always `role: "user"` content and never concatenated into the system prompt, applying the prompt-injection-defence pattern. |
| 5 | RAG, LangChain & AI Agents (vector retrieval, agentic tool-use, LangGraph, retrieval poisoning, tool-call hijacking threats) | **Done — tool layer and natural-language orchestration complete.** | `src/camel_sentinel/agent/tools.py` exposes the allow-listed `score_bank` tool; the backend provides `/agent/tools` and `/agent/score`. Phase 9's `ChatAgent` (`src/camel_sentinel/chat/agent.py`) is the natural-language orchestration layer: it wraps Azure OpenAI's function-calling API with `score_bank` as the single allowed tool. A LangChain/LangGraph wrapper was not needed — the direct OpenAI tool-calling loop is simpler and sufficient for a single-tool use case. No retrieval/vector store is needed for this numeric problem. Arbitrary filesystem/database access remains deliberately excluded; unknown tool names still fail closed. |
| 6 | Explainable & Responsible AI (SHAP, LIME, counterfactuals, fairness auditing, model governance) | **XAI and core RAI audit done.** | XAI: `src/camel_sentinel/xai/explain.py` provides SHAP and optional LIME, while `src/camel_sentinel/xai/counterfactual.py` provides bounded what-if search. The scorer exposes all three views. RAI: `src/camel_sentinel/rai/audit.py` audits asset-size and region performance at the deployed threshold, `/rai-audit` exposes the results, and `docs/model_card.md` documents intended use, validation, limitations, and missing attributes. Charter-type auditing remains unavailable because the current panel does not contain that field. |
| 7 | Federated & Privacy-Preserving ML (federated averaging, differential privacy, Byzantine-robust aggregation defences) | **Core simulation done.** | `src/camel_sentinel/federated/simulation.py` partitions the panel by state as simulated data owners, trains a standardized linear classifier with FedAvg, clips client updates, and adds Gaussian noise. The UI/model card document that this is a teaching mechanism simulation, not a formal epsilon certificate or a replacement for the tuned centralized tree model. Byzantine-robust aggregation remains a stretch goal. |
| 8 | Multimodal AI (speech, vision-language models, document intelligence) | **Partially — speech.** | The chat assistant's voice mode (`src/camel_sentinel/chat/speech.py`) uses Azure speech-to-text and text-to-speech. Vision and documents aren't needed for numeric bank ratios. Deepfake-voice detection is demonstrated in the AI Security Lab. |
| 9 | Graph Neural Networks (fraud-ring detection, knowledge graphs, GraphRAG, GNN intrusion detection) | **Yes — separate fraud-detection use case.** | `src/camel_sentinel/gnn/` + `backend/gnn_router.py` + the 🕸️ Fraud Detection page: a heterogeneous GraphSAGE model on synthetic transaction graphs, compared against tabular baselines. See `docs/05_gnn_fraud_detection.md`. |
| 10 | MLOps & On-Premises Deployment (FastAPI serving, Docker, Kubernetes, Trivy, secrets, mTLS) | **Yes.** | FastAPI backend, `Dockerfile`, `docker-compose.yml`, `deploy/k8s/`, and images published to Docker Hub and Azure Container Registry. Walkthrough on the 🚢 Deploy & Containerize page; design in `docs/06_deployment_containerization.md`. |
| 11 | AI Cybersecurity (STRIDE, MITRE ATLAS, model stealing, deepfake detection, AI-powered SOC) | **Yes.** | `src/camel_sentinel/security/` + `backend/security_router.py` + the 🛡️ AI Security Lab page attack and defend this very system. Design in `docs/07_ai_security_lab.md`. |
| 12 | Final Capstone (one integrated, secured system) | **This project as a whole.** | Stages 1–11 combined into CAMEL Sentinel. The 🗺️ AI/ML Roadmap page (`docs/08_ai_ml_roadmap.md`) shows how every stage connects. |

## Phase 9 — Conversational Chat Interface

Not a separate learning-path stage, but a new product layer added after Phases 0–8.
Full design is in [`04_chat_interface.md`](04_chat_interface.md).

| Component | Status | Notes |
|---|---|---|
| `src/camel_sentinel/chat/` | **Done** | agent.py (Azure OpenAI tool-calling), history.py (rolling 20-turn window), prompts.py (system prompt + caveats), speech.py (Azure STT + TTS) |
| `POST /chat` endpoint | **Done** | Stateless; rolling history sent by client each request |
| `POST /speech/transcribe` | **Done** | Azure STT via `recognize_once()`; accepts 16 kHz WAV from `st.audio_input` |
| `POST /speech/synthesize` | **Done** | Azure TTS via `SpeechSynthesizer`; returns `audio/wav` bytes |
| `frontend/views/2_💬_Chat_Assistant.py` | **Done** | Text + voice; mirror-input rule; SHAP chart + risk tier inline |

All four implementation slices (A — LLM core, B — chat UI, C — voice input, D — voice output) complete.

---

## Reading this table

- "Yes" rows are full, built phases with their own directory and, later,
  their own notebook.
- "Partially" rows are genuine, working code — not stubs — but scoped to
  the one place in the pipeline where that module's technique actually
  solves a piece of *this* problem, rather than a standalone exercise.
- Stages 8–12 were added after the original "first seven stages" build; each row above links to its design doc.
