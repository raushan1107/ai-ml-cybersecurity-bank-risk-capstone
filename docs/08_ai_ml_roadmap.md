# AI/ML Roadmap — Design

> The **🗺️ AI/ML Roadmap** page summarises the whole learning path behind this project, as an
> interactive tree. The fundamentals of every topic are taught in the
> [AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)
> ([source on GitHub](https://github.com/raushan1107/AI-Machine-Learning-Handbook)) by **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**, the developer
> of this project. The roadmap is the map; the handbook is the textbook.

---

## 1. Naming rule

Stages are referred to **by name with a plain sequence number**, e.g. *"6 · Explainable & Responsible
AI"*. We never write "Module 6". Names say what you learn, while numbers only show the order.

## 2. The 12 stages and their topics

| # | Stage | Topics (tree leaves) |
|---|---|---|
| 1 | Data Science & Financial Data Analysis | EDA · Feature engineering · Regression · Classification · Clustering · Anomaly detection · Data-poisoning defence |
| 2 | Deep Neural Networks | Perceptrons & ANNs · Backpropagation · Optimizers · CNNs · Adversarial examples & training |
| 3 | NLP & Financial Text AI | Text representations · Embeddings · Named-entity recognition · Sentiment analysis · Prompt injection · PII redaction |
| 4 | Generative AI, LLMs & RLHF | Transformers & attention · Fine-tuning (LoRA/QLoRA) · Alignment (SFT → RM → PPO/DPO) · LLM red-teaming |
| 5 | RAG, LangChain & AI Agents | Retrieval-augmented generation · Agentic tool use · LangChain & LangGraph · Retrieval poisoning · Tool-call hijacking |
| 6 | Explainable & Responsible AI | SHAP · LIME · Counterfactuals · Fairness auditing · Model governance |
| 7 | Federated & Privacy-Preserving ML | Federated averaging · Differential privacy · Byzantine-robust aggregation |
| 8 | Multimodal AI | Speech AI · Vision-language models · Document intelligence |
| 9 | Graph Neural Networks | Message passing & GraphSAGE · Fraud-ring detection · Knowledge graphs & GraphRAG · GNN intrusion detection |
| 10 | MLOps & Deployment | Model serving (FastAPI) · Containers (Docker) · Orchestration (Kubernetes) · DevSecOps scanning · Secrets management · mTLS |
| 11 | AI Cybersecurity | STRIDE · MITRE ATLAS · Model stealing · Deepfake detection · AI-powered SOC |
| 12 | Capstone: One Secured System | End-to-end architecture · Security by design · Production readiness |

## 3. What each leaf reveals (the topic card)

| Lens | Content |
|---|---|
| **In one line** | A 1–2 line definition |
| **Real world** | 2–3 named, publicly documented uses, e.g. *Gemini is a natively multimodal Transformer*, *Gboard trains next-word prediction with federated learning*, *Google Maps ETAs use GNNs* |
| **Analogy** | An everyday analogy |
| **Math** | The core equation in plain Unicode notation (renders offline, no MathJax/CDN) |
| **Code** | A 5–15 line minimal snippet |
| **In CAMEL Sentinel** | Where the topic is used in this project (file or page), or honestly "not used, because…" |
| **Security angle** | How this topic is attacked or defended (links into the AI Security Lab) |
| **Learn it** | Deep link to the handbook chapter |

Real-world claims are limited to publicly documented examples (product blogs, papers, major news).
When a claim is general rather than tied to a named product, the card says so.

## 4. Interaction design

- **Views:**
  - **Tree**: a horizontal, collapsible SVG tree (root → stages → topics). Branches animate open, and a stage node's colour marks its track (Data & Models, Language & GenAI, Trust & Privacy, Production & Security).
  - **Journey**: the 12 stages as a flowing path, showing how each one hands over to the next ("the model from 1 is explained in 6, served in 10 and attacked in 11").
- **Click a topic:** the card slides in, with lens tabs across the top.
- **Search box:** filters and highlights matching topics.
- **Responsive:** on narrow screens the card stacks below the tree and the tree scrolls horizontally inside its own panel. The page itself never scrolls sideways.
- **Theme:** follows light/dark mode.

## 5. Implementation

| File | Purpose |
|---|---|
| `frontend/roadmap_content/stages_*.py` | The content, as plain Python dicts (easy for anyone to edit) |
| `frontend/roadmap_content/__init__.py` | `load_curriculum()` → list of stages, with validation of required fields |
| `frontend/views/7_🗺️_AI_ML_Roadmap.py` | Renders the tree + card as one self-contained HTML/JS component (no external scripts, so it works inside the container) |

Content is written in batches of about 3 stages and reviewed per batch.
