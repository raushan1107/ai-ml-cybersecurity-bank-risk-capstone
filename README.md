<div align="center">

# 🐫 CAMEL Sentinel

### An end-to-end AI/ML + Cybersecurity capstone: explainable bank-risk AI you can run, attack, defend and deploy

Predict a bank's failure risk from public FDIC data, see **why** the model decided, chat with it,
catch fraud rings with a graph neural network, attack it in a live **AI Security Lab**,
and ship it anywhere as **one Docker container**.

[![Docker Pulls](https://img.shields.io/docker/pulls/raushanranjan/camel-sentinel?logo=docker&logoColor=white&label=Docker%20pulls&color=2496ED)](https://hub.docker.com/r/raushanranjan/camel-sentinel)
[![Docker Image Version](https://img.shields.io/docker/v/raushanranjan/camel-sentinel?sort=semver&logo=docker&logoColor=white&label=image&color=2496ED)](https://hub.docker.com/r/raushanranjan/camel-sentinel/tags)
[![Image Size](https://img.shields.io/docker/image-size/raushanranjan/camel-sentinel/latest?logo=docker&logoColor=white&label=size&color=2496ED)](https://hub.docker.com/r/raushanranjan/camel-sentinel)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/raushan1107/ai-ml-cybersecurity-bank-risk-capstone?style=social)](https://github.com/raushan1107/ai-ml-cybersecurity-bank-risk-capstone/stargazers)
[![Follow Raushan Ranjan](https://img.shields.io/github/followers/raushan1107?label=Follow%20Raushan%20Ranjan&style=social)](https://github.com/raushan1107)

<a href="https://hub.docker.com/r/raushanranjan/camel-sentinel"><img alt="Get it on Docker Hub" src="https://img.shields.io/badge/🐳_Get_the_image-Docker_Hub-2496ED?style=for-the-badge"></a>
<a href="https://github.com/raushan1107/ai-ml-cybersecurity-bank-risk-capstone/stargazers"><img alt="Star this repo" src="https://img.shields.io/badge/⭐_Star-this_repo-f5c518?style=for-the-badge"></a>
<a href="https://github.com/raushan1107"><img alt="Follow on GitHub" src="https://img.shields.io/badge/👤_Follow-Raushan_Ranjan-181717?style=for-the-badge&logo=github"></a>
<a href="https://raushan1107.github.io/AI-Machine-Learning-Handbook/"><img alt="Read the handbook" src="https://img.shields.io/badge/📘_Learn-The_Handbook-7a4cc2?style=for-the-badge"></a>

**👨‍💻 Developer: [Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)** · 📘 Books/Handbook/Notes written by **Raushan Ranjan**

</div>

---

> [!TIP]
> **Found this useful?** ⭐ **Star this repository** and 👤 **follow [Raushan Ranjan](https://github.com/raushan1107)**.
> It helps other learners find the project, and it's the best way to say thanks.
> **Using it in your own project, course or talk?** Please keep the credit (see [Credits](#-credits--how-to-cite)).

## 📑 Contents

[What is this?](#-what-is-this) · [Screenshots](#-screenshots) · [Features](#-what-you-can-do) · [Quick start](#-quick-start) ·
[Chat & voice](#-optional-turn-on-chat--voice) · [Configuration](#%EF%B8%8F-configuration) · [How it works](#%EF%B8%8F-how-it-works) ·
[The model](#-the-model-in-numbers) · [Project structure](#%EF%B8%8F-project-structure) · [Documentation](#-documentation) ·
[Learning path](#-the-learning-path-behind-it) · [Security](#-security) · [Troubleshooting](#-troubleshooting) ·
[Versions](#-docker-image-versions) · [Credits](#-credits--how-to-cite)

---

## 🤔 What is this?

Banks are rated on **CAMEL(S)**: **C**apital, **A**sset quality, **M**anagement, **E**arnings, **L**iquidity (and
**S**ensitivity). Real ratings are confidential, so this project learns an **early-warning signal** from public data:
six financial ratios from **265,114 real FDIC bank-quarters (2005–2023)**, predicting whether a bank fails within two years.

It is built as a **complete, realistic AI system**, the kind learners rarely get to see end to end:

- 🧠 a trained, validated model with **SHAP, LIME and counterfactual** explanations and a fairness audit;
- 💬 an **LLM assistant** (text and voice) that explains the model but is never allowed to invent ratings;
- 🕸️ a **graph neural network** that catches fraud rings tabular models miss;
- 🛡️ a **security lab** that attacks the app itself (prompt injection, model stealing, deepfakes…) and shows the defences;
- 🚢 **one Docker image** on Docker Hub and Azure, deployable in minutes.

Every concept is taught step by step in the **[AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)**
by the same author. The handbook explains the ideas; this project shows them working together.

---

## 📸 Screenshots

| Score a bank + explanation | Interactive AI/ML roadmap |
|---|---|
| ![Score a bank](docs/images/score-a-bank.png) | ![AI/ML roadmap](docs/images/ai-ml-roadmap.png) |
| **AI Security Lab: prompt injection** | **AI Security Lab: adversarial attack on the real model** |
| ![Prompt injection lab](docs/images/security-lab-prompt-injection.png) | ![Adversarial lab](docs/images/security-lab-adversarial.png) |
| **Security Posture (high-contrast theme)** | **Deploy & Containerize (ocean theme)** |
| ![Security posture](docs/images/security-posture.png) | ![Deploy page](docs/images/deploy-and-containerize.png) |

---

## ✨ What you can do

| Page | What it does | Needs Azure keys? |
|---|---|:---:|
| 🐫 **Score a Bank** | Pick a real FDIC bank or type in ratios → failure-risk score, **SHAP** chart, **LIME** cross-check, **counterfactual** "what would make it safe?" | No |
| 💬 **Chat Assistant** | Ask about a bank in plain English, by text or **voice**. The LLM only explains the model's numbers | Yes |
| 📚 **Documentation** | Model card, fairness (RAI) audit, glossary, caveats | No |
| 🕸️ **Fraud Detection (GNN)** | Heterogeneous GraphSAGE on a synthetic transaction graph, compared with tabular baselines | No |
| 🛡️ **AI Security Lab** | STRIDE + MITRE ATLAS threat map and **10 attack → defence labs** against this app | No |
| 🗺️ **AI/ML Roadmap** | Clickable tree of **12 stages, 56 topics**, each with real-world use, analogy, maths, code and security angle | No |
| 🎓 **Build Guide** | How the whole project was built, with copy-paste AI prompts for every phase | No |
| 🚢 **Deploy & Containerize** | The Dockerfile explained line by line, registries, Azure Container Apps, Kubernetes, Trivy | No |
| 🔐 **Security Posture** | How the app protects itself, audit findings, CVE scan, and a **live self-test** | No |

Also: 🎨 **6 themes** (System, Light, Dark, **High contrast**, Sepia, Ocean) and a 🔌 **backend switch** to point the UI at any CAMEL API (local or cloud).

---

## 🚀 Quick start

Pick **one** way. **A** is the easiest: no Python needed.

### A. Run with Docker (recommended, one command)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), start it, then:

```bash
docker run -d --name camel -p 8501:8501 raushanranjan/camel-sentinel:latest
docker ps        # wait until STATUS shows (healthy), about 1–2 minutes
```

Open **<http://localhost:8501>** 🎉

```bash
docker logs -f camel      # watch it start
docker stop camel         # stop it
docker start camel        # start it again later
docker rm -f camel        # remove it
```

### B. Run with Python (to read and change the code)

**1. Get the code**
```bash
git clone https://github.com/raushan1107/ai-ml-cybersecurity-bank-risk-capstone.git
cd ai-ml-cybersecurity-bank-risk-capstone
```

**2. Create a virtual environment** (Python **3.10** recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the dependencies** (5–15 minutes)
```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**4. Start the backend and the UI**: two terminals, both in the project folder with the virtual environment active
```bash
# Terminal 1 — API (wait for "Model loaded. ... sample banks cached.")
python -m uvicorn main:app --app-dir backend --port 8000

# Terminal 2 — UI
streamlit run frontend/app.py
```

Open **<http://localhost:8501>**. API docs are at **<http://localhost:8000/docs>**.
The trained models and data are included, so there's nothing to train first.

### C. Two containers with Docker Compose (API and UI as separate services)

```bash
docker compose up -d      # UI on http://localhost:8501, API on http://localhost:8000
docker compose down
```

### D. Deploy to Azure Container Apps (the whole app in one container)

```bash
az login
az group create -n rg-camel-sentinel -l centralindia
az extension add --name containerapp --upgrade
az containerapp up -n camel-sentinel -g rg-camel-sentinel \
  --image docker.io/raushanranjan/camel-sentinel:latest \
  --ingress external --target-port 8501
az containerapp update -g rg-camel-sentinel -n camel-sentinel --cpu 2 --memory 4Gi
```

> [!IMPORTANT]
> Use **target port 8501** (the UI). Port 8000 is only the API, and in this all-in-one mode it stays inside the container.
> Give it **2 CPU / 4 GiB**. The full walkthrough, including Azure Container Registry, is in
> [docs/06_deployment_containerization.md](docs/06_deployment_containerization.md).

---

## 🔑 Optional: turn on chat & voice

Scoring, explanations, fraud detection, the security lab and the roadmap all work **without any keys**.
Only the 💬 Chat Assistant needs Azure OpenAI, and its voice mode needs Azure Speech.

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Fill in your own values in `.env`:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_API_VERSION=2025-08-07-preview
AZURE_SPEECH_KEY=<your-speech-key>
AZURE_SPEECH_REGION=<your-region>
```

Then run with it: `docker run -d --name camel -p 8501:8501 --env-file .env raushanranjan/camel-sentinel:latest`
(or restart the Python backend).

> [!CAUTION]
> Never commit `.env` or paste keys into chats. `.env` is already in `.gitignore` and is **never** baked into the Docker image.

---

## ⚙️ Configuration

| Variable | Default | What it does |
|---|---|---|
| `MODE` | `all` | Docker only: `all` = UI + API · `api` = API only · `ui` = UI only |
| `CAMEL_BACKEND_URL` | `http://localhost:8000` | Default API address the UI uses |
| `CAMEL_BACKEND_ALLOWLIST` | *(empty)* | Extra hosts the UI may switch to (built in: localhost, `api`, `*.azurecontainerapps.io`, `*.azurewebsites.net`, `*.azure-api.net`) |
| `CAMEL_LOCK_BACKEND` | `0` | `1` pins the API address and hides the sidebar switch |
| `CAMEL_API_PUBLIC` | `0` | Docker `MODE=all`: `1` exposes the API outside the container (add `-p 8000:8000`) |
| `CAMEL_CORS_ORIGINS` | `http://localhost:8501` | Browser origins allowed to call the API |

**Point the UI at another API** (for example one running in Azure): use the sidebar's **🔌 Backend API** panel, or open
`http://localhost:8501/?backend=https://<your-api>.azurecontainerapps.io`.

---

## 🏗️ How it works

```mermaid
flowchart LR
    U["👩‍💼 Browser"] --> UI["🖥️ Streamlit UI<br/>:8501"]
    UI -- "HTTP only" --> API["⚡ FastAPI<br/>:8000"]
    API --> M["🧠 CAMEL model<br/>HistGradientBoosting<br/>+ SHAP · LIME · counterfactual"]
    API --> G["🕸️ GNN fraud model<br/>GraphSAGE"]
    API --> S["🛡️ Security Lab engine"]
    API -. "optional" .-> AO["☁️ Azure OpenAI + Speech"]
    F[("🏦 FDIC BankFind API")] --> API
```

- The **UI never loads a model**. It only talks HTTP to the API, so each side can be tested, scaled and replaced independently.
- The **LLM is a translator, not a decision-maker**: risk numbers come only from the ML model through one allow-listed tool.
- One Docker image contains everything. `MODE` decides whether it runs as one container or as separate services.

---

## 📊 The model in numbers

| | |
|---|---|
| **Data** | 265,114 FDIC bank-quarters, 2005–2023 · 4,165 "failed within 2 years" examples |
| **Inputs** | 6 CAMEL ratios: Tier 1 capital, non-performing loans, efficiency, ROA, loan/deposit, loan concentration |
| **Model** | Tuned `HistGradientBoostingClassifier`, chosen from 10 compared families (incl. RandomForest, XGBoost, LightGBM, ExtraTrees, logistic regression, imbalanced-learn ensembles) |
| **Validation** | Tuned only on 2005–2011; tested on the **untouched 2012–2023** period |
| **Results** | **ROC-AUC 97.3%** · **PR-AUC 54.9%** (PR-AUC is the main metric because failures are rare) |
| **Explainability** | SHAP (global + per bank), LIME cross-check, bounded counterfactual search |
| **Responsible AI** | Subgroup audit by asset size and region; [model card](docs/model_card.md) |

> [!WARNING]
> A **teaching project**. Scores are a relative ranking from public financial-statement proxies. They are **not** official
> CAMELS supervisory ratings and **not** financial advice. The fraud-detection and some security-lab data are **synthetic**
> and clearly labelled.

---

## 🗂️ Project structure

```
ai-ml-cybersecurity-bank-risk-capstone/
├── README.md                  ← you are here
├── LICENSE                    ← MIT (keep the copyright notice = keep the credit)
├── Dockerfile                 ← one image: API + UI + models
├── docker-compose.yml         ← API and UI as two services from the same image
├── requirements.txt           ← local Python setup
├── requirements-docker.txt    ← exact pins used in the image
├── .env.example               ← template for optional Azure keys
├── backend/                   ← FastAPI: main.py, gnn_router.py, security_router.py
├── frontend/                  ← Streamlit
│   ├── app.py                 ←   router: navigation, themes, credits
│   ├── home.py                ←   🐫 Score a Bank
│   ├── views/                 ←   every other page
│   ├── roadmap_content/       ←   the 56 roadmap topic cards (easy to edit)
│   └── ui_kit.py              ←   shared helpers (themes, backend switch, safe rendering)
├── src/camel_sentinel/        ← all real logic, reusable from notebooks and the API
│   ├── data/  features/  models/        ← FDIC pull, CAMEL ratios, training
│   ├── xai/  rai/  federated/           ← explanations, fairness audit, federated learning
│   ├── agent/  chat/                    ← allow-listed tool, LLM assistant, speech
│   ├── gnn/                             ← graph neural network fraud detection
│   └── security/                        ← AI Security Lab (injection, PII, poisoning, stealing…)
├── notebooks/                 ← step-by-step walkthroughs 00 → 04
├── models/                    ← trained models (included, ~3 MB)
├── data/                      ← cached FDIC panel + synthetic GNN data (included)
├── docker/  deploy/           ← entrypoint, health check, Kubernetes manifest, Trivy results
└── docs/                      ← ALL documentation (start at docs/README.md)
```

---

## 📚 Documentation

All documents are in **[`docs/`](docs/README.md)**. Start with the **[documentation guide](docs/README.md)**.

| Start here | Then |
|---|---|
| 🚀 [Student setup guide](docs/STUDENT_SETUP.md) | 🧩 [Chat interface](docs/04_chat_interface.md) · [GNN fraud detection](docs/05_gnn_fraud_detection.md) |
| 🎯 [Problem statement](docs/00_problem_statement.md) | 🚢 [Deployment & containerization](docs/06_deployment_containerization.md) |
| 📖 [Glossary](docs/01_glossary.md) | 🛡️ [AI Security Lab](docs/07_ai_security_lab.md) · 🔐 [Security posture](docs/09_security_posture.md) |
| 🏦 [Data sources](docs/02_data_sources.md) | 🗺️ [AI/ML roadmap design](docs/08_ai_ml_roadmap.md) |
| 🧭 [Learning-path mapping](docs/03_module_mapping.md) | 📋 [Model card](docs/model_card.md) |

**Notebooks** (run `jupyter notebook notebooks/` to retrain everything yourself):
`00_dataset_research` → `01_eda` → `02_baseline_modeling` → `03_model_tuning` → `04_gnn_fraud_detection`.

---

## 🎓 The learning path behind it

This project walks the full learning path of the
**[AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)**
([GitHub](https://github.com/raushan1107/AI-Machine-Learning-Handbook)). Each stage is used where it genuinely fits:

| # | Stage | Where it lives here |
|:--:|---|---|
| 1 | Data Science & Financial Data Analysis | FDIC pipeline, CAMEL ratios, EDA, baseline + tuned models |
| 2 | Deep Neural Networks | Neural layers, backprop and Adam inside the GNN; adversarial-example ideas in the Security Lab |
| 3 | NLP & Financial Text AI | Prompt-injection detection, PII redaction |
| 4 | Generative AI, LLMs & RLHF | Chat assistant with server-side guardrails |
| 5 | RAG, LangChain & AI Agents | Allow-listed `score_bank` tool and tool-calling agent |
| 6 | Explainable & Responsible AI | SHAP, LIME, counterfactuals, fairness audit, model card |
| 7 | Federated & Privacy-Preserving ML | FedAvg simulation with clipping and noise |
| 8 | Multimodal AI | Voice input/output; deepfake-voice lab |
| 9 | Graph Neural Networks | GraphSAGE fraud-ring detection |
| 10 | MLOps & Deployment | FastAPI, Docker, Docker Hub, ACR, Azure Container Apps, Kubernetes, Trivy |
| 11 | AI Cybersecurity | AI Security Lab, STRIDE, MITRE ATLAS, Security Posture |
| 12 | Capstone | This whole project, one integrated and secured system |

---

## 🔐 Security

- The container runs as a **non-root user**, with **no secrets in the image**, pinned dependencies, and no `curl` or `pip` at run time.
- **LLM guardrails**: an injection pre-check, **PII redaction** before anything reaches the provider, an allow-listed tool with
  validated arguments, and sanitised output.
- **API hardening**: input limits, no echoed input in errors, private by default, CORS allow-list, SSRF-safe backend switch.
- **Scans**: Trivy config **27/27 pass**; image scan shows **0 Python-package CVEs** (remaining OS findings have no Debian fix yet).
- Full audit, fixes and future work: **[docs/09_security_posture.md](docs/09_security_posture.md)**, plus a live self-test on the 🔐 Security Posture page.

Found a security issue? Please open a GitHub issue **without** exploit details and ask for a private contact.

---

## 🧯 Troubleshooting

| Problem | Fix |
|---|---|
| `docker: ... dockerDesktopLinuxEngine: cannot find the file` | Start **Docker Desktop** and wait for "Engine running" |
| Container stays `(health: starting)` | Normal for 1–2 minutes while models load: `docker logs -f camel` |
| UI says it can't reach the backend | Start the API first (Terminal 1) and wait for `Model loaded` |
| Chat says "not configured" | Add your keys to `.env` and restart (see [Chat & voice](#-optional-turn-on-chat--voice)) |
| `ModuleNotFoundError` | Activate the virtual environment again (step B.2) |
| Error loading the model (scikit-learn version) | Keep `scikit-learn==1.7.2`, the version the models were saved with |
| Azure app only shows Swagger | Set the target port to **8501**, not 8000 |
| Banks are named "Synthetic Bank N" | The FDIC API was unreachable, so demo data is used |

More in the [setup guide](docs/STUDENT_SETUP.md#8-troubleshooting) and the [deployment runbook](docs/06_deployment_containerization.md).

---

## 🐳 Docker image versions

Image: **[`raushanranjan/camel-sentinel`](https://hub.docker.com/r/raushanranjan/camel-sentinel)**

| Tag | What's in it |
|---|---|
| `latest` | The newest release (recommended) |
| `v2.3` | Accurate SHAP base-rate display; matches this GitHub repository |
| `v2.2` | Switchable backend API (sidebar or `?backend=` link, allow-listed and verified) |
| `v2.1` | Security-hardened release, Security Posture page, 6 themes, grouped navigation, credits |
| `v2.0` | Deploy & Containerize page, AI Security Lab, AI/ML Roadmap |
| `v1.0` | Core app: scoring, XAI, RAI, chat, GNN, build guide |

```bash
docker pull raushanranjan/camel-sentinel:latest
```

---

## 🙌 Credits & how to cite

<div align="center">

**👨‍💻 Developer: [Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**
**📘 Books/Handbook/Notes written by Raushan Ranjan**: [AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)

</div>

This project is free to learn from and build on under the [MIT License](LICENSE). If you use it in a project, course,
article, video or talk, **please credit it**:

```text
Based on "CAMEL Sentinel" by Raushan Ranjan
https://github.com/raushan1107/ai-ml-cybersecurity-bank-risk-capstone
```

```bibtex
@software{ranjan_camel_sentinel_2026,
  author = {Ranjan, Raushan},
  title  = {CAMEL Sentinel: An AI/ML + Cybersecurity Bank-Risk Capstone},
  year   = {2026},
  url    = {https://github.com/raushan1107/ai-ml-cybersecurity-bank-risk-capstone}
}
```

Data: public FDIC BankFind Suite data. Product names mentioned in the learning material belong to their owners.

---

<div align="center">

### ⭐ If this project helped you, please star it and share it with someone learning AI/ML

<a href="https://github.com/raushan1107/ai-ml-cybersecurity-bank-risk-capstone/stargazers"><img alt="Star this repo" src="https://img.shields.io/badge/⭐_Star-this_repo-f5c518?style=for-the-badge"></a>
<a href="https://github.com/raushan1107"><img alt="Follow on GitHub" src="https://img.shields.io/badge/👤_Follow-Raushan_Ranjan-181717?style=for-the-badge&logo=github"></a>
<a href="https://hub.docker.com/r/raushanranjan/camel-sentinel"><img alt="Docker Hub" src="https://img.shields.io/badge/🐳_Pull-Docker_image-2496ED?style=for-the-badge"></a>

Made with ❤️ for AI/ML learners by **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**

</div>
