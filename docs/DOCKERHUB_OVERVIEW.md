# 🐫 CAMEL Sentinel: Explainable, Secured AI for Bank Risk

**One container, a complete AI/ML + cybersecurity learning path.** CAMEL Sentinel scores a bank's
financial health from six CAMEL ratios, explains every prediction, and shows how to deploy, attack and
defend a real AI system, all from your browser.

> ⭐ **If this helps you learn or teach, please star this repository** (the ☆ button at the top of this page).
> It helps others find it, and it's the best way to say thanks.

👨‍💻 **Developer:** [Raushan Ranjan](https://raushan-ranjan.azurewebsites.net) · 📘 **Books/Handbook/Notes written by Raushan Ranjan**:
[AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/) ·
[GitHub](https://github.com/raushan1107/AI-Machine-Learning-Handbook) (⭐ it too!)

---

## 🚀 Quick start (30 seconds)

```bash
docker run -d --name camel -p 8501:8501 raushanranjan/camel-sentinel:latest
docker ps        # wait until STATUS shows (healthy), about 1–2 minutes
```

Then open **http://localhost:8501** 🎉

Everything works without any keys. To enable the AI **Chat Assistant** (Azure OpenAI) and **voice** (Azure Speech),
pass your own keys *at run time*; they are never baked into the image:

```bash
docker run -d --name camel -p 8501:8501 --env-file .env raushanranjan/camel-sentinel:latest
```

```env
# .env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_API_VERSION=2025-08-07-preview
AZURE_SPEECH_KEY=<your-speech-key>
AZURE_SPEECH_REGION=<your-region>
```

## 🧭 What's inside

| Page | What you can do |
|---|---|
| 🐫 **Score a Bank** | Pick a real FDIC bank or enter ratios → failure-risk score with **SHAP**, **LIME** and a **counterfactual** "what would make it safe?" |
| 💬 **Chat Assistant** | Ask about a bank in plain English (text or voice). The LLM only explains the ML model's numbers; it never invents ratings |
| 📚 **Documentation** | Model card, fairness (RAI) audit, glossary |
| 🕸️ **Fraud Detection (GNN)** | Heterogeneous GraphSAGE catches fraud rings that tabular models miss (synthetic data) |
| 🛡️ **AI Security Lab** | 10 live attack → defence labs: prompt injection, PII leakage, tool hijacking, adversarial evasion, data poisoning, model stealing, deepfake voice, AI SOC, supply chain, with a STRIDE + MITRE ATLAS threat map |
| 🔐 **Security Posture** | How the app protects itself, audit findings, CVE scan, a **live self-test** you can run, and future work |
| 🗺️ **AI/ML Roadmap** | Interactive tree of 12 stages and 56 topics, each with real-world uses, an analogy, the maths, code and a security angle |
| 🎓 **Build Guide** | How the whole project was built, with copy-paste AI prompts |
| 🎨 **Themes** | Six themes in the sidebar: System, Light, Dark, High contrast, Sepia (reading), Ocean |
| 🚢 **Deploy & Containerize** | This image explained: Dockerfile line by line, layers, registries, Azure Container Apps, Kubernetes, Trivy |

**Stack:** Python 3.10 · FastAPI · Streamlit · scikit-learn (HistGradientBoosting) · SHAP · LIME · PyTorch Geometric · Azure OpenAI · Azure Speech.
**Data:** 265,114 real FDIC bank-quarters (2005–2023); the fraud and security-lab data are clearly-labelled synthetic sets.

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODE` | `all` | `all` = UI + API in one container · `api` = FastAPI only · `ui` = Streamlit only |
| `CAMEL_BACKEND_URL` | `http://localhost:8000` | Default API address for the UI (use `http://api:8000` with compose) |
| `CAMEL_BACKEND_ALLOWLIST` | empty | Extra host patterns the UI may switch to (built in: localhost, `api`, `*.azurecontainerapps.io`, `*.azurewebsites.net`, `*.azure-api.net`) |
| `CAMEL_LOCK_BACKEND` | `0` | `1` pins the API address and hides the sidebar switch (production) |
| `CAMEL_API_PUBLIC` | `0` | In `all` mode the API is private to the container. Set `1` (and add `-p 8000:8000`) to explore `http://localhost:8000/docs` |
| `CAMEL_CORS_ORIGINS` | `http://localhost:8501` | Allowed browser origins for the API |

**Ports:** `8501` (web UI) · `8000` (REST API, private by default).

### Two containers with docker compose

```yaml
services:
  api:
    image: raushanranjan/camel-sentinel:latest
    environment: { MODE: api }
  ui:
    image: raushanranjan/camel-sentinel:latest
    environment: { MODE: ui, CAMEL_BACKEND_URL: "http://api:8000" }
    ports: ["8501:8501"]
    depends_on: { api: { condition: service_healthy } }
```

### Azure Container Apps: the whole app in one go

Expose **port 8501** (the UI). The API runs in the same container and stays private. Give it **2 CPU / 4 GiB**.

```bash
az containerapp up -n camel-sentinel -g <resource-group> \
  --image docker.io/raushanranjan/camel-sentinel:latest --ingress external --target-port 8501
# then: az containerapp update -g <resource-group> -n camel-sentinel --cpu 2 --memory 4Gi
```

Exposing port 8000 shows only the API; from v2.1 a `MODE=all` app's API listens only inside the container.

### Test a cloud API from a local UI

```bash
# API-only app in Azure: MODE=api, target port 8000 (Swagger at /docs)
docker run -d -p 8501:8501 -e MODE=ui raushanranjan/camel-sentinel:latest
# open http://localhost:8501/?backend=https://<api-app>.azurecontainerapps.io
```

Or use the sidebar **🔌 Backend API** panel. It switches every page, accepts only allow-listed hosts that answer `/health` like a
CAMEL API, and can be pinned with `CAMEL_LOCK_BACKEND=1`.

## 🏷️ Tags

| Tag | Contents |
|---|---|
| `latest` | Newest release (recommended) = `v2.2` |
| `v2.2` | Switchable backend API (sidebar or `?backend=` link, allow-listed + verified); one-go Azure Container Apps guide |
| `v2.1` | Security-hardened release: SSRF fix, guarded chat, input limits, upgraded web stack, credits, grouped navigation, Security Posture page |
| `v2.0` | Adds Deploy & Containerize, AI Security Lab and AI/ML Roadmap |
| `v1.0` | Core app: scoring, XAI, RAI, chat, GNN, build guide |

## 🔐 Security

- Runs as a **non-root** user (uid 10001), with a health check (the API must report `model_loaded=true`)
- **No secrets inside the image**: keys are injected at run time only
- Trivy config scan: **27/27 checks pass**; image CVE scan: **all Python-package CVEs fixed**, no `curl` or `pip` in the image
- LLM guardrails: injection pre-check, PII redaction before the provider, allow-listed tool with validated arguments, sanitised output

## ⚠️ Disclaimer

A **teaching project**. Risk scores are a model's relative ranking from public financial-statement proxies. They are **not**
official CAMELS supervisory ratings and not financial advice.

---

## 📜 Version history: what each version contains

**v2.2 · switchable backend** (latest)
- 🔌 A **Backend API** panel in the sidebar points the whole UI at any CAMEL Sentinel API (local, compose, or Azure Container Apps)
- 🔗 One-click links: `http://localhost:8501/?backend=https://<api-app>.azurecontainerapps.io`
- 🛡️ SSRF-safe: http(s) only, host allow-list (`CAMEL_BACKEND_ALLOWLIST`), must answer `/health` like a CAMEL API, no credentials in URLs; `CAMEL_LOCK_BACKEND=1` pins it
- ☁️ Deploy page and runbook: the whole app in one Container App from ACR (port 8501, 2 CPU / 4 GiB), plus the API-only + local-UI test path

**v2.1 · security-hardened + new UI**
- 🔐 **Security Posture** page: assets, controls by layer, audit findings, CVE scan, a **live self-test** (9 attacks) and future scope
- Audit fixes: SSRF lock, validated agent tool calls, sanitised LLM output, PII redaction + injection pre-check in the real chat, input limits, 422s never echo input, private API by default, CORS allow-list, generic errors
- Web stack upgraded (FastAPI 0.141, Starlette 1.7, python-multipart 0.0.32); `curl` and `pip` removed → **0 Python-package CVEs** (was 9)
- 🧭 Grouped sidebar navigation · 🎨 6 themes (System, Light, Dark, High contrast, Sepia, Ocean) · credits on every page

**v2.0 · deploy, attack, learn**
- 🚢 **Deploy & Containerize**: Dockerfile line by line, layer-cache simulator, manual + AI-prompt steps, Docker Hub + ACR, Container Apps, compose, Kubernetes, Trivy (27/27)
- 🛡️ **AI Security Lab**: STRIDE + MITRE ATLAS threat map and 10 attack → defence labs against the app itself
- 🗺️ **AI/ML Roadmap**: an interactive tree of 12 stages and 56 topic cards (real-world use, analogy, maths, code, security angle)
- Smaller image (`COPY --chown`), mode-aware health check, one image with three modes (`all` / `api` / `ui`)

**v1.0 · the core app**
- 🐫 Bank risk scoring on 265k real FDIC bank-quarters with **SHAP**, **LIME** and **counterfactual** explanations
- 📚 Model card + fairness (RAI) audit · 💬 LLM chat assistant with voice (Azure OpenAI + Speech) · 🕸️ GNN fraud-ring detection · 🎓 Build Guide

---

⭐ **Star this repo** if you found it useful · 👨‍💻 Developer: [Raushan Ranjan](https://raushan-ranjan.azurewebsites.net) · 📘 Books/Handbook/Notes written by [Raushan Ranjan](https://raushan-ranjan.azurewebsites.net):
[AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/).
