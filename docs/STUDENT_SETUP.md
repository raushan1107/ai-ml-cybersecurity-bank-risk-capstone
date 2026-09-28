# CAMEL Sentinel — Student Setup Guide

Get the app running on your own computer in about 15 minutes.
Everything you need is in the repository, including the **already-trained models**,
so you do not have to train anything before trying the app.

---

## What you get

| Feature | Where | Needs Azure keys? |
|---|---|---|
| Score a bank's failure risk from 6 CAMEL ratios | Main page | No |
| "Why?" — SHAP chart, LIME cross-check, counterfactual what-if | Main page | No |
| Model card, fairness (RAI) audit, glossary | 📚 Documentation | No |
| GNN fraud detection on a **synthetic** transaction graph | 🕸️ Fraud Detection | No |
| Step-by-step build walkthrough | 🎓 Build Guide | No |
| Chat with an AI assistant about banks and the model | 💬 Chat Assistant | Yes (Azure OpenAI) |
| Talk to the assistant by voice | 💬 Chat Assistant | Yes (Azure Speech) |
| Package, publish and deploy the app with Docker | 🚢 Deploy & Containerize | No |
| Attack the app and watch the defences hold | 🛡️ AI Security Lab | No (optional live chat test needs Azure OpenAI) |
| Explore the whole AI/ML learning path as a clickable tree | 🗺️ AI/ML Roadmap | No |
| See how the app protects itself and run a live security self-test | 🔐 Security Posture | No |

> 📘 **Learn the fundamentals first:** every concept used here is taught step by step in the
> [AI/ML + Cybersecurity Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/)
> ([GitHub](https://github.com/raushan1107/AI-Machine-Learning-Handbook)) by **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**,
> the developer of this project.

### Fastest start: Docker (skip sections 1–6)

If you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running:

```bash
docker run -d --name camel -p 8501:8501 raushanranjan/camel-sentinel:latest
docker ps          # wait for (healthy), then open http://localhost:8501
```

Add `--env-file .env` to the command if you want the Chat Assistant (see section 5).

> **Important:** risk scores are a teaching model's output, not an official
> regulatory CAMELS rating. The fraud-detection module uses entirely
> **synthetic (made-up) data**. Read the caveats on the Documentation page.

---

## 1. Requirements

- **Python 3.10, 3.11 or 3.12** (tested on 3.10). Check with `python --version`.
- About **3 GB of free disk space** (PyTorch is large).
- Internet access during install. The app also downloads current bank
  data from the public FDIC API when it starts. If that fails, it falls
  back to a **synthetic** sample (banks named "Synthetic Bank N").

## 2. Get the code and open a terminal in the project folder

```bash
git clone https://github.com/raushan1107/ai-ml-cybersecurity-bank-risk-capstone.git
cd ai-ml-cybersecurity-bank-risk-capstone
```

No git? On the GitHub page click **Code → Download ZIP**, unzip it, and open a terminal in that folder.

## 3. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your prompt.

## 4. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The first line installs the smaller, CPU-only build of PyTorch. The whole
install takes 5–15 minutes, depending on your internet speed.

## 5. (Optional) Add Azure keys for the Chat Assistant

Skip this step if you only want scoring, explanations and fraud detection.

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Open `.env` in any text editor and replace each `<...>` placeholder with
your own values. The comments in the file say where to find each one in the
Azure portal. **Never share your `.env` file.**

## 6. Run the app (two terminals)

Both terminals must be in the project folder with the virtual environment
activated (step 3).

**Terminal 1: backend API**
```bash
python -m uvicorn main:app --app-dir backend --port 8000
```
Wait until you see `Model loaded. ... sample banks cached.`. The first start
can take about a minute.

**Terminal 2: user interface**
```bash
streamlit run frontend/app.py
```

Your browser should open **http://localhost:8501**. If it doesn't, open
that address yourself. Interactive API docs are at
**http://localhost:8000/docs**.

To stop either one, press `Ctrl+C` in its terminal.

---

## 7. Things to try

1. **Main page**: choose *Pick a real bank*, then **Score this bank**.
   Drag *Tier 1 capital ratio* down or *NPL ratio* up, score again, and
   see how the SHAP bars change.
2. Open the **Counterfactual** section to see which ratio changes would
   lower the risk score.
3. Switch **Scoring model** in the sidebar (tuned vs baseline) and compare
   their scores for the same bank.
4. **🕸️ Fraud Detection**: run the three demo transactions, then build
   your own using the presets.
5. **📚 Documentation → Important caveats**: read why this is a *proxy*
   score and what the model cannot do.
6. **🛡️ AI Security Lab**: launch a prompt injection, then an adversarial attack on
   the real model, and watch the defences hold.
7. **🗺️ AI/ML Roadmap**: open a stage, click a topic, and switch between the
   *Real world*, *Analogy*, *Math* and *Code* tabs.
8. **🔐 Security Posture → Live self-test**: run 9 real attacks against your backend.
9. Try the **🎨 Theme** picker in the sidebar (High contrast, Sepia, Ocean…).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "Can't reach the backend" | Start Terminal 1 first and wait for `Model loaded`. |
| `Address already in use` / port 8000 busy | Stop the other process, or start the backend with `--port 8010` and enter `http://localhost:8010` in the sidebar's **🔌 Backend API** panel. |
| Fraud demo says "Demo cache is still building" | Normal for about 60 seconds after the backend starts. Try again. |
| Chat says Azure OpenAI is "not configured" | Complete step 5, then **restart** the backend. |
| Voice returns a server error | Your Speech key or region is wrong. The region must match your Azure Speech resource exactly. |
| First score takes about 30 seconds | Normal: the explainers warm up on the first request. |
| `ModuleNotFoundError` | Your virtual environment isn't active. Repeat the activate command from step 3. |
| Banks are named "Synthetic Bank N" | The FDIC API couldn't be reached, so the app is using generated demo data. |

## 9. Want to retrain the models yourself?

Run the notebooks in order (`pip install jupyter` is already covered by requirements):

```bash
jupyter notebook notebooks/
```

| Notebook | Produces |
|---|---|
| `00_dataset_research.ipynb`, `01_eda.ipynb` | Data pull and exploration |
| `02_baseline_modeling.ipynb` | `models/camel_baseline_model.joblib` |
| `03_model_tuning.ipynb` | `models/camel_tuned_model.joblib`, `data/processed/camel_panel.parquet` (downloads about 265k bank-quarters from FDIC; can take about an hour) |
| `04_gnn_fraud_detection.ipynb` | `models/gnn_fraud_model.pt`, `gnn_fraud_meta.pkl`, `gnn_baselines.pkl`, `data/gnn/*.parquet` |

## 10. Where things live

```
backend/            FastAPI server (main.py, gnn_router.py, security_router.py)
frontend/           Streamlit app (app.py router + home.py + views/)
src/camel_sentinel/ All real logic: data, features, models, xai, rai,
                    federated, agent, chat, gnn, security
docs/               Problem statement, glossary, data sources, design docs, model card
Dockerfile          Container recipe (+ docker-compose.yml, docker/, deploy/k8s/)
frontend/roadmap_content/  The AI/ML Roadmap's topic cards (plain Python, easy to edit)
notebooks/          Numbered, step-by-step walkthroughs
models/             Pre-trained model files
data/               Cached bank panel + synthetic GNN transaction data
```

Start with [00_problem_statement.md](00_problem_statement.md) and [01_glossary.md](01_glossary.md), or see the [documentation guide](README.md).
