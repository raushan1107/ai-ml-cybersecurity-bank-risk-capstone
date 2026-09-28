# CAMEL(S) Rating Prediction — Project Blueprint
### Client capstone: an explainable, privacy-aware ML agent that scores bank health on the CAMEL(S) framework

---

## 0. TL;DR for the client conversation

**The one thing to say up front:** actual CAMELS composite ratings (1–5) assigned by regulators are **confidential supervisory information** — they are never published for individual banks, in the US (FDIC/Fed/OCC/NCUA) or almost anywhere else. So no dataset on Kaggle or elsewhere contains real, regulator-issued CAMEL labels. This is well documented and is *why* every academic paper on this topic builds a **proxy label** instead (usually "failed vs. not failed" or a rules-based composite score built from CAMEL-style ratios), and it's exactly the approach we'll take.

That's not a weakness — it's the standard, defensible approach used in peer-reviewed bank-failure-prediction literature, and it gives us a much better story for XAI/RAI (we can be fully transparent about what the label means) than pretending we have ground-truth ratings we don't.

**Recommended path:** build on **real FDIC bank financial data** (free public API, US, quarterly, since 1992, plus a full bank-failure history since the 1930s) to construct CAMEL ratios and a proxy rating, with a Kaggle ratio→rating dataset as a fast local sandbox while the FDIC pull is being engineered.

---

## 1. Problem framing

**Client ask (as I understand it):**
- An "agent" that predicts an outcome/rating grounded in the CAMEL(S) rating methodology, from a bank's financials.
- Must demonstrate: EDA → feature engineering → regression/classification/clustering/anomaly detection → **Explainable AI (XAI)** → **Responsible AI (RAI)** → **federated / privacy-preserving ML**, using the learnings from our [AI/ML Handbook](https://raushan1107.github.io/AI-Machine-Learning-Handbook/) — but not mechanically walking every module; solve the real problem, pull in what's needed.
- Not mandatory to be a strict re-run of the curriculum — this is a real deliverable, curriculum is the toolbox, not the spec.

**Restated as a solvable ML problem:**
> Given a bank's periodic financial statement data (balance sheet + income statement ratios), predict a CAMEL-style composite health rating / risk bucket, explain *why* the model said what it said (XAI), audit the model for fairness and document it for governance (RAI), and demonstrate that the same pipeline can be trained in a privacy-preserving / federated way across data that different institutions won't pool directly (banks, regulators, or business units won't share raw balance-sheet data with each other).

---

## 2. Dataset research (deep dive)

### 2.1 Why there's no direct "CAMELS-labelled" dataset
- CAMELS ratings are assigned via on-site supervisory exams plus ratio analysis, and are **shared only with the institution's top management and its regulator** — this is explicit in the FFIEC/FDIC framework (UFIRS, adopted 1979, CAMELS since ~1996 when "Sensitivity" was added) — precisely to avoid triggering bank runs on downgraded banks.
- Academic bank-failure-prediction papers (Gaul, OCC working paper; the 2024 counterfactual-explanation paper on arXiv; Turkish/Indonesian CAMEL studies) all confirm this and instead use **failure/non-failure** or a **hand-built composite ratio score** as the target.

### 2.2 Path A — Real bank financial data (recommended primary source)
**FDIC BankFind Suite API** — free, public, no mandatory API key for basic use (a free key is recommended for higher rate limits):
- Institutions endpoint — bank identity/metadata (charter type, state, asset size, holding company flag)
- **Financials endpoint** — quarterly Call Report data since **1992**, all the raw fields needed to build CAMEL ratios (equity, Tier 1 capital, RWA, NPLs, provisions, net income, interest income/expense, deposits, loans, liquid assets, etc.)
- **Failures endpoint** — the full catalog of FDIC-insured bank failures **back to the 1930s** — this is our proxy label source (bank failed within N quarters = proxy for a "4/5" rating; healthy survivors = proxy for "1/2")
- Bulk data downloads and an interactive report builder are also available if a pure `requests`/API pull is inconvenient.
- Docs: `https://banks.data.fdic.gov/bankfind-suite/bulkdata`, `https://api.fdic.gov/banks/docs`
- No network/API-key setup cost, well documented, quarterly granularity, decades of history, thousands of institutions — this is the credible, "real-world" backbone for the whole build.

*Caveat to plan for*: severe class imbalance (a few hundred failures against tens of thousands of bank-quarters), and reporting lag (quarterly, ~45–75 days after quarter-end).

### 2.3 Path B — Kaggle datasets to use as fast local sandboxes / structural templates
Use these **while the FDIC pipeline is being engineered**, or as an additional generalization check — they won't replace Path A, but they let Module-1-style EDA/feature-engineering/modeling work start on day one without waiting on any external pull:

| Dataset | Kaggle link | Why it's useful here |
|---|---|---|
| Corporate Credit Rating with Financial Ratios | `kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios` | Ratio → rating supervised structure — near-identical shape to what we need (financial ratios in, an ordinal rating label out) |
| Corporate Credit Rating | `kaggle.com/datasets/agewerc/corporate-credit-rating` | Same idea, different companies/ratios — good for a second opinion / robustness check on the ratio→rating classifier design |
| Predict Bankruptcy in Poland (Tomczak / Zięba et al., EMIS 2000–2013) | `kaggle.com/datasets/stealthtechnologies/predict-bankruptcy-in-poland` | The most-cited proxy dataset in CAMEL-style bankruptcy literature — 64 accounting ratios + bankrupt/not label, heavily imbalanced (mirrors real bank-failure imbalance), good for prototyping the classification + anomaly-detection + imbalance-handling steps before pointing the same code at FDIC data |
| Banking Dataset Classification (term-deposit marketing) | `kaggle.com/datasets/rashmiranu/banking-dataset-classification` | Not CAMEL-relevant directly — useful only if the client also wants a customer-level cross-sell/marketing angle later; otherwise skip |

### 2.4 Path C — Published academic CAMEL(S) datasets (benchmarking only)
Smaller, manually compiled from paper appendices — good for sanity-checking our ratio construction and composite-scoring formula, not as a primary training set:
- Turkish commercial banks (Bank Association of Turkey, 1994–2004; 44 active + 21 bankrupt) — 20 CAMELS predictors
- Indonesian rural banks (OJK, 2013–2019; 43 bankrupt + 43 active) — 5 CAMEL ratios
- US FDIC-insured banks 2008–2023 (`fdicdata` R package wrapper around the same FDIC API) used in the 2024 counterfactual-explanations paper, with an explicit table of CAMELS predictor ratios (TICRC, PLLL, TIE, ROA, LDR, etc.) — this table is the best available public "recipe" for turning Call Report fields into CAMEL ratios and is worth mirroring almost exactly.

### 2.5 Recommendation
Use **FDIC BankFind (Path A)** as the dataset of record for the client deliverable — real, large, longitudinal, well-documented, free, and it lets us honestly say "trained on real US bank supervisory financial data" rather than a synthetic proxy. Use the **Polish bankruptcy dataset and the two Kaggle rating datasets (Path B)** as an immediate, zero-setup sandbox for building and unit-testing the modeling pipeline this week, and the **academic CAMELS ratio tables (Path C)** as the reference recipe for feature engineering. If the client later wants an India-specific angle, RBI publishes aggregate bank-level financial parameters (Statistical Tables Relating to Banks of India) that can extend the same pipeline with a second data source — flagged as a fast-follow, not part of the MVP.

---

## 3. CAMEL(S) → feature map (the "recipe")

| Component | What it measures | Example ratios (from FDIC Call Report fields) |
|---|---|---|
| **C**apital adequacy | Buffer against losses | Tier 1 capital / risk-weighted assets; equity / total assets; leverage ratio |
| **A**sset quality | Risk in the loan/investment book | Non-performing loans / total loans; loan-loss provisions / total loans; charge-offs / average loans |
| **M**anagement quality | Proxy — regulators observe this directly, we can't | Efficiency ratio (non-interest expense / revenue); asset growth rate; loan growth rate (as a proxy for risk appetite) |
| **E**arnings | Profitability & sustainability | Return on assets (ROA); return on equity (ROE); net interest margin |
| **L**iquidity | Ability to meet obligations | Loans / deposits; liquid assets / total assets; core deposits / total assets |
| **S**ensitivity to market risk | Exposure to rate/market moves | Securities / total assets; short-term vs. long-term asset mix; uninsured deposit ratio |

"Management" has no direct financial-statement proxy (it's a supervisory judgment call from on-site exams) — this is an important, honest caveat to put in the model card: our model approximates M via efficiency/growth ratios, which is standard practice in the literature, but is explicitly the weakest of the six components.

**Composite proxy label**, two options, use both and compare:
1. **Rule-based composite score** (weighted z-scores of the 6 components, literature-standard weights, bucketed into 1–5) → gives a continuous target for **regression** and an ordinal target for **classification**.
2. **Failure-within-N-quarters** (binary, from the FDIC Failures endpoint) → an independent, harder ground truth for validating that the rule-based composite actually tracks real distress.

---

## 4. Solution architecture — "CAMEL Sentinel"

```
┌─────────────┐   ┌───────────────────┐   ┌────────────────────┐   ┌──────────────┐
│ FDIC Financ. │──▶│ Feature Store      │──▶│ Modeling Layer      │──▶│ XAI Layer     │
│ + Failures   │   │ (CAMEL ratios per │   │ - composite score   │   │ SHAP / LIME / │
│ + Institution│   │  bank-quarter)     │   │   (regression)      │   │ counterfactual│
└─────────────┘   └───────────────────┘   │ - rating bucket      │   └──────┬───────┘
                                          │   (classification)  │          │
                                          │ - peer clustering    │          ▼
                                          │ - anomaly detection  │   ┌──────────────┐
                                          └──────────┬───────────┘   │ RAI Layer     │
                                                     │               │ fairness audit│
                                                     ▼               │ model card    │
                                          ┌──────────────────────┐   └──────┬───────┘
                                          │ Federated / DP Layer │          │
                                          │ FedAvg across bank    │◀────────┘
                                          │ charter-type/region   │
                                          │ silos + DP noise      │
                                          └──────────┬───────────┘
                                                     ▼
                                          ┌──────────────────────┐
                                          │ Agent Layer           │
                                          │ tool-calling wrapper: │
                                          │ fetch → score →        │
                                          │ explain → flag risk    │
                                          └──────────────────────┘
```

- **Data layer**: scheduled pull from FDIC Financials/Failures/Institutions endpoints → versioned feature store keyed by `(CERT, quarter)`.
- **Modeling layer**: gradient-boosted trees (XGBoost/LightGBM) as the primary model (tabular, imbalanced, needs to pair well with SHAP); logistic/linear regression as an interpretable baseline; KMeans/HDBSCAN for peer-group clustering; Isolation Forest / autoencoder for anomaly detection (banks whose ratios look nothing like their peer group — an early-warning signal independent of the rating model).
- **XAI layer**: SHAP for global feature importance + per-bank waterfall explanations; LIME as a cross-check; counterfactual explanations ("what would this bank need to change to move from a 3 to a 2") — directly mirrors the 2024 arXiv paper's approach and is a strong, concrete client-facing feature.
- **RAI layer**: fairness audit — does the model systematically score small community banks or a particular region worse *for reasons other than genuine risk*? Model card documenting the proxy-label caveat, known blind spot (Management), and intended use ("decision-support / early-warning, not a replacement for supervisory judgment").
- **Federated / privacy-preserving layer**: simulate the realistic client scenario — multiple banks (or business units, or regional regulators) that won't pool raw financial data. Partition the FDIC data by charter type or region as separate "clients," train via **Federated Averaging (Flower or a hand-rolled FedAvg loop)**, and add **differential privacy** (`diffprivlib` or `Opacus`) on the shared updates. This is presented as "how the same model could be trained across institutions without centralizing sensitive financial data," not as a strict requirement to reproduce every curriculum sub-topic.
- **Agent layer**: a thin tool-calling agent (LangChain/LangGraph, matching Module 5) that: (1) looks up or accepts a bank's latest financials, (2) computes CAMEL ratios, (3) returns predicted rating + composite score + SHAP-based explanation + any fairness/anomaly flags, in natural language plus structured JSON.

---

## 5. Step-by-step build plan

| Phase | Focus | Key tasks | Deliverable | Est. effort |
|---|---|---|---|---|
| 0 | Problem framing & data contract | Confirm proxy-label approach with client in writing; define what "rating" means in the deliverable | 1-page data contract / assumptions doc | 0.5 day |
| 1 | Data acquisition & EDA | Pull FDIC Financials + Failures + Institutions; sandbox on Polish/Kaggle ratio datasets in parallel; missingness audit, failure-rate distribution, time coverage checks | `module0_camel_dataset_research.ipynb` (this deliverable) | 1–2 days |
| 2 | Feature engineering | Build the 6 CAMEL ratio groups per bank-quarter; build the rule-based composite score; build the failure-within-N-quarters label | Feature store + labeling notebook | 1 day |
| 3 | Baseline modeling | Regression (composite score), classification (rating bucket), peer clustering, anomaly detection | Baseline model notebook + metrics report | 2 days |
| 4 | Model selection & tuning | XGBoost/LightGBM with proper imbalance handling (class weights / focal loss / SMOTE-variant), cross-validated on time (no look-ahead) | Tuned model + validation report | 1 day |
| 5 | XAI | SHAP global + local explanations, LIME cross-check, counterfactual generator | Explainability notebook + example client-facing explanation output | 1 day |
| 6 | RAI | Fairness audit across bank-size/region/charter-type groups, bias mitigation if needed, model card | Model card + fairness report | 1 day |
| 7 | Federated / privacy-preserving simulation | Partition data as simulated institutions, FedAvg training loop, DP noise, compare accuracy/privacy trade-off vs. centralized model | Federated learning notebook + write-up | 1–2 days |
| 8 | Agent wrapper | Tool-calling agent exposing "score this bank" as a callable action, with explanation + flags in the response | Working agent demo | 1–2 days |
| 9 | Validation & handoff | End-to-end test, documentation, limitations/caveats section, handoff walkthrough | Final report + slide summary for client | 1 day |

Total: roughly **2–2.5 weeks** of focused build time for an MVP that genuinely covers dataset → model → XAI → RAI → federated learning → agent, without force-fitting every curriculum topic.

---

## 6. Risks & caveats to put in writing for the client (do this before Phase 1 is "done")

1. **No public ground-truth CAMELS scores exist anywhere** — every claim of "predicting the CAMEL rating" must be qualified as predicting a **transparent, documented proxy** for it. This protects both us and the client from over-claiming regulatory-grade accuracy.
2. **Severe class imbalance** on the failure label (failures are a small fraction of bank-quarters) — accuracy alone will look great and mean nothing; report precision/recall/F1/AUC on the minority class, and consider cost-sensitive framing (false negatives = missed distress = more expensive than false positives).
3. **"Management" component has no clean financial-statement proxy** — call this out explicitly rather than quietly substituting an efficiency ratio and calling it done.
4. **Reporting lag** — Call Report data lands 45–75 days after quarter-end, so any "real-time" framing of the agent needs a caveat about data freshness.
5. **Position the deliverable as decision-support / early-warning**, not a replacement for supervisory judgment — this is also the responsible-AI framing that keeps the RAI section honest rather than a compliance checkbox.

---

## 7. Immediate next actions

1. Confirm with the client that the proxy-label framing (Section 0 / Section 6.1) is acceptable before any modeling work is presented as "the CAMEL rating."
2. Run `camel_rating_module0_dataset_research.ipynb` (companion notebook) locally — it pulls a starter sample from the FDIC API, builds the first-pass CAMEL ratio table, and does initial EDA. (Built and documented here; not executed in this sandbox since the FDIC domain isn't reachable from this container's network — run it from your own machine or Koenig VM.)
3. Stand up the Kaggle sandbox datasets (Section 2.3) in parallel so Phase 3 baseline modeling isn't blocked on the FDIC pull finishing.
4. Decide FedAvg tooling: Flower (more production-shaped) vs. a hand-rolled loop (faster to build, easier to explain to the client in plain language) — recommend starting hand-rolled for the MVP, matching the "trusted foundation first" philosophy already used in Module 1.

---

## 8. Key sources consulted

- FDIC BankFind Suite — bulk data & API docs: `banks.data.fdic.gov/bankfind-suite/bulkdata`, `api.fdic.gov/banks/docs`
- Wikipedia — CAMELS rating system (history, confidentiality, six components)
- CFI — CAMELS Rating System overview
- Gaul, L. — "CAMELS Ratings and Their Information Content" (OCC working paper)
- arXiv 2407.11089 — "Explainable bank failure prediction models: Counterfactual explanations to reduce the failure risk" (FDIC 2008–2023 data, CAMELS predictor tables)
- arXiv 2510.06852 — "Enhancing Bankruptcy Prediction of Banks through Advanced Machine Learning Techniques" (Turkish + Indonesian CAMEL ratio tables)
- arXiv 2512.05148 — CAMEL literature review (Turkish banks)
- Kaggle: `kirtandelwadia/corporate-credit-rating-with-financial-ratios`, `agewerc/corporate-credit-rating`, `stealthtechnologies/predict-bankruptcy-in-poland`, `rashmiranu/banking-dataset-classification`
- Tomczak / Zięba et al. (2016) — Polish companies bankruptcy dataset methodology
