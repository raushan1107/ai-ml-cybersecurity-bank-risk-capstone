"""
frontend/views/1_📚_Documentation.py — the "how this works" page.

WHAT this file does: a second Streamlit page (Streamlit auto-discovers
anything under frontend/views/ and adds it to the sidebar nav next to
app.py) that explains the whole project in plain language — problem,
data, method, results, folder structure, and which file holds which
piece of logic — for a reader who has never opened the codebase.

WHY a separate page instead of an expander on the scorer: the scorer
page (app.py) is for *using* the model; this page is for *understanding*
it — different audiences, different reading speed, so they get different
screens instead of being crammed onto one.

HOW it connects: purely descriptive — it doesn't call the backend for
anything the scorer page needs, so it renders even if the backend is
down. It does try to fetch a live model card for the "current numbers"
section, but falls back to the hand-written summary below if that fails.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from ui_kit import backend_url  # noqa: E402

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="CAMEL Sentinel — Documentation", page_icon="\U0001F4DA", layout="centered")

st.title("\U0001F4DA CAMEL Sentinel — How This Works")
st.caption("A plain-language walkthrough of the problem, the data, the method, and where every piece of logic lives in the codebase.")

st.info("**Developer: [Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**", icon="\U0001F468‍\U0001F4BB")

# ---------------------------------------------------------------------------
st.subheader("Choose a learning view")
overview_tab, abbreviations_tab = st.tabs(["Project walkthrough", "Abbreviations Explained"])

with abbreviations_tab:
    st.markdown(
        "Use this tab as a human-readable field guide. The full source glossary is "
        "maintained in `docs/01_glossary.md` so it remains available outside the UI."
    )
    glossary_rows = [
        ("AI", "Artificial Intelligence", "The broad field of building systems that perform tasks usually requiring human reasoning, such as prediction or explanation."),
        ("ML", "Machine Learning", "A method within AI where a model learns patterns from examples instead of being given every rule explicitly."),
        ("CAMELS", "Capital, Asset quality, Management, Earnings, Liquidity, Sensitivity", "The six-part bank-supervision framework. This project uses public financial ratios as proxies, not confidential regulator ratings."),
        ("FDIC", "Federal Deposit Insurance Corporation", "The U.S. agency whose BankFind data supplies institution records, quarterly financials, and historical failures."),
        ("EDA", "Exploratory Data Analysis", "Inspecting distributions, missing values, outliers, correlations, and class balance before training."),
        ("ROC-AUC", "Receiver Operating Characteristic - Area Under the Curve", "How well scores rank positives above negatives across every classification threshold. 0.5 is random ranking; 1.0 is perfect ranking."),
        ("PR-AUC", "Precision-Recall Area Under the Curve", "How well a model maintains precision while finding positives across thresholds. It is more informative here because failures are rare."),
        ("Precision", "Positive Predictive Value", "Among banks flagged as likely to fail, the fraction that actually fail: TP / (TP + FP)."),
        ("Recall", "True Positive Rate / Sensitivity", "Among banks that actually fail, the fraction the model flags: TP / (TP + FN)."),
        ("F1", "F1 score", "The harmonic mean of precision and recall: 2 * precision * recall / (precision + recall). It is high only when both are useful."),
        ("TP / FP / TN / FN", "True/False Positive/Negative", "The four cells of a confusion matrix: correct or incorrect failure flags and correct or incorrect healthy predictions."),
        ("CV", "Cross-Validation", "Repeated train/test splits used to estimate how a model generalizes. Our grouped CV keeps one bank from appearing in both sides."),
        ("No-look-ahead", "Time-based validation", "Training only on earlier dates and testing on later dates so future information cannot influence model selection."),
        ("XAI", "Explainable Artificial Intelligence", "Methods that show why a model produced an individual prediction or which features matter globally."),
        ("SHAP", "SHapley Additive exPlanations", "A game-theoretic explanation method that allocates a prediction's change from a baseline among input features."),
        ("RAI", "Responsible Artificial Intelligence", "The broader practice of checking appropriate use, limitations, subgroup behavior, governance, and accountability."),
        ("ROC/PR threshold", "Decision threshold", "The probability cutoff used to turn a continuous risk score into a flag. Changing it trades missed failures against false alarms."),
        ("Calibration", "Score-to-probability reliability", "Whether a predicted 20% risk corresponds to roughly 20% outcomes in comparable cases. Class balancing makes our scores rankings, not literal odds."),
        ("Data leakage", "Information contamination", "When training or tuning uses information unavailable at prediction time, making evaluation look better than real deployment."),
        ("Proxy label", "Transparent stand-in target", "A documented substitute for an unavailable ground-truth label. Here, failure within two years stands in for distress, not a CAMELS rating."),
        ("FedAvg", "Federated Averaging", "A federated-learning method where local model updates are averaged without sending raw participant data to the coordinator."),
        ("DP / epsilon", "Differential Privacy / privacy budget", "A formal way to limit how much one record can influence a released result; smaller epsilon generally means stronger privacy and more noise."),
        ("API", "Application Programming Interface", "A defined way for programs to exchange data or actions. The UI calls the FastAPI backend through `/score`, `/models`, and `/rai-audit`."),
    ]
    glossary_df = pd.DataFrame(glossary_rows, columns=["Term", "Full form", "Meaning in this project"])
    search_term = st.text_input("Filter terms", placeholder="Try: AUC, SHAP, privacy, bank")
    if search_term:
        searchable = glossary_df.astype(str).apply(lambda column: column.str.contains(search_term, case=False, na=False))
        glossary_df = glossary_df[searchable.any(axis=1)]
    st.dataframe(glossary_df, use_container_width=True, hide_index=True)
    st.markdown(
        "**References:** [FDIC BankFind Suite](https://banks.data.fdic.gov/bankfind-suite/), "
        "[scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), "
        "[SHAP documentation](https://shap.readthedocs.io/), "
        "[NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)."
    )

with overview_tab:
    st.caption("The complete project walkthrough continues below.")

# ---------------------------------------------------------------------------
st.header("1. What problem is this solving?")
st.markdown(
    """
Bank regulators score every bank on **CAMELS** — Capital, Asset quality, Management,
Earnings, Liquidity, Sensitivity to market risk — but those real ratings are
**confidential**. There is no public dataset of "here is bank X's real CAMELS score"
to learn from.

**CAMEL Sentinel's approach**: build the same six ratios from banks' *public* quarterly
financial filings (Call Reports, via the FDIC's own API), combine them into a transparent
composite score as a stand-in for the real rating, and — separately — train a machine
learning model on a real, independent, unambiguous outcome instead: **did this bank
actually fail?** That second part is the genuine prediction problem this project solves;
the composite score is a supporting, rule-based sanity check, not the model's target.

This is explicitly positioned as **decision-support / early-warning**, not a replacement
for a real regulator's supervisory judgment — see the caveats in Section 5.
"""
)

# ---------------------------------------------------------------------------
st.header("2. The data")
st.markdown(
    """
All data comes from the **FDIC BankFind Suite API** (`banks.data.fdic.gov`), free and
public, no key required. Three endpoints:

| Endpoint | What it gives us |
|---|---|
| `institutions` | Bank identity — name, state, charter type, total assets |
| `financials` | One quarter's Call Report numbers per bank — capital, loans, income, deposits, etc. |
| `failures` | The full historical list of every FDIC bank failure since 1934, with the failure date |

**Why more than one quarter matters**: a single quarter's snapshot (e.g. just
2023-12-31) only contains ~15 banks that ever went on to fail, out of ~4,700 — far too
few examples for any model to learn from, or for comparing algorithms to mean anything.
So the model is instead trained on a **pooled panel: 265,114 bank-quarters spanning
2005-2023**, including the 2008-2012 financial crisis (where the overwhelming majority
of real U.S. bank failures happened) — giving **4,165** real positive examples instead
of 15.
"""
)

# ---------------------------------------------------------------------------
st.header("3. The six CAMEL ratios (the model's only inputs)")
st.markdown("Every prediction is based on exactly six numbers, one per CAMEL letter:")
st.table(
    {
        "Letter": ["C", "A", "M", "E", "L", "S"],
        "Component": [
            "Capital adequacy", "Asset quality", "Management",
            "Earnings", "Liquidity", "Sensitivity",
        ],
        "Ratio used": [
            "Tier 1 risk-based capital ratio",
            "Non-performing loans / total loans",
            "Efficiency ratio (non-interest expense / revenue)",
            "Return on assets (ROA)",
            "Loans / deposits",
            "Loans / total assets (loan concentration)",
        ],
        "Healthier direction": [
            "Higher", "Lower", "Lower", "Higher", "Lower", "Lower",
        ],
    }
)
st.warning(
    "**Honest weak point**: 'Management' quality has no real financial-statement proxy — "
    "regulators observe it on-site, we can't. The efficiency ratio is a standard "
    "literature substitute, but it's explicitly the weakest link in this whole pipeline.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
st.header("4. Method — how the model was built")

with st.expander("Step 1 — Feature engineering", expanded=False):
    st.markdown(
        """
Each bank-quarter's six raw ratios are also turned into a **composite score (0-100)**
and a **1-5 proxy rating** (1 = healthiest), purely as a transparent, rule-based
reference point: each ratio is converted to a z-score pointed in the "healthier"
direction, the six z-scores are averaged, and that average is rescaled to 0-100 and
bucketed into quintiles. This composite score is *not* what the ML model predicts —
see Section 1 for why.
"""
    )

with st.expander("Step 2 — The real prediction target", expanded=False):
    st.markdown(
        """
The model predicts **P(this bank fails within the next 2 years)**, using each bank's
ratios *at the time*, matched against the FDIC's actual failure history — a bank-quarter
is labeled `failed=1` only if a real failure happened within 2 years *after* that specific
quarter (not "ever, at any point in FDIC history" — that distinction is what makes the
panel usable instead of just 15 positive examples with the naive version of this label).
"""
    )

with st.expander("Step 3 — Model comparison", expanded=False):
    st.markdown(
        """
10 model families were compared with **grouped 5-fold cross-validation** (grouped by
bank, since the same bank appears across many quarters — a plain random split would let
a bank's data leak between train and test):

`LogisticRegression`, `RandomForest`, `ExtraTrees`, `HistGradientBoosting`, `AdaBoost`,
`XGBoost`, `LightGBM`, `GaussianNB`, and two imbalance-specialist ensembles from
`imbalanced-learn` — `BalancedRandomForest` and `EasyEnsemble`.

Ranked by **PR-AUC** (precision-recall area under curve — the right metric when
positives are ~1.6% of the data; plain accuracy is meaningless here since "always
predict healthy" would already score >98%), gradient-boosted trees won, with
**HistGradientBoosting** on top.
"""
    )

with st.expander("Step 4 — No-look-ahead validation (the strict test)", expanded=False):
    st.markdown(
        """
Cross-validation folds still mix years together, so a model could look good partly by
having seen the *same crisis era* in both train and test. The stricter check: train
**only on 2005-2011**, evaluate **only on 2012-2023** — a period with a visibly
different (much calmer) failure rate that the model never saw during training at all.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| RandomForest (untuned) | 0.963 | 0.495 |
| HistGradientBoosting (untuned) | 0.969 | 0.548 |
| **HistGradientBoosting (tuned)** | **0.973** | **0.549** |

This confirmed the improvement from switching model families is real, not a
same-era coincidence — and that hyperparameter tuning (a 25-iteration randomized
search, restricted to only the 2005-2011 training rows so tuning couldn't peek at
the future either) adds a smaller additional gain on top.
"""
    )

with st.expander("Step 5 — Explainability (XAI)", expanded=False):
    st.markdown(
        """
Every prediction has three explanation views. **SHAP** (`shap.TreeExplainer`)
computes exact Shapley values for the tree model and shows how each ratio moved
the score from its baseline. **LIME** creates nearby variations and fits a simple
local surrogate as an independent cross-check. The **counterfactual** search asks
whether bounded, one-ratio-at-a-time changes can move the score below the medium-risk
threshold. Counterfactuals are model what-ifs, not causal recommendations: financial
ratios can be correlated and cannot necessarily be changed independently.
"""
    )

# ---------------------------------------------------------------------------
st.header("5. Compare models and understand the algorithms")
st.caption(
    "The scorer can run the two saved artifacts. The benchmark also contains "
    "algorithms evaluated in the notebook but not exported for live scoring."
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
panel_results = pd.read_csv(PROJECT_ROOT / "notebooks" / "model_comparison_results_panel.csv")
holdout_results = pd.read_csv(PROJECT_ROOT / "notebooks" / "model_comparison_results_time_holdout.csv")

THEORY = {
    "HistGradientBoosting": (
        "Boosted decision trees",
        "Many small trees correct one another's errors and learn nonlinear distress patterns.",
        "F(x) = F_prev(x) + eta * tree_m(x)",
        "Sequentially adds histogram-binned trees that follow the current loss gradient; class balancing increases the cost of missed failures.",
        "Strong ranking and interactions, but less transparent than a linear equation and exposed to regime shift.",
    ),
    "RandomForest": (
        "Bagged decision trees",
        "Averaging many decorrelated trees reduces variance and gives a stable tabular baseline.",
        "p_hat(x) = (1/T) * sum_t p_t(y=1|x)",
        "Each tree sees a bootstrap sample and random feature subset; probabilities are averaged across trees.",
        "Robust and practical, but usually less precise than boosting for subtle interactions.",
    ),
    "LogisticRegression": (
        "Linear probabilistic classifier",
        "Failure risk is a sigmoid of a weighted combination of the six ratios.",
        "P(y=1|x) = 1 / (1 + exp(-(w^T x + b)))",
        "Learns one coefficient per feature by minimizing log loss; standardization makes ratio scales comparable.",
        "Fast and interpretable, but cannot naturally model thresholds or interactions.",
    ),
    "ExtraTrees": (
        "Randomized tree ensemble",
        "More random split points can lower tree correlation and improve generalization.",
        "p_hat(x) = (1/T) * sum_t p_t(y=1|x)",
        "Grows trees using randomized candidate thresholds, then aggregates their predictions.",
        "Fast and nonlinear, but extra randomization can trade bias for lower variance.",
    ),
    "AdaBoost": (
        "Sequential boosting",
        "Keep increasing attention on examples previous weak learners classified incorrectly.",
        "F(x) = sum_m alpha_m * h_m(x)",
        "Reweights difficult training rows after every learner, then combines weak classifiers.",
        "Can produce high precision, but is more affected by noisy labels and outliers.",
    ),
    "XGBoost": (
        "Regularized gradient boosting",
        "Gradient boosting plus explicit regularization learns high-quality nonlinear ranking functions.",
        "objective = loss + gamma * leaves + lambda * ||weights||^2",
        "Adds trees using first and second loss derivatives, with shrinkage, depth, and leaf penalties.",
        "Powerful and tunable, but more parameters increase operational complexity.",
    ),
    "LightGBM": (
        "Efficient gradient boosting",
        "Histogram-based, leaf-wise growth optimizes large tabular datasets quickly.",
        "F(x) = F_prev(x) + eta * tree_m(x)",
        "Bins continuous values and grows the leaf with the largest loss reduction.",
        "Fast and accurate, but leaf-wise growth can overfit without constraints.",
    ),
    "GaussianNB": (
        "Generative probabilistic classifier",
        "Within each class, each ratio is approximately Gaussian and conditionally independent.",
        "P(y|x) proportional to P(y) * product_j Normal(x_j; mean_yj, variance_yj)",
        "Estimates a mean and variance for every feature/class pair, then applies Bayes' rule.",
        "Very fast, but its independence and Gaussian assumptions are restrictive.",
    ),
    "BalancedRandomForest": (
        "Imbalance-aware bagged trees",
        "Each tree sees a more balanced sample so rare failures influence learning.",
        "p_hat(x) = (1/T) * sum_t p_t(y=1|x)",
        "Resamples minority and majority classes for each tree before fitting a Random Forest.",
        "Improves minority recall, but false positives can rise.",
    ),
    "EasyEnsemble": (
        "Imbalance-aware ensemble",
        "Several balanced subsets preserve different majority examples, then vote together.",
        "p_hat(x) = (1/K) * sum_k p_k(y=1|x)",
        "Under-samples the majority repeatedly and trains an AdaBoost learner on each subset.",
        "Strong minority coverage, but broad early-warning predictions may have low precision.",
    ),
}

algorithm = st.selectbox("Choose an algorithm to study", list(THEORY))
family, thesis, equation, internal, tradeoff = THEORY[algorithm]
st.markdown(f"**Family:** {family}  \n**Thesis:** {thesis}")
st.markdown(f"**Core mathematics:** `{equation}`")
st.markdown(f"**Internals:** {internal}  \n**Trade-off:** {tradeoff}")

panel_display = panel_results.rename(columns={"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Balanced_Accuracy": "Balanced accuracy"})
panel_display["Live scorer"] = panel_display["model"].isin(["HistGradientBoosting", "RandomForest (Phase 3 baseline)"])
metric_columns = ["ROC-AUC", "PR-AUC", "F1", "Recall", "Precision", "Balanced accuracy"]
st.markdown("**Grouped cross-validation benchmark**")
st.dataframe(
    panel_display[["model", *metric_columns, "Live scorer"]].style.format({column: "{:.1%}" for column in metric_columns}),
    use_container_width=True,
    hide_index=True,
)
st.caption("Only rows marked `Live scorer` have saved artifacts available in the scorer dropdown.")

holdout_display = holdout_results.rename(columns={"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Balanced_Accuracy": "Balanced accuracy"})
st.markdown("**Strict future holdout: trained on 2005–2011, tested on untouched 2012–2023**")
st.dataframe(
    holdout_display[["model", *metric_columns]].style.format({column: "{:.1%}" for column in metric_columns}),
    use_container_width=True,
    hide_index=True,
)
st.info("PR-AUC is the primary choice metric because only about 1.6% of bank-quarters are failures. Accuracy would reward predicting 'healthy' almost every time.")

# ---------------------------------------------------------------------------
st.header("6. Important caveats (read before trusting a score)")


def _fetch_live_model_card() -> dict | None:
    for url in (backend_url(), "http://127.0.0.1:8000"):
        try:
            resp = requests.get(f"{url}/model-card", timeout=3)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            continue
    return None


live_card = _fetch_live_model_card()
if live_card:
    st.success("Live model card fetched from the running backend:", icon="✅")
    for key, value in live_card.items():
        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
else:
    st.caption("(Backend not reachable from this page right now — showing the hand-written summary below instead.)")
    st.markdown(
        """
- **Proxy label, not a real CAMELS rating**: real ratings are confidential; this project's
  composite score and failure label are transparent, documented stand-ins.
- **Predicted probabilities are a relative risk ranking, not calibrated real-world odds** —
  the model is trained with class-balanced weighting to counter severe class imbalance, so
  a "60%" score does not mean a 60% real-world chance of failure.
- **The failure rate itself shifted ~5x** between the 2005-2011 training era (2.1%) and the
  2012-2023 period (0.43%) — a model calibrated to one era over-flags under a calmer one, which
  is why the deployed risk tiers use calibrated cutoffs (0.60 / 0.98), not a flat 0.5.
- **Intended use**: decision-support / early-warning signal, not a replacement for supervisory judgment.
"""
    )

# ---------------------------------------------------------------------------
st.header("7. Responsible-AI audit")


def _fetch_live_rai_audit() -> dict | None:
    for url in (backend_url(), "http://127.0.0.1:8000"):
        try:
            resp = requests.get(f"{url}/rai-audit", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            continue
    return None


rai_audit = _fetch_live_rai_audit()
if rai_audit and rai_audit.get("groups"):
    st.caption(
        f"The audit uses the medium-risk threshold ({rai_audit['threshold']:.0%}) and excludes groups "
        f"with fewer than {rai_audit['minimum_group_support']} observations."
    )
    for dimension, rows in rai_audit["groups"].items():
        st.subheader(dimension.replace("_", " ").title())
        audit_table = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="group")
        audit_table = audit_table.rename(
            columns={
                "positive_rate": "Failure rate",
                "selection_rate": "Flag rate",
                "recall": "Recall",
                "precision": "Precision",
                "false_positive_rate": "False-positive rate",
                "balanced_accuracy": "Balanced accuracy",
                "selection_rate_ratio": "Flag-rate ratio",
                "recall_ratio": "Recall ratio",
            }
        )
        percentage_columns = [
            "Failure rate", "Flag rate", "Recall", "Precision",
            "False-positive rate", "Balanced accuracy",
        ]
        st.dataframe(
            audit_table.style.format({column: "{:.1%}" for column in percentage_columns}),
            use_container_width=True,
            hide_index=True,
        )
    st.warning(
        "These are descriptive model-performance disparities, not a legal fairness determination. "
        "The panel does not contain charter type or protected demographic attributes."
    )
else:
    st.info("The live RAI audit is available when the backend is running with the cached panel.")

# ---------------------------------------------------------------------------
st.header("8. Federated and privacy-preserving training")
st.markdown(
    """
The repository includes a runnable teaching simulation in
`src/camel_sentinel/federated/simulation.py`. It partitions the public panel
by state to represent separate data owners, standardizes the six ratios locally
for a linear classifier, trains a model at each client, clips each client update,
adds Gaussian noise, and averages only the updates with FedAvg. Raw client rows
are not sent to the coordinator.

This simulation is intentionally separate from the deployed tuned tree model:
tree structures are not safely averaged as simple weight vectors. The simulated
model is a privacy-learning demonstration, not an automatic replacement for the
centralized production artifact.

The `clip_norm` and `noise_multiplier` settings control the mechanism. They are
not, by themselves, a formal end-to-end epsilon guarantee; a production privacy
claim requires a complete accounting of sampling, rounds, composition, and all
released outputs.
"""
)

# ---------------------------------------------------------------------------
st.header("9. Tool-calling agent layer")
st.markdown(
    """
The agent boundary is explicit and allow-listed. `GET /agent/tools` describes
the available `score_bank` tool, and `POST /agent/score` invokes it with the
same model, SHAP, LIME, and counterfactual pipeline used by the UI. The tool is
read-only: it predicts and explains, but it cannot change data or call arbitrary
functions. Unknown tool names fail closed in
`src/camel_sentinel/agent/tools.py`.

This is the foundation for the Phase 9 conversational chat layer — now built
and available on the **💬 Chat Assistant** page (sidebar). The `ChatAgent` in
`src/camel_sentinel/chat/agent.py` wraps Azure OpenAI's function-calling API
with `score_bank` as the single allowed tool. See Section 10 for how it works.
"""
)

# ---------------------------------------------------------------------------
st.header("10. Conversational Chat Interface (Phase 9)")
st.markdown(
    """
The **💬 Chat Assistant** page (sidebar) adds a natural-language layer on top of the
scoring engine. Ask questions in plain English — or speak them — and get back a
plain-English explanation of what the CAMEL model found, with the SHAP chart inline.

**How a turn works:**

| Input | Path |
|---|---|
| Text question | typed text → `POST /chat` → GPT agent → reply text |
| Voice question | mic → `POST /speech/transcribe` → Azure STT → same GPT path → `POST /speech/synthesize` → Azure TTS → audio |

**Mirror-input rule:** voice questions get audio replies (autoplay). Text questions
get text replies only. Old voice replies show a replay button in the history but
don't re-play automatically on every page load.

**The LLM is a translator, not a decision-maker.** Risk numbers come from the CAMEL
ML model via the `score_bank` tool. The language model rephrases those numbers; it
never generates a risk rating from its own weights. Four calibration caveats are
baked into the server-side system prompt and cannot be removed by user input.

**Components:**

| File | Job |
|---|---|
| `src/camel_sentinel/chat/agent.py` | Azure OpenAI client, `score_bank` tool schema, two-step tool-calling loop |
| `src/camel_sentinel/chat/speech.py` | Azure Speech STT (`transcribe_audio`) and TTS (`synthesize_text`) |
| `src/camel_sentinel/chat/history.py` | Rolling 20-turn conversation window, role validation |
| `src/camel_sentinel/chat/prompts.py` | Server-side system prompt: role + tool discipline + caveats + scope boundary |
| `backend/main.py` | Adds `POST /chat`, `POST /speech/transcribe`, `POST /speech/synthesize` |
| `frontend/views/2_💬_Chat_Assistant.py` | Chat UI: text + voice input, SHAP + risk tier inline, conversation history |

**Requires** (in `.env`, see `.env.example`): `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_DEPLOYMENT` for the LLM; `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` for voice.
Text-only mode works with just the OpenAI credentials.
"""
)

# ---------------------------------------------------------------------------
st.header("11. Where everything lives in the codebase")
st.markdown("Folder structure, and which file has the logic for which specific job:")

st.code(
    """
Project/
├── backend/
│   └── main.py              # FastAPI: /score, /banks, /model-card, /rai-audit,
│                             #          /chat, /speech/transcribe, /speech/synthesize
├── frontend/
│   ├── app.py               # Streamlit scorer page (home)
│   └── pages/
│       ├── 1_Documentation.py          # ← this page
│       └── 2_Chat_Assistant.py         # Phase 9 chat: text + voice, SHAP inline
├── src/camel_sentinel/
│   ├── data/
│   │   └── fdic_client.py   # FDIC API pull + synthetic fallback
│   ├── features/
│   │   ├── camel_ratios.py  # Builds the 6 CAMEL ratios + composite score/rating
│   │   └── panel.py         # Multi-quarter panel + "failed within 2 years" label
│   ├── models/
│   │   ├── baseline.py      # Train/evaluate the classifier
│   │   └── registry.py      # Save/load model artifacts (joblib)
│   ├── xai/
│   │   ├── explain.py       # SHAP (TreeExplainer) + LIME local surrogates
│   │   └── counterfactual.py  # Bounded what-if ratio search
│   ├── rai/
│   │   └── audit.py         # Fairness audit by asset size and region
│   ├── federated/
│   │   └── simulation.py    # FedAvg simulation across state partitions
│   ├── agent/
│   │   └── tools.py         # Allow-listed score_bank tool (called by ChatAgent)
│   └── chat/                # Phase 9 — conversational layer
│       ├── agent.py         # Azure OpenAI tool-calling; 2-step score_bank loop
│       ├── history.py       # Rolling 20-turn window; role validation
│       ├── prompts.py       # Server-side system prompt (caveats + scope)
│       └── speech.py        # Azure Speech STT (transcribe_audio) + TTS (synthesize_text)
├── notebooks/
│   ├── 00_dataset_research.ipynb  # Phase 1 — data acquisition + CAMEL engineering
│   ├── 01_eda.ipynb               # Phase 2 — exploratory data analysis
│   ├── 02_baseline_modeling.ipynb # Phase 3 — RandomForest baseline
│   └── 03_model_tuning.ipynb      # Phase 4 — panel model, comparison, no-look-ahead
├── models/
│   ├── camel_baseline_model.joblib  # Phase 3 (reference)
│   └── camel_tuned_model.joblib     # Phase 4 (served)
├── data/processed/
│   └── camel_panel.parquet      # 265k-row multi-quarter training panel
├── .env.example             # Required environment variables (keys not committed)
└── docs/                    # Problem statement, glossary, data sources, module mapping,
                             # chat interface design (04_chat_interface.md)
""",
    language="text",
)

st.markdown(
    """
**Quick "I want to change X, where do I look" guide**:

| I want to... | Look in |
|---|---|
| Change how a CAMEL ratio is calculated | `src/camel_sentinel/features/camel_ratios.py` |
| Add a new data source / fix an API field | `src/camel_sentinel/data/fdic_client.py` |
| Change which quarters are in the training panel | `src/camel_sentinel/features/panel.py` |
| Retrain or swap the model | `notebooks/03_model_tuning.ipynb`, then `src/camel_sentinel/models/registry.py` |
| Change what the API returns | `backend/main.py` |
| Change the SHAP explanation logic | `src/camel_sentinel/xai/explain.py` |
| Change the scorer UI | `frontend/app.py` |
| Change this documentation page | `frontend/views/1_📚_Documentation.py` |
| Change how the chat agent responds or its caveats | `src/camel_sentinel/chat/prompts.py` |
| Swap the LLM model or Azure endpoint | `.env` (`AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_ENDPOINT`) |
| Change the TTS voice | `src/camel_sentinel/chat/speech.py` (`synthesize_text` default voice arg) |
| Change the chat UI | `frontend/views/2_💬_Chat_Assistant.py` |
"""
)

st.divider()
st.markdown(
    "<div style='text-align:center; opacity:0.7;'>CAMEL Sentinel — built as a hands-on "
    "AI/ML + Cybersecurity project.<br><b>Developer: <a href='https://raushan-ranjan.azurewebsites.net' target='_blank'>Raushan Ranjan</a></b></div>",
    unsafe_allow_html=True,
)
