# Glossary

> Every term used elsewhere in this project, defined once, here, in plain
> language. If you hit an unfamiliar term in a notebook or in `src/`, it
> should be defined on this page — if it isn't, that's a gap to fill, not a
> reason to guess.

## Domain terms (banking / CAMEL)

**CAMELS rating** — A 1 (best/strongest) to 5 (worst/weakest) composite
score US bank regulators assign to every supervised bank, covering six
components (hence the acronym):
- **C**apital adequacy — does the bank have enough of a capital buffer to
  absorb losses?
- **A**sset quality — how risky is the bank's loan/investment book (how
  many of its loans are going bad)?
- **M**anagement quality — is the bank well-run? (Judged by regulators
  on-site; the one component with no direct financial-statement number.)
- **E**arnings — is the bank consistently profitable?
- **L**iquidity — can the bank meet its short-term obligations (deposit
  withdrawals, etc.) without a fire sale of assets?
- **S**ensitivity to market risk — how exposed is the bank to interest-rate
  or market moves?

The "S" was added in 1996 (the framework started as "CAMEL" in 1979); both
names are used interchangeably in the literature and in this project.

**Call Report** — The standardized quarterly financial statement every
FDIC-insured bank must file (officially the "Consolidated Reports of
Condition and Income"). This is where every raw number used to build our
CAMEL ratios comes from — total assets, equity, non-performing loans, net
income, deposits, etc.

**Proxy label** — Because real CAMELS ratings are confidential (see
[`00_problem_statement.md`](00_problem_statement.md)), we build a stand-in
target from public data instead: either a rule-based composite ratio score,
or a "failed within N quarters" flag. "Proxy" means: transparent,
documented, defensible — but explicitly *not* the real regulator-assigned
number, and every result must be described as predicting the proxy, not the
real rating.

**Composite z-score** — A way to combine six ratios that are on totally
different scales (a percentage, a ratio, a dollar-based margin) into one
comparable number. Each ratio is converted to a z-score
(`(value - mean) / standard_deviation`, i.e. "how many standard deviations
above/below the average bank is this one"), flipped in sign if a lower raw
value is actually healthier (e.g. a lower efficiency ratio is better), and
the six z-scores are averaged into one composite score per bank-quarter.

**Charter type / BKCLASS** — FDIC codes for what kind of institution a bank
is (national bank, state member bank, savings bank, etc.) and its
regulatory category — used later to check the model isn't systematically
unfair to one charter type (see RAI, below), and to partition data into
simulated institutions for federated learning.

## Machine learning terms

**EDA (Exploratory Data Analysis)** — Looking at the data before modeling
it: distributions, missing values, correlations, class balance — to catch
problems (like the severe class imbalance on bank failures) before they
silently wreck a model.

**Feature engineering** — Turning raw Call Report fields into the ratios a
model can actually learn from (e.g. turning "non-performing loans" and
"total loans" into a single `npl_ratio` percentage).

**Class imbalance** — When one outcome is rare compared to the other(s) —
here, bank failures are a tiny fraction of all bank-quarters. Plain
accuracy is misleading under imbalance (a model that always predicts
"healthy" can still score >95% accuracy while catching zero real failures),
so this project reports precision/recall/F1/AUC on the minority (failure)
class instead.

## Explainable AI (XAI)

**XAI (Explainable AI)** — Techniques that answer "why did the model predict
this?" for a specific prediction, not just "how accurate is the model
overall?"

**SHAP (SHapley Additive exPlanations)** — An XAI method, based on game
theory, that assigns each input feature (each CAMEL ratio) a number showing
how much it pushed *this specific bank's* prediction up or down relative to
the average prediction. Used here for both global explanations ("which
ratios matter most across all banks?") and local/per-bank explanations
("why did *this* bank get a 4?").

**LIME (Local Interpretable Model-agnostic Explanations)** — A second,
independent XAI method used as a cross-check against SHAP: it explains one
prediction by fitting a simple, interpretable model locally around that one
data point.

**Counterfactual explanation** — "What would need to change about this
bank's ratios for the model to predict a better rating?" — a concrete,
actionable form of explanation (e.g. "raising the Tier 1 capital ratio by
2 points would move this bank from a 3 to a 2").

## Responsible AI (RAI)

**RAI (Responsible AI)** — The practice of checking a model isn't just
accurate, but fair, documented, and used appropriately — as opposed to XAI,
which explains individual predictions, RAI is about the model and its
deployment as a whole.

**Fairness audit** — Checking whether the model's errors or scores are
systematically worse for one group (e.g. small community banks, or banks in
a particular region) than another, *for reasons unrelated to actual
financial risk*.

**Model card** — A short, standard document describing what a model does,
what data it was trained on, its known limitations (here: the proxy-label
caveat and the "Management has no clean proxy" caveat), and its intended
use — written so someone who didn't build the model can decide whether it's
appropriate for their use case.

## Federated / privacy-preserving ML

**Federated learning** — Training one shared model across multiple data
"owners" (here: simulated banks or regulatory regions) **without any of
them sending their raw data to a central server** — only model updates are
shared and combined.

**FedAvg (Federated Averaging)** — The standard federated-learning
algorithm: each participant trains the shared model locally on its own
data for a few steps, then only the resulting model *weight updates* (not
the data) are sent to a coordinator and averaged together into an improved
shared model, repeated over multiple rounds.

**Differential privacy (DP)** — Adding carefully calibrated random noise to
data or model updates so that no individual record can be reverse-engineered
from the output, with a mathematical guarantee (the "privacy budget",
often called epsilon) bounding how much any single bank's data could have
influenced the shared model.

## Agent

**Agent / tool-calling agent** — A thin wrapper around the model that
accepts a natural-language or structured request ("score this bank"),
calls the right underlying functions (fetch financials → compute ratios →
predict → explain) as callable "tools", and returns a response that
combines the prediction, its SHAP-based explanation, and any RAI/privacy
caveats — the piece that makes the pipeline usable as more than a script.

## Additional evaluation and engineering terms

**AI (Artificial Intelligence)** — The broad field of building systems that
perform tasks associated with human reasoning, such as prediction,
classification, explanation, and planning.

**ML (Machine Learning)** — A subfield of AI where a model learns patterns
from examples rather than receiving every decision rule explicitly.

**ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)** — A
threshold-independent measure of ranking quality. It summarizes how often a
random positive receives a higher score than a random negative. 0.5 is random
ranking; 1.0 is perfect ranking. See the
[scikit-learn ROC-AUC documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics).

**PR-AUC (Precision-Recall Area Under the Curve)** — A summary of the
precision/recall trade-off across thresholds. It is usually more informative
than ROC-AUC when positives are rare. This project uses average precision as
its computable PR-AUC summary. See the
[scikit-learn precision-recall documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics).

**Precision (positive predictive value)** — Of all cases flagged positive,
the fraction that is actually positive: `TP / (TP + FP)`.

**Recall (sensitivity / true positive rate)** — Of all actual positive cases,
the fraction flagged by the model: `TP / (TP + FN)`.

**F1 score** — The harmonic mean of precision and recall:
`2 * precision * recall / (precision + recall)`. It is high only when both
precision and recall are useful.

**Confusion matrix (TP, FP, TN, FN)** — A four-cell count of correct and
incorrect predictions: true positives, false positives, true negatives, and
false negatives.

**Cross-validation (CV)** — Repeated train/test splits used to estimate how
well a model generalizes. This project uses grouped CV so rows from the same
bank do not leak across a fold.

**No-look-ahead validation** — A time-respecting test: train on earlier
quarters and evaluate on later quarters that did not influence training or
tuning.

**Decision threshold** — The cutoff that turns a continuous risk score into
a yes/no flag. Lowering it generally increases recall and false alarms; raising
it generally increases precision and missed positives.

**Calibration** — Whether predicted probabilities match observed frequencies.
This project warns that class-balanced training makes scores better
interpreted as relative rankings than literal failure odds.

**Data leakage** — Information from outside the prediction-time boundary
accidentally entering features, labels, training, or tuning. Leakage makes
evaluation look stronger than real deployment.

**NIST AI RMF (AI Risk Management Framework)** — A voluntary NIST framework
for managing AI risks through governance, mapping, measurement, and
management. See the
[NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).

**Privacy budget (epsilon)** — The epsilon parameter in differential privacy.
It bounds how much the output can change when one record is added or removed;
smaller epsilon generally provides stronger privacy but requires more noise.

**API (Application Programming Interface)** — A defined contract through
which programs exchange data or invoke actions. The Streamlit UI calls the
FastAPI backend through endpoints such as `/score`, `/models`, and
`/rai-audit`.

**LIME (Local Interpretable Model-agnostic Explanations)** — A method that
creates nearby variations of one input, observes the black-box model's
outputs, and fits a simple local surrogate. Its weights explain the local
neighborhood, not the whole model.

**Counterfactual explanation** — A model what-if showing a small input change
that would cross a selected decision threshold. It is not proof that changing
that input causes the outcome to change in real life.

**Local surrogate** — A simple model, often linear, fitted only around one
case so a complex model's local behavior can be described to a human.
