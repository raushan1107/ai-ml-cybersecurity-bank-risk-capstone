# Problem Statement

> Start here. Everything else in `docs/`, `src/`, and `notebooks/` exists to solve
> the problem defined on this page. If a piece of code doesn't trace back to
> something here, it doesn't belong in the project.

## 1. The real-world problem

Bank regulators (in the US: the FDIC, the Federal Reserve, the OCC, the NCUA)
score every bank they supervise on **CAMELS** — Capital adequacy, Asset
quality, Management, Earnings, Liquidity, Sensitivity to market risk — a 1
(best) to 5 (worst) composite rating produced from on-site exams plus ratio
analysis. A bad rating is an early warning that a bank is in danger of
failing.

The catch: **CAMELS ratings are confidential supervisory information.** They
are shared only with a bank's own top management and its regulator, by
design — publishing a downgrade could trigger the very bank run the exam was
trying to prevent. No dataset, on Kaggle or anywhere else, contains real,
regulator-issued CAMELS labels for individual banks. This is well documented
and is *why* every peer-reviewed bank-failure-prediction paper builds a
**proxy label** instead of training on the real thing (see
[`02_data_sources.md`](02_data_sources.md) for the sources that confirm
this).

## 2. Restated as a solvable ML problem

> Given a bank's periodic financial-statement data (balance sheet + income
> statement ratios), predict a CAMEL-style composite health rating / risk
> bucket, **explain why** the model said what it said (XAI), **audit** the
> model for fairness and document it for governance (RAI), and demonstrate
> that the same pipeline can be **trained without centralizing sensitive
> data** across institutions that won't pool raw financial records with each
> other (federated / privacy-preserving ML).

Two labels are built and compared, not one — because neither alone is fully
trustworthy:

1. **Rule-based composite score** — a weighted combination of the six CAMEL
   components, built from transparent financial ratios, bucketed 1 (best) to
   5 (worst). This is the primary training target: continuous for
   regression, ordinal for classification.
2. **Failure-within-N-quarters** — a real, independently-verifiable ground
   truth (did the bank actually fail?), pulled from the FDIC's public
   failure history. Used to *validate* that the rule-based composite score
   actually tracks real distress, not just to train on directly (it's far
   too rare and too lagging to be the only target).

## 3. What "done" looks like (success criteria)

- A feature table of CAMEL ratios per bank-quarter, built from real FDIC
  data (with a documented synthetic fallback for offline development —
  see [`02_data_sources.md`](02_data_sources.md)).
- A composite score whose 5 (worst) bucket captures a disproportionately
  high share of real bank failures — i.e. the proxy label is doing its job
  as an early-warning signal, not just producing plausible-looking numbers.
- A trained, tuned model (not just the rule-based score) that predicts the
  composite rating / failure risk, evaluated with metrics that make sense
  under severe class imbalance (precision/recall/F1/AUC on the minority
  class — not raw accuracy).
- Every prediction is explainable: given one bank, the system can show
  *which ratios pushed the score up or down, and by how much* (SHAP), not
  just a number.
- A written fairness audit and model card: does the model score small
  community banks, or banks in a particular region, worse for reasons other
  than genuine financial risk?
- A working demonstration that the same model can be trained in a
  federated way (data partitioned across simulated institutions, never
  centralized) with a measured accuracy/privacy trade-off against the
  centralized version.
- A model you can actually query — an exported artifact served behind an
  API, with a UI that shows the prediction, the explanation, and the
  privacy/fairness caveats together — not just a notebook cell's output.

## 4. Scope boundary — what this project is *not*

- **Not a replacement for supervisory judgment.** This is a decision-support
  / early-warning tool built on a transparent proxy label, not the real
  CAMELS rating. Every prediction the system produces should be presented
  with that caveat attached.
- **Not claiming real-time monitoring.** FDIC Call Report data lands
  45–75 days after quarter-end. The system reflects that reporting lag
  honestly rather than implying live monitoring.
- **Not pretending "Management" is measured directly.** Of the six CAMEL
  components, Management is a supervisory judgment call with no clean
  financial-statement proxy. We substitute the efficiency ratio (and
  growth-rate signals) as a standard, literature-backed approximation — and
  say so everywhere that component is used, rather than quietly treating it
  as equivalent to the real thing.

## 5. Where this plan came from

This problem statement is the sharpened, standalone version of Sections 0,
1, and 6 of [`CAMEL-Rating-ML-Project-Blueprint.md`](CAMEL-Rating-ML-Project-Blueprint.md),
the original planning document. The blueprint stays in place as the
project's planning history / audit trail; this `docs/` folder is the
maintained reference going forward. See
[`03_module_mapping.md`](03_module_mapping.md) for how each build phase maps
to a phase of work and to the learning-path stages behind it.
