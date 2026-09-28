# CAMEL Sentinel Model Card

## Model

The deployed model is a tuned `HistGradientBoostingClassifier`. It estimates a relative risk score for whether a bank fails within two years of a financial snapshot. It is not a regulator-assigned CAMELS rating and its probability is not calibrated as a literal real-world probability.

The model uses six CAMEL-style ratios: Tier 1 capital, non-performing loans, efficiency, ROA, loans/deposits, and loan concentration. The training panel contains 265,114 bank-quarters from 2005-2023 and 4,165 forward-looking failure examples.

## Intended use

Use the output as a decision-support and early-warning ranking for investigation. The medium-risk threshold is 0.60 and the high-risk threshold is 0.98. Missing a failure and generating a false alarm have different costs, so users should choose the operating threshold deliberately.

Do not use the score as an automatic supervisory decision, a lending decision, or a substitute for confidential supervisory judgment.

## Validation

Hyperparameters were selected using only 2005-2011 data. The untouched 2012-2023 period produced:

| Metric | Tuned HistGradientBoosting |
|---|---:|
| ROC-AUC | 97.27% |
| PR-AUC | 54.89% |
| Recall | 92.01% |
| Precision | 9.44% |
| F1 | 17.12% |
| Balanced accuracy | 94.09% |

PR-AUC is the primary model-selection metric because failures are rare. Ordinary accuracy would hide poor minority-class performance.

## Responsible-AI audit

The audit was run on the cached training panel at the deployed medium-risk threshold of 0.60. Overall results were:

| Metric | Value |
|---|---:|
| Failure rate | 1.57% |
| Flag rate | 7.80% |
| Recall | 88.81% |
| Precision | 17.89% |
| False-positive rate | 6.50% |
| Balanced accuracy | 91.15% |

The audit compares descriptive performance across asset-size tertiles and broad U.S. regions. Small banks had 83.88% recall versus 90.73% for large banks. Regional precision varied substantially, with the Northeast estimate especially unstable because its observed failure rate was only 0.02%.

These are disparity indicators, not a legal fairness determination. Base-rate differences, sample composition, missing data, and threshold choice affect the comparison. The panel does not contain charter type, race, gender, or other protected attributes, so those dimensions are not audited. A large share of rows also has an unmapped/unknown region value and should be resolved before making regional governance claims.

## Explanation methods

The scorer provides three views of an individual result:

- **SHAP** allocates the model's change from its baseline among the six ratios using a tree-aware Shapley calculation.
- **LIME** fits a simple local surrogate around the selected bank and reports which ratios influence that nearby approximation. It is a cross-check, not a guarantee that the underlying model is linear.
- **Counterfactual** searches bounded, one-feature-at-a-time CAMEL changes for a path below the medium-risk threshold. It is a model what-if, not a causal recommendation; financial ratios can be correlated and cannot necessarily be changed independently.

## Data and label limitations

- Public FDIC data does not contain confidential regulator-issued CAMELS ratings.
- `failed within two years` is an independent proxy outcome, not a CAMELS label.
- Management is approximated with an efficiency ratio because supervisory management assessments are not public.
- Call Report data arrives with reporting lag and historical failure regimes changed over time.
- Class-balanced training makes scores useful for ranking but weakly calibrated as probabilities.

## Monitoring and review

Re-run the grouped performance and RAI audit whenever the panel, feature recipe, threshold, or deployed model changes. Review calibration and subgroup sample sizes after a regime shift or when new geography/charter metadata becomes available.

## Federated-learning demonstration

`src/camel_sentinel/federated/simulation.py` contains a hand-rolled FedAvg
demonstration. It partitions the public panel by state, trains a standardized
linear classifier locally, clips client updates, adds Gaussian noise, and
averages updates without sending raw client rows to the coordinator.

This is not the deployed model and is not a formal differential-privacy
certificate. The reported noise and clipping settings are mechanism
parameters; a production epsilon claim requires privacy accounting across
sampling, rounds, composition, and released outputs.

## Agent interface

The read-only `score_bank` tool is exposed through `GET /agent/tools` and
`POST /agent/score`. It is allow-listed and can only score the six declared
CAMEL inputs using a selected persisted model, then return explanations. It
cannot write data, access arbitrary files, or invoke arbitrary functions.
