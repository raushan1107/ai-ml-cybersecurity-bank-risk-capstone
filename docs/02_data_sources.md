# Data Sources

> What we're pulling data from, why each source was chosen, and what its
> caveats are. See [`00_problem_statement.md`](00_problem_statement.md) for
> *why* no dataset anywhere has real CAMELS labels, and
> [`01_glossary.md`](01_glossary.md) for any unfamiliar term below.

## Primary source — FDIC BankFind Suite API (dataset of record)

Free, public, no mandatory API key for light use (a free key is recommended
for higher rate limits in later phases). Docs:
`https://api.fdic.gov/banks/docs`, bulk downloads at
`https://banks.data.fdic.gov/bankfind-suite/bulkdata`.

Three endpoints, used together and joined on `CERT` (a bank's unique FDIC
certificate number):

| Endpoint | What it gives us | Used for |
|---|---|---|
| `institutions` | Bank identity/metadata — name, state, charter type, asset size, holding-company flag | Grouping banks for EDA and for the fairness audit (Phase 6) and federated-learning partitions (Phase 7) |
| `financials` | Quarterly Call Report data since **1992** — equity, Tier 1 capital, risk-weighted assets, non-performing loans, provisions, net income, interest income/expense, deposits, loans, liquid assets | The raw inputs to every CAMEL ratio (see `src/camel_sentinel/features/camel_ratios.py`) |
| `failures` | The full catalog of FDIC-insured bank failures, back to the **1930s** | Our proxy-label ground truth: "did this bank fail within N quarters?" |

**Why this source**: real, large (thousands of institutions), longitudinal
(decades of quarterly history), free, and well documented — it lets this
project honestly say "trained on real US bank supervisory financial data"
rather than a synthetic stand-in.

**Caveats to design around**:
- **Severe class imbalance** — a few hundred failures against tens of
  thousands of bank-quarters. See "Class imbalance" in the glossary.
- **Reporting lag** — Call Report data lands 45–75 days after quarter-end,
  so the system is never claiming real-time monitoring.
- **Join the failure label against the right base population** — a real
  bug hit during Phase 1/3: an early version pulled `institutions` with an
  `ACTIVE:1` filter and joined it as the *base* table onto `financials`.
  Since a failed bank is by definition not on today's active roster, that
  join silently dropped almost every bank that had ever failed, and the
  resulting `failed` flag was ~0% everywhere regardless of rating quality.
  Fix: `financials` (every bank that filed a Call Report that quarter) is
  the base table, institution metadata is left-joined onto it, and
  `fetch_institutions()` no longer filters to `ACTIVE:1` by default — see
  the "Notes" section of `camel_ratios.build_camel_ratios()`'s docstring.
  Keep this in mind for any future query against a regulatory API that has
  its own notion of "current" vs. "historical" records.
- **Verify a field's actual meaning, not just its shape** — a second real
  bug, found while wiring up the backend: `RBCT1J` was used as
  `tier1_ratio` since Phase 0, and it *looks* like a ratio column (a
  reasonable-looking number per row) but is actually **Tier 1 capital in
  dollars** (thousands) — confirmed by pulling one bank's raw fields
  directly and noticing `RBCT1J` sat right next to `EQ` (also a dollar
  figure) while `RBC1RWAJ` was clearly a percentage. `RBC1RWAJ` (Tier 1
  risk-based capital ratio, %) is the field actually used now, clipped to
  `[0, 50]` to guard against a handful of outlier records in FDIC's own
  data. Lesson: a column full of plausible-*looking* numbers is not the
  same as a column with the *right* numbers — check a value's magnitude
  against a real-world sense of the unit, not just whether the code runs.
- **A "liquid assets" ratio was never actually requested** — `FINANCIAL_FIELDS`
  in `fdic_client.py` never included a real liquid-assets Call Report
  field, so `liquid_asset_ratio` was 100% missing (`NaN`) on every live
  pull — invisible in a single `.head()` preview, since a NaN cell doesn't
  look different from a normal one until you check `.notna().sum()`.
  `loan_deposit_ratio` (loans / deposits, computed from `LNLSNET` and
  `DEP` — both real, always-present fields) is now the Liquidity ("L")
  input everywhere; `liquid_asset_ratio` is kept only as a descriptive,
  unused column in case a real field is added later. Dividing by `DEP`
  can itself produce `inf` for the rare bank reporting zero deposits —
  handled by replacing `[inf, -inf]` with `NaN` before any fillna/plotting
  step touches the column (see `baseline.prepare_training_data()`).
- **Network access** — some sandboxed/CI environments block outbound
  requests to `banks.data.fdic.gov` (this one, at time of writing, does
  not). Every function in `src/camel_sentinel/data/fdic_client.py` that
  calls the live API has a clearly-labeled synthetic fallback with the
  *same schema and realistic distributions*, so the full pipeline
  (ratios → composite score → EDA → modeling/XAI/RAI) still runs wherever
  it's executed. The synthetic sample is never to be treated as a finding
  about real banks.

## Sandbox source — Kaggle ratio→rating datasets

Used **only** as a fast, zero-setup local sandbox to prototype the
modeling pipeline's *shape* (classification/regression on financial ratios,
handling imbalance) before pointing the same code at the FDIC data — not as
a replacement for it.

| Dataset | Why it's useful here |
|---|---|
| [Corporate Credit Rating with Financial Ratios](https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios) | Ratio-in, ordinal-rating-out — nearly identical supervised-learning shape to what we need |
| [Corporate Credit Rating](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating) | Same idea, different companies/ratios — a second opinion / robustness check on the ratio→rating classifier design |
| [Predict Bankruptcy in Poland](https://www.kaggle.com/datasets/stealthtechnologies/predict-bankruptcy-in-poland) (Tomczak/Zięba et al., EMIS 2000–2013) | The most-cited proxy dataset in CAMEL-style bankruptcy literature — 64 accounting ratios + bankrupt/not label, heavily imbalanced (mirrors real bank-failure imbalance) |

## Benchmarking-only sources — published academic CAMEL(S) tables

Small, manually compiled datasets from academic paper appendices. Not used
for training — used to sanity-check that our ratio construction and
composite-scoring formula match the literature-standard recipe:

- Turkish commercial banks (Bank Association of Turkey, 1994–2004)
- Indonesian rural banks (OJK, 2013–2019)
- arXiv 2407.11089 — "Explainable bank failure prediction models:
  Counterfactual explanations to reduce the failure risk" (FDIC 2008–2023
  data, with an explicit CAMELS predictor-ratio table this project mirrors
  closely)

## Why not just use real CAMELS ratings?

Because they don't exist in any public dataset — see
[`00_problem_statement.md`](00_problem_statement.md) section 1. This isn't
a data-access problem to work around; it's the reason the whole project is
built around a transparent proxy label in the first place.
