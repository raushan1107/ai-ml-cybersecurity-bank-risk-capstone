"""
camel_ratios.py — Phase 1/2: turn raw Call Report fields into the CAMEL
ratio "recipe", then into a single composite health score / proxy rating.

WHAT this file does: two steps.
  1. `build_camel_ratios()` — merges the institution/financial data pulled
     by `camel_sentinel.data.fdic_client` and computes one ratio per CAMEL
     component per bank-quarter.
  2. `score_composite()` — combines those six ratios into one comparable
     "composite_score_0_100" (a regression target) and buckets it into a
     "proxy_rating_1_5" (a classification target, 1 = healthiest).

WHY the ratios are built this way: real CAMELS ratings are confidential
(see docs/00_problem_statement.md), so there is nothing to train against
directly — the composite score built here *is* the label, which is why its
construction has to be transparent and literature-backed rather than a
black box. See docs/CAMEL-Rating-ML-Project-Blueprint.md Section 3 for the
ratio table this mirrors.

HOW it connects to the rest of the pipeline: takes the three DataFrames
from `fdic_client.py` as input; its output (`camel_df`) is what
`notebooks/00_dataset_research.ipynb` runs EDA on, and what Phase 3's
baseline-modeling notebook will train the real (non-rule-based) models on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_camel_ratios(
    institutions: pd.DataFrame,
    financials: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge institution + financial data and compute the six CAMEL ratio
    groups, one row per bank-quarter.

    Parameters
    ----------
    institutions : pandas.DataFrame
        Output of `fdic_client.fetch_institutions()` (or the synthetic
        equivalent) — must contain CERT, NAME, STALP, ASSET.
    financials : pandas.DataFrame
        Output of `fdic_client.fetch_financials()` (or synthetic) — must
        contain CERT plus the Call Report fields listed in
        `fdic_client.FINANCIAL_FIELDS`.
    failures : pandas.DataFrame
        Output of `fdic_client.fetch_failures()` (or synthetic) — must
        contain CERT. Only used here to attach the `failed` flag; the
        proxy-rating construction itself never looks at this column, so
        the rating can be validated *against* it afterwards without
        circularity (see the sanity-check step in the notebook).

    Returns
    -------
    pandas.DataFrame
        One row per bank, columns: CERT, NAME, STALP, ASSET, the six raw
        ratio columns (one or more per CAMEL component, named below), and
        `failed` (1 if this CERT appears in `failures`, else 0).

    Notes
    -----
    `financials` — not `institutions` — is the base table this merges
    onto, and the merge is a `left` join. This is deliberate: `financials`
    is the actual population we're building ratios for (every bank that
    filed a Call Report that quarter); `institutions` only *enriches* each
    row with identity metadata where available. Using `institutions` as
    the base (or an inner join) would silently drop any bank whose CERT
    isn't in whatever `institutions` pull was passed in — which, if that
    pull was limited or filtered to currently-active banks, is exactly the
    set of banks most likely to have since failed. Losing precisely the
    rows this whole project cares about most is the bug this ordering
    avoids.
    """
    # DataFrame.merge(other, on=..., how="left", suffixes=...) — a SQL-style
    # left join; `on="CERT"` joins the two tables on the shared bank ID,
    # `how="left"` keeps every row of `financials` (the left/base table)
    # even if no matching institution metadata exists, and `suffixes`
    # disambiguates any column name that appears in both frames (e.g. both
    # may carry an ASSET-like field from different vintages of the API).
    df = financials.merge(institutions, on="CERT", how="left", suffixes=("", "_inst"))

    # Series.isin(other_series) — True for every CERT that also appears in
    # the failures table; .astype(int) turns the resulting bool column
    # into a plain 0/1 flag the rest of the pipeline can treat as numeric.
    df["failed"] = df["CERT"].isin(failures["CERT"]).astype(int)

    camel = pd.DataFrame({
        "CERT": df["CERT"],
        "NAME": df["NAME"],
        "STALP": df["STALP"],
        "ASSET": df["ASSET"],
    })

    # --- C: Capital adequacy — buffer against losses ---
    # Tier 1 risk-based capital ratio (%): higher = more capital cushion =
    # healthier. Deliberately RBC1RWAJ, not RBCT1J — RBCT1J is a DOLLAR
    # AMOUNT (Tier 1 capital in thousands), not a percentage; an earlier
    # version of this pipeline used it directly as "tier1_ratio" and it
    # went undetected through several phases because a dollar figure still
    # *looks* like a plausible-shaped column until you check its scale
    # against a bank's actual assets. See the comment on RBCT1J in
    # fdic_client.py's FINANCIAL_FIELDS for how this was confirmed.
    #
    # .clip(lower=0, upper=50): a small number of raw Call Report records
    # report implausible outlier values for this field (a real bank's
    # Tier 1 ratio above ~30% is already exceptional; values in the
    # thousands are almost certainly a reporting/vintage artifact in
    # FDIC's own data, not a real bank). Capping at 50% keeps a handful of
    # bad rows from dominating z_capital's mean/std below — z-scores are
    # sensitive to exactly this kind of outlier. This is a blunt guard,
    # not a data-cleaning pass; Phase 4 should audit which specific CERTs
    # get clipped and why, rather than silently capping them forever.
    camel["tier1_ratio"] = df["RBC1RWAJ"].clip(lower=0, upper=50)

    # --- A: Asset quality — risk in the loan/investment book ---
    camel["npl_ratio"] = df["NPERFV"]           # non-performing loans / total loans — higher = riskier
    camel["provision_ratio"] = df["LNATRES"]     # loan-loss allowance — higher usually tracks higher recognized risk

    # --- M: Management quality — regulators observe this on-site; we can't ---
    # No CAMEL component has a weaker financial-statement proxy than this
    # one. We use the efficiency ratio (non-interest expense / revenue) as
    # a standard literature substitute — a well-run bank tends to control
    # costs relative to revenue — but this is explicitly the weakest link
    # in the composite score and must be called out in every downstream
    # explanation (see docs/00_problem_statement.md section 4 and
    # docs/01_glossary.md).
    camel["efficiency_ratio"] = df["EEFFR"]      # lower = more efficiently run = healthier

    # --- E: Earnings — profitability & sustainability ---
    camel["roa"] = df["ROA"]     # return on assets — higher = healthier
    camel["roe"] = df["ROE"]     # return on equity — higher = healthier
    camel["nim"] = df["NIMY"]    # net interest margin — higher = healthier

    # --- L: Liquidity — ability to meet obligations without a fire sale ---
    # This is the ratio actually used for scoring/modeling below. Prefer a
    # precomputed "LDR" column if the source provides one (our synthetic
    # sample does); otherwise derive loans/deposits directly from
    # LNLSNET and DEP, which ARE always present in FINANCIAL_FIELDS —
    # unlike liquid_asset_ratio just below, this ratio is reliably
    # available on real FDIC data.
    camel["loan_deposit_ratio"] = (
        df["LDR"] if "LDR" in df.columns else (df["LNLSNET"] / df["DEP"] * 100)
    )
    # Descriptive only — NOT used for scoring or modeling. FINANCIAL_FIELDS
    # (fdic_client.py) never requests a liquid-assets Call Report field, so
    # this is 100% missing on real FDIC data (only the synthetic fallback
    # populates it, as "LIQ_RATIO"). Discovered by running the backend
    # against live data: a model/score built on this column alone would be
    # trained on an entirely empty feature. `loan_deposit_ratio` above is
    # the real Liquidity input everywhere downstream; this column is kept
    # only in case a future FDIC field pull adds true liquid-asset data.
    camel["liquid_asset_ratio"] = df["LIQ_RATIO"] if "LIQ_RATIO" in df.columns else np.nan

    # --- S: Sensitivity to market risk — exposure to rate/market moves ---
    # Loan concentration as a share of total assets, as a simple proxy:
    # a bank whose balance sheet is almost entirely loans has less room
    # to absorb a rate shock than one with a more diversified asset mix.
    camel["loan_concentration"] = df["LNLSNET"] / df["ASSET"] * 100

    camel["failed"] = df["failed"]
    return camel


def _healthy_z(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Convert one raw ratio column into a z-score pointed in the "healthier
    direction", so every component can be averaged on the same scale
    regardless of its original units or whether a high or low raw value is
    actually good news (see "Composite z-score" in docs/01_glossary.md).

    Parameters
    ----------
    series : pandas.Series
        The raw ratio column to standardize (e.g. camel["tier1_ratio"]).
    higher_is_better : bool
        True if a larger raw value means a healthier bank (e.g. Tier 1
        capital ratio); False if a smaller raw value is healthier (e.g.
        the efficiency ratio, or the non-performing-loan ratio) — the
        z-score is negated in that case so "higher composite = healthier"
        holds consistently across every component.

    Returns
    -------
    pandas.Series
        The z-scored (and sign-flipped if needed) column: mean 0, std 1,
        higher = healthier in every case.
    """
    z = (series - series.mean()) / series.std()
    return z if higher_is_better else -z


def score_composite(camel: pd.DataFrame) -> pd.DataFrame:
    """
    Build the composite CAMEL score (0-100, regression target) and the
    proxy 1-5 rating (classification target) from the ratio columns
    produced by `build_camel_ratios()`.

    Parameters
    ----------
    camel : pandas.DataFrame
        Output of `build_camel_ratios()` — must contain tier1_ratio,
        npl_ratio, efficiency_ratio, roa, liquid_asset_ratio, and
        loan_concentration.

    Returns
    -------
    pandas.DataFrame
        `camel` with six new z-score columns (`z_capital` ... `z_sensitivity`),
        `composite_z` (their mean), `composite_score_0_100` (rescaled for
        readability), and `proxy_rating_1_5` (quintile-bucketed, 1 =
        healthiest, matching CAMELS' own 1-best/5-worst convention).
    """
    camel = camel.copy()

    camel["z_capital"] = _healthy_z(camel["tier1_ratio"], higher_is_better=True)
    camel["z_asset_quality"] = _healthy_z(camel["npl_ratio"], higher_is_better=False)
    camel["z_management"] = _healthy_z(camel["efficiency_ratio"], higher_is_better=False)
    camel["z_earnings"] = _healthy_z(camel["roa"], higher_is_better=True)
    # loan_deposit_ratio, not liquid_asset_ratio: see the comment on
    # liquid_asset_ratio in build_camel_ratios() — it's 100% missing on
    # real FDIC data. A lower loan/deposit ratio means more of a bank's
    # deposits are backed by liquid assets rather than tied up in loans,
    # so lower is healthier here.
    camel["z_liquidity"] = _healthy_z(camel["loan_deposit_ratio"], higher_is_better=False)
    camel["z_sensitivity"] = _healthy_z(camel["loan_concentration"], higher_is_better=False)

    component_cols = [
        "z_capital", "z_asset_quality", "z_management",
        "z_earnings", "z_liquidity", "z_sensitivity",
    ]
    # DataFrame[cols].mean(axis=1) — row-wise average across the six
    # component z-scores; axis=1 means "average across columns, per row"
    # (axis=0, the default, would average each column down its rows instead).
    camel["composite_z"] = camel[component_cols].mean(axis=1)

    # Min-max rescale composite_z into a 0-100 range purely for human
    # readability in reports/UI — it does not change the ranking of banks,
    # only the units the score is expressed in.
    camel["composite_score_0_100"] = (
        (camel["composite_z"] - camel["composite_z"].min())
        / (camel["composite_z"].max() - camel["composite_z"].min())
        * 100
    ).round(1)

    # pd.qcut(series, q, labels) — splits the data into `q` equal-sized
    # buckets (quintiles here, q=5) by value, not by a fixed threshold —
    # so exactly ~20% of banks fall in each rating bucket by construction.
    # `labels=[5, 4, 3, 2, 1]` assigns the *lowest* composite_z quintile to
    # label 5 (worst) and the *highest* to label 1 (best), matching the
    # real CAMELS 1-best/5-worst convention.
    camel["proxy_rating_1_5"] = pd.qcut(
        camel["composite_z"], q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    return camel
