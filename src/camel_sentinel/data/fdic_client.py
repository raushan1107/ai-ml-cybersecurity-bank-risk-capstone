"""
fdic_client.py — Phase 1: pull the raw data the whole project is built on.

WHAT this file does: fetches bank identity (`institutions`), quarterly
financial fields (`financials`), and historical bank failures (`failures`)
from the free, public FDIC BankFind Suite API. If that API can't be
reached (e.g. this development sandbox's network doesn't allow it — see
docs/02_data_sources.md), `build_synthetic_sample()` generates a
schema-matched, realistic-distribution stand-in so every downstream step
(feature engineering, EDA, and later modeling/XAI/RAI) can still be built
and demonstrated end-to-end.

WHY it's a separate module instead of notebook cells: this logic needs to
be reused identically from `notebooks/00_dataset_research.ipynb`, from the
FastAPI backend (built in a later phase), and from the agent's "fetch a
bank's financials" tool — one tested implementation, three callers.

HOW it connects to the rest of the pipeline: the three DataFrames returned
here (`institutions`, `financials`, `failures`) are the direct input to
`camel_sentinel.features.camel_ratios.build_camel_ratios()`, the next step
in Phase 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

# The FDIC BankFind Suite API's base URL. All three endpoints we use
# (institutions, financials, failures) hang off this one base.
# Docs: https://api.fdic.gov/banks/docs
FDIC_BASE_URL = "https://banks.data.fdic.gov/api"

# The Call Report field codes we need to build the six CAMEL ratio groups
# (see camel_sentinel.features.camel_ratios for what each ratio is used
# for). Requesting exactly these fields — instead of every field the API
# offers — keeps each response small and keeps this file the single place
# that documents what each code means.
FINANCIAL_FIELDS = [
    "CERT",       # unique bank certificate number — the join key across all three endpoints
    "REPDTE",     # report date (quarter end), e.g. "20231231"
    "ASSET",      # total assets (thousands of $) — the denominator for most ratios
    "EQ",         # total equity capital — numerator for basic capital-adequacy ratios
    "RBCT1J",     # Tier 1 (core) capital — a DOLLAR AMOUNT (thousands), not a ratio.
                  # Confirmed against the live API: for one bank, RBCT1J=19554
                  # sits right next to EQ=22294 (both dollar figures) while
                  # RBC1RWAJ=9.76 is clearly the percentage. Kept only in case a
                  # leverage ratio (RBCT1J / ASSET) is wanted later — capital
                  # adequacy scoring below uses RBC1RWAJ, never this field directly.
    "RBC1RWAJ",   # Tier 1 risk-based capital ratio (%) — the actual ratio used
                  # for Capital adequacy (the "C" in CAMEL); see camel_ratios.py.
    "NPERFV",     # total nonperforming loans and leases — Asset quality (the "A")
    "LNATRES",    # loan & lease loss allowance — Asset quality (loss-absorption reserve)
    "NETINC",     # net income (thousands of $) — Earnings (the "E")
    "NIMY",       # net interest margin (%) — Earnings
    "ROA",        # return on assets (%), FDIC-precomputed — Earnings
    "ROE",        # return on equity (%), FDIC-precomputed — Earnings
    "EEFFR",      # efficiency ratio (%) — Management proxy (the "M"; see camel_ratios.py for the caveat)
    "LNLSNET",    # net loans and leases (thousands of $) — Liquidity / Sensitivity inputs
    "DEP",        # total deposits (thousands of $) — Liquidity (the "L")
]


def _fdic_get(endpoint: str, params: dict, timeout: int = 30) -> pd.DataFrame:
    """
    Thin, shared wrapper around one FDIC API GET request.

    Parameters
    ----------
    endpoint : str
        Which API endpoint to call — one of "institutions", "financials",
        "failures". Appended to FDIC_BASE_URL to form the full URL.
    params : dict
        Query-string parameters passed straight through to `requests.get`.
        The FDIC API expects, in particular:
          - "filters": a string like "REPDTE:20231231" or "ACTIVE:1" that
            narrows which rows come back (required for financials/failures
            to avoid pulling the entire multi-decade history at once).
          - "fields": a comma-separated string of field codes to return
            (keeps the response to just what we need — see FINANCIAL_FIELDS
            above for the financials example).
          - "limit": max rows to return in one call (the API paginates
            beyond this; Phase 2 will add pagination when pulling multiple
            quarters, one page is enough for this single-quarter Phase 1
            pull).
          - "format": "json" — the response shape this function assumes.
    timeout : int, default 30
        Seconds to wait for the API before giving up — passed straight to
        `requests.get(timeout=...)`. Optional; 30s is generous for a
        single-page JSON response.

    Returns
    -------
    pandas.DataFrame
        One row per record returned by the API, columns = the requested
        `fields`.
    """
    url = f"{FDIC_BASE_URL}/{endpoint}"
    # requests.get(url, params=..., timeout=...) — a plain HTTP GET; params
    # becomes the URL's query string, timeout bounds how long we'll wait.
    resp = requests.get(url, params=params, timeout=timeout)
    # raise_for_status() turns a non-2xx HTTP response into an exception
    # immediately, instead of silently returning an error payload as if it
    # were data — callers rely on this to fall back to the synthetic sample.
    resp.raise_for_status()
    payload = resp.json()
    # The FDIC API wraps each record as {"data": {...actual fields...}} —
    # unwrap that envelope so the DataFrame has plain column names.
    rows = [record["data"] for record in payload.get("data", [])]
    return pd.DataFrame(rows)


def fetch_institutions(limit: int = 10000, active_only: bool = False) -> pd.DataFrame:
    """
    Fetch bank identity/metadata from the `institutions` endpoint.

    Parameters
    ----------
    limit : int, default 10000
        Max number of institutions to return in this single-page pull.
        Optional — raised well above the ~4,600 banks a single quarter's
        `financials` pull typically returns, specifically so a later merge
        against `financials` on CERT doesn't silently truncate to whichever
        institutions happened to be returned first (see `active_only` below
        for the other half of that bug).
    active_only : bool, default False
        If True, adds an `ACTIVE:1` filter, returning only banks operating
        *today*. **Deliberately off by default**: a bank that failed after
        the financials snapshot you're joining against (e.g. failed in 2024
        after filing a Q4-2023 Call Report) is, by definition, not active
        today — filtering to `ACTIVE:1` silently removes every such bank's
        identity metadata, and because `camel_ratios.build_camel_ratios()`
        merges on CERT, that removal used to drop the bank from the CAMEL
        table entirely, not just mislabel it. Leave this False unless you
        specifically only want banks that are still open right now.

    Returns
    -------
    pandas.DataFrame
        Columns: CERT (join key), NAME, STALP (state), CHARTER (charter
        type code), ASSET (total assets), BKCLASS (regulatory class) — used
        later for grouping in EDA and for the Phase 6 fairness audit.
    """
    params = {
        "fields": "CERT,NAME,STALP,CHARTER,ASSET,BKCLASS",
        "limit": limit,
        "format": "json",
    }
    if active_only:
        params["filters"] = "ACTIVE:1"
    return _fdic_get("institutions", params=params)


def fetch_financials(report_date: str = "20231231", limit: int = 5000) -> pd.DataFrame:
    """
    Fetch one quarter of Call Report financial fields from the
    `financials` endpoint — the raw inputs to every CAMEL ratio.

    Parameters
    ----------
    report_date : str, default "20231231"
        Which quarter to pull, in FDIC's "YYYYMMDD" quarter-end format
        (e.g. "20231231" = Q4 2023). Required by the API's REPDTE filter;
        defaults to the most recent full year-end in the blueprint's
        starter pull. Phase 2 will loop this over multiple quarters to
        build the "failed within N quarters" label properly.
    limit : int, default 5000
        Max rows for this single-page pull. Optional.

    Returns
    -------
    pandas.DataFrame
        One row per bank for the given quarter, columns = FINANCIAL_FIELDS.
    """
    return _fdic_get(
        "financials",
        params={
            "filters": f"REPDTE:{report_date}",
            "fields": ",".join(FINANCIAL_FIELDS),
            "limit": limit,
            "format": "json",
        },
    )


def fetch_failures(limit: int = 5000) -> pd.DataFrame:
    """
    Fetch the historical bank-failure catalog from the `failures` endpoint
    — this is the project's proxy-label ground truth (see
    docs/00_problem_statement.md, label option 2).

    Parameters
    ----------
    limit : int, default 5000
        Max rows to return. Optional; the full FDIC failure history since
        the 1930s is a few thousand records, well under this default.

    Returns
    -------
    pandas.DataFrame
        Columns: CERT (join key), NAME, FAILDATE, CITYST, SAVR (receiver),
        COST (estimated cost to the deposit insurance fund).
    """
    return _fdic_get(
        "failures",
        params={
            "fields": "CERT,NAME,FAILDATE,CITYST,SAVR,COST",
            "limit": limit,
            "format": "json",
        },
    )


def try_live_pull(report_date: str = "20231231") -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, bool]:
    """
    Attempt the real, three-endpoint FDIC pull, and report whether it
    actually succeeded — the caller decides whether to fall back to
    `build_synthetic_sample()` based on the returned success flag.

    Parameters
    ----------
    report_date : str, default "20231231"
        Passed straight through to `fetch_financials()` — see its
        docstring above.

    Returns
    -------
    tuple(institutions, financials, failures, live_pull_ok)
        The three DataFrames (each None if the pull raised before
        completing), and a bool that is True only if all three came back
        non-empty.
    """
    try:
        institutions = fetch_institutions()
        financials = fetch_financials(report_date=report_date)
        failures = fetch_failures()
        live_pull_ok = all(
            df is not None and len(df) > 0 for df in (institutions, financials, failures)
        )
        return institutions, financials, failures, live_pull_ok
    except Exception as exc:
        # Broad except is deliberate here: any failure mode (no network,
        # DNS blocked, timeout, HTTP error) means the same thing to the
        # caller — "live pull didn't work, use the synthetic fallback" —
        # and we don't want an environment-specific network exception to
        # crash the whole pipeline.
        print(f"Live FDIC pull not available ({type(exc).__name__}: {exc}). Falling back to synthetic sample.")
        return None, None, None, False


def build_synthetic_sample(n_banks: int = 1500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a schema-matched, realistically-distributed synthetic stand-in
    for (institutions, financials, failures) — used only when the live
    FDIC pull above isn't reachable.

    Do not treat this as a finding about real banks (see
    docs/02_data_sources.md) — swap in `try_live_pull()` the moment this
    runs somewhere with internet access to `banks.data.fdic.gov`.

    Parameters
    ----------
    n_banks : int, default 1500
        How many synthetic banks to generate. Optional.
    seed : int, default 42
        Seed for `numpy.random.default_rng` — fixed by default so the
        synthetic sample (and every ratio/score built on top of it) is
        reproducible across runs.

    Returns
    -------
    tuple(institutions, financials, failures)
        Same columns/shape as the real endpoints would return, so every
        downstream function (camel_ratios.build_camel_ratios, etc.) works
        identically on either source.
    """
    # np.random.default_rng(seed) — NumPy's recommended random generator
    # (replaces the older np.random.seed global-state API); `rng.<dist>`
    # calls below draw from it deterministically for the given seed.
    rng = np.random.default_rng(seed)
    cert = np.arange(100000, 100000 + n_banks)

    institutions = pd.DataFrame({
        "CERT": cert,
        "NAME": [f"Synthetic Bank {i}" for i in range(n_banks)],
        # rng.choice(options, size) — uniform random pick from a fixed list,
        # one per bank; stands in for real state/charter-type metadata.
        "STALP": rng.choice(["NY", "CA", "TX", "IL", "OH", "GA", "FL", "PA"], n_banks),
        "CHARTER": rng.choice(["N", "SM", "SB", "NM"], n_banks),
        # rng.lognormal(mean, sigma, size) — a right-skewed distribution
        # (many small banks, a few huge ones), which is how real bank
        # asset sizes are actually shaped, unlike a symmetric normal draw.
        "ASSET": rng.lognormal(mean=12, sigma=1.5, size=n_banks).round(0),
        "BKCLASS": rng.choice(["N", "SM", "SB", "NM"], n_banks),
    })

    # Each ratio drawn from a distribution shaped like its real-world
    # counterpart (e.g. non-performing loans is right-skewed and can't go
    # negative, so gamma + clip; capital ratio is roughly bell-shaped, so
    # normal + clip to a plausible range).
    tier1 = rng.normal(11, 3, n_banks).clip(2, 25)               # Tier 1 risk-based capital ratio (%)
    npl_ratio = rng.gamma(1.5, 0.8, n_banks).clip(0, 15)          # nonperforming loans / total loans (%)
    provision_ratio = (npl_ratio * rng.uniform(0.3, 0.6, n_banks)).clip(0, 8)
    # ROA is built with a deliberate negative relationship to npl_ratio —
    # banks with more bad loans tend to earn less — so downstream EDA and
    # modeling have a real, known signal to find, not pure noise.
    roa = rng.normal(1.0, 0.6, n_banks) - 0.15 * (npl_ratio - npl_ratio.mean())
    efficiency_ratio = rng.normal(62, 12, n_banks).clip(30, 110)  # non-interest expense / revenue (%)
    ldr = rng.normal(80, 15, n_banks).clip(30, 130)               # loans / deposits (%)
    liquid_asset_ratio = rng.normal(18, 8, n_banks).clip(2, 50)   # liquid assets / total assets (%)

    financials = pd.DataFrame({
        "CERT": cert,
        "REPDTE": "20231231",
        "ASSET": institutions["ASSET"],
        "EQ": (institutions["ASSET"] * (tier1 / 100) * rng.uniform(0.8, 1.1, n_banks)).round(0),
        "RBCT1J": tier1.round(2),
        "RBC1RWAJ": (tier1 * rng.uniform(0.9, 1.1, n_banks)).round(2),
        "NPERFV": npl_ratio.round(2),
        "LNATRES": provision_ratio.round(2),
        "NETINC": (roa / 100 * institutions["ASSET"]).round(0),
        "NIMY": rng.normal(3.3, 0.7, n_banks).clip(0.5, 6).round(2),
        "ROA": roa.round(2),
        "ROE": (roa * rng.uniform(8, 12, n_banks)).round(2),
        "EEFFR": efficiency_ratio.round(2),
        "LNLSNET": (institutions["ASSET"] * rng.uniform(0.5, 0.8, n_banks)).round(0),
        "DEP": (institutions["ASSET"] * rng.uniform(0.75, 0.92, n_banks)).round(0),
        "LDR": ldr.round(2),
        "LIQ_RATIO": liquid_asset_ratio.round(2),
    })

    # Build a synthetic "distress score" as a weighted combination of the
    # same signals a real composite CAMEL score would use, then flag the
    # worst ~3% as "failed" — matching the real-world FDIC failure rate's
    # order of magnitude, so downstream class-imbalance handling (Phase 3+)
    # has a realistic imbalance to practice on, not an arbitrary one.
    distress_score = (
        -0.35 * (tier1 - tier1.mean()) / tier1.std()
        + 0.30 * (npl_ratio - npl_ratio.mean()) / npl_ratio.std()
        - 0.20 * (roa - roa.mean()) / roa.std()
        + 0.15 * (efficiency_ratio - efficiency_ratio.mean()) / efficiency_ratio.std()
        - 0.10 * (liquid_asset_ratio - liquid_asset_ratio.mean()) / liquid_asset_ratio.std()
        + rng.normal(0, 0.5, n_banks)
    )
    # np.quantile(array, 0.97) — the value below which 97% of scores fall;
    # everything at/above it becomes our synthetic "failed" flag, i.e. a
    # 3% synthetic failure rate.
    fail_threshold = np.quantile(distress_score, 0.97)
    failed_mask = distress_score >= fail_threshold

    failures = pd.DataFrame({
        "CERT": cert[failed_mask],
        "NAME": institutions.loc[failed_mask, "NAME"].values,
        "FAILDATE": "2024-06-30",
        "CITYST": institutions.loc[failed_mask, "STALP"].values,
        "SAVR": "SYN",  # marks this as synthetic, never a real receiver code
        "COST": rng.uniform(1e5, 5e7, failed_mask.sum()).round(0),
    })

    return institutions, financials, failures
