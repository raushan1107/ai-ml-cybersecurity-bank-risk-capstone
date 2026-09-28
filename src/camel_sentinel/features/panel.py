"""
panel.py — Phase 4: pool many quarters into one training panel.

WHAT this file does: `build_camel_ratios()` (Phase 1) only ever sees one
quarter's financials joined against the *full* failure history, so its
`failed` flag really means "this bank failed at some point, ever" — most
of FDIC's ~4,100 historical failures are banks that closed decades before
2023-12-31 and simply aren't in that one snapshot at all. Only 15 of the
4,658 banks in the 2023-12-31 snapshot are labeled `failed`, which is too
few positive examples for any model comparison to be more than noise (see
notebooks/02_baseline_modeling.ipynb's caveat).

This module fixes that by pulling *many* quarters of financials — enough
to include the 2008-2012 crisis, where the bulk of real U.S. bank failures
happened — and labeling each bank-quarter with a proper forward-looking
target: "did this bank fail within the next `horizon_days`?" instead of
"has this bank ever failed, at any point in FDIC history?". That is both
more data (hundreds of positive bank-quarters instead of 15) and a more
honest label (no look-ahead: a 2005 snapshot is judged against failures
through ~2007, not against a 2023 snapshot's failures list).

HOW it connects: reuses `fdic_client.fetch_financials/fetch_failures` and
`camel_ratios.build_camel_ratios` per quarter (so the ratio math itself is
never duplicated), then overwrites each quarter's `failed` column with the
windowed label below and concatenates every quarter into one long panel —
the same shape `models.baseline.prepare_training_data()` already expects.
"""

from __future__ import annotations

import pandas as pd

from camel_sentinel.data import fdic_client
from camel_sentinel.features import camel_ratios


def label_failed_within_horizon(
    camel_quarter: pd.DataFrame,
    failures: pd.DataFrame,
    report_date: str,
    horizon_days: int = 730,
) -> pd.Series:
    """
    Replace "ever failed" with "failed within `horizon_days` of this
    specific quarter" — the forward-looking, leakage-free target.

    Parameters
    ----------
    camel_quarter : pandas.DataFrame
        One quarter's output of `camel_ratios.build_camel_ratios()` — only
        the `CERT` column is used here.
    failures : pandas.DataFrame
        Full failure history from `fdic_client.fetch_failures()` — must
        contain `CERT` and `FAILDATE`.
    report_date : str
        The quarter this snapshot was taken, "YYYYMMDD" (matches
        `fdic_client.fetch_financials(report_date=...)`).
    horizon_days : int, default 730
        How far forward to look for a failure (730 days = 2 years — long
        enough to catch failures a bank's ratios were already signaling
        trouble for, short enough that "failed" still means something
        close in time to the ratios being scored).

    Returns
    -------
    pandas.Series of int (0/1)
        1 if this CERT's earliest failure date falls in
        `(report_date, report_date + horizon_days]`; 0 otherwise —
        including banks that failed *before* this report_date (already
        gone, not a forward-looking failure) or more than `horizon_days`
        after it (too far out to attribute to this quarter's ratios).
    """
    report_dt = pd.to_datetime(report_date, format="%Y%m%d")
    horizon_end = report_dt + pd.Timedelta(days=horizon_days)

    # groupby("CERT")["FAILDATE"].min() — a bank appears once per failure
    # in the raw table (almost always once, period), but .min() guards
    # against any duplicate/rescue-then-refail rows collapsing to the
    # earliest date, which is the one relevant "did it fail" event.
    first_faildate = (
        failures.assign(FAILDATE=pd.to_datetime(failures["FAILDATE"]))
        .groupby("CERT")["FAILDATE"]
        .min()
    )
    matched_faildate = camel_quarter["CERT"].map(first_faildate)
    in_window = (matched_faildate > report_dt) & (matched_faildate <= horizon_end)
    return in_window.astype(int)


def build_multi_quarter_panel(
    report_dates: list[str],
    horizon_days: int = 730,
    limit: int = 10000,
) -> pd.DataFrame:
    """
    Fetch and stack many quarters of CAMEL ratios into one training panel,
    each row labeled with the forward-looking `failed` target above.

    Parameters
    ----------
    report_dates : list[str]
        Quarters to pull, each "YYYYMMDD" (e.g. "20081231") — passed
        straight to `fdic_client.fetch_financials(report_date=...)`.
    horizon_days : int, default 730
        Passed to `label_failed_within_horizon()`.
    limit : int, default 10000
        Passed to `fdic_client.fetch_financials(limit=...)` — 10000 is the
        FDIC API's own request cap and comfortably covers even 2005's
        ~9,000 filing banks (a single quarter never exceeds that today).

    Returns
    -------
    pandas.DataFrame
        Every quarter's `camel_ratios.build_camel_ratios()` output,
        concatenated, with `REPDTE` added and `failed` overwritten by the
        windowed label. A quarter that fails to fetch (rare API hiccup) is
        skipped with a printed warning rather than aborting the whole
        panel — one bad quarter shouldn't cost every other quarter's data.

    Notes
    -----
    `institutions` and `failures` are each fetched once (not once per
    quarter) since both are already the full, quarter-independent table —
    refetching them 20+ times would be wasted API calls for identical data.
    """
    institutions = fdic_client.fetch_institutions()
    failures = fdic_client.fetch_failures()

    frames = []
    for report_date in report_dates:
        try:
            financials = fdic_client.fetch_financials(report_date=report_date, limit=limit)
        except Exception as exc:
            print(f"Skipping {report_date}: fetch failed ({type(exc).__name__}: {exc})")
            continue
        if financials.empty:
            print(f"Skipping {report_date}: no financials returned")
            continue

        camel_quarter = camel_ratios.build_camel_ratios(institutions, financials, failures)
        camel_quarter["REPDTE"] = report_date
        camel_quarter["failed"] = label_failed_within_horizon(
            camel_quarter, failures, report_date, horizon_days
        )
        frames.append(camel_quarter)
        print(f"{report_date}: {len(camel_quarter)} banks, {camel_quarter['failed'].sum()} failed-within-{horizon_days}d")

    panel = pd.concat(frames, ignore_index=True)
    return panel
