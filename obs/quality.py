"""Data-quality & drift checks (build brief §5b). Mostly ``warn`` -- trends, not blocks.

Surfaces the out-of-time shift (train spans 18 months; val/test are a later
13-week window) as per-feature PSI / KS, plus null-rate deviation, unseen
categorical levels, and range sanity.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from . import contracts as C
from .loader import Artifacts
from .results import CheckResult, Severity

WARN = Severity.WARN
T = C.THRESHOLDS


# --------------------------------------------------------------------------- #
# Drift statistics
# --------------------------------------------------------------------------- #


def psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = T.psi_bins) -> float:
    """Population Stability Index using quantile bins fitted on ``expected``."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_cnt = np.histogram(expected, edges)[0].astype(float)
    a_cnt = np.histogram(actual, edges)[0].astype(float)
    eps = 1e-6
    e_pct = e_cnt / e_cnt.sum() + eps
    a_pct = a_cnt / a_cnt.sum() + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_categorical(expected: pl.Series, actual: pl.Series) -> float:
    """PSI over categorical levels (union of observed levels)."""
    e = expected.drop_nulls()
    a = actual.drop_nulls()
    levels = sorted(set(e.unique().to_list()) | set(a.unique().to_list()))
    if not levels:
        return float("nan")
    eps = 1e-6
    e_pct = np.array([(e == lvl).sum() for lvl in levels], float) / max(len(e), 1) + eps
    a_pct = np.array([(a == lvl).sum() for lvl in levels], float) / max(len(a), 1) + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _band(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value >= T.psi_alert:
        return "large"
    if value >= T.psi_warn:
        return "moderate"
    return "stable"


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_null_rates(art: Artifacts) -> CheckResult:
    """Per-feature null rate by split; flag splits deviating from train."""
    train = art.raw["train"]
    rows = []
    offenders = 0
    for feat in C.FEATURES:
        base = train[feat].null_count() / len(train)
        rec = {"feature": feat, "train": round(base, 4)}
        flagged = False
        for split in ("validation", "test"):
            r = art.raw[split][feat].null_count() / len(art.raw[split])
            rec[split] = round(r, 4)
            if abs(r - base) > T.null_rate_abs_delta:
                flagged = True
        rec["flagged"] = flagged
        offenders += int(flagged)
        rows.append(rec)
    rows.sort(key=lambda d: max(abs(d.get("validation", 0) - d["train"]),
                                abs(d.get("test", 0) - d["train"])), reverse=True)
    passed = offenders == 0
    return CheckResult("quality", "null_rates", passed, WARN,
                       f"{offenders} feature(s) shift null-rate >|{T.null_rate_abs_delta}| vs train",
                       n_offending=offenders, details={"table": rows})


def check_distribution_drift(art: Artifacts) -> CheckResult:
    """PSI (all features) + KS (continuous) for train -> test, ranked by drift."""
    train, test = art.raw["train"], art.raw["test"]
    rows = []
    for feat in C.NUMERIC_FEATURES:
        e = train[feat].to_numpy().astype(float)
        a = test[feat].to_numpy().astype(float)
        p = psi(e, a)
        em, am = e[~np.isnan(e)], a[~np.isnan(a)]
        ks_stat, ks_p = (stats.ks_2samp(em, am)[:2] if em.size and am.size else (float("nan"),) * 2)
        rows.append({"feature": feat, "kind": "numeric", "psi": round(p, 4),
                     "psi_band": _band(p), "ks_stat": round(float(ks_stat), 4),
                     "ks_pvalue": float(ks_p)})
    for feat in C.CATEGORICAL_FEATURES + C.BOOL_FEATURES:
        p = psi_categorical(train[feat], test[feat])
        rows.append({"feature": feat, "kind": "categorical", "psi": round(p, 4),
                     "psi_band": _band(p), "ks_stat": None, "ks_pvalue": None})
    rows.sort(key=lambda d: (d["psi"] if not np.isnan(d["psi"]) else -1), reverse=True)
    n_moderate = sum(1 for r in rows if r["psi_band"] in ("moderate", "large"))
    passed = n_moderate == 0
    top = rows[0]["feature"] if rows else "n/a"
    return CheckResult("quality", "distribution_drift", passed, WARN,
                       f"{n_moderate} feature(s) with PSI>={T.psi_warn} (train->test); top drift: {top}",
                       n_offending=n_moderate, details={"table": rows})


def check_level_coverage(art: Artifacts) -> CheckResult:
    """Any categorical level in val/test that was unseen in train (-> all-zero one-hot)."""
    train = art.raw["train"]
    unseen = {}
    for cat in C.CATEGORICAL_FEATURES:
        seen = set(train[cat].drop_nulls().unique().to_list())
        for split in ("validation", "test"):
            extra = sorted(set(art.raw[split][cat].drop_nulls().unique().to_list()) - seen)
            if extra:
                unseen[f"{split}/{cat}"] = extra
    passed = not unseen
    return CheckResult("quality", "level_coverage", passed, WARN,
                       "no unseen categorical levels in val/test"
                       if passed else f"unseen levels (all-zero one-hot): {unseen}",
                       n_offending=len(unseen), details=unseen)


def check_range_sanity(art: Artifacts) -> CheckResult:
    """Range / sanity rules: requested_amount band, rates in [0,1], counts >=0."""
    rate_cols = ["aggregate_credit_utilization", "invoice_payment_delinquency_rate"]
    count_cols = ["recent_inquiries_count_6mo", "prior_loans_count", "prior_loans_default_count",
                  "multi_lender_inquiry_count_30d", "repeat_application_count",
                  "observed_overdraft_count_3mo"]
    nonneg_cols = ["requested_amount", "stated_annual_revenue", "existing_debt_obligations",
                   "prior_loans_amount_total", "requested_amount_to_observed_revenue"]
    violations = {}
    for split in C.SPLITS:
        df = art.raw[split]
        ra = df["requested_amount"].drop_nulls()
        n_band = int(((ra < 4000) | (ra > 60000)).sum())  # ~[5k,50k] product band, loose
        if n_band:
            violations[f"{split}/requested_amount_band"] = n_band
        for c in rate_cols:
            x = df[c].drop_nulls()
            n = int(((x < 0) | (x > 1)).sum())
            if n:
                violations[f"{split}/{c}_in[0,1]"] = n
        for c in count_cols + nonneg_cols:
            x = df[c].drop_nulls()
            n = int((x < 0).sum())
            if n:
                violations[f"{split}/{c}_nonneg"] = n
    passed = not violations
    return CheckResult("quality", "range_sanity", passed, WARN,
                       "all range / sanity rules hold"
                       if passed else f"range violations: {violations}",
                       n_offending=sum(violations.values()), details=violations)


CHECKS = [check_null_rates, check_distribution_drift, check_level_coverage, check_range_sanity]


def run_all(art: Artifacts) -> list[CheckResult]:
    return [chk(art) for chk in CHECKS]
