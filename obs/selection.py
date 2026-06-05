"""Selection & labeling diagnostics (build brief §5c) + default-timing helpers.

Surfaces the selective-labels reality: ~60.6% of train was historically funded
and outcomes exist only for funded+matured loans. Includes a funding-propensity
``e(x)=P(funded|x)`` overlap diagnostic (positivity / common support) and the
default-timing object behind Deliverable B.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import contracts as C
from .loader import Artifacts
from .results import CheckResult, Severity

WARN = Severity.WARN
T = C.THRESHOLDS


def _funded_mask(df: pl.DataFrame) -> pl.Series:
    """Funded == historically approved (prior_decision == 1)."""
    return df["prior_decision"] == 1


# --------------------------------------------------------------------------- #
# Funded / declined / default-rate summary
# --------------------------------------------------------------------------- #


def funded_declined_summary(art: Artifacts) -> CheckResult:
    rows = []
    mismatches = {}
    g = C.GOLDEN["selection"]
    for split in C.SPLITS:
        df = art.raw[split]
        funded = int(_funded_mask(df).sum())
        declined = len(df) - funded
        labeled = int(df["default_flag"].drop_nulls().len())
        mean_dr = df.filter(_funded_mask(df))["default_flag"].mean()  # None if all withheld
        dr = float(mean_dr) if mean_dr is not None else float("nan")
        rows.append({"split": split, "n": len(df), "funded": funded, "declined": declined,
                     "labeled": labeled, "default_rate": None if np.isnan(dr) else round(dr, 4)})
    # Regression against golden values
    tr, va = rows[0], rows[1]
    checks = {
        "train_funded": (tr["funded"], g["train_funded"]),
        "train_declined": (tr["declined"], g["train_declined"]),
        "train_default_rate": (tr["default_rate"], g["train_default_rate"]),
        "val_funded": (va["funded"], g["val_funded"]),
        "val_default_rate": (va["default_rate"], g["val_default_rate"]),
        "test_labeled": (rows[2]["labeled"], g["test_labeled"]),
    }
    for k, (got, exp) in checks.items():
        if got is None or not C.golden_close(float(got), float(exp)):
            mismatches[k] = {"got": got, "golden": exp}
    passed = not mismatches
    return CheckResult("selection", "funded_declined_summary", passed, WARN,
                       "funded/declined/default-rate reproduce golden values"
                       if passed else f"golden mismatch: {mismatches}",
                       n_offending=len(mismatches),
                       details={"table": rows, "mismatches": mismatches})


def default_rate_by_cohort(art: Artifacts) -> CheckResult:
    """Default rate by cohort_week on labeled funded val loans (train cohort is null)."""
    va = art.frame("validation", "raw")  # encoded frame carries the derived cohort_week
    funded = va.filter(_funded_mask(va) & va["default_flag"].is_not_null())
    by = (
        funded.group_by("cohort_week")
        .agg(pl.len().alias("n"), pl.col("default_flag").mean().alias("default_rate"))
        .sort("cohort_week")
    )
    rows = [{"cohort_week": r["cohort_week"], "n": r["n"],
             "default_rate": round(float(r["default_rate"]), 4)} for r in by.iter_rows(named=True)]
    n_weeks = C.GOLDEN["cohorts"]["n_weeks"]
    covered = {r["cohort_week"] for r in rows}
    missing = sorted(set(range(1, n_weeks + 1)) - covered)
    passed = not missing
    return CheckResult("selection", "default_rate_by_cohort", passed, WARN,
                       f"val default rate computed for {len(rows)}/{n_weeks} cohorts"
                       + (f"; missing {missing}" if missing else ""),
                       n_offending=len(missing), details={"table": rows, "missing": missing})


def label_availability(art: Artifacts) -> CheckResult:
    """Which rows are scorable (have a default_flag) per split."""
    rows = []
    for split in C.SPLITS:
        df = art.raw[split]
        n = len(df)
        labeled = int(df["default_flag"].drop_nulls().len())
        rows.append({"split": split, "rows": n, "labeled": labeled,
                     "labeled_frac": round(labeled / n, 4), "withheld": n - labeled})
    return CheckResult("selection", "label_availability", True, WARN,
                       f"labeled rows -- train {rows[0]['labeled']:,}, "
                       f"val {rows[1]['labeled']:,}, test {rows[2]['labeled']:,} (withheld)",
                       details={"table": rows})


# --------------------------------------------------------------------------- #
# Funding-propensity e(x) overlap diagnostic
# --------------------------------------------------------------------------- #


def _propensity_matrix(art: Artifacts, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Build standardized X (excluding prior_decision -- it *is* the target) and y=funded."""
    dense = art.frame(split, "dense")
    # Exclude variables *determined by* the funding decision, else e(x) is degenerate:
    #  - prior_decision dummies (this IS the target), and
    #  - prior_approved_amount (+ its __ismissing flag): null iff declined, so the
    #    flag perfectly reveals the label. Both are post-decision, not as-of-application.
    excl = {f"{C.SELECTION_TRAP_COL}_{lvl}" for lvl in art.manifest["cat_levels"][C.SELECTION_TRAP_COL]}
    excl |= {"prior_approved_amount", f"prior_approved_amount{C.MISSING_SUFFIX}"}
    cols = [c for c in art.manifest["dense_feature_cols"] if c not in excl]
    X = dense.select(cols).to_numpy().astype(float)
    y = (_funded_mask(art.raw[split]).to_numpy()).astype(float)
    return X, y, cols


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _fit_logreg(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iters: int = 400,
                lr: float = 0.5) -> np.ndarray:
    """Compact L2-regularized logistic regression (full-batch GD, deterministic)."""
    Xb = np.c_[np.ones(len(X)), _standardize(X)]
    w = np.zeros(Xb.shape[1])
    n = len(Xb)
    reg = np.ones(Xb.shape[1]); reg[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        grad = Xb.T @ (p - y) / n + l2 * reg * w / n
        w -= lr * grad
    return Xb, w


def propensity_overlap(art: Artifacts) -> CheckResult:
    """Fit e(x)=P(funded|x) on train; report common support + positivity-violation mass."""
    X, y, _cols = _propensity_matrix(art, "train")
    Xb, w = _fit_logreg(X, y)
    ex = 1.0 / (1.0 + np.exp(-(Xb @ w)))

    funded = ex[y == 1]
    declined = ex[y == 0]
    eps = T.overlap_epsilon

    # Histogram-overlap coefficient between funded & declined score distributions.
    edges = np.linspace(0, 1, 21)
    hf = np.histogram(funded, edges, density=True)[0]
    hd = np.histogram(declined, edges, density=True)[0]
    width = edges[1] - edges[0]
    overlap = float(np.sum(np.minimum(hf, hd)) * width)

    below_eps = float((ex < eps).mean())          # population near zero propensity
    declined_high = float((declined > 1 - eps).mean())
    # crude AUC via Mann-Whitney for a separability read
    try:
        from scipy import stats as _st
        auc = float(_st.mannwhitneyu(funded, declined, alternative="greater").statistic
                    / (len(funded) * len(declined)))
    except Exception:
        auc = float("nan")

    hist_rows = [{"bin_lo": round(edges[i], 3), "funded": float(hf[i]), "declined": float(hd[i])}
                 for i in range(len(hf))]
    # Positivity violation is *expected* (legacy policy never funded some regions);
    # warn only if a large share of the population sits below epsilon.
    deterministic = overlap < 0.05 and auc > 0.99
    note = (" -- legacy funding policy is near-deterministic: no common support, "
            "so PD/NPV are only partially identified off-policy") if deterministic else ""
    passed = below_eps < 0.05
    return CheckResult("selection", "propensity_overlap", passed, WARN,
                       f"e(x) AUC~{auc:.3f}; common-support overlap {overlap:.2f}; "
                       f"{below_eps:.1%} of population has e(x)<{eps} (positivity-violation region)"
                       + note,
                       n_offending=int((ex < eps).sum()),
                       details={"auc": auc, "overlap_coef": round(overlap, 4),
                                "share_below_eps": round(below_eps, 4),
                                "declined_above_1_minus_eps": round(declined_high, 4),
                                "near_deterministic_policy": deterministic,
                                "epsilon": eps, "score_hist": hist_rows})


# --------------------------------------------------------------------------- #
# Default-timing object (Deliverable B / build brief §6)
# --------------------------------------------------------------------------- #


def default_timing(art: Artifacts) -> dict:
    """CDR-by-day + median (train labeled defaults) and the cohort x age grid (val)."""
    tr = art.raw["train"]
    dd = tr.filter(tr["default_flag"] == 1)["days_to_default"].drop_nulls().to_numpy()
    median_day = float(np.median(dd)) if dd.size else float("nan")
    cdr_by_day = {d: round(float((dd <= d).mean()), 4) for d in (7, 14, 28, 60, 90)}

    # Discrete-time cumulative default rate, cohort_week x loan_age_weeks, on val funded.
    va = art.frame("validation", "raw")  # encoded frame carries cohort_week
    funded = va.filter(_funded_mask(va) & va["default_flag"].is_not_null())
    grid = []
    n_weeks = C.GOLDEN["cohorts"]["n_weeks"]
    for cw in range(1, n_weeks + 1):
        sub = funded.filter(funded["cohort_week"] == cw)
        n = len(sub)
        dtd = sub.filter(sub["default_flag"] == 1)["days_to_default"].drop_nulls().to_numpy()
        for age in range(1, n_weeks + 1):
            cdr = float((dtd <= age * 7).sum() / n) if n else float("nan")
            grid.append({"cohort_week": cw, "loan_age_weeks": age, "n": n,
                         "cumulative_default_rate": None if np.isnan(cdr) else round(cdr, 4)})
    return {"median_day": median_day, "cdr_by_day": cdr_by_day, "cohort_age_grid": grid}


CHECKS = [funded_declined_summary, default_rate_by_cohort, label_availability, propensity_overlap]


def run_all(art: Artifacts) -> list[CheckResult]:
    return [chk(art) for chk in CHECKS]
