"""Intervention (Deliverable-C) diagnostics (build brief §5d).

For each of the 900 single-feature ``do()`` queries: the applicant's current
value, the intervention value's percentile in the train marginal, an in-support
flag, a no-op flag, and a structural-feature flag (``intervenable == False`` in
the data dictionary -- those require DAG propagation, not a single-column edit).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import contracts as C
from .loader import Artifacts
from .results import CheckResult, Severity

WARN = Severity.WARN
T = C.THRESHOLDS


def _train_marginals(art: Artifacts) -> dict[str, np.ndarray]:
    train = art.raw["train"]
    feats = set(art.interventions["feature_name"].unique().to_list())
    return {f: train[f].drop_nulls().to_numpy().astype(float) for f in feats if f in train.columns}


def _applicant_lookup(art: Artifacts) -> dict[str, dict]:
    """applicant_id -> row dict, searching val then test (queries target deployment pop)."""
    needed = set(art.interventions["applicant_id"].unique().to_list())
    lookup: dict[str, dict] = {}
    for split in ("validation", "test", "train"):
        df = art.raw[split]
        hit = df.filter(pl.col("applicant_id").is_in(list(needed - set(lookup))))
        for row in hit.iter_rows(named=True):
            lookup[row["applicant_id"]] = row
    return lookup


def build_query_diagnostics(art: Artifacts) -> pl.DataFrame:
    """One row per query with current value, percentile, support / no-op / structural flags."""
    marg = _train_marginals(art)
    lookup = _applicant_lookup(art)
    structural = set(art.data_dictionary.filter(pl.col("intervenable") == False)["field"].to_list())  # noqa: E712
    # discrete-count features get an exception from the p1-p99 in-support band
    discrete = {"prior_loans_count", "prior_loans_default_count", "recent_inquiries_count_6mo",
                "multi_lender_inquiry_count_30d", "repeat_application_count",
                "observed_overdraft_count_3mo", "employee_count_bucket"}

    out = []
    for q in art.interventions.iter_rows(named=True):
        feat = q["feature_name"]
        val = q["intervention_value"]
        cur = lookup.get(q["applicant_id"], {}).get(feat)
        col = marg.get(feat)
        pct = lo = hi = None
        in_support = None
        out_of_range = None
        if col is not None and col.size:
            pct = float((col <= val).mean() * 100.0)
            lo, hi = float(np.percentile(col, T.support_lo_pct)), float(np.percentile(col, T.support_hi_pct))
            # Authoritative in-support rule: train p1-p99 band (matches golden fixtures).
            in_support = bool(lo <= val <= hi)
            out_of_range = bool(val < col.min() or val > col.max())
        cur_f = float(cur) if isinstance(cur, (int, float)) and cur is not None else None
        no_op = bool(cur_f is not None and abs(cur_f - float(val)) < 1e-9)
        out.append({
            "query_id": q["query_id"], "applicant_id": q["applicant_id"], "feature_name": feat,
            "current_value": cur_f, "intervention_value": float(val),
            "percentile_in_train": None if pct is None else round(pct, 2),
            "in_support": in_support, "no_op": no_op,
            "structural": feat in structural, "out_of_range": out_of_range,
            # discrete-count features have lumpy marginals where the p1-p99 band can be
            # degenerate; flagged so analysts can read the support call with that caveat.
            "discrete_feature": feat in discrete,
        })
    return pl.DataFrame(out)


# --------------------------------------------------------------------------- #
# Checks (each compares against the §6 golden fixtures)
# --------------------------------------------------------------------------- #


def check_intervention_design(art: Artifacts) -> CheckResult:
    diag = build_query_diagnostics(art)
    g = C.GOLDEN["interventions"]
    n = len(diag)
    structural = int(diag["structural"].sum())
    intervenable = n - structural
    in_support = int(diag["in_support"].fill_null(False).sum())
    no_ops = int(diag["no_op"].sum())
    out_of_range = int(diag["out_of_range"].fill_null(False).sum())
    n_applicants = diag["applicant_id"].n_unique()

    mismatches = {}
    for label, got, exp in [
        ("n_queries", n, g["n_queries"]),
        ("n_applicants", n_applicants, g["n_applicants"]),
        ("structural", structural, g["structural"]),
        ("intervenable", intervenable, g["intervenable"]),
    ]:
        if got != exp:
            mismatches[label] = {"got": got, "golden": exp}
    if not C.golden_close(in_support / n, g["in_support_frac"]):
        mismatches["in_support_frac"] = {"got": round(in_support / n, 4), "golden": g["in_support_frac"]}

    passed = not mismatches
    return CheckResult("intervention", "intervention_design", passed, WARN,
                       f"{n} queries / {n_applicants} applicants | structural {structural} "
                       f"({structural / n:.1%}) | in-support {in_support / n:.1%} | "
                       f"no-ops {no_ops} ({no_ops / n:.1%}) | out-of-range {out_of_range}"
                       + ("" if passed else f" | GOLDEN MISMATCH {mismatches}"),
                       n_offending=len(mismatches),
                       details={"counts": {"n_queries": n, "n_applicants": n_applicants,
                                           "structural": structural, "intervenable": intervenable,
                                           "in_support": in_support, "no_ops": no_ops,
                                           "out_of_range": out_of_range},
                                "mismatches": mismatches})


def check_feature_support_coverage(art: Artifacts) -> CheckResult:
    """Per-feature support coverage: share of each feature's queries that land in-support."""
    diag = build_query_diagnostics(art)
    by = (
        diag.group_by("feature_name")
        .agg(pl.len().alias("n_queries"),
             pl.col("in_support").fill_null(False).mean().alias("in_support_frac"),
             pl.col("structural").mean().alias("structural_frac"),
             pl.col("out_of_range").fill_null(False).sum().alias("out_of_range"))
        .sort("in_support_frac")
    )
    rows = [{"feature": r["feature_name"], "n_queries": r["n_queries"],
             "in_support_frac": round(float(r["in_support_frac"]), 3),
             "structural": bool(r["structural_frac"] > 0.5),
             "out_of_range": int(r["out_of_range"])}
            for r in by.iter_rows(named=True)]
    low = [r for r in rows if r["in_support_frac"] < 0.8]
    return CheckResult("intervention", "feature_support_coverage", not low, WARN,
                       f"{len(rows)} intervened features; "
                       f"{len(low)} have <80% of queries in-support",
                       n_offending=len(low), details={"table": rows})


def check_structural_interventions(art: Artifacts) -> CheckResult:
    """Surface the structural-feature interventions that require DAG propagation."""
    diag = build_query_diagnostics(art)
    struct = diag.filter(pl.col("structural"))
    by_feat = (struct.group_by("feature_name").agg(pl.len().alias("n"))
               .sort("n", descending=True))
    feats = [{"feature": r["feature_name"], "n_queries": r["n"]} for r in by_feat.iter_rows(named=True)]
    n = len(struct)
    g = C.GOLDEN["interventions"]
    passed = n == g["structural"]
    bank = int((struct["feature_name"] == "has_linked_bank_feed").sum())
    return CheckResult("intervention", "structural_interventions", passed, WARN,
                       f"{n} structural-feature interventions ({n / len(diag):.1%}) need DAG "
                       f"propagation (e.g. {bank}x do(has_linked_bank_feed) must move the whole "
                       f"bank-feed block)",
                       n_offending=0 if passed else abs(n - g["structural"]),
                       details={"by_feature": feats, "golden_structural": g["structural"]})


CHECKS = [check_intervention_design, check_feature_support_coverage, check_structural_interventions]


def run_all(art: Artifacts) -> list[CheckResult]:
    return [chk(art) for chk in CHECKS]
