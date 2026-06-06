"""Pipeline-integrity assertions (build brief §5a). Any failure is an ``error``.

These are the leakage-safety and encoding invariants the pipeline must keep
guaranteeing. Each check recomputes from first principles where possible, so it
cross-checks the pipeline rather than trusting it.
"""

from __future__ import annotations

import polars as pl

from . import contracts as C
from .loader import Artifacts
from .results import CheckResult, Severity

ERROR = Severity.ERROR


def _ok(name: str, msg: str) -> CheckResult:
    return CheckResult("integrity", name, True, ERROR, msg)


def _fail(name: str, msg: str, n: int = 0, details: dict | None = None) -> CheckResult:
    return CheckResult("integrity", name, False, ERROR, msg, n_offending=n, details=details or {})


def _feature_space(art: Artifacts, split: str, kind: str) -> list[str]:
    """Columns that make up the feature space of a frame (excludes ids/cohort/outcomes)."""
    if kind == "dense":
        feat = list(art.manifest["dense_feature_cols"])
    else:
        feat = list(C.FEATURES) + [f"{c}{C.MISSING_SUFFIX}" for c in C.MISSING_FLAG_COLS]
    cols = set(art.frame(split, kind).columns)
    return [c for c in feat if c in cols]


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def check_no_leakage(art: Artifacts) -> CheckResult:
    """None of the 6 outcome columns appear in any frame's feature space."""
    offenders = {}
    for split in C.SPLITS:
        for kind in ("raw", "native", "dense"):
            leaked = sorted(set(_feature_space(art, split, kind)) & set(C.OUTCOME))
            if leaked:
                offenders[f"{split}/{kind}"] = leaked
    if offenders:
        return _fail("no_leakage", f"outcome columns leaked into feature space: {offenders}",
                     n=len(offenders), details=offenders)
    return _ok("no_leakage", "no outcome column appears in any feature space")


def check_manifest_agreement(art: Artifacts) -> CheckResult:
    """contracts.py column roles == feature_manifest.json roles."""
    m = art.manifest["column_roles"]
    expected = {
        "ids": C.IDS,
        "outcome": C.OUTCOME,
        "categorical_features": C.CATEGORICAL_FEATURES,
        "bool_features": C.BOOL_FEATURES,
        "numeric_features": C.NUMERIC_FEATURES,
    }
    diffs = {}
    for role, exp in expected.items():
        got = m.get(role, [])
        if set(got) != set(exp):
            diffs[role] = {"only_manifest": sorted(set(got) - set(exp)),
                           "only_contract": sorted(set(exp) - set(got))}
    if set(art.manifest["missing_flag_cols"]) != set(C.MISSING_FLAG_COLS):
        diffs["missing_flag_cols"] = {
            "only_manifest": sorted(set(art.manifest["missing_flag_cols"]) - set(C.MISSING_FLAG_COLS)),
            "only_contract": sorted(set(C.MISSING_FLAG_COLS) - set(art.manifest["missing_flag_cols"])),
        }
    if diffs:
        return _fail("manifest_agreement", f"contracts vs manifest disagree: {diffs}",
                     n=len(diffs), details=diffs)
    return _ok("manifest_agreement", "contracts.py roles match feature_manifest.json")


def check_fit_on_train(art: Artifacts) -> CheckResult:
    """One-hot levels and impute medians in the manifest are reproducible from train alone."""
    train = art.raw["train"]
    bad = {}
    # recompute fitted cat levels
    for c in C.CATEGORICAL_FEATURES:
        recomputed = sorted(train[c].drop_nulls().unique().to_list())
        stored = art.manifest["cat_levels"][c]
        if recomputed != stored:
            bad[f"levels:{c}"] = {"recomputed": recomputed, "manifest": stored}
    # recompute impute medians
    for c in C.NUMERIC_FEATURES:
        recomputed = float(train[c].median())
        stored = float(art.manifest["impute_medians"][c])
        if abs(recomputed - stored) > 1e-9:
            bad[f"median:{c}"] = {"recomputed": recomputed, "manifest": stored}
    if bad:
        return _fail("fit_on_train", f"manifest fitted state not reproducible from train: {bad}",
                     n=len(bad), details=bad)
    return _ok("fit_on_train", "fitted one-hot levels + impute medians reproduce from train only")


def check_row_counts(art: Artifacts) -> CheckResult:
    """Encoded rows per split == raw rows (and == golden shapes)."""
    bad = {}
    for split in C.SPLITS:
        n_raw = len(art.raw[split])
        golden = C.GOLDEN["shapes"][split][0]
        if n_raw != golden:
            bad[f"{split}/raw_csv"] = {"got": n_raw, "golden": golden}
        for kind in ("raw", "native", "dense"):
            n = len(art.frame(split, kind))
            if n != n_raw:
                bad[f"{split}/{kind}"] = {"got": n, "expected": n_raw}
    if bad:
        return _fail("row_counts", f"row-count mismatch: {bad}", n=len(bad), details=bad)
    return _ok("row_counts", "encoded row counts preserved across all splits")


def check_dense_completeness(art: Artifacts) -> CheckResult:
    """Dense feature columns have zero nulls in every split."""
    feat = art.manifest["dense_feature_cols"]
    bad = {}
    for split in C.SPLITS:
        df = art.frame(split, "dense")
        nz = {c: df[c].null_count() for c in feat if df[c].null_count() > 0}
        if nz:
            bad[split] = nz
    if bad:
        return _fail("dense_completeness", f"dense feature columns have nulls: {bad}",
                     n=sum(len(v) for v in bad.values()), details=bad)
    return _ok("dense_completeness", "dense feature columns are null-free in every split")


def check_native_fidelity(art: Artifacts) -> CheckResult:
    """For each missing-flag col, native null count == its __ismissing sum."""
    bad = {}
    for split in C.SPLITS:
        nat = art.frame(split, "native")
        for c in C.MISSING_FLAG_COLS:
            nulls = nat[c].null_count()
            flagsum = int(nat[f"{c}{C.MISSING_SUFFIX}"].sum())
            if nulls != flagsum:
                bad[f"{split}/{c}"] = {"native_nulls": nulls, "ismissing_sum": flagsum}
    if bad:
        return _fail("native_fidelity", f"missing-flag mismatch in native frame: {bad}",
                     n=len(bad), details=bad)
    return _ok("native_fidelity", "native nulls match __ismissing flags for all 9 columns")


def check_onehot_validity(art: Artifacts) -> CheckResult:
    """Within each categorical, dense dummies sum to <=1 per row (==1 where in-support)."""
    bad = {}
    unseen = {}
    for split in C.SPLITS:
        dense = art.frame(split, "dense")
        for cat in C.CATEGORICAL_FEATURES:
            dummies = [f"{cat}_{lvl}" for lvl in art.manifest["cat_levels"][cat]]
            s = dense.select(dummies).sum_horizontal()
            n_over = int((s > 1).sum())
            n_zero = int((s == 0).sum())
            if n_over:
                bad[f"{split}/{cat}"] = n_over
            if n_zero:
                unseen[f"{split}/{cat}"] = n_zero
    if bad:
        return _fail("onehot_validity", f"categoricals with dummies summing to >1: {bad}",
                     n=sum(bad.values()), details={"over_one": bad, "all_zero": unseen})
    msg = "one-hot dummies sum to <=1 in every row"
    if unseen:
        msg += f" ({sum(unseen.values())} rows all-zero from unseen levels -- see quality coverage)"
    return CheckResult("integrity", "onehot_validity", True, ERROR, msg,
                       details={"all_zero": unseen})


def check_cohort_assignment(art: Artifacts) -> CheckResult:
    """train.cohort_week all null; val/test fully in 1..13."""
    bad = {}
    tr = art.frame("train", "raw")["cohort_week"]
    if tr.null_count() != len(tr):
        bad["train"] = f"{len(tr) - tr.null_count()} non-null cohort_week values (expected all null)"
    for split in ("validation", "test"):
        cw = art.frame(split, "raw")["cohort_week"]
        in_grid = cw.is_between(1, C.GOLDEN["cohorts"]["n_weeks"]).fill_null(False)
        n_bad = int((~in_grid).sum())
        if n_bad:
            bad[split] = f"{n_bad} rows with cohort_week outside 1..13 (or null)"
    if bad:
        return _fail("cohort_assignment", f"cohort assignment broken: {bad}", n=len(bad), details=bad)
    return _ok("cohort_assignment", "train cohort_week all null; val/test fully assigned to 1..13")


def check_encoder_determinism(art: Artifacts) -> CheckResult:
    """Re-running transform reproduces the on-disk frames exactly."""
    from preprocess import Preprocessor

    pre = Preprocessor(art.data_dictionary, art.cohorts).fit(art.raw["train"])
    bad = {}
    for split in C.SPLITS:
        redo = pre.transform(art.raw[split])
        for kind in ("raw", "native", "dense"):
            stored = art.frame(split, kind)
            if not redo[kind].equals(stored):
                # locate first differing column for a useful message
                diff_cols = [c for c in stored.columns
                             if c in redo[kind].columns and not redo[kind][c].equals(stored[c])]
                bad[f"{split}/{kind}"] = diff_cols[:5] or ["schema/shape differs"]
    if bad:
        return _fail("encoder_determinism", f"re-transform not byte-identical: {bad}",
                     n=len(bad), details=bad)
    return _ok("encoder_determinism", "re-running transform reproduces every frame exactly")


CHECKS = [
    check_no_leakage,
    check_manifest_agreement,
    check_fit_on_train,
    check_row_counts,
    check_dense_completeness,
    check_native_fidelity,
    check_onehot_validity,
    check_cohort_assignment,
    check_encoder_determinism,
]


def run_all(art: Artifacts) -> list[CheckResult]:
    return [chk(art) for chk in CHECKS]
