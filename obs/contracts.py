"""Single source of truth: column roles, golden values, and thresholds.

These declarations are *independent* of ``preprocess.py``: the integrity layer
asserts that the runtime ``feature_manifest.json`` agrees with what is declared
here, so a future edit to either side that drifts the contract is caught loudly.

Every number in ``GOLDEN`` was verified against the shipped data; treat them as
regression fixtures (see the build brief §6).
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Column contract (build brief §3)
# --------------------------------------------------------------------------- #

IDS = ["business_id", "applicant_id"]

OUTCOME = [
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
]

DERIVED = ["cohort_week"]            # from application_timestamp; null outside window
DROPPED = ["application_timestamp"]  # not a feature

CATEGORICAL_FEATURES = [
    "sector",
    "geography_region",
    "employee_count_bucket",
    "intended_use_of_funds",
    "owner_personal_credit_band",
    "application_channel",
    "prior_decision",
]

BOOL_FEATURES = ["has_linked_bank_feed"]

NUMERIC_FEATURES = [
    "vintage_years",
    "stated_annual_revenue",
    "stated_time_in_business",
    "requested_amount",
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "observed_overdraft_count_3mo",
    "payroll_regularity_score",
    "aggregate_credit_utilization",
    "recent_inquiries_count_6mo",
    "existing_debt_obligations",
    "days_since_last_external_decline",
    "account_age_days",
    "platform_active_months",
    "bookkeeping_recency_days",
    "invoice_payment_delinquency_rate",
    "prior_loans_count",
    "prior_loans_default_count",
    "prior_loans_amount_total",
    "multi_lender_inquiry_count_30d",
    "days_since_last_inquiry_elsewhere",
    "repeat_application_count",
    "requested_amount_to_observed_revenue",
    "prior_underwriter_score",
    "prior_approved_amount",
]

# The 9 feature columns that carry train nulls (6 bank-feed + 2 days_since_* +
# prior_approved_amount). Missingness is signal -> each gets a `__ismissing` flag.
MISSING_FLAG_COLS = [
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "observed_overdraft_count_3mo",
    "payroll_regularity_score",
    "days_since_last_external_decline",
    "days_since_last_inquiry_elsewhere",
    "prior_approved_amount",
]

FEATURES = CATEGORICAL_FEATURES + BOOL_FEATURES + NUMERIC_FEATURES
MISSING_SUFFIX = "__ismissing"

# `prior_decision` is constant on labeled rows -> drop from outcome models, but
# it is the target-adjacent variable for the funding-propensity model e(x).
SELECTION_TRAP_COL = "prior_decision"

# Encoded frame shapes per split (build brief §3).
RAW_COLS = 53
NATIVE_COLS = 53
DENSE_COLS = 73
N_ONEHOT = 27  # 7 categoricals expand to 27 dummies

SPLITS = ["train", "validation", "test"]


# --------------------------------------------------------------------------- #
# Golden values (build brief §6) -- regression fixtures
# --------------------------------------------------------------------------- #

GOLDEN = {
    "shapes": {
        "train": (85340, 44),
        "validation": (4489, 44),
        "test": (8817, 44),
    },
    "encoded_shapes": {"raw": RAW_COLS, "native": NATIVE_COLS, "dense": DENSE_COLS},
    "selection": {
        "train_funded": 51722,
        "train_declined": 33618,
        "train_funded_frac": 0.606,
        "train_labeled": 51722,
        "train_default_rate": 0.1745,
        "train_defaulted": 9024,
        "train_paid": 42698,
        "val_funded": 2551,
        "val_declined": 1938,
        "val_default_rate": 0.2062,
        "test_labeled": 0,
    },
    "bank_feed": {
        "linked_true": 54887,
        "linked_false": 30453,
        "false_frac": 0.357,
    },
    "timing": {
        "days_to_default_median": 37.0,
        "cdr_by_day": {7: 0.072, 14: 0.19, 28: 0.388, 60: 0.775, 90: 1.0},
    },
    "cohorts": {
        "n_weeks": 13,
        "start": "2025-06-30",
        "end": "2025-09-28",
    },
    "interventions": {
        "n_queries": 900,
        "n_applicants": 300,
        "per_applicant": 3,
        "intervenable": 726,
        "intervenable_frac": 0.807,
        "structural": 174,
        "structural_frac": 0.193,
        "in_support_frac": 0.918,
        "noop_frac": 0.07,
    },
}


# --------------------------------------------------------------------------- #
# Thresholds (tunable; used by quality / intervention checks)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Thresholds:
    # null-rate deviation between a split and train before we warn
    null_rate_abs_delta: float = 0.05
    # PSI bands (industry convention): <0.1 stable, 0.1-0.25 moderate, >0.25 large
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    psi_bins: int = 10
    # KS p-value below which we flag a continuous shift
    ks_pvalue_warn: float = 0.01
    # in-support window for intervention values (train percentile band)
    support_lo_pct: float = 1.0
    support_hi_pct: float = 99.0
    # positivity / overlap: applicants with e(x) below this lack common support
    overlap_epsilon: float = 0.05
    # tolerance when reproducing golden fractions / rates
    golden_rtol: float = 0.01
    golden_atol: float = 0.002


THRESHOLDS = Thresholds()


def golden_close(actual: float, expected: float, *, t: Thresholds = THRESHOLDS) -> bool:
    """True if ``actual`` reproduces a golden value within tolerance."""
    return abs(actual - expected) <= max(t.golden_atol, t.golden_rtol * abs(expected))
