"""Build brief §6 golden values as regression fixtures -- a data refresh that
shifts any of these should fail loudly."""

from __future__ import annotations

import polars as pl
import pytest

from obs import intervention, selection
from obs.contracts import GOLDEN, golden_close


def _funded(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("prior_decision") == 1)


# -- shapes ----------------------------------------------------------------- #

@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_raw_shapes(art, split):
    assert art.raw[split].shape == GOLDEN["shapes"][split]


# -- selection / labeling --------------------------------------------------- #

def test_train_funded_and_default_rate(art):
    tr = art.raw["train"]
    g = GOLDEN["selection"]
    funded = _funded(tr)
    assert len(funded) == g["train_funded"]
    assert len(tr) - len(funded) == g["train_declined"]
    assert golden_close(float(funded["default_flag"].mean()), g["train_default_rate"])
    assert int(funded.filter(pl.col("default_flag") == 1).height) == g["train_defaulted"]


def test_val_funded_and_default_rate(art):
    va = art.raw["validation"]
    g = GOLDEN["selection"]
    funded = _funded(va)
    assert len(funded) == g["val_funded"]
    assert golden_close(float(funded["default_flag"].mean()), g["val_default_rate"])


def test_test_labels_withheld(art):
    assert art.raw["test"]["default_flag"].drop_nulls().len() == GOLDEN["selection"]["test_labeled"]


# -- bank-feed MNAR --------------------------------------------------------- #

def test_bank_feed_mnar(art):
    tr = art.raw["train"]
    g = GOLDEN["bank_feed"]
    assert int((tr["has_linked_bank_feed"]).sum()) == g["linked_true"]
    assert int((~tr["has_linked_bank_feed"]).sum()) == g["linked_false"]
    # observed_* null iff no linked feed
    by = tr.group_by("has_linked_bank_feed").agg(
        pl.col("observed_monthly_revenue_avg_3mo").null_count().alias("nulls"), pl.len())
    for row in by.iter_rows(named=True):
        assert row["nulls"] == (0 if row["has_linked_bank_feed"] else row["len"])


# -- default timing --------------------------------------------------------- #

def test_default_timing(art):
    t = selection.default_timing(art)
    assert t["median_day"] == GOLDEN["timing"]["days_to_default_median"]
    for day, exp in GOLDEN["timing"]["cdr_by_day"].items():
        assert golden_close(t["cdr_by_day"][day], exp), f"CDR@{day}"


# -- cohorts ---------------------------------------------------------------- #

def test_cohorts(art):
    g = GOLDEN["cohorts"]
    assert art.frame("train", "raw")["cohort_week"].null_count() == len(art.raw["train"])
    for split in ("validation", "test"):
        cw = art.frame(split, "raw")["cohort_week"]
        assert cw.is_between(1, g["n_weeks"]).all()


# -- interventions ---------------------------------------------------------- #

def test_intervention_design_fixtures(art):
    diag = intervention.build_query_diagnostics(art)
    g = GOLDEN["interventions"]
    assert len(diag) == g["n_queries"]
    assert diag["applicant_id"].n_unique() == g["n_applicants"]
    assert int(diag["structural"].sum()) == g["structural"]
    assert g["n_queries"] - int(diag["structural"].sum()) == g["intervenable"]
    assert golden_close(float(diag["in_support"].fill_null(False).mean()), g["in_support_frac"])
    assert golden_close(float(diag["no_op"].mean()), g["noop_frac"])


# -- encoded shapes --------------------------------------------------------- #

def test_encoded_shapes(art):
    g = GOLDEN["encoded_shapes"]
    for split in ["train", "validation", "test"]:
        assert art.frame(split, "raw").width == g["raw"]
        assert art.frame(split, "native").width == g["native"]
        assert art.frame(split, "dense").width == g["dense"]
