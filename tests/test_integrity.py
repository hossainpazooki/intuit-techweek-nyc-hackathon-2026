"""Build brief §5a invariants as automated tests -- every integrity check must pass."""

from __future__ import annotations

import pytest

from obs import integrity
from obs.contracts import DENSE_COLS, NATIVE_COLS, RAW_COLS, SPLITS


@pytest.mark.parametrize("check", integrity.CHECKS, ids=lambda c: c.__name__)
def test_integrity_invariant_passes(art, check):
    result = check(art)
    assert result.passed, f"{result.name} failed: {result.message}"


def test_no_blocking_failures(art):
    results = integrity.run_all(art)
    blocking = [r for r in results if r.blocking]
    assert not blocking, [r.message for r in blocking]


@pytest.mark.parametrize("split", SPLITS)
def test_encoded_shapes(art, split):
    assert art.frame(split, "raw").width == RAW_COLS
    assert art.frame(split, "native").width == NATIVE_COLS
    assert art.frame(split, "dense").width == DENSE_COLS


def test_dense_feature_space_is_null_free(art):
    for split in SPLITS:
        df = art.frame(split, "dense")
        for col in art.manifest["dense_feature_cols"]:
            assert df[col].null_count() == 0, f"{split}/{col} has nulls in dense"


def test_no_outcome_in_dense_feature_space(art):
    outcomes = set(art.manifest["column_roles"]["outcome"])
    feats = set(art.manifest["dense_feature_cols"])
    assert not (outcomes & feats)
