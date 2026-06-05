#!/usr/bin/env python3
"""Leakage-safe preprocessing pipeline for the SMB Underwriting Challenge.

The pipeline is built so that *no statistic ever crosses the train/val/test
boundary*. One-hot category sets and dense-frame impute medians are fitted on
``train`` only and applied unchanged to ``validation`` / ``test``. Six
post-origination outcome columns are quarantined as labels and never enter the
feature space. The raw ``application_timestamp`` is dropped and replaced by a
derived ``cohort_week``.

Every split is emitted in three deterministic encodings:

    {split}_raw.parquet      53 cols  integer-coded categoricals, NaNs preserved
    {split}_native.parquet   53 cols  native categorical dtype, NaNs preserved
    {split}_dense.parquet     73 cols  one-hot (7->27) + median-imputed, NaN-free

plus ``feature_manifest.json`` recording the fitted state (column roles, cat
levels, impute medians, missing-flag list, input hashes, caveats).

Usage:
    python preprocess.py --data data/ --dict dataset/data_dictionary.csv \
        --cohorts dataset/cohort_week_definitions.csv --out artifacts/
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PIPELINE_VERSION = "1.0.0"
MISSING_SUFFIX = "__ismissing"
SPLITS = ("train", "validation", "test")


# --------------------------------------------------------------------------- #
# Column-role derivation (from the data dictionary -- the pipeline's own view).
# contracts.py declares the *expected* roles independently; the observability
# layer asserts the two agree, so this derivation is meaningfully checked.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ColumnRoles:
    ids: list[str]
    outcome: list[str]
    dropped: list[str]
    derived: list[str]
    categorical: list[str]
    boolean: list[str]
    numeric: list[str]

    @property
    def features(self) -> list[str]:
        """Feature columns in canonical (dictionary) order."""
        return self._ordered(set(self.categorical) | set(self.boolean) | set(self.numeric))

    _order: tuple[str, ...] = field(default=(), repr=False)

    def _ordered(self, cols: set[str]) -> list[str]:
        return [c for c in self._order if c in cols]


def derive_roles(data_dictionary: pl.DataFrame) -> ColumnRoles:
    """Assign each raw column a role from its dictionary ``dtype`` / ``group``."""
    order = data_dictionary["field"].to_list()
    rows = {r["field"]: r for r in data_dictionary.iter_rows(named=True)}

    ids, outcome, dropped, categorical, boolean, numeric = [], [], [], [], [], []
    for fld in order:
        r = rows[fld]
        dtype, group = r["dtype"], r["group"]
        if group == "outcome":
            outcome.append(fld)
        elif dtype == "string":
            ids.append(fld)
        elif dtype == "timestamp":
            dropped.append(fld)
        elif dtype == "categorical":
            categorical.append(fld)
        elif dtype == "bool":
            boolean.append(fld)
        elif dtype in ("float", "int"):
            numeric.append(fld)
        else:  # pragma: no cover - dictionary is closed-world
            raise ValueError(f"unhandled dtype {dtype!r} for field {fld!r}")

    return ColumnRoles(
        ids=ids,
        outcome=outcome,
        dropped=dropped,
        derived=["cohort_week"],
        categorical=categorical,
        boolean=boolean,
        numeric=numeric,
        _order=tuple(order),
    )


# --------------------------------------------------------------------------- #
# Cohort assignment
# --------------------------------------------------------------------------- #


def _cohort_expr(cohorts: pl.DataFrame) -> pl.Expr:
    """Build a (date -> cohort_week) expression; null outside every window."""
    d = pl.col("application_timestamp").str.slice(0, 10).str.to_date("%Y-%m-%d")
    expr = pl.lit(None, dtype=pl.Int64)
    for row in cohorts.iter_rows(named=True):
        start = datetime.strptime(row["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(row["end_date"], "%Y-%m-%d").date()
        expr = (
            pl.when((d >= pl.lit(start)) & (d <= pl.lit(end)))
            .then(pl.lit(int(row["cohort_week"]), dtype=pl.Int64))
            .otherwise(expr)
        )
    return expr.alias("cohort_week")


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #


class Preprocessor:
    """``Preprocessor(data_dictionary, cohorts).fit(train).transform(split)``."""

    def __init__(self, data_dictionary: pl.DataFrame, cohorts: pl.DataFrame):
        self.dictionary = data_dictionary
        self.cohorts = cohorts
        self.roles = derive_roles(data_dictionary)
        # Fitted state (populated by .fit) -- everything below is train-only.
        self.cat_levels: dict[str, list] = {}
        self.impute_medians: dict[str, float] = {}
        self.missing_flag_cols: list[str] = []
        self._fitted = False
        self._train_hash: str | None = None

    # -- fit ------------------------------------------------------------------
    def fit(self, train: pl.DataFrame) -> "Preprocessor":
        feats = self.roles.features

        # Fitted one-hot levels: sorted observed train levels per categorical.
        self.cat_levels = {
            c: sorted(train[c].drop_nulls().unique().to_list())
            for c in self.roles.categorical
        }
        # Dense impute medians: train median per numeric feature.
        self.impute_medians = {
            c: float(train[c].median()) for c in self.roles.numeric  # type: ignore[arg-type]
        }
        # Missing-flag columns: feature cols that carry nulls in train, in
        # canonical order. Missingness is signal -- every such column gets a flag.
        self.missing_flag_cols = [c for c in feats if train[c].null_count() > 0]

        self._fitted = True
        self._train_hash = _frame_hash(train)
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Preprocessor.fit(train) must be called before transform().")

    # -- transform ------------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Return {'raw', 'native', 'dense'} frames for one split."""
        self._require_fitted()
        roles = self.roles
        feats = roles.features

        cohort = df.select(_cohort_expr(self.cohorts)).to_series()

        # Missing-flag indicators (shared by all three encodings).
        flags = {
            f"{c}{MISSING_SUFFIX}": df[c].is_null().cast(pl.Int8) for c in self.missing_flag_cols
        }

        base_cols = (
            [df[c] for c in roles.ids]
            + [cohort]
            + [df[c] for c in feats]
            + [s.alias(name) for name, s in flags.items()]
            + [df[c] for c in roles.outcome]
        )

        raw = pl.DataFrame(base_cols)

        # native: categoricals -> native categorical dtype, NaNs preserved.
        native = raw.with_columns(
            [pl.col(c).cast(pl.String).cast(pl.Categorical) for c in roles.categorical]
        )

        dense = self._densify(raw)

        return {"raw": raw, "native": native, "dense": dense}

    def _densify(self, raw: pl.DataFrame) -> pl.DataFrame:
        roles = self.roles
        # One-hot the 7 categoricals using *fitted train levels* (deterministic
        # column set). Unseen val/test levels produce all-zero dummies.
        onehot_cols: list[pl.Expr] = []
        for c in roles.categorical:
            for lvl in self.cat_levels[c]:
                name = f"{c}_{lvl}"
                onehot_cols.append((pl.col(c) == pl.lit(lvl)).cast(pl.Int8).alias(name))

        # Median-impute numeric features (incl. bank-feed) with TRAIN medians.
        numeric_imputed = [
            pl.col(c).fill_null(self.impute_medians[c]).alias(c) for c in roles.numeric
        ]
        bool_int = [pl.col(c).cast(pl.Int8).alias(c) for c in roles.boolean]

        keep = (
            roles.ids
            + ["cohort_week"]
            + [c for c in raw.columns if c.endswith(MISSING_SUFFIX)]
            + roles.outcome
        )
        dense = (
            raw.select(keep)
            .hstack(raw.select(numeric_imputed))
            .hstack(raw.select(bool_int))
            .hstack(raw.select(onehot_cols))
        )
        # Stable column order: ids, cohort, numeric, bool, one-hot, flags, outcomes.
        ordered = (
            roles.ids
            + ["cohort_week"]
            + roles.numeric
            + roles.boolean
            + [f"{c}_{lvl}" for c in roles.categorical for lvl in self.cat_levels[c]]
            + [c for c in raw.columns if c.endswith(MISSING_SUFFIX)]
            + roles.outcome
        )
        return dense.select(ordered)

    # -- manifest -------------------------------------------------------------
    def manifest(self, input_hashes: dict[str, str] | None = None) -> dict:
        self._require_fitted()
        roles = self.roles
        dense_feature_cols = (
            roles.numeric
            + roles.boolean
            + [f"{c}_{lvl}" for c in roles.categorical for lvl in self.cat_levels[c]]
            + [f"{c}{MISSING_SUFFIX}" for c in self.missing_flag_cols]
        )
        return {
            "pipeline_version": PIPELINE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "column_roles": {
                "ids": roles.ids,
                "outcome": roles.outcome,
                "dropped": roles.dropped,
                "derived": roles.derived,
                "categorical_features": roles.categorical,
                "bool_features": roles.boolean,
                "numeric_features": roles.numeric,
            },
            "cat_levels": {c: self.cat_levels[c] for c in roles.categorical},
            "impute_medians": self.impute_medians,
            "missing_flag_cols": self.missing_flag_cols,
            "dense_feature_cols": dense_feature_cols,
            "shapes": {
                "raw_cols": 2 + 1 + len(roles.features) + len(self.missing_flag_cols) + len(roles.outcome),
                "native_cols": 2 + 1 + len(roles.features) + len(self.missing_flag_cols) + len(roles.outcome),
                "dense_cols": 2 + 1 + len(dense_feature_cols) + len(roles.outcome),
            },
            "train_hash": self._train_hash,
            "input_hashes": input_hashes or {},
            "caveats": [
                "Outcome columns are labels only; never use them as features.",
                "prior_decision is constant (=1) on labeled rows -> zero variance in any "
                "outcome model fit on funded data; it is the target-adjacent variable for "
                "the funding-propensity model e(x). Drop it from outcome models.",
                "Bank-feed columns are MNAR: null iff has_linked_bank_feed is False (~35.7%).",
                "days_since_* nulls mean 'no such prior event' (~50%), captured by __ismissing.",
                "cohort_week is null for train (out-of-window) and 1..13 for val/test.",
            ],
        }


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def _frame_hash(df: pl.DataFrame) -> str:
    """Order-sensitive content hash of a frame (for provenance / determinism)."""
    h = hashlib.sha256()
    h.update(",".join(df.columns).encode())
    h.update(df.write_csv(None).encode())
    return h.hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run(data_dir: Path, dict_path: Path, cohorts_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dictionary = pl.read_csv(dict_path)
    cohorts = pl.read_csv(cohorts_path)
    frames = {s: pl.read_csv(data_dir / f"{s}.csv", infer_schema_length=20000) for s in SPLITS}

    pre = Preprocessor(data_dictionary, cohorts).fit(frames["train"])

    input_hashes = {s: _frame_hash(frames[s]) for s in SPLITS}
    for split in SPLITS:
        encoded = pre.transform(frames[split])
        for kind, frame in encoded.items():
            frame.write_parquet(out_dir / f"{split}_{kind}.parquet")

    manifest = pre.manifest(input_hashes=input_hashes)
    (out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--dict", type=Path, default=Path("dataset/data_dictionary.csv"))
    p.add_argument("--cohorts", type=Path, default=Path("dataset/cohort_week_definitions.csv"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    args = p.parse_args(argv)

    manifest = run(args.data, args.dict, args.cohorts, args.out)
    s = manifest["shapes"]
    print(f"Wrote encoded frames to {args.out}/")
    print(f"  raw={s['raw_cols']} native={s['native_cols']} dense={s['dense_cols']} cols/split")
    print(f"  missing-flag cols: {len(manifest['missing_flag_cols'])}")
    print(f"  manifest: {args.out / 'feature_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
