# Data Observability & Dataset Exploration

A monitoring + exploration layer over the leakage-safe preprocessing pipeline for
the SMB Underwriting Challenge. It does two things:

1. **Observability** — an automated check suite that *guarantees the pipeline is
   sound* (integrity assertions) and *surfaces the data realities* (quality,
   drift, selection, intervention diagnostics), pinned to verified golden values
   so a future data refresh that breaks an assumption is caught loudly.
2. **Exploration** — an interactive UI for *seeing* the data: column profiles,
   selection/labeling, default timing, interventions, and drift.

> Scope note: this layer is only about seeing the data and proving the pipeline
> is correct. It does **not** build the A/B/C/D models — the funding-propensity
> `e(x)` here is a *diagnostic for overlap only*, not a deliverable.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-obs.txt

# 1. unzip the raw data
python -c "import zipfile; zipfile.ZipFile('dataset/dataset-compressed.zip').extractall('data')"

# 2. run the leakage-safe pipeline -> artifacts/*.parquet + feature_manifest.json
python preprocess.py --data data --out artifacts

# 3. run the observability suite -> report/observability_report.{json,html}
python run_observability.py --data data --artifacts artifacts --out report

# 4. (optional) launch the interactive explorer
streamlit run explore/app.py

# 5. run the regression tests (invariants + golden fixtures)
pytest tests/
```

`run_observability.py` exits non-zero if any **integrity** (`error`) check fails.

## What the pipeline produces

`preprocess.py` is `Preprocessor(data_dictionary, cohorts).fit(train).transform(split)`.
It fits one-hot levels and impute medians on **train only**, quarantines the six
outcome columns as labels, drops `application_timestamp` in favor of a derived
`cohort_week`, and emits three deterministic encodings per split:

| Frame | Cols | Categoricals | Missing values |
|---|---|---|---|
| `*_raw.parquet`    | 53 | integer codes              | preserved (NaN) + `__ismissing` flags |
| `*_native.parquet` | 53 | native categorical dtype   | preserved (NaN) + `__ismissing` flags |
| `*_dense.parquet`  | 73 | one-hot (7 → 27)           | median-imputed, NaN-free (+ flags keep the fact) |

Plus `feature_manifest.json` (column roles, fitted cat levels, impute medians,
missing-flag list, shapes, input hashes, caveats).

## The checks (`obs/`)

- **`integrity.py` (§5a — blocking)**: no outcome leakage; `contracts.py` roles ==
  manifest; fit-on-train reproducible; row-count preservation; dense completeness;
  native fidelity (`null == __ismissing`); one-hot validity; cohort assignment;
  encoder determinism.
- **`quality.py` (§5b — warn)**: per-feature null-rate deviation; train→test PSI
  (all) + KS (continuous), ranked; unseen categorical levels; range/sanity.
- **`selection.py` (§5c — warn)**: funded/declined + default rate by split and by
  cohort; label-availability map; `e(x)=P(funded|x)` overlap / positivity diagnostic.
- **`intervention.py` (§5d — warn)**: per-query current value, train percentile,
  in-support (p1–p99), no-op, and structural-feature flags; support coverage; the
  structural interventions needing DAG propagation.
- **`report.py`**: collects everything into JSON + a self-contained HTML report
  with provenance (input/manifest hashes, version, timestamp, golden diffs).

## Golden fixtures (`obs/contracts.py` → `GOLDEN`)

Verified against the shipped data and asserted by `tests/test_golden.py`:

```
shapes        train 85,340×44 | val 4,489×44 | test 8,817×44
encoded       raw 53 | native 53 | dense 73
funded(train) 51,722 (60.6%)   default_rate 0.1745
val funded    2,551            default_rate 0.2062  | test labels withheld
bank feed     linked False 30,453 (35.7%); observed_* null iff False
timing        days_to_default median 37; CDR@7/14/28/60/90 .072/.19/.388/.775/1.0
interventions 900 queries / 300 applicants | structural 174 (19.3%) | in-support 91.8% | no-ops 7%
```

## Data realities the tools surface

- **Out-of-time split** — train spans 18 months; val/test are a later 13-week
  window. The drift dashboard ranks the resulting shift (top: `observed_revenue_trend_3mo`).
- **Selective labels** — outcomes exist only for funded+matured loans; test
  labels are withheld. The selection explorer separates labeled vs full population.
- **The `prior_decision` trap** — constant (=1) on labeled rows: zero variance in
  any outcome model, but it *is* the target for `e(x)`.
- **Bank-feed MNAR** — the six `observed_*` columns are null **iff** no linked feed.
  Never silently imputed in a display; the dense frame imputes but keeps `__ismissing`.
- **Default timing** — defaults span days 3–90 (median 37); the timing explorer
  shows the cohort × age trajectory (the Deliverable-B object) and weekly hazard.
- **Positivity / overlap** — the legacy policy is **near-deterministic** (funds iff
  `prior_underwriter_score ≳ 0.273`), so `e(x)` has essentially no common support
  and PD/NPV are only partially identified off-policy. The overlap diagnostic flags this.

## Layout

```
preprocess.py            leakage-safe polars pipeline (fit/transform + manifest)
obs/
  contracts.py           column roles, golden values, thresholds (source of truth)
  loader.py              load raw + encoded frames + manifest
  results.py             CheckResult / CheckReport types
  integrity.py           §5a assertions (blocking)
  quality.py             §5b null / PSI / KS / range
  selection.py           §5c funded-declined, e(x) overlap, default timing
  intervention.py        §5d query diagnostics
  report.py              collect -> JSON + static HTML
run_observability.py     CLI
explore/app.py           Streamlit exploration UI (5 views)
tests/                   §5a invariants + §6 golden fixtures
```
