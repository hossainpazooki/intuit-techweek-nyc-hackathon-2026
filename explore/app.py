"""Dataset exploration UI for the SMB Underwriting Challenge (build brief §7).

Analyst-facing surface over the encoded parquet artifacts. Reads parquet (fast),
respects the train/val/test boundary, and never silently imputes in a display --
missingness is shown as its own category, consistent with the pipeline policy.

    streamlit run explore/app.py

Run ``python preprocess.py`` first so the artifacts exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obs import contracts as C  # noqa: E402
from obs import intervention, quality, selection  # noqa: E402
from obs.loader import load_artifacts  # noqa: E402

st.set_page_config(page_title="SMB Underwriting — Dataset Explorer", layout="wide")
SPLIT_COLORS = {"train": "#2471a3", "validation": "#27ae60", "test": "#b9770e"}


@st.cache_resource(show_spinner="Loading artifacts…")
def _art():
    return load_artifacts(ROOT / "data", ROOT / "artifacts", ROOT / "dataset")


@st.cache_data(show_spinner=False)
def _dict_note(field: str) -> tuple[str, str]:
    art = _art()
    row = art.data_dictionary.filter(pl.col("field") == field)
    if row.height:
        r = row.row(0, named=True)
        return r["group"], r["notes"]
    return "", ""


@st.cache_data(show_spinner=False)
def _intervenable_map() -> dict:
    return _art().intervenable_map()


def _intervenability_badge(field: str) -> str:
    m = _intervenable_map()
    if field not in m:
        return ""
    return ("🍃 intervenable — a valid do() target" if m[field]
            else "🧩 structural (intervenable=False) — needs DAG propagation, not a column edit")


try:
    art = _art()
except Exception as e:  # pragma: no cover - UI guard
    st.error(f"Could not load artifacts: {e}\n\nRun `python preprocess.py` first.")
    st.stop()

st.sidebar.title("SMB Underwriting")
st.sidebar.caption("Dataset & pipeline explorer")
view = st.sidebar.radio(
    "View",
    ["Column profiler", "Selection explorer", "Loan rejection", "Timing explorer",
     "Days to default", "Intervention explorer", "Drift dashboard"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Reads parquet artifacts. Missingness is shown as its own "
                   "category — never silently imputed.")


# --------------------------------------------------------------------------- #
# 1. Column profiler
# --------------------------------------------------------------------------- #
def column_profiler():
    st.header("Column profiler")
    imap = _intervenable_map()
    f1, f2 = st.columns(2)
    groups = ["(all)"] + sorted(art.data_dictionary["group"].unique().to_list())
    g = f1.selectbox("Filter by group", groups)
    iv = f2.selectbox("Filter by intervenability",
                      ["(all)", "🍃 intervenable", "🧩 structural (non-intervenable)"])
    fields = art.data_dictionary["field"].to_list()
    if g != "(all)":
        fields = [f for f in fields if _dict_note(f)[0] == g]
    if iv.startswith("🍃"):
        fields = [f for f in fields if imap.get(f, False)]
    elif iv.startswith("🧩"):
        fields = [f for f in fields if not imap.get(f, True)]
    if not fields:
        st.info("No columns match this filter combination.")
        return
    col = st.selectbox("Column", fields)
    group, note = _dict_note(col)
    dtype = art.data_dictionary.filter(pl.col("field") == col).row(0, named=True)["dtype"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("group", group)
    c2.metric("dtype", dtype)
    c3.metric("intervenable", "yes" if imap.get(col) else "no")
    nulls = {s: art.raw[s][col].null_count() / len(art.raw[s]) for s in C.SPLITS}
    c4.metric("train null rate", f"{nulls['train']:.1%}")
    st.caption(f"{_intervenability_badge(col)}  ·  {note}")

    is_cat = dtype in ("categorical", "bool") or col in C.CATEGORICAL_FEATURES
    fig = go.Figure()
    for split in C.SPLITS:
        s = art.raw[split][col]
        if is_cat:
            # show missing as its own bucket
            vc = (s.cast(pl.String).fill_null("(missing)").value_counts())
            vc = vc.sort(vc.columns[0])
            total = len(s)
            fig.add_bar(x=vc[vc.columns[0]].cast(pl.String).to_list(),
                        y=(vc["count"] / total).to_list(), name=split,
                        marker_color=SPLIT_COLORS[split], opacity=0.7)
        else:
            vals = s.drop_nulls().to_numpy()
            if vals.size:
                fig.add_histogram(x=vals, name=split, histnorm="probability density",
                                  marker_color=SPLIT_COLORS[split], opacity=0.55, nbinsx=40)
    fig.update_layout(barmode="overlay" if is_cat else "overlay", height=420,
                      title=f"{col} — split overlay (train / val / test)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Null rate by split")
    st.dataframe(pl.DataFrame({"split": list(nulls), "null_rate": [round(v, 4) for v in nulls.values()]}),
                 hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# 2. Selection explorer
# --------------------------------------------------------------------------- #
def selection_explorer():
    st.header("Selection explorer")
    st.caption("~60.6% of train was historically funded; outcomes exist only for "
               "funded+matured loans. Default rate is over funded rows only.")
    slice_by = st.selectbox("Slice by", ["sector", "geography_region",
                                         "owner_personal_credit_band", "cohort_week"])
    population = st.radio("Population", ["labeled (funded)", "full applicant population"],
                          horizontal=True)
    split = st.selectbox("Split", C.SPLITS, index=0)

    df = art.frame(split, "raw")  # encoded frame carries cohort_week
    if slice_by == "cohort_week" and split == "train":
        st.info("train has no cohort_week (out-of-window). Pick validation/test for cohort slices.")
    funded = df.filter(pl.col("prior_decision") == 1)
    base = funded if population.startswith("labeled") else df

    grp = base.group_by(slice_by).agg(
        pl.len().alias("n"),
        (pl.col("prior_decision") == 1).sum().alias("funded"),
        pl.col("default_flag").mean().alias("default_rate"),
    ).sort(slice_by)
    pdf = grp.to_pandas()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(pdf, x=slice_by, y="n", title="Count by slice",
                           color_discrete_sequence=[SPLIT_COLORS[split]]), width="stretch")
    c2.plotly_chart(px.bar(pdf, x=slice_by, y="default_rate", title="Default rate by slice",
                           color_discrete_sequence=["#c0392b"]), width="stretch")
    st.dataframe(grp, hide_index=True, width="stretch")
    st.caption("→ For the funding decision itself (rejection rate, the legacy policy "
               "threshold, and the e(x) overlap/positivity diagnostic) see the "
               "**Loan rejection** view.")


# --------------------------------------------------------------------------- #
# 3. Timing explorer
# --------------------------------------------------------------------------- #
def timing_explorer():
    st.header("Timing explorer — Deliverable B object")
    st.caption("Defaults span days 3–90 (median 37) — not an early-only spike. "
               "Cumulative default rate by cohort_week × loan_age_weeks (val funded loans).")
    t = selection.default_timing(art)
    c1, c2 = st.columns([1, 2])
    c1.metric("median days_to_default", f"{t['median_day']:.0f}")
    cdr = pl.DataFrame({"day": list(t["cdr_by_day"]), "cdr": list(t["cdr_by_day"].values())})
    c1.plotly_chart(px.line(cdr.to_pandas(), x="day", y="cdr", markers=True,
                            title="CDR by day (train)"), width="stretch")

    n = C.GOLDEN["cohorts"]["n_weeks"]
    z = [[None] * n for _ in range(n)]
    for cell in t["cohort_age_grid"]:
        z[cell["cohort_week"] - 1][cell["loan_age_weeks"] - 1] = cell["cumulative_default_rate"]
    heat = go.Figure(go.Heatmap(z=z, x=[f"age {a}" for a in range(1, n + 1)],
                                y=[f"wk {w}" for w in range(1, n + 1)], colorscale="Reds"))
    heat.update_layout(title="Cumulative default rate: cohort_week × loan_age_weeks", height=460)
    c2.plotly_chart(heat, width="stretch")

    st.subheader("Discrete-time hazard (val funded)")
    grid = pl.DataFrame(t["cohort_age_grid"])
    hz = (grid.group_by("loan_age_weeks")
          .agg(pl.col("cumulative_default_rate").mean().alias("cdr"))
          .sort("loan_age_weeks"))
    hz = hz.with_columns((pl.col("cdr") - pl.col("cdr").shift(1).fill_null(0)).alias("hazard"))
    st.plotly_chart(px.bar(hz.to_pandas(), x="loan_age_weeks", y="hazard",
                           title="Marginal weekly hazard"), width="stretch")


# --------------------------------------------------------------------------- #
# 4. Intervention explorer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Computing query diagnostics…")
def _diag():
    return intervention.build_query_diagnostics(_art())


def intervention_explorer():
    st.header("Intervention explorer — Deliverable C")
    diag = _diag()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("queries", diag.height)
    c2.metric("structural", int(diag["structural"].sum()),
              help="intervenable=False → need DAG propagation")
    c3.metric("in-support", f"{diag['in_support'].fill_null(False).mean():.1%}")
    c4.metric("no-ops", int(diag["no_op"].sum()))

    st.caption("🍃 **intervenable** features are valid single-feature do() targets; "
               "🧩 **structural** (intervenable=False) interventions move an upstream node and "
               "must propagate through the causal DAG — e.g. do(has_linked_bank_feed=True) "
               "changes the entire bank-feed block, not one column.")
    iv_feats = sorted(diag.filter(~pl.col("structural"))["feature_name"].unique().to_list())
    st_feats = sorted(diag.filter(pl.col("structural"))["feature_name"].unique().to_list())
    k1, k2 = st.columns(2)
    k1.markdown(f"**🍃 Intervenable features intervened on ({len(iv_feats)})**")
    k1.write(", ".join(iv_feats) or "—")
    k2.markdown(f"**🧩 Structural features intervened on ({len(st_feats)})**")
    k2.write(", ".join(st_feats) or "—")

    only = st.radio("Browse queries", ["all", "🍃 intervenable only", "🧩 structural only"],
                    horizontal=True)
    pool = diag
    if only.startswith("🍃"):
        pool = diag.filter(~pl.col("structural"))
    elif only.startswith("🧩"):
        pool = diag.filter(pl.col("structural"))
    q = st.selectbox("Query", pool["query_id"].to_list())
    row = diag.filter(pl.col("query_id") == q).row(0, named=True)
    feat = row["feature_name"]
    badge = "🧩 structural (DAG)" if row["structural"] else "🍃 leaf (intervenable)"
    st.write(f"**{feat}** — {badge} · current `{row['current_value']}` → intervention "
             f"`{row['intervention_value']}` · percentile {row['percentile_in_train']} · "
             f"{'in-support' if row['in_support'] else 'OUT-OF-SUPPORT'}"
             f"{' · NO-OP' if row['no_op'] else ''}")

    col = art.raw["train"][feat].drop_nulls().to_numpy()
    fig = go.Figure()
    fig.add_histogram(x=col, nbinsx=40, marker_color="#2471a3", opacity=0.6, name="train marginal")
    if row["current_value"] is not None:
        fig.add_vline(x=row["current_value"], line_color="#27ae60", annotation_text="current")
    fig.add_vline(x=row["intervention_value"], line_color="#c0392b", annotation_text="do()")
    fig.update_layout(height=360, title=f"{feat}: intervention vs train marginal")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Per-feature support coverage")
    cov = pl.DataFrame(intervention.check_feature_support_coverage(art).details["table"])
    cov = cov.with_columns((~pl.col("structural")).alias("intervenable")).select(
        ["feature", "intervenable", "structural", "n_queries", "in_support_frac", "out_of_range"])
    st.dataframe(cov, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# 5. Drift dashboard
# --------------------------------------------------------------------------- #
def drift_dashboard():
    st.header("Drift dashboard — out-of-time shift (train → test)")
    res = quality.check_distribution_drift(art)
    st.caption(res.message)
    tbl = pl.DataFrame(res.details["table"])
    st.dataframe(tbl, hide_index=True, width="stretch")

    feat = st.selectbox("Inspect feature distribution", tbl["feature"].to_list())
    fig = go.Figure()
    for split in ("train", "test"):
        vals = art.raw[split][feat].drop_nulls().to_numpy()
        if vals.size:
            fig.add_histogram(x=vals, name=split, histnorm="probability density",
                              marker_color=SPLIT_COLORS[split], opacity=0.55, nbinsx=40)
    fig.update_layout(barmode="overlay", height=400, title=f"{feat}: train vs test")
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# 2b. Loan rejection (the funding decision)
# --------------------------------------------------------------------------- #
def loan_rejection():
    st.header("Loan rejection — the funding decision")
    st.caption("Funded == legacy `prior_decision == 1`. Declined applicants are "
               "unlabeled (selective labels). The legacy policy is near-deterministic "
               "in `prior_underwriter_score`, so there is little common support.")
    split = st.selectbox("Split", C.SPLITS, index=0)
    df = art.frame(split, "raw")  # encoded frame carries cohort_week
    n = len(df)
    declined = int((df["prior_decision"] == 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("applications", f"{n:,}")
    c2.metric("declined", f"{declined:,}")
    c3.metric("rejection rate", f"{declined / n:.1%}")

    slice_by = st.selectbox("Rejection rate by", ["sector", "geography_region",
                            "owner_personal_credit_band", "employee_count_bucket", "cohort_week"])
    if slice_by == "cohort_week" and split == "train":
        st.info("train has no cohort_week (out-of-window). Pick validation/test for cohort slices.")
    grp = (df.group_by(slice_by)
           .agg(pl.len().alias("n"), (pl.col("prior_decision") == 0).mean().alias("rejection_rate"))
           .sort(slice_by))
    st.plotly_chart(px.bar(grp.to_pandas(), x=slice_by, y="rejection_rate",
                           title=f"Rejection rate by {slice_by}",
                           color_discrete_sequence=["#c0392b"]), width="stretch")

    st.subheader("The (near-deterministic) legacy policy")
    funded = df.filter(pl.col("prior_decision") == 1)["prior_underwriter_score"].drop_nulls().to_numpy()
    decl = df.filter(pl.col("prior_decision") == 0)["prior_underwriter_score"].drop_nulls().to_numpy()
    fig = go.Figure()
    fig.add_histogram(x=decl, name="declined", marker_color="#c0392b", opacity=0.6, nbinsx=50)
    fig.add_histogram(x=funded, name="funded", marker_color="#27ae60", opacity=0.6, nbinsx=50)
    if funded.size and decl.size:
        thr = 0.5 * (funded.min() + decl.max())
        fig.add_vline(x=thr, line_dash="dash", annotation_text=f"~{thr:.3f}")
        gap = funded.min() - decl.max()
        st.caption(f"Funded iff `prior_underwriter_score ≳ {thr:.3f}` — declined max "
                   f"{decl.max():.4f} < funded min {funded.min():.4f} (gap {gap:.4f}). "
                   f"Zero overlap → severe positivity violation.")
    fig.update_layout(barmode="overlay", height=360,
                      title="prior_underwriter_score: funded vs declined")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Funding-propensity e(x) — overlap / positivity diagnostic"):
        r = selection.propensity_overlap(art)
        st.write(r.message)
        h = pl.DataFrame(r.details["score_hist"])
        fig2 = go.Figure()
        fig2.add_bar(x=h["bin_lo"], y=h["funded"], name="funded", marker_color="#27ae60", opacity=0.6)
        fig2.add_bar(x=h["bin_lo"], y=h["declined"], name="declined", marker_color="#c0392b", opacity=0.6)
        fig2.update_layout(barmode="overlay", height=300, title="e(x): funded vs declined")
        st.plotly_chart(fig2, width="stretch")


# --------------------------------------------------------------------------- #
# 3b. Days to default (timing distribution)
# --------------------------------------------------------------------------- #
def days_to_default_view():
    st.header("Days to default — timing of defaults")
    st.caption("`days_to_default` is observed only for defaulted, labeled loans. "
               "Defaults span days 3–90 (median 37) — not an early-only spike, so a "
               "binary default flag throws away signal Deliverable B needs.")
    split = st.radio("Labeled split", ["train", "validation"], horizontal=True)
    df = art.frame(split, "raw")
    dflt = df.filter(pl.col("default_flag") == 1)
    dd = dflt["days_to_default"].drop_nulls().to_numpy().astype(float)
    if dd.size == 0:
        st.info("No labeled defaults in this split.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("defaulted loans", f"{dd.size:,}")
    c2.metric("median day", f"{np.median(dd):.0f}")
    c3.metric("IQR (days)", f"{np.percentile(dd, 25):.0f}–{np.percentile(dd, 75):.0f}")

    fig = go.Figure(go.Histogram(x=dd, nbinsx=45, marker_color="#c0392b"))
    fig.update_layout(height=320, title="days_to_default distribution",
                      xaxis_title="day (1–90)", yaxis_title="defaults")
    st.plotly_chart(fig, width="stretch")

    days = np.arange(1, 91)
    cdr = [float((dd <= d).mean()) for d in days]
    st.plotly_chart(px.line(x=days, y=cdr, markers=False,
                            labels={"x": "loan day", "y": "cumulative default rate"},
                            title="Cumulative default rate by day"), width="stretch")

    dims = ["sector", "owner_personal_credit_band", "geography_region"]
    if split == "validation":
        dims.append("cohort_week")
    slice_by = st.selectbox("Compare timing distribution by", dims)
    sub = dflt.select([slice_by, "days_to_default"]).drop_nulls().to_pandas()
    st.plotly_chart(px.box(sub, x=slice_by, y="days_to_default",
                           title=f"days_to_default by {slice_by}",
                           color_discrete_sequence=["#2471a3"]), width="stretch")
    st.caption("Compare against the cohort × loan-age cumulative grid (the Deliverable-B "
               "object) in the **Timing explorer** view.")


VIEWS = {
    "Column profiler": column_profiler,
    "Selection explorer": selection_explorer,
    "Loan rejection": loan_rejection,
    "Timing explorer": timing_explorer,
    "Days to default": days_to_default_view,
    "Intervention explorer": intervention_explorer,
    "Drift dashboard": drift_dashboard,
}
VIEWS[view]()
