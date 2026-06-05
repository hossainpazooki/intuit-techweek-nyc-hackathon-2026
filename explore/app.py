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


try:
    art = _art()
except Exception as e:  # pragma: no cover - UI guard
    st.error(f"Could not load artifacts: {e}\n\nRun `python preprocess.py` first.")
    st.stop()

st.sidebar.title("SMB Underwriting")
st.sidebar.caption("Dataset & pipeline explorer")
view = st.sidebar.radio(
    "View",
    ["Column profiler", "Selection explorer", "Timing explorer",
     "Intervention explorer", "Drift dashboard"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Reads parquet artifacts. Missingness is shown as its own "
                   "category — never silently imputed.")


# --------------------------------------------------------------------------- #
# 1. Column profiler
# --------------------------------------------------------------------------- #
def column_profiler():
    st.header("Column profiler")
    groups = ["(all)"] + sorted(art.data_dictionary["group"].unique().to_list())
    g = st.selectbox("Filter by group", groups)
    fields = art.data_dictionary["field"].to_list()
    if g != "(all)":
        fields = art.data_dictionary.filter(pl.col("group") == g)["field"].to_list()
    col = st.selectbox("Column", fields)
    group, note = _dict_note(col)
    dtype = art.data_dictionary.filter(pl.col("field") == col).row(0, named=True)["dtype"]

    c1, c2, c3 = st.columns(3)
    c1.metric("group", group)
    c2.metric("dtype", dtype)
    nulls = {s: art.raw[s][col].null_count() / len(art.raw[s]) for s in C.SPLITS}
    c3.metric("train null rate", f"{nulls['train']:.1%}")
    st.caption(note)

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

    with st.expander("Funding-propensity overlap e(x) — positivity diagnostic"):
        r = selection.propensity_overlap(art)
        st.write(r.message)
        h = pl.DataFrame(r.details["score_hist"])
        fig = go.Figure()
        fig.add_bar(x=h["bin_lo"], y=h["funded"], name="funded", marker_color="#27ae60", opacity=0.6)
        fig.add_bar(x=h["bin_lo"], y=h["declined"], name="declined", marker_color="#c0392b", opacity=0.6)
        fig.update_layout(barmode="overlay", height=320, title="e(x): funded vs declined")
        st.plotly_chart(fig, width="stretch")


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

    q = st.selectbox("Query", diag["query_id"].to_list())
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
    cov = intervention.check_feature_support_coverage(art).details["table"]
    st.dataframe(pl.DataFrame(cov), hide_index=True, width="stretch")


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


VIEWS = {
    "Column profiler": column_profiler,
    "Selection explorer": selection_explorer,
    "Timing explorer": timing_explorer,
    "Intervention explorer": intervention_explorer,
    "Drift dashboard": drift_dashboard,
}
VIEWS[view]()
