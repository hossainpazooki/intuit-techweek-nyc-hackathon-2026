"""Collect all checks into a structured report: JSON (machine) + static HTML (human).

Includes provenance (build brief §5e): input + manifest hashes, pipeline version,
timestamp, and a golden-value diff table, so two runs are directly comparable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from . import contracts as C
from . import integrity, intervention, quality, selection
from .loader import Artifacts, load_artifacts
from .results import CheckReport, CheckResult, Severity

SEV_COLOR = {"error": "#c0392b", "warn": "#b9770e", "info": "#2471a3"}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def provenance(art: Artifacts) -> dict:
    inputs = {s: _hash_file(art.data_dir / f"{s}.csv") for s in C.SPLITS}
    inputs["feature_manifest.json"] = _hash_file(art.artifacts_dir / "feature_manifest.json")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": art.manifest.get("pipeline_version", "unknown"),
        "input_hashes": inputs,
        "encoded_shapes": art.manifest.get("shapes", {}),
    }


def golden_diff(art: Artifacts) -> list[dict]:
    """Actual vs golden for the headline fixtures (build brief §6)."""
    g = C.GOLDEN
    sel = selection.funded_declined_summary(art).details["table"]
    tr, va = sel[0], sel[1]
    timing = selection.default_timing(art)
    diag = intervention.build_query_diagnostics(art)
    rows = [
        ("train_funded", tr["funded"], g["selection"]["train_funded"]),
        ("train_default_rate", tr["default_rate"], g["selection"]["train_default_rate"]),
        ("val_funded", va["funded"], g["selection"]["val_funded"]),
        ("val_default_rate", va["default_rate"], g["selection"]["val_default_rate"]),
        ("days_to_default_median", timing["median_day"], g["timing"]["days_to_default_median"]),
        ("n_queries", len(diag), g["interventions"]["n_queries"]),
        ("structural_interventions", int(diag["structural"].sum()), g["interventions"]["structural"]),
        ("in_support_frac", round(float(diag["in_support"].fill_null(False).mean()), 4),
         g["interventions"]["in_support_frac"]),
    ]
    out = []
    for name, actual, golden in rows:
        ok = actual is not None and C.golden_close(float(actual), float(golden))
        out.append({"fixture": name, "actual": actual, "golden": golden, "match": ok})
    return out


# --------------------------------------------------------------------------- #
# Build report
# --------------------------------------------------------------------------- #


def run_checks(art: Artifacts) -> CheckReport:
    rep = CheckReport()
    rep.extend(integrity.run_all(art))
    rep.extend(quality.run_all(art))
    rep.extend(selection.run_all(art))
    rep.extend(intervention.run_all(art))
    return rep


def build_report(art: Artifacts) -> dict:
    rep = run_checks(art)
    golden = golden_diff(art)
    return {
        "provenance": provenance(art),
        "summary": {
            "passed": rep.passed,
            "n_checks": len(rep.results),
            "n_error": rep.n_error,
            "n_warn": rep.n_warn,
            "golden_all_match": all(r["match"] for r in golden),
        },
        "golden_diff": golden,
        "results": [r.to_dict() for r in rep.results],
    }, rep


# --------------------------------------------------------------------------- #
# Charts (embedded in the HTML)
# --------------------------------------------------------------------------- #


def _fig_html(fig: go.Figure, *, first: bool) -> str:
    return pio.to_html(fig, include_plotlyjs=("inline" if first else False), full_html=False,
                       config={"displayModeBar": False})


def _charts(art: Artifacts, rep: CheckReport) -> str:
    parts = []
    # 1. Drift ranking
    drift = next((r for r in rep.results if r.name == "distribution_drift"), None)
    if drift:
        tbl = drift.details["table"][:15][::-1]
        fig = go.Figure(go.Bar(x=[r["psi"] for r in tbl], y=[r["feature"] for r in tbl],
                               orientation="h", marker_color="#2471a3"))
        fig.update_layout(title="Top train->test drift (PSI)", height=420,
                          margin=dict(l=10, r=10, t=40, b=10))
        parts.append(_fig_html(fig, first=True))
    # 2. e(x) overlap
    ov = next((r for r in rep.results if r.name == "propensity_overlap"), None)
    if ov:
        h = ov.details["score_hist"]
        fig = go.Figure()
        fig.add_bar(x=[r["bin_lo"] for r in h], y=[r["funded"] for r in h], name="funded",
                    marker_color="#27ae60", opacity=0.6)
        fig.add_bar(x=[r["bin_lo"] for r in h], y=[r["declined"] for r in h], name="declined",
                    marker_color="#c0392b", opacity=0.6)
        fig.update_layout(title="Funding propensity e(x): funded vs declined (overlap)",
                          barmode="overlay", height=360, margin=dict(l=10, r=10, t=40, b=10))
        parts.append(_fig_html(fig, first=False))
    # 3. Cohort x age default trajectory (Deliverable B object)
    timing = selection.default_timing(art)
    n = C.GOLDEN["cohorts"]["n_weeks"]
    z = [[None] * n for _ in range(n)]
    for cell in timing["cohort_age_grid"]:
        z[cell["cohort_week"] - 1][cell["loan_age_weeks"] - 1] = cell["cumulative_default_rate"]
    fig = go.Figure(go.Heatmap(z=z, x=[f"age {a}" for a in range(1, n + 1)],
                               y=[f"wk {w}" for w in range(1, n + 1)], colorscale="Reds"))
    fig.update_layout(title="Cumulative default rate: cohort_week x loan_age_weeks (val)",
                      height=420, margin=dict(l=10, r=10, t=40, b=10))
    parts.append(_fig_html(fig, first=False))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML rendering (self-contained, no server)
# --------------------------------------------------------------------------- #


def _results_rows(results: list[CheckResult]) -> str:
    out = []
    for r in results:
        status = "PASS" if r.passed else r.severity.value.upper()
        color = "#27ae60" if r.passed else SEV_COLOR[r.severity.value]
        out.append(
            f"<tr><td><span style='color:{color};font-weight:600'>{status}</span></td>"
            f"<td>{r.section}</td><td>{r.name}</td><td>{r.n_offending}</td>"
            f"<td>{_esc(r.message)}</td></tr>"
        )
    return "\n".join(out)


def _golden_rows(golden: list[dict]) -> str:
    out = []
    for g in golden:
        mark = "MATCH" if g["match"] else "DRIFT"
        color = "#27ae60" if g["match"] else "#c0392b"
        out.append(f"<tr><td>{g['fixture']}</td><td>{g['actual']}</td><td>{g['golden']}</td>"
                   f"<td style='color:{color}'>{mark}</td></tr>")
    return "\n".join(out)


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_html(report: dict, rep: CheckReport, art: Artifacts) -> str:
    s = report["summary"]
    banner = "PASS" if s["passed"] else "FAIL"
    banner_color = "#27ae60" if s["passed"] else "#c0392b"
    prov = report["provenance"]
    charts = _charts(art, rep)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Pipeline Observability Report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f6f8;color:#1c2833}}
 .wrap{{max-width:1100px;margin:0 auto;padding:24px}}
 h1{{margin:0 0 4px}} .sub{{color:#5d6d7e;font-size:13px;margin-bottom:16px}}
 .banner{{display:inline-block;padding:8px 18px;border-radius:6px;color:#fff;font-weight:700;
   font-size:18px;background:{banner_color}}}
 .cards{{display:flex;gap:12px;margin:16px 0}}
 .card{{background:#fff;border-radius:8px;padding:12px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08);flex:1}}
 .card b{{font-size:22px}} .card span{{color:#5d6d7e;font-size:12px;display:block}}
 table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;
   box-shadow:0 1px 3px rgba(0,0,0,.08);margin:10px 0 24px;font-size:13px}}
 th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #eef1f4;vertical-align:top}}
 th{{background:#eaeef2;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
 h2{{margin:24px 0 6px;font-size:16px}} code{{background:#eef1f4;padding:1px 5px;border-radius:4px}}
 .charts > div{{background:#fff;border-radius:8px;margin:12px 0;padding:8px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
</style></head><body><div class="wrap">
 <h1>SMB Underwriting — Pipeline Observability</h1>
 <div class="sub">generated {prov['generated_utc']} · pipeline v{prov['pipeline_version']} ·
   train/val/test hashes {prov['input_hashes']['train']} / {prov['input_hashes']['validation']} /
   {prov['input_hashes']['test']} · manifest {prov['input_hashes']['feature_manifest.json']}</div>
 <span class="banner">{banner}</span>
 <div class="cards">
  <div class="card"><b>{s['n_checks']}</b><span>checks run</span></div>
  <div class="card"><b style="color:#c0392b">{s['n_error']}</b><span>blocking errors</span></div>
  <div class="card"><b style="color:#b9770e">{s['n_warn']}</b><span>warnings</span></div>
  <div class="card"><b>{'✔' if s['golden_all_match'] else '✗'}</b><span>golden fixtures</span></div>
 </div>
 <h2>Golden-value regression (§6)</h2>
 <table><tr><th>fixture</th><th>actual</th><th>golden</th><th>match</th></tr>
  {_golden_rows(report['golden_diff'])}</table>
 <h2>Checks</h2>
 <table><tr><th>status</th><th>section</th><th>check</th><th>offending</th><th>message</th></tr>
  {_results_rows(rep.results)}</table>
 <h2>Visuals</h2>
 <div class="charts">{charts}</div>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def generate(data_dir="data", artifacts_dir="artifacts", dataset_dir="dataset",
             out_dir="report") -> tuple[dict, CheckReport]:
    art = load_artifacts(data_dir, artifacts_dir, dataset_dir)
    report, rep = build_report(art)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "observability_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "observability_report.html").write_text(render_html(report, rep, art))
    return report, rep
