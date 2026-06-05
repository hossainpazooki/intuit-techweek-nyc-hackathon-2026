#!/usr/bin/env python3
"""Run the preprocessing-pipeline observability suite.

Loads the raw splits + encoded parquet frames + ``feature_manifest.json``, runs
the integrity (§5a), quality/drift (§5b), selection (§5c), and intervention
(§5d) checks, and writes a machine-readable JSON report and a self-contained
HTML report. Exits non-zero if any blocking (``error``) integrity check fails.

    python run_observability.py --data data --artifacts artifacts --out report

Run ``python preprocess.py`` first to produce the encoded frames + manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from obs.report import generate
from obs.results import Severity


def _print_summary(report: dict, rep) -> None:
    s = report["summary"]
    print("\n" + "=" * 78)
    print("PIPELINE OBSERVABILITY")
    print("=" * 78)
    for r in rep.results:
        status = "PASS " if r.passed else r.severity.value.upper().ljust(5)
        print(f"  [{status}] {r.section:>12} / {r.name}: {r.message}")

    print("-" * 78)
    print("Golden-value regression (§6):")
    for g in report["golden_diff"]:
        mark = "ok  " if g["match"] else "DRIFT"
        print(f"  [{mark}] {g['fixture']:>26}: actual={g['actual']}  golden={g['golden']}")
    print("-" * 78)

    verdict = "PASS" if s["passed"] else "FAIL"
    print(f"RESULT: {verdict}  ({s['n_error']} error(s), {s['n_warn']} warning(s), "
          f"golden {'all match' if s['golden_all_match'] else 'DRIFTED'})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data", help="dir with train/validation/test.csv")
    p.add_argument("--artifacts", default="artifacts", help="dir with *_{raw,native,dense}.parquet + manifest")
    p.add_argument("--dataset", default="dataset", help="dir with data_dictionary / cohort / intervention csvs")
    p.add_argument("--out", default="report", help="output dir for JSON + HTML report")
    args = p.parse_args(argv)

    for f in ("train.csv", "validation.csv", "test.csv"):
        if not (Path(args.data) / f).exists():
            print(f"ERROR: {Path(args.data) / f} not found. Unzip dataset/dataset-compressed.zip "
                  f"into {args.data}/.", file=sys.stderr)
            return 2
    if not (Path(args.artifacts) / "feature_manifest.json").exists():
        print(f"ERROR: {Path(args.artifacts) / 'feature_manifest.json'} not found. "
              f"Run: python preprocess.py --data {args.data} --out {args.artifacts}", file=sys.stderr)
        return 2

    report, rep = generate(args.data, args.artifacts, args.dataset, args.out)
    _print_summary(report, rep)
    print(f"\nWrote {args.out}/observability_report.json and {args.out}/observability_report.html")

    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
