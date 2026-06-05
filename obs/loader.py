"""Load raw splits, encoded frames, and the fitted manifest for the checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from . import contracts as C


@dataclass
class Artifacts:
    """All inputs the observability layer reads (no pipeline recompute needed)."""

    raw: dict[str, pl.DataFrame]                      # split -> original csv frame
    encoded: dict[str, dict[str, pl.DataFrame]]       # split -> kind -> frame
    manifest: dict
    data_dictionary: pl.DataFrame
    cohorts: pl.DataFrame
    interventions: pl.DataFrame
    data_dir: Path
    artifacts_dir: Path

    def frame(self, split: str, kind: str = "raw") -> pl.DataFrame:
        return self.encoded[split][kind]


def load_artifacts(
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    dataset_dir: str | Path = "dataset",
) -> Artifacts:
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    dataset_dir = Path(dataset_dir)

    raw = {s: pl.read_csv(data_dir / f"{s}.csv", infer_schema_length=20000) for s in C.SPLITS}
    encoded = {
        s: {k: pl.read_parquet(artifacts_dir / f"{s}_{k}.parquet") for k in ("raw", "native", "dense")}
        for s in C.SPLITS
    }
    manifest = json.loads((artifacts_dir / "feature_manifest.json").read_text())
    data_dictionary = pl.read_csv(dataset_dir / "data_dictionary.csv")
    cohorts = pl.read_csv(dataset_dir / "cohort_week_definitions.csv")
    interventions = pl.read_csv(dataset_dir / "intervention_queries.csv")

    return Artifacts(
        raw=raw,
        encoded=encoded,
        manifest=manifest,
        data_dictionary=data_dictionary,
        cohorts=cohorts,
        interventions=interventions,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
