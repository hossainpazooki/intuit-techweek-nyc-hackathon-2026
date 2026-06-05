"""Shared fixtures: ensure data is unzipped + encoded, then load artifacts once."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "dataset"


def _ensure_data() -> None:
    if all((DATA / f"{s}.csv").exists() for s in ("train", "validation", "test")):
        return
    DATA.mkdir(exist_ok=True)
    with zipfile.ZipFile(DATASET / "dataset-compressed.zip") as z:
        for name in z.namelist():
            if name.endswith(".csv") and not name.startswith("__MACOSX"):
                (DATA / Path(name).name).write_bytes(z.read(name))


def _ensure_artifacts() -> None:
    if (ARTIFACTS / "feature_manifest.json").exists():
        return
    from preprocess import run
    run(DATA, DATASET / "data_dictionary.csv", DATASET / "cohort_week_definitions.csv", ARTIFACTS)


@pytest.fixture(scope="session")
def art():
    _ensure_data()
    _ensure_artifacts()
    from obs.loader import load_artifacts
    return load_artifacts(DATA, ARTIFACTS, DATASET)
