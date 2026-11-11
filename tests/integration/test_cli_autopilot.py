"""The README quickstart command runs on the CSV that ships with the repo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO / "examples" / "data" / "iris.csv"


def test_sample_csv_is_bundled():
    frame = pd.read_csv(SAMPLE_CSV)
    assert frame.shape == (150, 5)
    assert frame["species"].nunique() == 3


def test_autopilot_cli_runs_on_bundled_csv(tmp_path):
    # Run from tmp_path so the default relative output paths land there, not in the repo.
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "main.py"),
            "--autopilot",
            str(SAMPLE_CSV),
            "--label",
            "species",
            "--max-candidates",
            "1",
            "--cv-folds",
            "3",
            "--ordering",
            "registry",
            "--no-use-proxy",
            "--no-use-meta-learning",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    results = pd.read_csv(tmp_path / "experiments" / "results.csv")
    assert len(results) >= 1
