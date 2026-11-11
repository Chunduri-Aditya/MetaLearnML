"""Grow the meta-model history by running autopilot on a list of Kaggle tabular datasets."""

import subprocess
import sys

DATASETS: list[tuple[str, str]] = []  # (owner/dataset, label column); filled at hardening


def main() -> int:
    if not DATASETS:
        raise NotImplementedError("kaggle_meta_history dataset list")
    for slug, label in DATASETS:
        cmd = [sys.executable, "main.py", "--autopilot", f"kaggle:{slug}", "--label", label]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
