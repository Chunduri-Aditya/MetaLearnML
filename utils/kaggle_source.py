"""Resolve `kaggle:<owner>/<dataset>` to a local CSV via the Kaggle REST API (stdlib only)."""

from __future__ import annotations

import base64
import os
import urllib.request
import zipfile
from pathlib import Path

PREFIX = "kaggle:"
API = "https://www.kaggle.com/api/v1/datasets/download/{slug}"
CACHE = Path("data/kaggle")


def _load_env(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def _download(slug: str, dest: Path) -> Path:
    _load_env()
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not user or not key:
        raise RuntimeError("Set KAGGLE_USERNAME and KAGGLE_KEY in .env to download from Kaggle")
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    req = urllib.request.Request(API.format(slug=slug), headers={"Authorization": f"Basic {token}"})
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "dataset.zip"
    with urllib.request.urlopen(req, timeout=120) as resp:
        archive.write_bytes(resp.read())
    return archive


def resolve(source: str, cache: Path = CACHE, download=_download) -> str:
    """Return a local CSV path. Plain paths pass through unchanged."""
    if not source.startswith(PREFIX):
        return source
    slug = source[len(PREFIX) :].strip("/")
    dest = cache / slug.replace("/", "__")
    csvs = sorted(dest.rglob("*.csv")) if dest.exists() else []
    if not csvs:
        archive = download(slug, dest)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        csvs = sorted(dest.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in Kaggle dataset {slug}")
    return str(max(csvs, key=lambda p: p.stat().st_size))
