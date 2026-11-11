import shutil
import zipfile
from pathlib import Path

from utils.kaggle_source import resolve

IRIS = Path(__file__).resolve().parents[2] / "examples" / "data" / "iris.csv"


def test_kaggle_slug_downloads_and_returns_csv(tmp_path):
    calls = []

    def fake_download(slug, dest):
        calls.append(slug)
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / "dataset.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(IRIS, "iris.csv")
        return archive

    path = resolve("kaggle:uciml/iris", cache=tmp_path, download=fake_download)
    assert Path(path).name == "iris.csv"
    assert Path(path).read_bytes() == IRIS.read_bytes()
    assert resolve("kaggle:uciml/iris", cache=tmp_path, download=fake_download) == path
    assert calls == ["uciml/iris"]  # second call hits the cache
    assert resolve(str(IRIS)) == str(IRIS)


def test_autopilot_runs_on_resolved_kaggle_csv(tmp_path):
    dest = tmp_path / "uciml__iris"
    dest.mkdir()
    shutil.copy(IRIS, dest / "iris.csv")
    assert resolve("kaggle:uciml/iris", cache=tmp_path).endswith("iris.csv")
