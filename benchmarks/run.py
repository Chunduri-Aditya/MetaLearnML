"""Command-line entry point for the evidence-driven benchmark."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from benchmarks.manifest import (
    BENCHMARK_ROOT,
    LockedDataset,
    fetch_and_lock,
    load_locked_frame,
    validate_locked_datasets,
)
from benchmarks.protocol import (
    MeasuredRun,
    candidate_rows,
    history_dataframe,
    history_records_for_run,
    leave_one_dataset_out,
    measure_dataset_split,
    replay_and_test,
)
from engine.meta_history import environment_fingerprint, repository_commit

DEFAULT_CONFIG = BENCHMARK_ROOT / "benchmark_config.yaml"
RESULTS_DIR = BENCHMARK_ROOT / "results"
CACHE_RUNS_DIR = BENCHMARK_ROOT / "cache" / "runs"
META_ARTIFACT_DIR = BENCHMARK_ROOT / "cache" / "meta_models"
REPLAY_CACHE_DIR = BENCHMARK_ROOT / "cache" / "replay"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = {
        "profile",
        "outer_seeds",
        "test_ratio",
        "cv_folds",
        "methods",
        "random_ordering_seeds",
        "absolute_budgets",
        "fractional_budgets",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Benchmark config is missing keys: {missing}")
    if config["profile"] != "balanced-15-v1":
        raise ValueError("Only the reviewed balanced-15-v1 profile can publish evidence")
    expected_methods = {"exhaustive", "proxy", "meta", "proxy_meta", "random"}
    if set(config["methods"]) != expected_methods:
        raise ValueError(f"Benchmark methods must be exactly {sorted(expected_methods)}")
    return config


def _write_csv_gzip(dataframe: pd.DataFrame, path: Path) -> None:
    """Write gzip with a fixed timestamp so identical rows have identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            dataframe.to_csv(compressed, index=False, lineterminator="\n")


def _checkpoint_path(record: LockedDataset, seed: int) -> Path:
    return CACHE_RUNS_DIR / f"{record.benchmark_id}_{seed}.joblib"


def _measure_or_load(
    record: LockedDataset,
    config: dict[str, Any],
    *,
    seed: int,
    force: bool,
) -> MeasuredRun:
    checkpoint = _checkpoint_path(record, seed)
    if checkpoint.exists() and not force:
        measured = joblib.load(checkpoint)
        if not isinstance(measured, MeasuredRun):
            raise TypeError(f"Invalid benchmark checkpoint: {checkpoint}")
        return measured
    X, y = load_locked_frame(record)
    measured = measure_dataset_split(
        record,
        X,
        y,
        outer_seed=seed,
        test_ratio=float(config["test_ratio"]),
        cv_folds=int(config["cv_folds"]),
        cv_repeats=int(config.get("cv_repeats", 1)),
        max_epochs=int(config["max_epochs"]),
        patience=int(config["early_stopping_patience"]),
        device=str(config.get("device", "cpu")),
        n_jobs=int(config.get("n_jobs", 1)),
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(measured, checkpoint)
    return measured


def run_full(*, force: bool = False) -> None:
    config = load_config()
    locked = validate_locked_datasets()
    if len(locked) < 10:
        raise RuntimeError("Published benchmark requires at least 10 pinned datasets")
    measured_runs: list[MeasuredRun] = []
    raw_rows: list[dict[str, Any]] = []
    history_records = []
    for record in locked:
        for seed in config["outer_seeds"]:
            print(f"[BENCHMARK] Measuring {record.benchmark_id}, outer seed {seed}")
            measured = _measure_or_load(record, config, seed=int(seed), force=force)
            measured_runs.append(measured)
            raw_rows.extend(candidate_rows(measured))
            history_records.extend(history_records_for_run(measured))

    history = history_dataframe(history_records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(RESULTS_DIR / "meta_history_v2.csv", index=False)
    method_rows: list[dict[str, Any]] = []
    final_by_run: dict[tuple[str, int, str], tuple[float, float]] = {}
    META_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for measured in measured_runs:
        print(
            f"[BENCHMARK] Replaying rankings {measured.dataset.benchmark_id}, "
            f"outer seed {measured.outer_seed}"
        )
        lodo = leave_one_dataset_out(
            history,
            target_checksum=measured.dataset.local_sha256,
            task_type=(
                "regression" if measured.dataset.task_type == "regression" else "classification"
            ),
            metric_name=measured.metric.name,
        )
        replay_checkpoint = REPLAY_CACHE_DIR / (
            f"{measured.dataset.benchmark_id}_{measured.outer_seed}.joblib"
        )
        if replay_checkpoint.exists() and not force:
            replay_rows, final_scores = joblib.load(replay_checkpoint)
        else:
            replay_rows, final_scores = replay_and_test(
                measured,
                lodo,
                absolute_budgets=config["absolute_budgets"],
                fractional_budgets=config["fractional_budgets"],
                random_seeds=config["random_ordering_seeds"],
                meta_artifact_dir=META_ARTIFACT_DIR,
                device=str(config.get("device", "cpu")),
            )
            replay_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump((replay_rows, final_scores), replay_checkpoint)
        method_rows.extend(replay_rows)
        for candidate_id, scores in final_scores.items():
            final_by_run[(measured.dataset.benchmark_id, measured.outer_seed, candidate_id)] = (
                scores
            )

    for row in raw_rows:
        scores = final_by_run.get((row["dataset_id"], int(row["outer_seed"]), row["candidate_id"]))
        if scores is not None:
            row["test_objective"], row["test_score"] = scores
    raw = pd.DataFrame(raw_rows).sort_values(
        ["dataset_id", "outer_seed", "candidate_id", "inner_fold_id"]
    )
    methods = pd.DataFrame(method_rows).sort_values(
        ["dataset_id", "outer_seed", "method", "ordering_seed", "candidate_budget"],
        na_position="first",
    )
    _write_csv_gzip(raw, RESULTS_DIR / "candidate_results.csv.gz")
    methods.to_csv(RESULTS_DIR / "method_results.csv", index=False, lineterminator="\n")
    environment_hash, environment = environment_fingerprint()
    metadata = {
        "profile": config["profile"],
        "repository_commit": repository_commit(),
        "environment_hash": environment_hash,
        "environment": environment,
        "configuration": config,
        "dataset_count": len(locked),
        "run_count": len(measured_runs),
        "candidate_row_count": len(raw),
        "method_row_count": len(methods),
    }
    (RESULTS_DIR / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from benchmarks.analyze import generate_evidence

    generate_evidence(results_dir=RESULTS_DIR, output_root=BENCHMARK_ROOT)


def run_measure_shard(*, shard_index: int, shard_count: int, force: bool = False) -> None:
    """Populate reusable measurement checkpoints for one reviewed dataset shard."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be within [0, shard_count)")
    config = load_config()
    locked = validate_locked_datasets()
    selected = [record for index, record in enumerate(locked) if index % shard_count == shard_index]
    if not selected:
        raise ValueError(f"Shard {shard_index}/{shard_count} contains no datasets")
    for record in selected:
        for seed in config["outer_seeds"]:
            print(
                f"[BENCHMARK] Shard {shard_index + 1}/{shard_count}: "
                f"measuring {record.benchmark_id}, outer seed {seed}"
            )
            _measure_or_load(record, config, seed=int(seed), force=force)


def _synthetic_record(identifier: str, task_id: int) -> LockedDataset:
    return LockedDataset(
        benchmark_id=identifier,
        openml_task_id=task_id,
        task_type="binary_classification",
        openml_dataset_id=task_id,
        openml_dataset_version=1,
        dataset_name=identifier,
        target_column="target",
        license="Generated test fixture (CC0)",
        license_source="local fixture generator",
        provider_checksum=f"fixture-{identifier}",
        local_sha256=(identifier.encode("utf-8").hex() * 64)[:64],
        cache_file="",
        n_rows=90,
        n_columns=5,
        retrieved_at="fixture",
    )


def _smoke_fixture(seed: int, *, mixed: bool) -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(seed)
    size = 90
    first = generator.normal(size=size)
    second = generator.normal(size=size)
    logits = first + 0.6 * second + generator.normal(scale=0.5, size=size)
    target = pd.Series(np.where(logits > np.quantile(logits, 0.62), "yes", "no"))
    frame = pd.DataFrame({"first": first, "second": second, "third": first * second})
    if mixed:
        frame["group"] = np.where(first > 0, "positive", "negative")
        frame.loc[::11, "group"] = None
    else:
        frame.loc[::13, "second"] = np.nan
    return frame, target


def run_smoke(output_dir: str | Path) -> None:
    """Run a deterministic two-dataset benchmark without network access."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[MeasuredRun] = []
    records = [_synthetic_record("smoke-a", 900001), _synthetic_record("smoke-b", 900002)]
    for index, record in enumerate(records):
        X, y = _smoke_fixture(100 + index, mixed=bool(index))
        runs.append(
            measure_dataset_split(
                record,
                X,
                y,
                outer_seed=42,
                test_ratio=0.2,
                cv_folds=3,
                cv_repeats=1,
                max_epochs=1,
                patience=1,
                n_jobs=1,
            )
        )
    history = history_dataframe(record for run in runs for record in history_records_for_run(run))
    history.to_csv(output_dir / "meta_history_v2.csv", index=False)
    raw_rows = [row for run in runs for row in candidate_rows(run)]
    method_rows = []
    meta_dir = output_dir / "meta_models"
    for run in runs:
        lodo = leave_one_dataset_out(
            history,
            target_checksum=run.dataset.local_sha256,
            task_type="classification",
            metric_name="f1_macro",
            min_training_datasets=1,
        )
        rows, scores = replay_and_test(
            run,
            lodo,
            absolute_budgets=[1, 3],
            fractional_budgets=[1.0],
            random_seeds=[0, 1],
            meta_artifact_dir=meta_dir,
        )
        method_rows.extend(rows)
        for row in raw_rows:
            if row["dataset_id"] == run.dataset.benchmark_id:
                result = scores.get(row["candidate_id"])
                if result is not None:
                    row["test_objective"], row["test_score"] = result
    _write_csv_gzip(pd.DataFrame(raw_rows), output_dir / "candidate_results.csv.gz")
    pd.DataFrame(method_rows).to_csv(output_dir / "method_results.csv", index=False)
    summary = {
        "dataset_count": 2,
        "candidate_rows": len(raw_rows),
        "method_rows": len(method_rows),
        "status": "passed",
    }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def report_check() -> None:
    from benchmarks.analyze import evidence_hashes, generate_evidence

    committed = evidence_hashes(BENCHMARK_ROOT)
    with tempfile.TemporaryDirectory(prefix="metalearnml-report-") as directory:
        temporary = Path(directory)
        (temporary / "results").mkdir(parents=True)
        for filename in ("candidate_results.csv.gz", "method_results.csv", "environment.json"):
            shutil.copy2(RESULTS_DIR / filename, temporary / "results" / filename)
        generate_evidence(results_dir=temporary / "results", output_root=temporary)
        regenerated = evidence_hashes(temporary)
    if committed != regenerated:
        raise RuntimeError(
            f"Deterministic report check failed: committed={committed}, regenerated={regenerated}"
        )
    print("[BENCHMARK] Deterministic report artifacts match committed hashes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="fetch and validate pinned OpenML snapshots")
    fetch.add_argument("--update-lock", action="store_true")
    full = subparsers.add_parser("full", help="run the reviewed balanced-15 benchmark")
    full.add_argument("--force", action="store_true", help="ignore reusable measurement cache")
    measure = subparsers.add_parser(
        "measure", help="measure one dataset shard for the manual benchmark workflow"
    )
    measure.add_argument("--shard-index", type=int, required=True)
    measure.add_argument("--shard-count", type=int, required=True)
    measure.add_argument("--force", action="store_true")
    smoke = subparsers.add_parser("smoke", help="run the offline two-dataset smoke profile")
    smoke.add_argument(
        "--output-dir",
        default=str(BENCHMARK_ROOT / "smoke-output"),
    )
    subparsers.add_parser("report-check", help="regenerate reports and compare hashes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fetch":
        records = fetch_and_lock(update_lock=args.update_lock)
        print(f"[BENCHMARK] Validated {len(records)} pinned OpenML datasets")
    elif args.command == "full":
        run_full(force=args.force)
    elif args.command == "measure":
        run_measure_shard(
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            force=args.force,
        )
    elif args.command == "smoke":
        run_smoke(args.output_dir)
        print(f"[BENCHMARK] Smoke artifacts written to {args.output_dir}")
    elif args.command == "report-check":
        report_check()


if __name__ == "__main__":
    main()
