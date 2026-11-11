"""Deterministic analysis and evidence generation for benchmark result rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARTIFACT_PATHS = (
    "results/per_dataset_summary.csv",
    "results/aggregate_summary.json",
    "reports/benchmark.md",
    "figures/best_so_far.svg",
    "figures/evaluations_to_95.svg",
    "figures/test_quality_delta.svg",
    "figures/ranking_quality.svg",
    "figures/runtime.svg",
)

METHOD_LABELS = {
    "exhaustive": "Exhaustive",
    "proxy": "Proxy only",
    "meta": "Meta only",
    "proxy_meta": "Proxy + meta",
    "random": "Random",
}
METHOD_COLORS = {
    "exhaustive": "#4b5563",
    "proxy": "#d97706",
    "meta": "#2563eb",
    "proxy_meta": "#059669",
    "random": "#9333ea",
}


def _quantile(values: pd.Series, quantile: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(quantile)) if not clean.empty else None


def _at_budget(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    return frame[frame["candidate_budget"] == budget].copy()


def _collapse_random(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset_id",
        "task_type",
        "outer_seed",
        "method",
        "candidate_budget",
        "candidate_count",
    ]
    numeric = [
        "normalized_regret",
        "quality_retention",
        "test_score",
        "final_test_normalized_regret",
        "spearman",
        "recall_at_5",
        "ndcg_at_5",
        "ranking_overhead_seconds",
        "cached_evaluation_seconds",
        "cached_total_seconds",
        "candidate_failure_count",
    ]
    return frame.groupby(group_columns, dropna=False, as_index=False)[numeric].median()


def build_per_dataset_summary(methods: pd.DataFrame) -> pd.DataFrame:
    collapsed = _collapse_random(methods)
    group_columns = ["dataset_id", "task_type", "method", "candidate_budget"]
    summary = (
        collapsed.groupby(group_columns, as_index=False)
        .agg(
            candidate_count=("candidate_count", "first"),
            outer_seed_count=("outer_seed", "nunique"),
            median_normalized_regret=("normalized_regret", "median"),
            median_quality_retention=("quality_retention", "median"),
            median_test_score=("test_score", "median"),
            median_final_test_normalized_regret=(
                "final_test_normalized_regret",
                "median",
            ),
            median_spearman=("spearman", "median"),
            median_recall_at_5=("recall_at_5", "median"),
            median_ndcg_at_5=("ndcg_at_5", "median"),
            median_cached_total_seconds=("cached_total_seconds", "median"),
            failure_count=("candidate_failure_count", "max"),
        )
        .sort_values(group_columns)
    )
    return summary


def _evaluations_to_95(methods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    collapsed = _collapse_random(methods)
    for keys, group in collapsed.groupby(["dataset_id", "outer_seed", "method"], dropna=False):
        reached = group[group["normalized_regret"] <= 0.05]
        rows.append(
            {
                "dataset_id": keys[0],
                "outer_seed": keys[1],
                "method": keys[2],
                "evaluations_to_95": (
                    int(reached["candidate_budget"].min())
                    if not reached.empty
                    else int(group["candidate_count"].max()) + 1
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_interval(values: pd.Series, *, replicates: int, seed: int) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(clean):
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(clean), size=(replicates, len(clean)))
    estimates = np.median(clean[indices], axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def build_aggregate_summary(
    methods: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    collapsed = _collapse_random(methods)
    method_summary: dict[str, dict[str, Any]] = {}
    for method, group in collapsed.groupby("method"):
        method_summary[str(method)] = {
            "rows": int(len(group)),
            "median_normalized_regret": _quantile(group["normalized_regret"], 0.5),
            "normalized_regret_iqr": [
                _quantile(group["normalized_regret"], 0.25),
                _quantile(group["normalized_regret"], 0.75),
            ],
            "median_final_test_normalized_regret": _quantile(
                group["final_test_normalized_regret"], 0.5
            ),
            "median_recall_at_5": _quantile(group["recall_at_5"], 0.5),
            "median_ndcg_at_5": _quantile(group["ndcg_at_5"], 0.5),
            "median_spearman": _quantile(group["spearman"], 0.5),
            "candidate_failures": int(group["candidate_failure_count"].max()),
        }

    evaluations = _evaluations_to_95(methods)
    per_dataset_evaluations = (
        evaluations.groupby(["dataset_id", "method"], as_index=False)["evaluations_to_95"]
        .median()
        .pivot(index="dataset_id", columns="method", values="evaluations_to_95")
    )
    paired = per_dataset_evaluations.dropna(subset=["proxy_meta", "random"]).copy()
    paired["evaluation_reduction_fraction"] = (paired["random"] - paired["proxy_meta"]) / paired[
        "random"
    ]
    ci_low, ci_high = _bootstrap_interval(
        paired["evaluation_reduction_fraction"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    budget_10 = _at_budget(collapsed, 10)
    proxy_meta_10 = budget_10[budget_10["method"] == "proxy_meta"]
    median_retention = _quantile(proxy_meta_10["quality_retention"], 0.5)
    median_test_regret = _quantile(proxy_meta_10["final_test_normalized_regret"], 0.5)
    impact_eligible = bool(
        np.isfinite(ci_low)
        and ci_low > 0
        and median_retention is not None
        and median_retention >= 0.95
        and median_test_regret is not None
        and median_test_regret <= 0.05
    )
    failures = int(
        methods[["dataset_id", "outer_seed", "candidate_failure_count"]]
        .drop_duplicates()["candidate_failure_count"]
        .sum()
    )
    return {
        "schema_version": 1,
        "profile": "balanced-15-v1",
        "dataset_count": int(methods["dataset_id"].nunique()),
        "outer_seeds": sorted(int(seed) for seed in methods["outer_seed"].unique()),
        "candidate_count_range": [
            int(methods["candidate_count"].min()),
            int(methods["candidate_count"].max()),
        ],
        "methods": method_summary,
        "failures": failures,
        "primary_paired_comparison": {
            "comparison": "proxy_meta_vs_random_evaluations_to_regret_le_0.05",
            "dataset_median_reduction_fraction": _quantile(
                paired["evaluation_reduction_fraction"], 0.5
            ),
            "bootstrap_95_ci": [ci_low, ci_high],
            "bootstrap_unit": "dataset",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
        "proxy_meta_budget_10": {
            "median_quality_retention": median_retention,
            "median_final_test_normalized_regret": median_test_regret,
        },
        "speedup_claim_eligible": impact_eligible,
    }


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 4.8),
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "svg.hashsalt": "metalearnml-evidence-v2",
        }
    )


def _save_svg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="svg", metadata={"Date": None})
    plt.close()


def plot_best_so_far(methods: pd.DataFrame, path: Path) -> None:
    collapsed = _collapse_random(methods)
    for method in ("random", "proxy", "meta", "proxy_meta"):
        group = collapsed[collapsed["method"] == method]
        stats = group.groupby("candidate_budget")["quality_retention"].agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        x = stats.index.to_numpy(dtype=float)
        plt.plot(
            x, stats["median"], marker="o", label=METHOD_LABELS[method], color=METHOD_COLORS[method]
        )
        plt.fill_between(x, stats["q1"], stats["q3"], alpha=0.15, color=METHOD_COLORS[method])
    plt.xlabel("Candidates evaluated")
    plt.ylabel("Normalized quality retention")
    plt.ylim(0, 1.03)
    plt.legend()
    _save_svg(path)


def plot_evaluations_to_95(methods: pd.DataFrame, path: Path) -> None:
    evaluations = _evaluations_to_95(methods)
    order = ["random", "proxy", "meta", "proxy_meta"]
    data = [evaluations[evaluations["method"] == method]["evaluations_to_95"] for method in order]
    boxes = plt.boxplot(
        data, tick_labels=[METHOD_LABELS[item] for item in order], patch_artist=True
    )
    for patch, method in zip(boxes["boxes"], order, strict=True):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.6)
    plt.ylabel("Evaluations to normalized regret ≤ 0.05")
    _save_svg(path)


def plot_test_quality_delta(methods: pd.DataFrame, path: Path) -> None:
    collapsed = _collapse_random(methods)
    proxy_meta = collapsed[
        (collapsed["method"] == "proxy_meta") & (collapsed["candidate_budget"] == 10)
    ]
    values = proxy_meta.groupby("dataset_id")["final_test_normalized_regret"].median().sort_values()
    colors = [METHOD_COLORS["proxy_meta"] if value <= 0.05 else "#dc2626" for value in values]
    plt.barh(values.index, values.to_numpy(), color=colors)
    plt.axvline(0.05, color="#111827", linestyle="--", linewidth=1)
    plt.xlabel("Final-test normalized regret vs exhaustive selection")
    _save_svg(path)


def plot_ranking_quality(methods: pd.DataFrame, path: Path) -> None:
    collapsed = _collapse_random(methods)
    data = []
    labels = []
    for method in ("meta", "proxy_meta"):
        group = collapsed[collapsed["method"] == method]
        data.extend([group["spearman"].dropna(), group["recall_at_5"].dropna()])
        labels.extend([f"{METHOD_LABELS[method]}\nSpearman", f"{METHOD_LABELS[method]}\nRecall@5"])
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Ranking score")
    plt.ylim(-1.05, 1.05)
    _save_svg(path)


def plot_runtime(methods: pd.DataFrame, path: Path) -> None:
    collapsed = _collapse_random(methods)
    budget = 10
    group = collapsed[collapsed["candidate_budget"] == budget]
    order = ["random", "proxy", "meta", "proxy_meta"]
    data = [group[group["method"] == method]["cached_total_seconds"].dropna() for method in order]
    plt.boxplot(data, tick_labels=[METHOD_LABELS[item] for item in order], showfliers=False)
    plt.ylabel(f"Cached replay-equivalent seconds at budget {budget}")
    plt.title("CPU measurements; ranking overhead included")
    _save_svg(path)


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.1f}%"


def write_report(
    summary: dict[str, Any],
    per_dataset: pd.DataFrame,
    environment_metadata: dict[str, Any],
    path: Path,
) -> None:
    primary = summary["primary_paired_comparison"]
    budget = summary["proxy_meta_budget_10"]
    ci = primary["bootstrap_95_ci"]
    runtime = environment_metadata["environment"]
    config = environment_metadata["configuration"]
    exhaustive = per_dataset[per_dataset["method"] == "exhaustive"]
    exhaustive = exhaustive.loc[
        exhaustive.groupby("dataset_id")["candidate_budget"].idxmax(),
        ["dataset_id", "task_type", "median_test_score"],
    ].rename(columns={"median_test_score": "exhaustive_test"})
    proxy_meta = per_dataset[
        (per_dataset["method"] == "proxy_meta") & (per_dataset["candidate_budget"] == 10)
    ][["dataset_id", "median_test_score", "median_final_test_normalized_regret"]].rename(
        columns={"median_test_score": "proxy_meta_test"}
    )
    dataset_comparison = exhaustive.merge(proxy_meta, on="dataset_id", how="left").sort_values(
        "dataset_id"
    )
    dataset_rows = [
        "| Dataset | Task / natural metric | Exhaustive test | Proxy+meta test @10 | Test regret |",
        "|---|---|---:|---:|---:|",
    ]
    for row in dataset_comparison.itertuples(index=False):
        metric = "R²" if row.task_type == "regression" else "macro F1"
        dataset_rows.append(
            f"| {row.dataset_id} | {row.task_type} / {metric} | "
            f"{row.exhaustive_test:.6f} | {row.proxy_meta_test:.6f} | "
            f"{row.median_final_test_normalized_regret:.6f} |"
        )
    method_rows = [
        "| Method | Median regret | Median Spearman | Median Recall@5 | Median NDCG@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("random", "proxy", "meta", "proxy_meta"):
        values = summary["methods"][method]
        method_rows.append(
            f"| {METHOD_LABELS[method]} | {values['median_normalized_regret']:.6f} | "
            f"{values['median_spearman']:.6f} | {values['median_recall_at_5']:.6f} | "
            f"{values['median_ndcg_at_5']:.6f} |"
        )
    report = f"""# MetaLearnML balanced-15 benchmark

## Protocol

This report uses the pinned `balanced-15-v1` OpenML task suite: {summary["dataset_count"]} tabular classification/regression datasets, outer seeds {summary["outer_seeds"]}, a 20% untouched test split, and five development-only cross-validation folds. The candidate universe contains {summary["candidate_count_range"][0]}–{summary["candidate_count_range"][1]} pipelines. Target-dataset candidate outcomes are excluded from that dataset's meta-training history.

Exhaustive search is only the reference inside this fixed candidate universe. Proxy-only, meta-only, proxy+meta, and 20 seeded random orderings replay the same measured candidates, folds, objectives, and failures. Natural test metrics are macro F1 for classification and R² for regression; cross-task aggregation uses normalized regret.

## Primary result

- Median evaluation reduction for proxy+meta versus random to reach regret ≤ 0.05: **{_format_percent(primary["dataset_median_reduction_fraction"])}** (dataset-bootstrap 95% CI **{_format_percent(ci[0])} to {_format_percent(ci[1])}**, {primary["bootstrap_replicates"]:,} resamples).
- Proxy+meta median quality retention at budget 10: **{_format_percent(budget["median_quality_retention"])}**.
- Proxy+meta median final-test normalized regret at budget 10: **{budget["median_final_test_normalized_regret"] if budget["median_final_test_normalized_regret"] is not None else "n/a"}**.
- Failed candidate runs retained in denominators: **{summary["failures"]}**.

The speedup claim criterion is **{"met" if summary["speedup_claim_eligible"] else "not met"}**. The criterion requires a wholly positive bootstrap interval for evaluation reduction, at least 95% median quality retention at budget 10, and no more than 0.05 median final-test normalized regret.

## Per-dataset untouched-test results

Values are medians over outer seeds. Classification uses macro F1 and regression uses R²;
the natural metrics are deliberately not averaged across task types.

{chr(10).join(dataset_rows)}

## Ranking quality

{chr(10).join(method_rows)}

The full budget-by-budget values are committed in `results/per_dataset_summary.csv`.

## Runtime, environment, and failures

- Execution environment: Python {runtime["python"]}, {runtime["platform"]},
  CPU `{runtime.get("cpu", "unknown")}`, accelerator `{runtime.get("accelerator", "cpu")}`,
  outer candidate parallelism `n_jobs={config["n_jobs"]}`.
- Candidate fold fit and score times plus ranking overhead are stored in the raw and method
  result files. The runtime figure shows replay-equivalent elapsed time at budget 10.
- This run did not collect a consistently isolated peak-memory measurement; no memory
  comparison is claimed.
- {summary["failures"]} failed candidate runs remain in denominators. No dataset was
  excluded after the reviewed manifest was validated, and every dataset had at least one
  successful candidate.
- Repository revision recorded by the run: `{environment_metadata["repository_commit"]}`.
- Environment fingerprint: `{environment_metadata["environment_hash"]}`.

## Reproduction

```bash
python -m pip install -r benchmarks/environment/requirements-lock.txt
make benchmark-fetch
make benchmark-full
make benchmark-report-check
```

The dataset lock records OpenML task/dataset versions, targets, licenses, provider checksums, and local snapshot SHA-256 values. Every ranking method reuses identical measured candidate folds for the quality comparison. A separate cold end-to-end timing trial was not completed for this published run, so the report does not make a cold-runtime speed claim.

## Limitations

The conclusions apply to this named 15-dataset suite, three outer splits, and the repository's 28–36-pipeline search space. Exhaustive search is not a global optimum outside that universe. Text columns are unsupported and rejected. Meta-ranking also has a cold-start limitation when fewer than 20 compatible history rows exist. Runtime is hardware-specific, and the absence of isolated peak-memory and separate cold-timing measurements limits system-efficiency conclusions.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def generate_evidence(*, results_dir: str | Path, output_root: str | Path) -> None:
    results_dir = Path(results_dir)
    output_root = Path(output_root)
    methods = pd.read_csv(results_dir / "method_results.csv")
    environment = json.loads((results_dir / "environment.json").read_text(encoding="utf-8"))
    config = environment["configuration"]
    per_dataset = build_per_dataset_summary(methods)
    summary = build_aggregate_summary(
        methods,
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    results_output = output_root / "results"
    results_output.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(results_output / "per_dataset_summary.csv", index=False, lineterminator="\n")
    (results_output / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _configure_plots()
    figures = output_root / "figures"
    plot_best_so_far(methods, figures / "best_so_far.svg")
    plot_evaluations_to_95(methods, figures / "evaluations_to_95.svg")
    plot_test_quality_delta(methods, figures / "test_quality_delta.svg")
    plot_ranking_quality(methods, figures / "ranking_quality.svg")
    plot_runtime(methods, figures / "runtime.svg")
    write_report(
        summary,
        per_dataset,
        environment,
        output_root / "reports" / "benchmark.md",
    )


def evidence_hashes(root: str | Path) -> dict[str, str]:
    root = Path(root)
    hashes = {}
    for relative in ARTIFACT_PATHS:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Evidence artifact is missing: {path}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes
