# MetaLearnML balanced-15 benchmark

## Protocol

This report uses the pinned `balanced-15-v1` OpenML task suite: 15 tabular classification/regression datasets, outer seeds [13, 42, 97], a 20% untouched test split, and five development-only cross-validation folds. The candidate universe contains 28–36 pipelines. Target-dataset candidate outcomes are excluded from that dataset's meta-training history.

Exhaustive search is only the reference inside this fixed candidate universe. Proxy-only, meta-only, proxy+meta, and 20 seeded random orderings replay the same measured candidates, folds, objectives, and failures. Natural test metrics are macro F1 for classification and R² for regression; cross-task aggregation uses normalized regret.

## Primary result

- Median evaluation reduction for proxy+meta versus random to reach regret ≤ 0.05: **0.0%** (dataset-bootstrap 95% CI **0.0% to 50.0%**, 10,000 resamples).
- Proxy+meta median quality retention at budget 10: **100.0%**.
- Proxy+meta median final-test normalized regret at budget 10: **0.0**.
- Failed candidate runs retained in denominators: **18**.

The resume impact criterion is **not met**. The criterion requires a wholly positive bootstrap interval for evaluation reduction, at least 95% median quality retention at budget 10, and no more than 0.05 median final-test normalized regret.

## Per-dataset untouched-test results

Values are medians over outer seeds. Classification uses macro F1 and regression uses R²;
the natural metrics are deliberately not averaged across task types.

| Dataset | Task / natural metric | Exhaustive test | Proxy+meta test @10 | Test regret |
|---|---|---:|---:|---:|
| abalone | regression / R² | 0.558041 | 0.538687 | 0.000622 |
| adult | classification / macro F1 | 0.802918 | 0.802918 | 0.000000 |
| bioresponse | classification / macro F1 | 0.780473 | 0.780473 | 0.000000 |
| colleges | regression / R² | 0.561996 | 0.542652 | 0.000040 |
| credit-g | classification / macro F1 | 0.669566 | 0.669566 | 0.000000 |
| dresses-sales | classification / macro F1 | 0.540441 | 0.552353 | 0.000000 |
| elevators | regression / R² | 0.813975 | 0.813975 | 0.000000 |
| eucalyptus | classification / macro F1 | 0.658518 | 0.661309 | 0.000000 |
| house-prices-nominal | regression / R² | 0.846918 | 0.846918 | 0.000000 |
| iris | classification / macro F1 | 1.000000 | 1.000000 | 0.000000 |
| micro-mass | classification / macro F1 | 0.915412 | 0.796795 | 0.154206 |
| phoneme | classification / macro F1 | 0.866004 | 0.866004 | 0.000000 |
| quake | regression / R² | -0.007005 | -0.007056 | 0.000000 |
| sensory | regression / R² | 0.164982 | 0.164982 | 0.000000 |
| wine-quality | regression / R² | 0.458106 | 0.458106 | 0.000000 |

## Ranking quality

| Method | Median regret | Median Spearman | Median Recall@5 | Median NDCG@5 |
|---|---:|---:|---:|---:|
| Random | 0.001023 | 0.021622 | 0.200000 | 0.044561 |
| Proxy only | 0.000004 | 0.028847 | 0.200000 | 0.082329 |
| Meta only | 0.000000 | 0.567568 | 0.400000 | 0.184970 |
| Proxy + meta | 0.000000 | 0.406976 | 0.400000 | 0.191430 |

The full budget-by-budget values are committed in `results/per_dataset_summary.csv`.

## Runtime, environment, and failures

- Execution environment: Python 3.11.5, macOS-26.5.2-arm64-arm-64bit,
  CPU `Apple M3 Pro`, accelerator `cpu`,
  outer candidate parallelism `n_jobs=4`.
- Candidate fold fit and score times plus ranking overhead are stored in the raw and method
  result files. The runtime figure shows replay-equivalent elapsed time at budget 10.
- This run did not collect a consistently isolated peak-memory measurement; no memory
  comparison is claimed.
- 18 failed candidate runs remain in denominators. No dataset was
  excluded after the reviewed manifest was validated, and every dataset had at least one
  successful candidate.
- Repository revision recorded by the run: `2ffdd12d7134a480152dfaa328d5fafabf8fbc01-dirty`.
- Environment fingerprint: `b2be90084aa0bc915abf5fd8da4814001bbabc986ad42237b607a483d9660618`.

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
