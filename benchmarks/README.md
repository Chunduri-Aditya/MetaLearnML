# MetaLearnML benchmark

The published `balanced-15-v1` profile compares exhaustive, proxy-only, meta-only,
proxy+meta, and 20 deterministic random candidate orderings over identical measured
candidate universes. OpenML task IDs pin both datasets and targets. The reviewed lock
adds the dataset version, license, provider checksum, and local snapshot SHA-256.

The protocol uses outer seeds 13, 42, and 97, a 20% untouched test partition, and five
development-only folds. Classification selects macro F1; regression selects R². Each
target dataset is excluded entirely from its meta-training history.

```bash
python -m pip install -r benchmarks/environment/requirements-lock.txt
make benchmark-fetch
make benchmark-smoke
make benchmark-full
make benchmark-report-check
```

`benchmark-fetch` validates the committed lock and downloads data only into the ignored
`benchmarks/cache/` directory. Updating the lock is an explicit maintainer operation:

```bash
python -m benchmarks.run fetch --update-lock
```

The smoke profile creates two deterministic local classification fixtures and does not
need OpenML or Neo4j. Published result rows, summaries, reports, and SVG figures live in
`benchmarks/results/`, `benchmarks/reports/`, and `benchmarks/figures/`.

Text features are unsupported in v2. Manifest validation rejects a dataset if text is
inferred, provenance changes, the target is missing, or licensing is ambiguous.
