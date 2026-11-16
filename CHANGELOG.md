# Changelog

All notable changes to this project are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/) (`0.x` while the API stabilises).

## [Unreleased]

### Added
- Focused `FidelityReport`, `MemorizationReport` and `UtilityReport` views share stored results and metadata.
- Interactive Plotly score, column, pair, slice, Concentration and MLE plots, plus bounded input-data distribution views and offline HTML export.
- Curated Sphinx documentation, executable offline report/visualization examples and detailed metric guides.
- DCR generated and independent holdout distance percentiles q05, q10, q25, q50, q75, q90, q95 and q99, retaining the generated median alias.
- Replication ratio counts training records reached by generated output; near-match rate uses generated-row denominators. Both use fixed inclusive Euclidean distance `1e-3` in training-fitted geometry.
- MLE exposes raw AUROC for classification and MAE for regression, with generated-to-real raw-score ratios, classification prediction agreement and regression prediction MSE through unified `evaluate`. Classification scores and ratios are higher-is-better; regression scores and ratios are lower-is-better. All outputs reuse shared model fits and scoring rows; real-only evaluation returns raw AUROC/MAE.
- Shared training-only CPU/CUDA preprocessing, native categorical predictors, precision provenance, nonfinite-transform diagnostics and storage-lifetime safeguards.

### Changed
- Nonlinear C2ST and MLE automatically select GPU execution when CUDA-enabled XGBoost, CuPy, and estimated workspace are available, otherwise CPU; reports record the routing reason and explicit CUDA remains strict.
- Low-level `XGBoostModel` accepts explicit devices only and defaults to CPU; metric execution owns automatic device selection before preprocessing.
- CUDA runtime selection also accepts the matched Torch 2.6/CUDA 11.8/Triton 3.2 stack; CUDA extras continue to target Torch 2.11/CUDA 13.0/Triton 3.6.
- Rename the distribution and CLI to `tabgen-eval`, the Python package to `tabgen_eval`, and documentation branding to **TabGenEval**. Native operator namespaces and deployment environment variables use the same name (`TABGEN_EVAL_*`); the previous names are removed.
- Consolidate repeated-score collection under `uncertainty`, share paired and test-only preparation, and keep report construction outside statistical execution.
- Tighten shared type contracts and add configured type checking.
- Refresh module and API documentation, including MLE score/ratio interpretation and metric-specific missing-data behavior.
- Remove code with no package, test, or tooling consumers while retaining useful numerical reference helpers.
- Evaluation options are direct keyword arguments. `uncertainty=None` computes fixed scores; an ordinary dictionary with required `mode="reference"` or `"both"` and optional `n_replicates` requests repetition. Reports expose `options_dict` and export resolved settings under `options`.
- Linear C2ST's direct-solver iteration limit and SAGA epoch limit default to 10,000, with convergence-based early stopping.
- Statistical `alpha` metadata and distribution/comparison arguments are now `confidence_interval`, storing central coverage (`0.95`) instead of excluded tail mass (`0.05`); public bands remain fixed at 95%.
- Public empirical bands are fixed at 95%; evaluation configuration and CLI no longer expose an alpha option. Raw repeated outcomes, unavailable counts and reasons remain retained.
- Concentration uses smoothed nonnegative candidate/reference ratios with fixed `epsilon=1e-8` and value-based fixed-radius keys. Fixed radius `1e-3` reports only the mean.
- MLE uses matched training draws and common disjoint real test rows. Reference mode fixes the generated model and rescores both models on fresh real test rows; both mode also redraws and refits the generated side. Ratios divide raw generated-trained scores by raw real-trained scores without chance adjustment or clipping. Positive scores over zero remain infinite and zero over zero remains undefined.
- Exact memorization searches retain explicit source caps and bounded temporary workspace without ANN dependencies or automatic work-count refusal.
- Traceability checks complete catalogue coverage and real top-level test definitions; installed-artifact checks include retained preprocessing contracts.

### Removed
- Optional local MCP server, its dependency extra and command-line entry point.
- Public `read_table` loader; Python callers load and prepare pandas DataFrames directly. CLI file loading remains internal.
- `EvalConfig` and the `config` evaluator argument. `UncertaintyOptions` is an optional TypedDict annotation for the ordinary uncertainty dictionary.
- Automatic fixed-score and distribution comparison rules/labels, the scalar `comparison` field, comparison export columns and interpretation diagrams. Raw scores, reference scores and uncertainty bands remain available.
- `generated_tail_allowance` configuration and CLI flag, together with comparison-only statistics helpers. Generated and reference distributions retain the same `confidence_interval` coverage without automatic decision rules.
- `sample_budget` configuration, CLI flag and calibration argument; evaluation and calibration always derive pool sizes automatically from their full inputs.
- `evaluate_utility` and the separate memorization entry point; both supervised and memorization metrics use unified `evaluate` with explicit input roles and metric selectors.
- Separate predictive-retention, global-utility and standalone prediction-agreement entry points/metric selectors; their supported shared-fit behavior is consolidated under MLE.
- Configurable near-match thresholds, approximate memorization options and retired training-coverage naming.

### Fixed
- CUDA fidelity kernels support Triton 3.2 reductions, histograms, and quantile edge selection without narrowing FP64 inputs.
- Native CUDA builds prefer Torch's matching NVIDIA libraries over incompatible system toolkits, and Cholesky operators reject nonfinite factor diagonals even when cuSOLVER reports success.
- CUDA Conditional Fidelity centering preserves FP32 subnormal values on Triton 3.2 instead of producing NaNs from flush-to-zero division.
- Paired uncertainty over both generated and holdout pools uses half the eligible holdout when the generated pool is larger than that size, and half the eligible generated pool otherwise, including equality. Fixed points and repetitions share the selected resolution; no-UQ and reference-only resolution rules are unchanged.
- Conditional Fidelity uses tensor division for equal-width bin edges, preserving CPU/CUDA boundary membership and rarity classification for tied values.
- CUDA self-distance launchers include runtime global row offsets across integer widths, allowing cold-cache installed execution without a C compiler.
- Missing-Torch installation checks distinguish supported numerical fallback from classifiers that require shared Torch preprocessing.

## [0.1.0] — 2026-06-28

First v1: sample-based metrics, end-to-end, with the value / CI / real-real ideal / flags contract.

### Added
- `evaluate(holdout, generated, schema, ...) -> Report` — the one-call entry point. `generated` may differ in row count from `holdout`.
- Two-sided **with-replacement bootstrap** on both inputs; the real–real ideal value at matched sizes — by default two **disjoint holdout halves** (`split_holdout_ideal`), so the reference and candidate are each evaluated at ~`n_holdout/2` per side and stay directly comparable; percentile CIs.
- Reliability-flag system (`INSUFFICIENT_DATA`, `UNRELIABLE_CI`, `SIZE_DISPARITY`, `IDEAL_OPTIMISTIC_DUP`, …) — never a number we don't trust.
- Metrics: **Shape**, **Trend** (fidelity), **Conditional** (subpopulation Shape, Trend & Recovery, all & rare slices), **PRDC** (diversity), **Sanity** (always-on).
- Statistical primitives implemented in core (KS complement, TV complement, correlation & contingency similarity) — no extra dependencies beyond numpy/scipy/scikit-learn.
- Numeric columns in Trend / Conditional Trend contingency tables are discretised into **quantile** (equal-frequency) bins anchored to the holdout, so a skewed column is not collapsed into one or two occupied bins. Conditional's numeric *conditioning slices* stay equal-width so sparse regions still surface as rare slices.
- Raw-data cleaning (`clean="safe"/"off"/"strict"`): coercion, `±inf`, category normalisation — explicit and logged.
- `Report`: `to_dataframe()`, `to_json()`, `to_markdown()`, `summary()`, and `dump_replicates()` (full per-replicate arrays when `store_replicates=True`).
- `read_table()` CSV/Parquet loader + `tabgen-eval` CLI mirroring evaluation options.
- `load_openml(name_or_id)` — fetch a real OpenML table and return a deterministic seeded `(train, holdout)` split (via scikit-learn's `fetch_openml`; no extra dependency).
- Test suite: unit, property (Hypothesis), metamorphic, calibration, fuzz.
