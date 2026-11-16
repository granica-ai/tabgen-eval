# Troubleshooting

Use this page to diagnose rejected inputs, missing results and execution problems.
For interpretation of successful results, see [The Sharp Bits](sharp-bits.md).

(cpu-build-prerequisites)=
## My first CPU evaluation fails to compile

Without a compatible precompiled library, the first CPU evaluation builds native
operators on Linux or Apple Silicon macOS. Follow the
{ref}`Linux <linux-cpu-installation>` or
{ref}`macOS installation instructions <macos-cpu-installation>` to prepare the
compiler and OpenMP. CPU compilation does not require a CUDA toolkit or GPU.

The exception identifies the failed step and its log path. Inspect the reported
`check-*.log` or `build.log`:

| Failed step | What to check |
|---|---|
| `check-cxx17` / `check-cxx20` | Install a compiler and standard-library headers supporting the standard selected for Torch; set `CXX` to the correct compiler if necessary. |
| `check-torch-headers` | Read the log for incompatibilities with the installed Torch headers. Update the compiler or Apple's Command Line Tools if needed. |
| `check-openmp` | Install OpenMP headers and runtime libraries for the selected compiler, following the installation instructions above. |
| `check-*-runtime` | Ensure the compiled program can find its runtime libraries and the cache filesystem permits execution. |
| Cache creation | Set `TABGEN_EVAL_CPU_CACHE` to a writable directory with enough space. |
| `build` | Read `build.log` for Torch-header/compiler compatibility errors. If the compiler is killed, check memory; for a timeout, check host load and compiler wrappers. |

Fix the reported problem and repeat the evaluation. Failed builds are retried.

(macos-cpu-build-troubleshooting)=
## macOS compiler, OpenMP and library-loading issues

On Apple Silicon, Python and Torch need to run natively as `arm64`. This command
checks the Python process:

```bash
python -c 'import platform; print(platform.machine())'
```

It must report `arm64`. An `x86_64` process requires reinstalling native arm64
Python and its dependencies; see the
{ref}`macOS installation instructions <macos-cpu-installation>`.

The required C++ standard is selected automatically. If `check-cxx20` or
`check-torch-headers` fails, inspect its log and update Apple's Command Line
Tools if needed. Set `CXX=/usr/bin/clang++` to select Apple Clang.

OpenMP is found in Torch or Homebrew. For a custom installation, set its prefix:

```bash
export TABGEN_EVAL_OPENMP_ROOT=/path/to/libomp
```

That prefix must contain `include/omp.h` and, when Torch does not bundle the
runtime, `lib/libomp.dylib`.

(cpu-build-cache)=
## CPU compilation is slow or repeats

Once compiled, the CPU operators are reused across Python processes without
repeating the build checks. Updating Torch, changing the source or selecting
another compiler with `CXX` can trigger a new build. The default cache is beneath
Torch's extension-cache directory; `TABGEN_EVAL_CPU_CACHE` selects another
location. `TORCH_EXTENSIONS_DIR` has no effect on this builder.

```bash
export TABGEN_EVAL_CPU_CACHE=/path/to/writable-executable-cache
```

The cache filesystem must permit writing, file locking and execution. A full
build has a 600-second timeout; check memory and host load if a compiler is
killed or times out.

(an-unfiltered-evaluation-requires-a-target-or-xgboost)=
## My evaluation unexpectedly requires a target

With no metric filters, `evaluate(data)` selects every metric supported by your
data roles. When `test` is supplied, that includes MLE, which requires
`Schema.target`.

Declare the prediction target, or select the metrics you intend to evaluate:

```python
report = evaluate(data, metrics=["shape", "trend"])
```

See the [metric selection guide](reference/evaluation.md#choose-metrics) for
selection rules.

(my-numeric-column-is-rejected)=
## My input is rejected

Check the exception against these requirements:

- Supply pandas DataFrames and an explicit `Schema`.
- Input tables must have the same complete set of uniquely named columns.
- Declared numeric columns must already have numeric storage and finite observed
  values. Missing values follow each metric's rules.
- Convert numeric text and choose units for temporal measurements before evaluation.
- Supply the data roles required by your selected metrics.

Schema declarations assign roles; they do not convert values. See
[data preparation](reference/data.md) for the input contract.

(my-score-is-undefined)=
## My score is undefined or a point is missing

The flags and point status help explain why a result is unavailable:

```python
stat = report["shape"]
print(stat.value, stat.flag_codes)
print(stat.point_status, stat.ideal_status)
```

An undefined computed score can indicate insufficient usable data, an
inapplicable statistic or an undefined ratio. Check the metric's definition and
diagnostics.

A point marked `"absent"` was intentionally not computed. Fixed test-only
evaluation stores one reference score in `.value` and marks `.ideal` absent.
Repeated test-only evaluation reports the finite mean in `.value`, leaves
`.ideal` absent, and retains the scores in `reference_distribution`. The same convention applies to MLE's real-trained
scores.

Distance metrics and MLE also report how many rows were excluded:

```python
print(report.row_usage())
```

Retained rows do not guarantee that every statistic is computable. MLE
classification scores, for example, also depend on the classes present in the
sampled training and scoring rows.

## My report has no bands

The default `uncertainty=None` requests fixed scores.

For ordinary paired metrics, the requested bands depend on the mode:

| Setting | Requested bands |
| --- | --- |
| `None` | None |
| `{"mode": "reference"}` | Real-reference band |
| `{"mode": "candidate"}` | Generated-real band |
| `{"mode": "both"}` | Both bands |

Memorization metrics remain point-only. MLE follows its
[own repetition protocol](metrics/utility/mle.md#repeated-evaluation); its
generated scores can vary in reference mode because scoring rows change.

Repeated test-only evaluation supports `"reference"` and `"both"` modes. It
reports the mean of the finite scores in `.value` and retains the individual
scores in `reference_distribution`. The reference band is available in
`.ideal_ci_low` and `.ideal_ci_high`. `"candidate"` mode requires generated data.

Each distribution records its finite and unavailable score counts, along with
any discarded reasons:

```python
distribution = report["shape"].reference_distribution
if distribution is not None:
    print(distribution.finite_count, distribution.unavailable_count)
    print(distribution.discarded_reasons)
```

Use `generated_distribution` to inspect generated-real scores. An undefined or
infinite score is retained in the samples and excluded from finite summaries.
Other valid statistics remain usable. The metric diagnostics retain available
failure reasons.

Read the reported interval from `ci_low`/`ci_high` or
`ideal_ci_low`/`ideal_ci_high`. These fields contain the same raw 2.5th and 97.5th
percentiles as the corresponding distribution's `.band_low` and `.band_high`.
They describe score variability across sampled comparisons and are not
confidence intervals for the population expected score.

Equal band endpoints indicate a zero-width interval. This can occur with a
single finite score or identical finite scores and is recorded
by `DEGENERATE_CI`.

(an-uncertainty-dictionary-is-rejected)=
## My uncertainty settings are rejected

Pass `None`, or a dictionary with:

- Required `mode`: `"reference"`, `"candidate"` or `"both"`.
- Optional `n_replicates`: a positive integer repetition count, defaulting to 10.

For example:

```python
uncertainty={
    "mode": "both",
    "n_replicates": 25,
}
```

Empty dictionaries, unknown keys, `mode="none"`, and nonpositive or noninteger
repetition counts are rejected. Use `uncertainty=None` to turn repetition off.

## My component table or plot is unavailable

Shape, Trend and Conditional component tables must be captured during evaluation:

```python
report = evaluate(data, metrics=["shape", "trend"], details=True)
columns = report.details("shape")
```

For Shape, Trend and Conditional Fidelity, `details=True` retains component
tables from the generated-real comparison. If the report was created without
that option, rerun the evaluation with it.

Aggregate score plots use stored scalar results and do not require component
tables. Concentration retains its radius profiles automatically.

See [component tables](reference/reports.md#component-tables) for available
detail levels.

(a-classifier-reached-its-training-budget)=
## A classifier did not converge

A finite Linear C2ST score is still reported when fitting finishes without
meeting its stopping criterion. The diagnostics record the solver, termination
reason and fold information:

```python
metric = next(
    result for result in report.results
    if result.name == "linear_c2st"
)
print(metric.diagnostics)
```

Solver limits are part of the metric's fixed definition.

For XGBoost metrics, `converged` records completion of the boosting budget. See
[convergence interpretation](sharp-bits.md#convergence-diagnostics-describe-fitting)
for what these diagnostics establish.

(my-device-or-optional-dependency-is-unavailable)=
## A GPU or dependency is unavailable

Check the [installation guide](getting-started/installation.md) against the
exception. Torch and XGBoost are core dependencies; GPU execution requires the
appropriate CUDA setup and GPU dependencies.

With `device="auto"`, execution is selected per metric. Tensor metrics can use
CPU for smaller workloads. Nonlinear C2ST and MLE prefer a usable GPU when their
dependencies and estimated memory permit.

The report shows which device was used for each metric:

```python
print(report.provenance["metric_execution"])
```

Explicit CUDA requests raise when their requirements cannot be met. Correct the
GPU setup or select `device="cpu"`.

## Evaluation is slow or runs out of memory

Reduce the requested work where appropriate:

- Select only the metrics you need.
- Start with fixed scores before requesting repeated evaluation.
- Reduce `n_replicates` to request fewer comparisons.
- Lower `n_jobs` to reduce concurrent CPU work.
- Use hardware with sufficient available memory.

Exact nearest-neighbor searches can make large DCR evaluations slow. Memory
planning limits temporary work, though individual allocations can still fail.
Reducing the input size changes which data are evaluated and may change the
comparison size. The [execution options](reference/configuration.md#set-execution-options)
and [DCR execution](metrics/memorization/dcr.md#execution) guides describe the
relevant controls.
