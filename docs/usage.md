# Use TabGenEval

The curated [quickstart](source/getting-started/quickstart.md),
[evaluation reference](source/reference/evaluation.md), and
[data preparation](source/reference/data.md) and
[uncertainty and execution](source/reference/configuration.md) describe current
public interfaces. The [visualization guide](source/getting-started/visualization.md)
contains interactive offline examples.

## Data roles

Use `evaluate(EvalData(test=holdout, generated=generated, schema=schema), metrics=...)` for fidelity,
detection, Concentration and MLE. The holdout is independent of generator training.
MLE requires `Schema.target` and retains raw MAE/AUROC, generated-to-real score ratios and
shared-test prediction comparison statistics. It reuses its generated/real-trained
models for all of those outputs. MAE and its ratio are lower-is-better; AUROC
and its ratio are higher-is-better. Missing supervised targets are excluded.

Use the same `evaluate` with `EvalData(train=training, generated=generated,
test=holdout, schema=schema)` and `group=["memorization"]` for actual training-record proximity.
`dcr_ratio` requires test data and exposes generated and test q05–q99
percentiles, a median alias and a median ratio. Near-match and replication count
different denominators at the fixed inclusive Euclidean threshold `1e-3` in
training-fitted geometry. They are exact and point-only; uncertainty settings
do not add score distributions to memorization results.

The other valid configurations are test alone, which evaluates disjoint real
subsets, and train plus generated, which supports near-match rate and replication
ratio without DCR. Generated alone is invalid. See the complete
[data roles](source/reference/data.md#assign-data-roles) and
[registered metric names](source/reference/evaluation.md#registered-names).

With both filters empty, every metric compatible with the roles is selected.
Test-based defaults include MLE and nonlinear C2ST and require a target for MLE.
XGBoost is included in the base installation. A nonempty `metrics` list overrides group selection, while
names in both filters are still validated. Use registered public names such as
`precision` or `cond_shape`; shared computation names such as `prdc` are not
accepted as metric selectors.

## Sampling and bands

An explicit schema is required. Inference is an optional helper whose output the
caller must check; inputs are not repaired or converted. With no UQ or
`"reference"`, paired resolution is the smaller of eligible generated size and
half the eligible test size, rounded down. `"both"` uses half the test size if
the eligible generated pool is larger than that size; otherwise it uses half
the generated size, rounded down. Fixed points and repetitions share the
selected mode's resolution, so changing modes can change point scores.
Distance procedures prefilter complete records, and MLE prefilters observed
targets. Fidelity keeps observed-value handling within full-pool draws.
`report.row_usage()` exposes source exclusions and percentages. All retained
source rows remain eligible. Fixed points use independent seeded
streams from repetitions. `uncertainty=None` computes fixed scores;
`uncertainty={"mode": "reference"}` and `uncertainty={"mode": "both"}` select
the shared or MLE-specific repetition protocol documented in the metric guide.
Generated and reference bands share the same fixed 95% central coverage; no public configuration or CLI confidence-interval
control is available. Raw samples, finite counts and unavailable reasons remain
inspectable through distributions and JSON exports.

## PyTorch execution

PyTorch is a core dependency. Explicit Python reference calculations remain
available; default execution uses compiled CPU operators for small workloads
and supported CUDA kernels for suitable larger tasks. CPU supports Torch 2.6–2.11;
the validated CUDA stack is Torch 2.11 / CUDA 13 / Triton 3.6 on Linux. Matching
binary artifacts validate ABI/source/checksums and avoid host compilation. Source
builds require the matching compiler/runtime dependencies.

XGBoost CPU models are included in the base installation. Install the supported
`xgboost-cuda` stack for CUDA models. Native categorical predictor matrices stay on their selected
device, with execution attribution in diagnostics. Explicit unsupported device
or precision combinations raise rather than silently changing requested work.
See [installation](source/getting-started/installation.md) and
[execution options](source/reference/configuration.md) for dependencies and execution settings.

## Data diagnostics

`diagnose(train=training, generated=generated, schema=schema)` returns a separate
`DiagnosticReport`. Its raw-input profiles and support checks use full training
data; `evaluate` does not invoke diagnostics.

## Reports

`report["metric/stat"]` retrieves a stored result; a public metric name resolves
its primary statistic. Unique bare statistic names
are also accepted. Ambiguous bare names list qualified alternatives. Empty or
multiply qualified keys are rejected. `summary`, `to_dataframe`, `to_markdown`,
`to_json` and `dump_replicates` retain the corresponding stored presentation.

Family views share results and metadata. `report.fidelity.plot(...)`,
`report.memorization.plot(...)`, and `report.utility.plot(...)` return interactive
Plotly figures without recomputation. Fidelity component plots require captured
details; undefined entries remain explicit. Input distribution plots are separate
utilities taking original frames, with deterministic bounded display samples.

## Command line

The installed command is `tabgen-eval`. It uses the same input roles, required
schema and registered names as Python:

```bash
tabgen-eval --test real_test.csv --generated generated.csv --schema schema.json \
  --metrics shape trend --device cpu \
  --uncertainty reference --n-replicates 25 --out evaluation.json
```

See the [CLI reference](source/reference/cli.md) for schema
JSON, exports and the distinction between stdout and discovery messages.
