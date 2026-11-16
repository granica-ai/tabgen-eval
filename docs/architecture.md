# Architecture

The public metric interface is `evaluate(EvalData(...), ...)`, with frozen
explicit data roles, a required declared schema, and registered groups/atomic
metric selectors. Input preparation belongs to the caller. Independent
`diagnose(train=..., generated=...)` returns its own data report. The public
planner validates input roles, prerequisites and selection before calling private
paired, test-only or memorization procedures. Statistical procedures own sampling; numerical implementations
consume assigned observations and return raw scores and diagnostics. Reports own
presentation. See the [user-facing API](source/reference/evaluation.md).

## Preparation and numerical execution

`preprocessing/` owns typed categories, immutable host columns, Arrow
preparation, precision matching, training-fitted fold/predictor state and CPU/CUDA
application. `execution/runtime/` owns tasks, contexts, routing and source draws.
`execution/python/` supplies NumPy/SciPy numerical primitives and custom-classifier fold execution;
`execution/xgboost/` owns shared supervised-model and nonlinear-classifier adapters;
`execution/torch/` owns tensor transfer, compiled operator adapters and CUDA/Triton
launches. `execution/csrc/` contains shared contracts and CPU/CUDA numerical sources.
Independent solver references used only by tests belong in `tests/support/`.

Caller frames, source positions and ambient numerical settings are preserved.
Encoders fit on assigned training roles only. Shared storage and task-local
workspace have distinct lifetimes; cached executors must not keep their owning
contexts alive through reference cycles. Each fold releases storage before the
next allocation, including failure paths.

`models/` owns XGBoost model fitting/prediction and typed fit failures. The same
training-only predictor backend supports MLE and nonlinear C2ST on CPU/CUDA.
Model outputs retain actual execution attribution, selected precision, target
vocabularies, seeds and stopping diagnostics. Schema, dependency and programming
errors propagate; recognized unusable fits become diagnosed undefined outcomes.

## Statistical protocols

`_evaluation/` validates and prepares metric computations, runs fixed-score
comparisons, and orchestrates paired, test-only, and memorization input roles.
`execution/runtime/preparation.py` prepares compatible sources and shared sessions
for paired and test-only evaluations. Both paths use the same grouping, classifier
preparation, and source-storage rules.

`uncertainty/` collects repeated raw scores for paired and test-only evaluations;
it does not perform public selection or assemble reports. `statistics/` owns
finite summaries and empirical percentile bands. `report/assembly.py` and
`report/evaluation.py` convert raw computation outputs into stored report objects.
Public evaluation retains descriptive bands at the fixed 95% level through
the shared internal
`DEFAULT_CONFIDENCE_INTERVAL`. The evaluator and CLI expose no confidence-interval
option. These bands are empirical score quantiles, without confidence-interval
interpretations for population expected scores. Finite summaries preserve raw
nonfinite observations, unavailable counts and reasons rather than hiding them.

Sampling domains own eligibility before point and UQ draws. Fidelity and detector
comparisons retain full sources and their metric-specific observed-value or
predictor treatment. Distance metrics use complete declared features; MLE uses
observed targets and retains missing predictors. Metrics share source positions
within a domain; adding another domain does not change existing draws. All
two-sample domains retain matched real-reference and real-candidate roles with
disjoint positions within each comparison.
The uncertainty mode determines each domain's comparison resolution after its
eligibility filter. Fixed points and repeated draws use the same resolution.
Changing to `"both"` can change point scores, while changing the repetition
count preserves them. Draws sample without replacement within a comparison;
all eligible source rows become available again for later draws.
MLE draws generated and fresh-real training samples and a shared real scoring
sample. The real roles are disjoint within each comparison and redrawn across
repetitions. Reference mode reuses the single generated-trained point model.
MLE's public statistics are raw
MAE/AUROC `score`, their generated-to-real `ratio`, and classification `prediction_agreement`
or regression `prediction_mse`. Each comparison calculates its ratio; the
distribution's mean ratio can differ from the quotient of raw-score means.
The ratio has no invented real-real ideal. Test-only MLE evaluation stores raw
`score` samples from disjoint real training/test draws. Regression MAE and its
ratio are lower-is-better; classification AUROC and its ratio are higher-is-better.

Memorization owns actual generator-training roles and is point-only. Its exact
search planner shares forward generated→training queries across DCR and near-match
rate and reverse training→generated queries for replication ratio. A holdout is
queried against the same training-fitted geometry. DCR summarizes generated and
holdout percentiles; near-match/replication use fixed inclusive distance `1e-3`.

Concentration retains target-smoothed nonnegative local-mass ratios with fixed
`epsilon=1e-8`, value-based fixed-radius keys and a mean-only fixed radius `1e-3`.
Captured profiles stay separate from uncertainty summaries.

## Selection and diagnostics

`registry/selection.py` declares public metric names, groups, output companions,
required roles and shared computation jobs. `report/evaluation.py` projects
completed jobs into independently selectable results; it performs no numerical
work. PRDC components and conditional components share preparation and jobs.
The `uncertainty` dictionary selects the comparisons to repeat and sets their
repetition count with `n_replicates`, defaulting to 10.

`diagnostics/` owns raw-input profiling, schema/structure observations and support
checks against full actual training data. It has no evaluator registration,
quality score or metric-sampling dependency.

## Presentation and documentation

`report/result.py` and `report/diagnostics.py` own stored result types and factual
flags. `report/io.py` owns lookup and exports; `report/rows.py` presents source
exclusions separately from selected row resolution and finite replicate counts.
`report/families.py` provides shared
focused views. `visualization/` reads stored scores/details or explicitly supplied
input frames. It never samples evaluation roles or fits a model.

`docs/source/` is the curated Sphinx site. Build hooks use small offline CPU
examples and retain source tables with interactive figures. The failure-mode
catalogue at `docs/failure-modes.md` uses stable IDs checked against AST-resolved
test references. Numerical oracles, runtime contracts and installed-wheel checks
have separate responsibilities; see `tests/README.md`.

## Types and test support

Shared annotations describe precision, seeds, arrays, execution
metadata, and source usage at module boundaries. Typed dictionaries describe
structured metadata; protocols describe execution capabilities without requiring
concrete context inheritance. The package ships `py.typed`. Run the configured
checker with `python -m pyright`; strict coverage applies to the paths listed
in `pyproject.toml` and expands as numerical and external-library boundaries
gain precise contracts. The checker targets the Python 3.13 development
dependency stubs. The package still supports Python 3.10, checked through
Ruff's Python 3.10 syntax target and the Python 3.10 CPU-wheel test matrix.

Ordinary CUDA host wrappers carry Tensor argument and return annotations; Triton `@jit` kernels retain the
compiler's pointer/scalar and `tl.constexpr` conventions rather than Python Tensor
annotations that would misdescribe device-language values.

Numerical reference helpers consumed by tests remain useful code. Remove code
only after checking package, tests, tools, documentation hooks, dynamic dispatch,
and native registration for consumers. Tests build explicitly configured metric
instances in `tests/support/`; production engines do not retain hidden defaults
or a second metric-selection API for test convenience.
