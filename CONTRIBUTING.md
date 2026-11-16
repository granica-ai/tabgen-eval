# Contributing

Thanks for helping improve `tabgen-eval`.

## Dev setup

We use [uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.13     # .venv + core deps + pytest, Hypothesis, Ruff and Pyright
```

Python 3.13 is the development/type-checking environment. The package supports
Python 3.10 and later; the installed-wheel CI matrix checks that compatibility.
XGBoost is a core dependency and is included by `uv sync`.

The Python package builds with Hatchling (`python -m build --wheel --no-isolation`
in a builder with `build` and `hatchling` installed). For an editable development
install, run `uv pip install --no-deps -e .`. Native operators are compiled or
packaged separately by the tensor wheel tooling below.

PyTorch is the default frontend. Tensor source builds need a Linux C++17/OpenMP
toolchain and the Torch build helpers (`setuptools`, needed by Torch 2.6); binary-enabled wheels automatically load a matching packaged library.
The C++/CUDA sources under `execution/csrc/{common,cpu,cuda}` also ship with wheels, so
`execution/torch/cpu/extension.py` can build and cache a release library against the installed Torch
ABI when no compatible binary is present. Set `TABGEN_EVAL_CPU_CACHE` to a writable location in restricted environments.
Run `pytest tests/native/cpu tests/contracts -m "not gpu"`
in the Torch environment. Native formatting uses clang-format 18.1.8 and the checked-in
`.clang-format` policy.
The numerical paths preserve caller thread/precision settings and retain failure
diagnostics. Measure warmed compiled execution separately from cold import,
source compilation and Triton device JIT. Independent Python/oracle comparisons
use small representative cases so the normal test lane stays fast.

## Tensor binary wheels

Build in the Torch environment that the artifact will support:

```bash
python tools/build_tensor_wheel.py --out dist/tensor
# Include CUDA C2ST and Triton host launchers; needs C/C++ and matching SDK headers:
python tools/build_tensor_wheel.py --out dist/tensor --cuda
# Reuse an existing Python source wheel instead of invoking Hatchling again:
python tools/build_tensor_wheel.py --out dist/tensor --cuda --wheel dist/tabgen_eval-*.whl
```

The tool packages baseline CPU instructions rather than `-march=native` and
relative runtime library paths. Libraries and metadata live under
`tabgen_eval/execution/_binaries/{cpu,cuda,triton}`. Loaders require an exact Torch
version/build, C++ ABI, platform/architecture, source digest and SHA256; CUDA
metadata also requires the matching Torch CUDA version. Minimum GLIBC/GLIBCXX
requirements are recorded; GLIBC is checked before loading and the dynamic linker
enforces GLIBCXX compatibility. Runtime `libstdc++` and CPU OpenMP `libgomp` are
still required. Install a matching
artifact to avoid first-use compiler/SDK requirements. Generic source wheels
still ship C++ sources and use compilation fallback. Keep external deployment
paths `TABGEN_EVAL_CPU_LIBRARY` / `TABGEN_EVAL_CUDA_LIBRARY` for explicitly selected,
compatible libraries.

Binary wheels pin the installed Torch distribution version and carry a runtime
Torch/CUDA build tag. Runtime metadata separately validates the full Torch build,
including its CUDA suffix: a PyPI distribution can report `2.11.0` while
`torch.__version__` is `2.11.0+cu130`. They currently use `linux_x86_64` tags
and do not claim manylinux compatibility.

CPU artifacts use `py3-none-linux_x86_64` because the operator library has no
CPython binding; supported Python/Torch combinations still apply. CUDA artifacts include
CPython Triton host modules and target the builder's exact Python ABI; the
current CUDA artifact is `cp313-cp313` and pins Triton `3.6.0`. Bundled host
metadata validates Python SOABI, Triton version, source digest, module checksums
and minimum GLIBC. Compatible driver/launcher modules populate Triton's normal
cache without replacing its compiler hooks. Device LLVM/PTX JIT compilation
remains normal and needs a writable Triton cache. These artifacts remove both
host C/C++ compiler and CUDA developer SDK requirements; generic/source CUDA
installs still need a C compiler for Triton's host wrappers and C++/SDK for C2ST.
Runtime `libstdc++`, `libgomp` and NVIDIA driver/libraries remain required.

CPU execution requires Torch `>=2.6`. CUDA extras require Torch `>=2.6` and
Triton `>=3.2`; runtime CUDA execution also requires a CUDA build `>=11.8`.
These are independent minimum versions, with no upper bounds or combination allowlist.
Portable Triton host bundles target Triton 3.6. Versions below the runtime minima
route to CPU automatically and raise on an explicit CUDA request. Current numerical/protocol validation
covers these Linux x86_64 environments:

| Python | Torch build | Execution |
|---|---|---|
| 3.10 / 3.11 | `2.6.0+cpu` | CPU numerical/protocol and compiler-free installed reports |
| 3.13 | `2.11.0+cpu` | CPU numerical/protocol and compiler-free installed reports |
| 3.13 | `2.11.0+cu130` | CPU and CUDA 13.0 / Triton 3.6.0 on A100 SXM4 80GB |
| 3.13 | `2.6.0+cu118` | CPU and CUDA 11.8 / Triton 3.2.0 on A100 80GB |

Fresh-consumer checks validate loading and complete reports with host compilers
disabled, extra SDK paths removed, and empty caches. CUDA checks cover FP32 and
FP64, explicit and automatic routing, and Triton's normal device JIT. The
validation scope is the artifact matrix above. Benchmark timings and their
measurement scope are recorded in [the benchmark artifact](docs/gpu-benchmarks.json).

The host CUDA library contains cuBLAS/cuSOLVER calls; Triton compiles device
kernels for the installed GPU. Broader platform, framework and device support
must be validated before extending this matrix. CPU execution works with the
matching CUDA-enabled Torch build without requiring a GPU; each CPU-only Torch
artifact must match its own exact version/build.

[`tensor-installation.yml`](.github/workflows/tensor-installation.yml) builds and
tests CPU Torch 2.6 on Python 3.10/3.11 and Torch 2.11 on Python 3.13
in fresh consumers on PRs. Its manually selected
CUDA lane requires a provisioned Linux x64 CUDA 13 runner. Its daily scheduled
statistical/scale job runs `pytest -m "calibration or perf"`; select `run_slow`
when dispatching the workflow to run that lane manually. The source CPU job
installs Arrow and XGBoost and runs the complete non-GPU fast suite. Memorization
search is exact and has no Faiss dependency. To inspect an
installed artifact locally, run from outside the checkout with `PYTHONPATH`
unset and fresh caches:

```bash
python /path/to/checkout/tools/check_installation.py --require-installed \
    --without-compiler --out installation-cpu.json
python /path/to/checkout/tools/check_installation.py --device cuda:0 \
    --require-installed --without-compiler --out installation-cuda.json
```

The installation tool checks fixed-score and two-replicate uncertainty reports, explicit
comparison backends, custom metric contracts, serialization and compiled CPU
loading without a compiler/extra SDK. Matching CUDA host bundles run without
C/C++ compilers or developer SDK while device JIT runs normally.
`--block-torch` / `--block-triton` exercise dependency absence. Fresh-install results are separate from numerical tests in
a development environment.

Validate artifacts in a fresh target environment, with compiler/extra SDK paths
removed, for CPU-only execution, explicit CUDA and automatic report routing.
Also check missing/incompatible artifact fallback and broken dependency errors.
Record import/build/JIT separately from warm task or complete-report timings;
[the repository benchmark artifact](docs/gpu-benchmarks.json) states each scope.

## Where things live

`evaluate()` validates the public selection and delegates to the required
paired, test-only and memorization procedures. The ownership map is in
[`docs/architecture.md`](docs/architecture.md). The short version:

- public names and data-role requirements live in `registry/selection.py`; computation adapters live in `metrics/`, with their internal registry in `registry/registry.py`;
- numerical calculations live in `execution/python/`, `execution/xgboost/` and `execution/torch/`, with shared raw outcomes in `execution/runtime/outcomes.py`;
- `_evaluation/` orchestrates fixed and repeated procedures, `uncertainty/` collects repeated raw scores, `statistics/` summarizes distributions, and `report/` assembles stored reports;
- the statistics behind value / CI / ideal / flags are in [`docs/methodology.md`](docs/methodology.md);
- the catalogue of failure modes the code defends against is in [`docs/failure-modes.md`](docs/failure-modes.md).

### Tensor and compiled-source ownership

`execution/csrc/common/` owns backend-neutral classifier sampling, rate policy,
and contracts. `execution/csrc/cpu/` separates runtime helpers, fidelity,
distance, classifier algorithms and operator bindings. `execution/csrc/cuda/`
separates runtime infrastructure, linear algebra, fidelity and classifier
algorithms. Device implementations must not include shared headers through a CPU
or CUDA directory. Python device adapters and loaders live under
`execution/torch/{cpu,cuda}`; shared classifier fold contracts stay under
`execution/classifier`.

The central inventory in `execution/torch/sources.py` defines compilation units
and source dependencies relative to `CSRC_ROOT` (`execution/csrc/`). Build caches
and packaged-library identities share that inventory. The `csrc/` tree contains
C++/CUDA sources only; Python build and loading helpers belong in `torch/`.
Update that inventory when adding or moving a source. Cache invalidation must
cover shared headers and public boundary changes, including installed wheels.
Keep handle guards, error helpers and PTX module loading in CUDA runtime
infrastructure, and keep algorithm-specific launches with their algorithms.

Native headers must compile with their own includes, declare ownership and
mutation clearly, and avoid depending on transitive includes. Decode serialized
tensor packets and diagnostics at adapter boundaries; numerical code should use
named views and fields rather than repeating positional indices.

Tests are grouped in `tests/native/{cpu,cuda,shared}`, cross-backend contracts in
`tests/contracts`, public reports in `tests/integration`, deployment checks in
`tests/packaging`, and statistical/scale checks in `tests/calibration`. Python
pytest tests may validate C++/CUDA operators through PyTorch; standalone C++ test
sources live beside their pytest driver. See [`tests/README.md`](tests/README.md)
for the directory map and fixture/oracle ownership. Keep numerical assertions,
selection markers and installed-package discovery intact during reorganizations.

## Before you open a PR

Check typing, lint and formatting, then run the fast and statistical test lanes:

```bash
uv run ruff check .                       # lint (whole repo)
uv run ruff format --check .              # format gate (whole repo)
uv run python -m pyright                  # configured strict public/core contracts
uv run pytest                             # fast lane: unit/property/metamorphic/fuzz
uv run pytest -m "calibration or perf"    # statistical and scale lane
```

Pyright's strict paths are listed in `pyproject.toml`; this is an incremental
gate, with some numerical and optional-library boundaries outside its scope.
It targets the Python 3.13 development dependency stubs. Keep meaningful shared
annotations with their domain owner, and use protocols or typed dictionaries
for structured boundaries instead of widening them to untyped containers.

Compiled source changes require clang-format **18.1.8** on the affected C++/CUDA
files and the compiled CPU/CUDA test lanes in their supported environments.
Install the pinned formatter with `uv tool install clang-format==18.1.8`, then run
`clang-format --dry-run --Werror path/to/file.cpp` (or `clang-format -i` to apply it).
The native-style CI lane checks all shipped native sources and standalone native
test sources against `.clang-format`. Keep formatting-only edits separate from
algorithm changes when possible. Document exported
and shared C++ interfaces with Doxygen comments; include shapes, precision,
thread/device ownership, mutation and error conditions where applicable.

Python entry points use NumPy-style docstrings. Registered PyTorch operators
state shape/dtype/device, mutation and fake-tensor contracts. Triton kernels state
launch geometry, memory layout, tail masks, arithmetic and temporary workspace.
Document meaningful interfaces rather than duplicating trivial helper bodies.

`pytest` with no args is the **fast lane** by default — `pyproject.toml` sets `addopts = -m "not calibration and not perf"`, so the slow statistical (`calibration`) and scale (`perf`) layers are *excluded* unless you select them with the second command. Run both locally before opening a PR. Installed-artifact, optional-dependency and CUDA checks have their own environment requirements; the workflow files define their CI triggers. (`ruff format .` without `--check` applies the formatting.)

## Test layers

The suite is layered fast → slow; each layer has a pytest marker:

| Layer | Marker | What it covers |
|---|---|---|
| unit | `unit` | each metric and helper on tiny hand-built frames (identical sets at the optimum, degraded sets worse) |
| CUDA | `gpu` | tensor preparation and FP32/FP64 Shape/PRDC/DCR/Trend/Conditional and cuBLAS/cuSOLVER C2ST kernels, complete reports, stream isolation and multi-draw workspace limits; requires PyTorch, Triton and a usable CUDA device; C2ST additionally needs matching CUDA developer headers/C++ or a compatible prebuilt host library |
| property | `property` | Hypothesis invariants over arbitrary valid frames: range, row/column-permutation invariance, symmetry and seeded sampling |
| metamorphic | `metamorphic` | controlled transformations and structural relations, including distance affine invariance and conditional-versus-global fidelity behavior |
| calibration | `calibration` | synthetic repeated-trial checks of reference variability, comparison behavior and sensitivity to distribution shifts |
| fuzz | `fuzz` | adversarial frames must yield a flagged `Report` or a typed error — never an uncaught crash |
| perf | `perf` | larger-frame execution, bounded distance workspaces, automatic resource planning and parallel/serial consistency |

`tests/test_traceability.py` is a meta-test: it asserts every catalogued failure mode maps to at least one covering test (and that every referenced test actually exists), so coverage gaps fail CI rather than going unnoticed. The catalogue lives in [`docs/failure-modes.md`](docs/failure-modes.md) — when you add or change a failure mode, update that table **and** the `COVERAGE` map in `test_traceability.py` (keyed by the failure-mode's slug). New behaviour ships with its tests in the same PR; don't assert exact float equality across platforms — use tolerances.

## Adding a metric

The public evaluator accepts registered names, not metric instances or per-metric
keyword dictionaries. Adding a built-in involves both its computation and the
public results that select that computation:

1. Implement a `Metric` subclass in `src/tabgen_eval/metrics/` with its name, category, statistic definitions and input requirements. The ordinary `compute(reference, candidate, schema, *, rng)` contract returns `{stat_name: value}`. Undefined statistical outcomes return NaN with diagnostics; invalid inputs, dependencies, resource failures and programming errors can raise.
2. Keep metric methods as adapters. Put numerical work in the appropriate `execution/python/`, `execution/xgboost/` or `execution/torch/` module. Numerical functions consume assigned data and explicit options; sampling and report construction belong to their own layers.
3. Register the computation with `MetricSpec(YourMetric)` in `registry/registry.py`. Add its public selector or selectors to `SELECTIONS` in `registry/selection.py`, declaring each group, required roles, output statistics and primary statistic. Several public names can share one computation, as PRDC and Conditional do.
4. Integrate any new data-role or fitted-model protocol explicitly in `_evaluation/` and `execution/runtime/`. Define source eligibility before uncertainty draws and record exclusions through the shared row-usage contract. Data diagnostics remain a separate `diagnose` workflow.
5. Add numerical, selection, role-validation and report tests appropriate to the change. Check that discovery through `list_metrics()` and `list_groups()` agrees with evaluation. Put independent Python numerical tests under `tests/execution/python/` and protocol/backend contracts in their corresponding suites.

Internal test adapters can implement the ordinary Python compute contract. Native
built-ins are selected explicitly in `execution.runtime.dispatch.supports`; a new metric does not
silently inherit native behavior. Classifier adapters subclass `ClassifierMetric`
and implement `_fit_predict(train, labels, test, *, seed, point, fit_threads=1)`
to reuse training-only transforms and fixed-source-fold bootstrap refits. Adapters accept the `fit_threads` keyword even when their numerical library
budgets threads externally.
New fitted protocols need their own explicit statistical rules. A noncomputable
fit raises `FitUnavailable` from `models/errors.py`, carrying a stable reason
and diagnostics for the metric's flagged result. Dependency/input errors must
propagate. Import `XGBoostModel` and `FitUnavailable` from `tabgen_eval.models`; models own fitting/prediction and do not depend on metric definitions.
Reuse classifier libraries through adapters;
the prepared numeric/category storage is independent of a model's encoding.

New metrics that need training, test, or other distinct data roles require an
explicit execution protocol and compatible `EvalData` roles through `evaluate`.
Do not infer an additional dataset or statistical procedure from the ordinary
two-frame metric interface. Numerical executors should consume assigned source
positions and return scores/diagnostics; the statistical layer owns repeated-score
summaries, and report assembly owns presentation.

Reuse `execution/python/encoding/features.py`
for Python reference-fitted geometry; PRDC and DCR share neighbor operations in
`execution/python/encoding/distances.py` and reductions in
`execution/python/distance.py`. Shared statistical helpers live in
`execution/python/stats.py`. Tensor geometry lives in `execution/torch/distance/geometry.py` and registered distance
operators in `execution/torch/distance/ops.py`; CUDA kernels live under
`execution/torch/cuda/triton` and compiled CPU operators under `execution/csrc/cpu`.

Standard paired metrics share point, reference and repeated-score assembly. Metrics with distinct data roles or statistical procedures need an explicit integration; report assembly must not invent a baseline or uncertainty interpretation.

## Principles

- Keep data roles and resolution explicit. Pairwise evaluation shows its matched real-real reference; uncertainty is optional, and training-referenced point metrics have no artificial reference or interval.
- Fail loud at the boundary (raise on misuse), fail safe inside (flag degenerate data).
- Use private seeded sampling and fitting streams. Record numerical precision, backend and approximation settings; do not promise bitwise identity across numerical implementations. GPU work respects the selected stream and bounded workspace.
