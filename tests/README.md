# Tests

Tests are grouped by the contract they exercise. Directory placement and pytest
markers are independent: moving a test does not change its selection or its
hardware requirements.

| Directory | Responsibility |
|---|---|
| `execution/python/` | Internal numerical helpers, statistics, reference distances and vectorized conditional calculations |
| `native/cpu/` | ATen/C++ numerical operators, classifier fitting, SAGA updates, thread and precision isolation |
| `native/cuda/` | CUDA/Triton numerical operators, cuBLAS/cuSOLVER fitting, native preparation, streams and device workspace |
| `native/shared/` | Backend-neutral rate policy and coherent sampled preparation on CPU/CUDA |
| `contracts/classifier/` | Source folds, training-only transforms, solver dispatch, stopping diagnostics and backoff behavior |
| `contracts/distance/` | Independent one-hot/direct-distance oracles, category presence and complete CPU/CUDA distance report parity |
| `contracts/fidelity/` | Independent correlations, KS/TV, occupied-cell counts, routing and complete fidelity report parity |
| `contracts/runtime/` | Shared preparation, tensor ownership, dependencies, execution selection and complete report parity |
| `integration/` | Public evaluate/report/CLI behavior, uncertainty orchestration and golden reports |
| `packaging/` | Binary ABI/source/checksum rejection, wheel integrity, compiler-free loading and Triton host bundles |
| `calibration/` | Statistical calibration and scale/budget checks |
| `preprocessing/` | Shared classifier and supervised predictor policies, compact device designs and CPU Triton interpreter safety checks |
| root Python/domain suites | Explicit schemas, input contracts, metrics, property/metamorphic/fuzz tests and failure-mode traceability |
| `support/` | Independent small mathematical oracles and explicit metric/options/backend fixtures |

`conftest.py` owns shared frame/schema fixtures. `helpers.py` provides deterministic
frame construction; `native/cpu/conftest.py` owns CPU operator fixtures and
`native/cpu/oracles.py` supplies small independent dense classifier references.
`support/linear_c2st.py` supplies the independent sklearn classifier reference,
using float64 optimization on the already precision-matched prepared values.
`native/cuda/conftest.py` owns identical availability, device, precision and
operator fixtures used by the CUDA operator suites.
Keep numerical oracles independent of implementation helpers. A mathematical
contract should have one shared oracle when the same reference applies to both
backends; device-specific assertions stay with the device tests.
`support/distances.py` constructs typed one-hot references independently and
distinguishes coordinate precision from accumulation precision. Its memorization
adapter calls the metric group with explicit training/generated and optional holdout roles; it is not a numerical oracle.
`support/backends.py` centralizes CUDA availability and explicit execution cases.
Native CPU/CUDA FP32 and FP64 remain separate execution cases. Internal
reference functions are tested directly, without a selectable Python backend.

The standalone C++ scalar-policy program is
`native/shared/saga_rate_probe.cpp`. Its pytest fixture compiles it against the
selected package's shipped C++ headers and checks the results against
independent Python expectations. It skips when a C++ compiler is unavailable.
Native operator tests otherwise enter through the normal Python/PyTorch API,
using the packaged library or source-build cache. This exercises operator
registration and deployment as well as numerical behavior.

## Selection

| Marker | Purpose |
|---|---|
| `unit` | Small numerical, protocol, integration and deployment checks |
| `gpu` | Requires a usable CUDA device and the supported Torch/Triton environment |
| `property` | Hypothesis invariants |
| `metamorphic` | Relations between related datasets |
| `fuzz` | Adversarial input behavior |
| `calibration` | Slow interval/power checks |
| `perf` | Scale and budget checks |

```bash
uv run pytest                              # default fast lane
uv run pytest -m 'calibration or perf'      # statistical and scale lane
uv run pytest tests/native/cpu tests/contracts -m 'not gpu'
uv run pytest tests/native/cuda tests/native/shared tests/contracts tests/preprocessing -m gpu
uv run pytest tests/preprocessing/test_cuda_bounds_interpreter.py  # needs Torch and Triton; runs on CPU
uv run pytest tests/packaging
```

The default lane excludes `calibration` and `perf` through `pyproject.toml`.
The existing installation workflow selects them in its daily scheduled
`statistical-and-scale` job, also available through the manual `run_slow` input.
GPU fixtures use device 1 when multiple devices are available, otherwise device 0.
Set `CUDA_VISIBLE_DEVICES` to one free device to isolate a local run.
The same-distribution band-overlap simulations also carry `calibration`; they
are statistical checks rather than deterministic metamorphic identities.

## Numerical and deployment coverage

Distance tests independently construct precision-specific one-hot geometry and
check unknown levels, duplicate tiles, strict boundaries, even medians, missing
queries, zero ratios, FP64 high offsets and ambient autocast. Fidelity tests
cover per-pair missingness/constants, tied reference quantiles, right-closed
bins, literal `nan` versus missing labels, and domains with 15,000 categories.
Shared identity contracts cover exact strings, distinct scalar families, NumPy
equivalents and a reserved missing category. Synthetic metric and model tests
check explicit probability tables, one-hot geometry, training-only vocabularies,
typed targets and parallel-worker serialization.
Complete reports compare every stored replicate, interval and flag for point-only,
small-replicate and multiple-batch evaluations. Memorization contracts compare
generated-to-training distances with an independent real holdout queried against the same training pool.

Classifier tests check training-only transforms, fixed source folds, independent
optimizer optima, direct-Newton/SketchySAGA dispatch and failure diagnostics.
SAGA tests check dense update oracles, online-gradient infinity-norm stopping,
initial-gradient normalization, coefficient-change diagnostics, batch-aware
rates, coherent sampled preparation, retry budgets, private RNG replay, chunk
invariance, stable inverse application and bounded memory. CPU and CUDA numerical
implementations share contract tests while retaining their own update oracles.

`test_traceability.py` maps every failure mode in
[`docs/failure-modes.md`](../docs/failure-modes.md) to a real test function. Update
its paths when moving tests; preserve the covered behavior and numerical checks.
Each catalogue row has a stable anchor ID matching a `COVERAGE` key. The harness
checks complete membership in both directions and resolves references to actual
top-level test definitions.

Source CPU builds require a C++17/OpenMP toolchain; set `TABGEN_EVAL_CPU_CACHE` to a
writable directory. CUDA host builds require matching CUDA/CRT/CCCL headers;
set `TABGEN_EVAL_CUDA_INCLUDE` and `TABGEN_EVAL_CUDA_CACHE`, or supply a compatible
`TABGEN_EVAL_CUDA_LIBRARY`. Native source ownership and compilation are defined by
`tabgen_eval.execution.torch.sources`; `CSRC_ROOT` points to
`execution/csrc/`, with shared headers under `csrc/common/`. Python tensor
adapters live under `execution/torch/`.

The [tensor installation workflow](../.github/workflows/tensor-installation.yml)
includes the complete non-GPU fast suite with Arrow and XGBoost installed. It also
builds exact Torch CPU artifacts and runs installed numerical/protocol tests,
including SketchySAGA, with compiler paths disabled. Its optional CUDA lane uses
one provisioned CUDA 13 GPU and requires Arrow, CuPy and CUDA-enabled XGBoost
before the optional-engine tests, so missing extras cannot silently skip them.
The Python 3.13 CPU wheel consumer installs Triton after the dependency-fallback
checks and runs the interpreter safety tests with GPUs hidden; the CUDA consumer
also runs the full shared preprocessing directory against the installed package.
Separate checks simulate unavailable Torch or Triton.
The missing-Torch check verifies that built-in metrics reject the unavailable
required dependency; custom metric callbacks can still run independently.
`tools/check_installation.py` checks cold-cache
reports outside the checkout with compiler/extra SDK paths removed. These
installed-artifact checks complement development numerical tests.
Its small CPU `--smoke` route is executed by a subprocess test, and
tracked examples have small command-line smoke tests with an offline OpenML loader.

XGBoost integration tests use the core dependency installed by `uv sync`.
Pure protocol and missing-dependency tests also cover incomplete installations.

## Type and maintenance checks

Run `python -m pyright` for the configured type-checking boundary and
`ruff check src tests tools examples` for linting. Shared annotations belong in
the package's type contracts; test-only fixture builders stay under `support/`.
Private evaluator fixtures construct metric instances and normalized options
explicitly. Numerical oracles remain independent of optimizer implementations.

Useful numerical reference helpers can be consumed only by tests. That does not
make them dead code. Before removing a helper, check package and test callers,
tooling, documentation hooks, dynamic registries, and native bindings. Production
Newton's stable Armijo-loss kernel is checked directly from its compiled PTX on
the current CUDA stream, alongside the separate Triton reference-kernel checks.
