# Building the documentation

The public site uses Sphinx Book Theme and MyST Markdown. Sources live in
`docs/source/`; metric guides and API pages are curated for package users.
Website source/edit links point to the project repository. Documentation tooling needs Python 3.11 or later.

## Install the tools

From a checkout with the package's Python dependencies already installed:

```bash
uv pip install --python .venv/bin/python --group docs
```

This installs the documentation dependency group without synchronizing the full
project environment or updating the lockfile.

## Build and preview

From the repository root:

```bash
make -C docs html PYTHON=../.venv/bin/python
```

Open `docs/_build/html/index.html`, or start a local server:

```bash
make -C docs serve PYTHON=../.venv/bin/python
```

Visit <http://localhost:8765>. The server binds to loopback. When working on a
remote machine, forward port 8765 through your SSH connection. Stop the server
with Ctrl+C.

## Keep examples and references accurate

The build fails on warnings, including broken documentation references. It runs
the small `examples/report_walkthrough.py` example with native CPU execution to
generate the displayed report tables and download. That example does not use a
GPU or download datasets. Without a compatible precompiled library, native CPU
operators compile on first use. Metric examples should be checked against the
current public interfaces. The site does not expand entire classes or publish
implementation source pages.

Generated report snippets and HTML live in ignored directories. Update the
example source instead of editing generated output. Only `docs/source/` is
published. Repository engineering guides remain outside
navigation, search and generated source downloads. Use `make -C docs clean`
before rebuilding after removing or relocating pages to clear stale HTML.

## Native CPU build details

The source builder supports Linux and native arm64 macOS. System prerequisites
belong in the [installation guide](source/getting-started/installation.md).
The builder selects the C++ standard required by Torch and checks the language,
OpenMP and installed Torch headers before compilation. Each stage retains its
command and output in the CPU cache; `build.json` records the final command,
Torch version and native source paths. Successful builds use a filesystem lock
and atomic library rename so other processes only see a completed artifact.

On macOS, the builder prefers Torch's bundled OpenMP runtime to avoid loading
another copy. `TABGEN_EVAL_OPENMP_ROOT` selects custom headers and supplies the
runtime when Torch has none. Apple Clang uses `-arch arm64`,
`-Xpreprocessor -fopenmp` and explicit OpenMP include/link paths. The generated
dynamic library links to the selected runtime with an rpath and reserves room
for install-name changes. The builder assembles these flags itself; `CPPFLAGS`
and `LDFLAGS` are not consumed.

Torch's bundled `libomp.dylib` can retain an absolute install name from its build
machine. The builder inspects the OpenMP probe and generated library with
`otool -L`, repairs stale dependencies with `install_name_tool -change`, and
refreshes the generated binary's ad-hoc signature with `codesign` when a repair
was needed. This happens before running the probe or publishing the library.
Only build-cache files are changed; Torch's installed runtime is untouched.
Dependency, relocation and signing commands have separate stage logs. macOS
support currently uses source builds; the native binary-wheel release tool
remains Linux-only.

## Methodology figures

The methodology page uses three independent standard-normal columns. Its first
figure compares one realized Shape and Trend score at each of ten sample sizes
with their exact finite-sample expectations. Both curves use the same sizes.
Its second compares repeated-score means and descriptive percentile bands with
the exact finite-sample means for real-real and generated-real comparisons. The toy
generator copies its first standard-normal column into its second and keeps an
independent third column.

The documentation build reads `docs/assets/methodology-gaussian.json`. Regenerate
that asset with the project environment and optimized CPU backend:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python examples/methodology_gaussian.py
```

The script calculates the exact curves and scores one fresh Gaussian sample
pair at each size with the package metrics. It also evaluates fixed-seed real
and generated tables of 10,000 rows each through the public API. The paired
example draws 10 matched 5,000-row comparisons without replacement within each
comparison, keeping the two real roles disjoint. Source rows remain available
for subsequent comparisons. The figure uses each repeated-score mean and the
raw 2.5th and 97.5th score percentiles. These bands describe score variability;
they are not confidence intervals for the expected score. Seeds, repeated
scores, fixed points, means, bands and exact means are retained in the asset
for reproduction.
Rebuild and review both figures after regenerating it.

This workflow builds a local website. Hosting configuration and Read the Docs
publication can be added separately once the pages have been reviewed.
