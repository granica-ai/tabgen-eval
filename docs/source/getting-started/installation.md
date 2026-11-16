# Install TabGenEval

Requires Python 3.10 or later.

## Install the package

Start a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Clone the TabGenEval repository and install the package:

```bash
git clone https://github.com/granica-ai/tabgen-eval.git
cd tabgen-eval
pip install .
```

When you compute metrics for the first time, the package automatically compiles the required native libraries and/or CUDA kernels. **To ensure successful compilation, please follow the instructions below to prepare your environment.**

**Platform support:** This version supports Linux (CPU and GPU) and macOS on Apple Silicon (CPU only). Other operating systems and hardware platforms are not yet supported.

(linux-cpu-installation)=
## Linux CPU Installation

Install a compiler supporting the C++ standard and OpenMP
development/runtime support before evaluating.

For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

For Fedora/RHEL:

```bash
sudo dnf install gcc-c++ libgomp
```

The default compiler is `c++` on `PATH`; standard GCC installations generally need
no environment configuration. If necessary, set a compiler with `CXX`:

```bash
export CXX=/usr/bin/g++
```

Verify the installation from the repository root.

```bash
python tools/check_installation.py --smoke --out /tmp/tabgen-eval.json
python examples/report_walkthrough.py
```

The first build takes additional time. 
The installation check prints "Installation checks passed"; the walkthrough
prints a CPU evaluation summary. 
Successful builds are cached and reused.
On failure, inspect `check-*.log` or `build.log` in the reported directory, fix the cause and retry.

For build checks and logs, see {ref}`CPU compilation troubleshooting <cpu-build-prerequisites>`.
For cache configuration, see {ref}`CPU build cache <cpu-build-cache>`.

(macos-cpu-installation)=
## Apple Silicon macOS CPU Installation

Install Apple's Command Line Tools and OpenMP with [Homebrew](https://brew.sh/):

```bash
xcode-select --install
brew install libomp
```

Select Apple Clang as the `c++` compiler:

```bash
export CXX=/usr/bin/clang++
```

Verify the installation from the repository root:

```bash
python tools/check_installation.py --smoke --out /tmp/tabgen-eval.json
python examples/report_walkthrough.py
```
The first build takes additional time. 
The installation check prints "Installation checks passed"; the walkthrough
prints a CPU evaluation summary. Repeat the installation check in a new Python
process using the same cache to verify reuse of the compiled library.

For build checks and logs, see {ref}`CPU compilation troubleshooting <cpu-build-prerequisites>`.
For cache configuration, see {ref}`CPU build cache <cpu-build-cache>`.

For compiler, OpenMP and library-loading details, see
{ref}`macOS build troubleshooting <macos-cpu-build-troubleshooting>`.

**CPU support has been tested on macOS Tahoe (26.4) with Apple M5.**

## Linux GPU Installation

### Nonlinear C2ST and MLE GPU Support

Install XGBoost GPU support for nonlinear C2ST and MLE, including native categorical GPU attribution:

```bash
pip install '.[xgboost-cuda]'
```

Requires Python >=3.10, XGBoost >=3.2 and CuPy.

CUDA XGBoost has been tested on Linux with an NVIDIA A100 using Python 3.10.12,
XGBoost 3.2.0, CuPy 13.6.0, Torch 2.6.0 (CUDA 11.8) and Triton 3.2.0.
The XGBoost CUDA build and CuPy runtime in that test were CUDA 12.9.

### Other Metrics GPU Support

Install support for CUDA and Triton:

```bash
pip install '.[cuda]'
```

CUDA execution requires Torch >=2.6, CUDA >=11.8 and Triton >= 3.2.
These minimums are checked independently. Numerical consistency checks have covered these Linux stacks:

| Torch | CUDA build | Triton |
| --- | --- | --- |
| 2.11 | 13.0 | 3.6 |
| 2.6 | 11.8 | 3.2 |

