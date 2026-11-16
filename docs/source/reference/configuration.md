# Uncertainty and execution

Pass these options directly to `evaluate`; the defaults are enough for a first
run. Prepare your [`EvalData`](data.md) and [choose metrics](evaluation.md) first.

## Choose uncertainty

For ordinary paired comparisons, `uncertainty` selects the repeated comparisons:

| Setting | What it requests |
| --- | --- |
| `None` (default) | Fixed generated-real and real-real scores |
| `{"mode": "reference"}` | Repeated real-real comparisons |
| `{"mode": "candidate"}` | Repeated generated-real comparisons only; retain the real-real point without a reference interval |
| `{"mode": "both"}` | Repeated generated-real and real-real comparisons |

Enabled uncertainty accepts one optional repetition count:

| Key | Default | Meaning |
| --- | --- | --- |
| `n_replicates` | `10` | Number of repeated score comparisons |

For example, request 25 repeated comparisons:

```python
from tabgen_eval import evaluate

report = evaluate(
    data,
    metrics=["shape", "trend"],
    uncertainty={
        "mode": "both",
        "n_replicates": 25,
    },
    seed=42,
)
```

`mode` is required, and `n_replicates` must be a positive integer.
Invalid dictionaries raise with the accepted keys and an example. Use `None`
to turn uncertainty off.

`"candidate"` requires generated data. Its generated-real draws, sample resolution,
and band match `"both"` for the same seed and repetition count, while skipping
repeated real-real comparisons. Read the candidate band through
`report["shape"].ci_low` and `.ci_high`; `.reference_distribution` is `None`.
Memorization metrics remain point-only in every mode.

For each repeated side, the reported score (`value` or `ideal`) is the mean of
the finite repeated scores. No additional point comparison is computed for that
side. An unrepeated side keeps a single score. Within each repetition, rows are selected without replacement and the
two real samples are disjoint. All eligible rows become available again for the
next repetition. The finite repeated scores are summarized by their mean and
their 2.5th and 97.5th percentiles.

These percentile bands describe score variability across the sampled
comparisons. They are not confidence intervals for the population expected
score, including the real-real mean $\mu_n(P)$. See
[Methodology](../methodology.md#seeing-resampling-at-work) for interpretation.

MLE draws disjoint real-training and scoring samples for each comparison.
Its [model-fitting and scoring protocol](../metrics/utility/mle.md#repeated-evaluation)
explains generated-model reuse in reference mode.

The uncertainty mode can change the
{ref}`comparison size <comparison-sample-size>` and reported scores. Changing
`n_replicates` can change the reported means and percentile bands. More
repetitions stabilize these summaries for the supplied data and require more
computation. See
[The Sharp Bits](../sharp-bits.md#scores-depend-on-comparison-size) and
[Methodology](../methodology.md) for interpreting the real-data reference and
score bands.

## Set execution options

| Argument | Default | What to use it for |
| --- | --- | --- |
| `seed` | `42` | Reproducible sampling and fitting; accepts a nonnegative integer or sequence of nonnegative integers |
| `device` | `"auto"` | Automatic placement, or explicitly `"cpu"`, `"cuda"`, or a CUDA device such as `"cuda:0"` |
| `dtype` | `"float64"` | Tensor numerical precision; `"float32"` reduces tensor memory use and can change numerical results |
| `n_jobs` | `8` | Upper bound on CPU parallelism; actual concurrency may be lower. Use `1` for serial execution or `-1` for all cores |
| `details` | `False` | Retain component tables for a generated/test point comparison; see [component tables](reports.md#component-tables) |

For example, add `device="cpu", details=True` to the call above to retain
component results using CPU execution. The
[installation guide](../getting-started/installation.md#linux-gpu-installation) covers
GPU setup.
