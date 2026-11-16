# Statistical methodology

TabGenEval grounds generated-data scores by comparing them with real-real scores
computed from samples of the same size. This page explains how those comparisons
are constructed and interpreted.

## The finite-sample yardstick

Finite-sample variation matters even when both samples come from the correct
population. For example, [Shape](metrics/fidelity/shape.md) has an ideal
population value of 1, but its expected real-real score at finite $n$ is
generally below 1. Comparing a generated score directly with 1 asks the
generator to meet a standard that even fresh real samples generally fall short
of because of sampling noise.

To make this concrete, let $R_{n},R'_{n},R''_{n},R'''_{n}$ be four independent
samples, each containing $n$ observations from the real population $P$.
For a metric $T$, the scores

$$
T(R_{n},R'_{n}) \qquad\text{and}\qquad T(R''_{n},R'''_{n})
$$

can differ through sampling noise alone, even though all four samples come
from $P$.

We therefore need a yardstick for the typical metric score at resolution $n$.
A natural choice is the expected score between two real samples of that size:

$$
\mu_n(P)=\mathbb{E}\!\left[T(R_n,R'_n)\right].
$$

Repeated real-real comparisons let us estimate this yardstick. Their score
distribution also describes variability across sampled comparisons.

For a running example, draw rows independently from a three-dimensional
standard Gaussian:

$$
P=\mathcal{N}(0,I_3).
$$

Each row has three independent standard-normal numeric entries. Each comparison
in the figure below uses two independent tables of $n$ rows from this population.

```{raw} html
:file: _generated/methodology/target.html
```

Each point on the score curve is one realization of $T(R_n,R'_n)$. These scores
fluctuate around $\mu_n(P)$, which remains below the population ideal of $1$ at
finite sample sizes. The mean $\mu_n(P)$ therefore provides a natural reference
for interpreting generated-data scores computed at the same sample size.

:::{dropdown} Exact means for this Gaussian example

Assume two independent samples of $n$ complete rows with three independent
standard-normal numeric columns. For Shape, let $D_{n,n}$ be the two-sample
Kolmogorov–Smirnov statistic for one column. Then

$$
\mu_n^{\mathrm{Shape}}
=1-\frac{1}{n}\sum_{k=1}^{n}\Pr\!\left(D_{n,n}\ge\frac{k}{n}\right),
$$

$$
\Pr\!\left(D_{n,n}\ge\frac{k}{n}\right)
=\frac{2}{\binom{2n}{n}}
\sum_{j=1}^{\lfloor n/k\rfloor}(-1)^{j-1}\binom{2n}{n-jk}.
$$

The exact tail is given by the equal-sample reflection formula in
Hodges (1958), Section 2(c).[^hodges1958]
See also the SciPy two-sample KS notes.[^scipy-ks]

For Trend, set $a=(n-2)/2$ with $n\ge3$. The
zero-correlation Pearson law[^scipy-pearson]
gives $(r+1)/2\sim\operatorname{Beta}(a,a)$. Applying this law to the pair score
$1-|r-r'|/2$ for independent sample correlations gives

$$
\mu_n^{\mathrm{Trend}}
=1-\frac{2\,\mathrm B(2a,2a)}{a\,\mathrm B(a,a)^2},
\qquad
\mathrm B(u,v)=\int_0^1 t^{u-1}(1-t)^{v-1}\,dt.
$$

Averaging the three column scores or the three pair scores preserves these
expectations.

:::

(resampling)=
## Estimating $\mu_n(P)$ and describing score variability

(metrics-with-other-data-roles)=
(other-evaluation-paths)=
The algorithms below describe two-sample comparisons.
[MLE](metrics/utility/mle.md#repeated-evaluation) repeats model fitting and scoring
with its own data roles.
[Memorization](metrics/memorization/index.md) remains point-only.

Draw $N$ comparisons without replacement within each comparison, returning all
rows to the available pool before the next. Average the scores to estimate
the expected score, and summarize their spread using empirical percentiles.
Below, $T(S,R)$ denotes the metric score for candidate sample $S$ against real
reference sample $R$. Pools contain the rows eligible for the selected metric.
The formulas below assume finite comparison scores. Reports summarize finite
outcomes and retain counts for unavailable scores.

::::{admonition} Estimating $\mu_n(P)$ and describing score variability
:class: note

**Inputs:** real holdout $H$, metric $T$, rows per sample $n$, and repetition
count $N$ (default 10). Require positive integers $n,N$ and $2n\le|H|$.

1. For each $i=1,\ldots,N$, draw disjoint $n$-row samples $R_i,R'_i$
   uniformly without replacement from $H$, and record

   $$
   C_i=T(R'_i,R_i).
   $$

   All rows become available again for the next comparison.

2. Estimate the expected score by the average:

   $$
   \widehat\mu_n=\frac1N\sum_{i=1}^{N}C_i.
   $$

3. Describe score variability using the 2.5th and 97.5th empirical percentiles:

   $$
   I_{\mathrm{ref}}=\left[q_{.025}(C),q_{.975}(C)\right].
   $$

   Here $q_p$ is the empirical $p$-quantile with linear interpolation.

**Outputs:** the mean estimate $\widehat\mu_n$, score band $I_{\mathrm{ref}}$, and
repeated scores $(C_1,\ldots,C_N)$.

::::

Increasing $N$ stabilizes the repeated-score mean and percentiles for the supplied
pool. It adds no observations and leaves the comparison resolution unchanged.
The band describes the spread of individual comparison scores. In general, it
does not provide a confidence interval for $\mu_n(P)$.

(data-roles-and-resolution)=
## A matched comparison

Let $Q$ be the generator's distribution and $G$ its generated data. Alongside
the real-real yardstick, we want the expected generated-real score at the same
resolution:

$$
\nu_n(P,Q)=\mathbb E\!\left[T(G_n,R_n)\right],
$$

where $G_n$ and $R_n$ are independent $n$-row samples from $Q$ and $P$.
The paired procedure estimates both expectations using the same real reference
in each comparison.
Use `uncertainty={"mode": "both"}` to request these matched repetitions.

(paired-resampling)=
::::{admonition} Repeated generated-real comparisons and their real-data reference
:class: note

**Inputs:** real holdout $H$ unseen by the generator, generated data $G$, metric
$T$, rows per sample $n$, and repetition count $N$ (default 10).
Require positive integers $n,N$, $2n\le|H|$ and $n\le|G|$.

1. For each $i=1,\ldots,N$, draw disjoint $n$-row real samples $R_i,R'_i$ from
   $H$ and an independent $n$-row generated sample $G_i$ from $G$, all without
   replacement. Record

   $$
   A_i=T(G_i,R_i),
   \qquad
   C_i=T(R'_i,R_i).
   $$

   Both scores share $R_i$. All rows become available again for each comparison.

2. Estimate the two expected scores by their averages:

   $$
   \widehat\nu_n=\frac1N\sum_{i=1}^{N}A_i,
   \qquad
   \widehat\mu_n=\frac1N\sum_{i=1}^{N}C_i.
   $$

3. Summarize each score distribution with its empirical percentile band:

   $$
   I_{\mathrm{gen}}=\left[q_{.025}(A),q_{.975}(A)\right],
   $$

   $$
   I_{\mathrm{ref}}=\left[q_{.025}(C),q_{.975}(C)\right].
   $$

   Quantiles use linear interpolation.

**Outputs:** the mean estimates $\widehat\nu_n,\widehat\mu_n$, score bands
$I_{\mathrm{gen}},I_{\mathrm{ref}}$, and repeated scores $(A_i,C_i)_{i=1}^{N}$.

::::

(comparison-sample-size)=
TabGenEval chooses the number of rows per sample, $n$, automatically:

- With `uncertainty=None` or `"reference"` mode, use half the real holdout,
  rounded down, capped by the number of generated rows.
- In `"candidate"` or `"both"` mode, use half the real holdout, rounded down,
  if the generated pool has more rows than that size. Otherwise, including
  equality, use half the generated pool, rounded down.

These counts refer to the rows available to the selected metric for sampling.
The generated-real comparison and its real-real reference both use the same $n$.
The [uncertainty options](reference/configuration.md#choose-uncertainty) select
which comparisons to repeat and set $N$.

Reports retain these repeated-score means in the distribution objects and use
them as `value` and `ideal` for the repeated sides. They do not compute an
additional point comparison for those sides. Unrepeated sides retain a single
score. See the [Report API](reference/reports.md#inspect-results)
for those fields.

(reference-confidence-interval)=

(empirical-score-bands)=

(reading-scores-and-bands)=
## Seeing resampling at work

Keep the real population $P=\mathcal N(0,I_3)$ and let the generator produce

$$
Q=\operatorname{Law}(Z,Z,W),
\qquad Z,W\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1).
$$

Every generated column has the correct marginal distribution, while the first
two columns are identical. This gives a singular Gaussian distribution with a
different dependence structure from $P$.

The finite-resolution Shape means are equal because the marginals match:

$$
\nu_n^{\mathrm{Shape}}(P,Q)=\mu_n^{\mathrm{Shape}}(P).
$$

For Trend, the copied pair has generated sample correlation exactly $1$.
Its expected similarity to the real pair is $1/2$, since the real sample
correlation has mean zero. The other two pairs each have the independent-Gaussian
correlation law used above. Therefore,

$$
\nu_n^{\mathrm{Trend}}(P,Q)
=\frac{\tfrac12+2\mu_n^{\mathrm{Trend}}(P)}{3}.
$$

Draw 10,000 real and 10,000 generated rows, and use $n=5{,}000$ rows per sample.
Repeat the matched comparison $N=10$ times, sampling without replacement
within each comparison. The evaluation is
reproducible with:

```python
import numpy as np
import pandas as pd
from tabgen_eval import EvalData, Schema, evaluate

columns = ["x", "y", "z"]
holdout = pd.DataFrame(
    np.random.default_rng(42).standard_normal((10_000, 3)), columns=columns
)
latent = np.random.default_rng(44).standard_normal((10_000, 2))
generated = pd.DataFrame(
    {"x": latent[:, 0], "y": latent[:, 0], "z": latent[:, 1]}
)
data = EvalData(test=holdout, generated=generated, schema=Schema(numeric=columns))
report = evaluate(
    data,
    metrics=["shape", "trend"],
    uncertainty={"mode": "both", "n_replicates": 10},
    seed=43,
    device="cpu",
    dtype="float64",
    n_jobs=2,
)
```

```{raw} html
:file: _generated/methodology/mean-estimates.html
```

Circles mark the repeated-score means, and thin vertical markers show the known
finite-resolution means. Horizontal bars run from the 2.5th to the 97.5th
percentiles of the repeated scores; the table gives their numerical values.

Three known means lie inside their score bands in this run. The generated–real
Trend mean lies below its band.

:::{warning}
These bands describe variability across individual sampled comparisons. They
are not guaranteed to provide 95% confidence intervals for $\mu_n(P)$ or
$\nu_n(P,Q)$, even asymptotically.
:::

See [The Sharp Bits](sharp-bits.md) for further interpretation details.

## References

[^hodges1958]: Hodges, J. L., Jr. (1958).
    [The significance probability of the Smirnov two-sample test](https://archive.ymsc.tsinghua.edu.cn/pacm_download/116/6944-11512_2007_Article_BF02589501.pdf).
    *Arkiv för Matematik*, 3, 469–486. Section 2(c) gives the equal-sample
    reflection formula.
[^scipy-ks]: SciPy documentation.
    [`scipy.stats.ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
    Notes on the exact two-sample Kolmogorov–Smirnov distribution.
[^scipy-pearson]: SciPy documentation.
    [`scipy.stats.pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).
    The exact distribution of the sample correlation under zero correlation.
