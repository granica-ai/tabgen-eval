# Concentration

Concentration compares the local mass of generated data with that of real data
at a fixed radius and at radii fitted from the real reference. It exposes
duplicates, tight clusters and changes in neighborhood density alongside
distributional fidelity.

For example, repeated generated records have many neighbors at a small radius,
even when their individual columns look realistic. A concentration ratio above
one reveals greater local mass than in the real reference; below one means
less. Matching profiles give one.

## Usage

With [prepared evaluation data](../reference/data.md):

```python
from tabgen_eval import evaluate

report = evaluate(data, metrics=["concentration"])
score = report["concentration/mean_radii_1e-3"]
print(score.value, score.ideal)
```

Compare the generated-real ratio with its real-real reference at the same sample
size.

## Local mass and ratios

For row $x_i$ in a pool of $n$ usable rows, local mass at radius $r$ is

$$m_i(r)=\frac{1}{n-1}\sum_{j\ne i}\mathbf1\{d(x_i,x_j)\le r\}.$$

The comparison is inclusive. A row excludes itself by position; another row with
identical values remains a neighbor. Mean mass averages each row's fraction of
neighbors. Upper mass is the 95th percentile of these fractions.

At each selected radius, the reported comparison is

$$\frac{\text{generated local-mass summary}+10^{-8}}
{\text{real local-mass summary}+10^{-8}}.$$

The fixed smoothing constant `1e-8` stabilizes small denominators. Two zero
masses give ratio one. Ratios are nonnegative and uncapped, with a target of one
from either direction.

Fewer than two sampled rows on either comparison side make the summaries
undefined.

## Fixed radii and result keys

Concentration uses a fixed radius `1e-3` plus the 5th and 25th percentiles
of sampled real-reference pair distances:

| Key | Summary |
|---|---|
| `concentration/mean_radii_1e-3` | Mean-mass ratio at fixed radius `1e-3` |
| `concentration/mean_q05`, `concentration/upper_q05` | Mean and upper-mass ratios at the real 5th-percentile radius |
| `concentration/mean_q25`, `concentration/upper_q25` | Mean and upper-mass ratios at the real 25th-percentile radius |

**The fixed radius `1e-3` reports only the mean.** Both fitted quantile radii
retain mean and upper summaries. For example, `upper_q05` compares the 95th
percentiles of local mass at the radius given by the 5th percentile of real pair
distances.

Fitted radii use 16,384 sampled pairs of distinct reference positions; pairs
can recur in this sample. Radius selection is approximate, while neighbor
counts at the selected radii are exact. Each scored comparison fits its geometry
and radii using only its real-reference sample.

## Distance space

Numeric coordinates use the reference minimum and range, with a zero range
replaced by one. Candidate values use that same transform without clipping.
Each categorical mismatch contributes two to squared Euclidean distance,
including a mismatch between two distinct generated-only categories. Categorical
labels retain their typed identities.

## Inspect profiles

The radius table is retained automatically for a generated-real point,
including when `details=False`. Inspect the actual radii, real and generated
mean/upper masses, ratios and usable row counts:

```python
profiles = report.details("concentration", level="radius")
print(profiles)
figure = report.fidelity.plot(kind="concentration")
```

These profiles describe the first generated-real repetition when that side is
repeated, or its single comparison otherwise. They are not repetition averages. The plot
shows the fixed small radius in the mean profile only.

<div id="interpretation"></div>

A radius spanning an entire pool gives mass one and can conceal structural
differences at smaller scales. Inspect the fixed small radius, fitted radii,
actual profile values and row counts together.

(missingness-and-interpretation)=
## Missing values

Records missing any declared distance feature are excluded before sampling or
uncertainty estimation. Ignored columns do not exclude records.
`report.row_usage()` shows excluded counts and percentages for each pool.
