# PRDC

PRDC measures how generated records fit the local geometry of real records.
It reports precision, recall, density and coverage using neighborhoods whose
radii are determined by nearest neighbors within each dataset.

For example, generated records concentrated in one real-data region can have
good precision while missing other regions, reducing recall. These complementary
scores help distinguish realistic records from broad coverage of the real data.

## Usage

With [prepared evaluation data](../../reference/data.md), select the whole
geometric fidelity group:

```python
from tabgen_eval import evaluate

report = evaluate(data, group=["geometric_fidelity"])

for name in ("precision", "recall", "density", "coverage"):
    score = report[name]
    print(name, score.value, score.ideal)
```

Select individual metrics with, for example, `metrics=["precision", "recall"]`.
Numeric and categorical columns are supported; no prediction target is required.

## Scores

| Report key | Meaning | Range |
| --- | --- | --- |
| `precision/precision` | Fraction of generated records inside at least one real neighborhood | $[0,1]$ |
| `recall/recall` | Fraction of real records inside at least one generated neighborhood | $[0,1]$ |
| `density/density` | Average number of real neighborhoods containing a generated record, divided by the fixed neighbor rank $k=5$ | $[0,\infty)$ |
| `coverage/coverage` | Fraction of real neighborhoods containing at least one generated record | $[0,1]$ |

All four statistics report higher values as better. Compare them with the
matched real-real reference: density can exceed $1$ when generated records lie
in overlapping real neighborhoods, so assess its size relative to that reference.
A generator that copies real records can also score well; use
[memorization metrics](../memorization/index.md) to investigate copying.

## Visualizing PRDC

```python
report.fidelity.plot(
    kind="scores", metrics=["precision", "recall", "density", "coverage"]
).show()
```

The [visual walkthrough](../../getting-started/visualization.md) shows all four
scores alongside their real-real reference and descriptive percentile bands:

```{raw} html
:file: ../../_generated/visualization/prdc.html
```

PRDC provides aggregate scores.

## Distance space

Each scored comparison fits its distance space on its sampled real-reference
rows, then applies it to both sides.
Distances are Euclidean in the transformed space. For numeric column $j$, let
$a_j$ and $b_j$ be its minimum and maximum in that real-reference sample.
Transform both datasets using

$$
\phi_j(x)=\frac{x_j-a_j}{s_j},
\qquad
s_j=\begin{cases}b_j-a_j,&b_j>a_j,\\1,&b_j=a_j.\end{cases}
$$

Generated values are not clipped to the real range. Each categorical column
uses a full set of indicator coordinates fitted on the real categories.
A generated category absent from the real vocabulary has all-zero coordinates
for that column. Vocabulary matching distinguishes labels such as `1` and `"1"`
and preserves text exactly; see [input preparation](../../reference/data.md).

Write $D(x,y)=\lVert\phi(x)-\phi(y)\rVert_2$. Consequently, distances depend
on the declared column roles and the observed real ranges. Two different known
categories contribute $2$ to squared distance; a known and an unknown category
contribute $1$. Two unknown values have identical all-zero coordinates for that
column, even if their original labels differ.

## Definition

For one comparison sampled from the complete rows, let $R=\{r_1,\ldots,r_m\}$ and
$G=\{g_1,\ldots,g_q\}$. For fixed neighbor rank $k=5$, let $\rho_i$ be the
distance from real record $r_i$ to its $k$th other real neighbor, and let
$\gamma_j$ be the corresponding within-generated distance for $g_j$.
The record itself is excluded; repeated records at other row positions remain
valid neighbors.

Let $\mathbf 1\{\cdot\}$ denote an indicator. The four statistics are

$$
\operatorname{Precision}
=\frac{1}{q}\sum_{j=1}^{q}
\mathbf 1\!\left\{\exists i:\ D(r_i,g_j)<\rho_i\right\},
$$

$$
\operatorname{Recall}
=\frac{1}{m}\sum_{i=1}^{m}
\mathbf 1\!\left\{\exists j:\ D(r_i,g_j)<\gamma_j\right\},
$$

$$
\operatorname{Density}
=\frac{1}{kq}\sum_{j=1}^{q}\sum_{i=1}^{m}
\mathbf 1\!\left\{D(r_i,g_j)<\rho_i\right\},
$$

$$
\operatorname{Coverage}
=\frac{1}{m}\sum_{i=1}^{m}
\mathbf 1\!\left\{\min_jD(r_i,g_j)<\rho_i\right\}.
$$

Neighborhood membership uses **strict** inequalities. A point exactly on the
radius does not count as inside. If a neighborhood radius is zero because of
duplicates, no point lies strictly inside it, even at distance zero. Thus
identical datasets do not necessarily score $1$ under ties or zero radii.

(missing-values-and-limitations)=
## Missing values

PRDC excludes rows missing any declared distance feature before sampling or
uncertainty estimation. Ignored columns do not affect eligibility.
`report.row_usage()` shows excluded counts and percentages for each pool.

## Limitations

- Precision, density and coverage require more than five sampled reference rows;
  recall requires more than five sampled candidate rows. Each also needs a
  nonempty opposite side. Public evaluation uses matched draws, so a per-side
  resolution of at most five makes all four scores undefined.
- PRDC uses Euclidean nearest-neighbor distances after numeric scaling and
  categorical encoding. This distance may not reflect which differences matter
  most for your data. High-dimensional geometry, range estimates and the fixed
  neighbor rank also affect its meaning. Use PRDC alongside marginal checks such as
  [Shape](../fidelity/shape.md) and relationship checks such as
  [Trend](../fidelity/trend.md).
- Ties and finite precision near a neighborhood boundary can change membership.
  Keep column roles and sample resolution consistent across comparisons.

## References

1. Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., and Aila, T. (2019).
   [Improved Precision and Recall Metric for Assessing Generative Models](https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html).
   *Advances in Neural Information Processing Systems*, 32.

2. Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., and Yoo, J. (2020).
   [Reliable Fidelity and Diversity Metrics for Generative Models](https://proceedings.mlr.press/v119/naeem20a.html).
   *Proceedings of the 37th International Conference on Machine Learning*,
   PMLR 119, 7176–7185.
