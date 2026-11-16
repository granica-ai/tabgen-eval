# Nonlinear C2ST

Nonlinear C2ST asks whether a boosted-tree classifier can distinguish generated
observations from real observations. It can detect nonlinear and interaction
effects that a [linear classifier](linear-c2st.md) may miss. For example,
generated measurements may follow a different curved relationship even while
their individual distributions resemble the real data.

## Usage

With [prepared evaluation data](../../reference/data.md), use the XGBoost engine
included in the base installation:

```python
from tabgen_eval import evaluate

report = evaluate(data, metrics=["nonlinear_c2st"])
score = report["nonlinear_c2st/auc"]
print(score.value, score.ideal)
```

The result ranges from $0.5$ to $1$; lower is closer to chance discrimination.
Numeric and categorical columns are supported. Source membership supplies the
classification labels.

## Definition

Let $R_n$ and $G_n$ be the selected real and generated observations. Pool their
features into a table $X$, with source labels $y_i=0$ for real rows and $y_i=1$
for generated rows. Split these rows into $K=3$ scoring folds using the
[shared fold convention](index.md#scoring-convention), and let $\mathcal I_k$
index the pooled rows in held-out fold $k$. Fit the classifier and its feature
preparation on the other two folds, then predict
decision scores $f_k(x)$ for the rows indexed by $\mathcal I_k$.

The score is the mean of the held-out ROC AUCs, each clipped at chance:

$$
T(G_n,R_n)=\frac{1}{K}\sum_{k=1}^{K}
\max\!\left\{\frac12,
\operatorname{AUC}\!\left(y_{\mathcal I_k},f_k(X_{\mathcal I_k})\right)
\right\}.
$$

Clipping happens within each fold before averaging. The
[shared scoring convention](index.md#scoring-convention) explains the matched
real-real reference and its interpretation.

### Fitting and features

The classifier uses XGBoost with the installed version's untuned hyperparameter
defaults and the categorical policy below. Keep the XGBoost version, device and
model settings consistent when comparing generators.

- Numeric features stay unscaled.
- Categorical features use native category-versus-rest splits. Each fit learns
  its vocabulary from both sources in the training folds. Categories first seen
  in the held-out fold are treated as unseen.

Before fitting, numeric observations from both sources are matched to a common
input precision per column. Note that XGBoost uses single-precision feature values
internally.

For GPU setup and device options, see the
[installation guide](../../getting-started/installation.md#nonlinear-c2st-and-mle-gpu-support) and
[execution reference](../../reference/configuration.md#set-execution-options).

CPU and GPU training can produce different trees and AUCs from the same draws
and seed. See
[CPU/GPU score variation](../../sharp-bits.md#xgboost-scores-can-differ-between-cpu-and-gpu).

## Missing values

Rows with missing features remain available for sampling. XGBoost handles
numeric missing values directly. Categorical missingness has a distinct level
when observed in the training folds.

(interpretation-and-unavailable-results)=
## Unavailable results

At least three sampled rows from each source and usable model inputs are
required. An unavailable fold makes the aggregate undefined (`NaN`). The report
stores the reason and fold fit diagnostics.

## References

1. Lopez-Paz, D., and Oquab, M. (2017).
   [Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545).
   *International Conference on Learning Representations*.
2. Chen, T., and Guestrin, C. (2016).
   [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining*.
3. [XGBoost Python Package](https://xgboost.readthedocs.io/en/stable/python/index.html).
   Official package documentation.
