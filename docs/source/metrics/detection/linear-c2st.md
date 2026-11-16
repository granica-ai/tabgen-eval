# Linear C2ST

Linear C2ST asks whether a linear classifier can distinguish generated records
from real records. It fits a regularized logistic classifier and measures its
ROC AUC on held-out rows. For example, small shifts spread across several
measurements may combine into a detectable difference even when each shift is
hard to see alone.

## Usage

With [prepared evaluation data](../../reference/data.md):

```python
from tabgen_eval import evaluate

report = evaluate(data, metrics=["linear_c2st"])

score = report["linear_c2st/auc"]
print(score.value)  # generated-versus-real AUC
print(score.ideal)  # real-versus-real AUC at the same resolution
```

The result `linear_c2st/auc` ranges from $0.5$ to $1$; lower is closer to chance
discrimination. Numeric and categorical columns are supported, without a
supervised target. The two sources supply the labels.

## Definition

Let $R_n$ and $G_n$ be the sampled real and generated datasets. Label real rows
$y=0$ and generated rows $y=1$. Divide these observations into $K=3$ scoring
folds using the [shared fold convention](index.md#scoring-convention).
Let $\mathcal I_k$ contain the row indices in held-out fold $k$, and
$\mathcal I_{-k}$ contain the indices in the other two folds.
For each fold $k$, fit preprocessing and a logistic classifier using only
$\mathcal I_{-k}$, then score $\mathcal I_k$.

For this fit, let $m_k=|\mathcal I_{-k}|$ and let $z_i$ denote row $i$'s features
after applying the preprocessing fitted on $\mathcal I_{-k}$. The classifier
minimizes

$$
\mathcal L_k(w,b)
=\frac{1}{m_k}\sum_{i\in\mathcal I_{-k}}
\left[\log\!\left(1+e^{w^\top z_i+b}\right)
-y_i(w^\top z_i+b)\right]
+\frac{\lambda_k}{2}\lVert w\rVert_2^2.
$$

The intercept $b$ is unpenalized. Linear C2ST uses
$\lambda_k=1/m_k$ in every training fold.

Using the held-out decision scores $f_k(z)=w_k^\top z+b_k$, the reported score is

$$
T_{\mathrm{linear\ C2ST}}(G_n,R_n)
=\frac{1}{K}\sum_{k=1}^{K}
\max\!\left\{\frac12,
\operatorname{AUC}\!\left(y_{\mathcal I_k},f_k(z_{\mathcal I_k})\right)
\right\}.
$$

Clipping happens **within each fold**, before averaging. See the
[shared scoring convention](index.md#scoring-convention) for how the real-real
reference provides context for this score.

### Feature preparation

Each fit learns preprocessing from the real and generated rows in its two
classifier-training folds, then applies it unchanged to the held-out fold:

- Numeric features are standardized.
- Categorical columns are one-hot encoded, with one binary feature per category
  observed in classifier training.

Numeric observations are first matched to a common precision per column,
limited by input dtypes and requested execution precision. This prevents
different storage precision from becoming a source label.

(optimization-accuracy)=
### Optimization

Each training fold automatically selects a solver for the regularized logistic
objective above. Smaller problems use Newton's method with backtracking line
search.

Larger problems use SketchySAGA, a variance-reduced stochastic gradient method.
Its updates use a preconditioner combining a Nyström low-rank approximation with
a diagonal correction.

Solver selection accounts for the encoded feature width, estimated computation
and available memory.

## Missing values

Rows with missing features remain available for sampling. Numeric features are
imputed, with separate binary indicators for missingness observed in classifier
training. These indicators let the classifier detect differences in missingness
between the sources. Missing categorical values form a category when observed
in classifier training.

## Interpreting the result

Finite-sample variation and fold clipping can put a real-real score above $0.5$.
Compare the generated score with the reported real-real reference at the same
sample resolution.

A linear detector can miss differences requiring nonlinear interactions; use
[Nonlinear C2ST](nonlinear-c2st.md) as a complementary detector.

## Results and diagnostics

There must be at least three sampled rows from each source and usable
training features. An unavailable fold makes the aggregate score undefined
(`NaN`).

When a fit misses its optimization stopping criterion, the report retains its
finite AUC and adds `CLASSIFIER_NOT_CONVERGED`. See
[classifier convergence](../../troubleshooting.md#a-classifier-did-not-converge)
for interpreting this diagnostic.

## References

1. Lopez-Paz, D., and Oquab, M. (2017).
   [Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545).
   *International Conference on Learning Representations*.
2. Boyd, S., and Vandenberghe, L. (2004).
   [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).
   Cambridge University Press. Section 9.5 covers Newton's method with
   backtracking line search.
3. Frangella, Z., Rathore, P., Zhao, S., and Udell, M. (2024).
   [PROMISE: Preconditioned Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates](https://www.jmlr.org/papers/v25/23-1187.html).
   *Journal of Machine Learning Research*, 25(346), 1–57. Section 3.3 introduces
   SketchySAGA.
4. Yeon, K., and Anitescu, M. (2025).
   [Beyond Low Rank: Fast Low-Rank + Diagonal Decomposition with a Spectral Approach](https://arxiv.org/pdf/2512.17120).
   arXiv:2512.17120. Background on low-rank-plus-diagonal approximation with
   Nyström sketching.
