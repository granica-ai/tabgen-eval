# Machine-learning efficacy (MLE)

MLE compares two XGBoost models: one trained on generated data and one trained
on fresh real data. Both predict the same real test observations.

Supply `test`, `generated`, and a prediction target in `Schema.target`. MLE
draws disjoint real model-training and scoring samples for each comparison.
The generator's actual training records belong to the separate
[memorization evaluation](../memorization/index.md).

## Example

With your [prepared evaluation data](../../reference/data.md), select `mle`:

```python
from tabgen_eval import evaluate

report = evaluate(data, metrics=["mle"])
score = report["mle/score"]
print(score.value, score.ideal)  # Generated-trained and real-trained raw scores
print(report["mle/ratio"].value)
```

For one comparison, the ratio divides the generated-trained score by the
real-trained score. For example:

| Task | Generated-trained score | Real-trained score | Ratio | Interpretation |
| --- | --- | --- | --- | --- |
| Classification | AUROC $0.6$ | AUROC $0.8$ | $0.75$ | Lower AUROC for the generated-trained model |
| Regression | MAE $3$ | MAE $2$ | $1.5$ | $50\%$ more prediction error for the generated-trained model |

**Higher AUROC and AUROC ratios are better; lower MAE and MAE ratios are better.**
A ratio of $1$ in one comparison means equal raw performance. Keep the raw
scores alongside the ratio to retain their scale and meaning.

(learner-and-interpretation)=
## Models and features

The models use the installed XGBoost version's default hyperparameters.
Numeric features stay unscaled. Each model learns categorical vocabularies from
its own training sample and treats categories first seen in scoring data as
unseen.

Numeric observations are matched to a common precision across input roles before
fitting. XGBoost uses single-precision feature values internally.

Keep the XGBoost version, model settings and execution device consistent when
comparing generators. CPU and GPU training can produce different trees and
scores from the same draws and seed; see
[CPU/GPU score variation](../../sharp-bits.md#xgboost-scores-can-differ-between-cpu-and-gpu).
For GPU setup and device selection, see
[installation](../../getting-started/installation.md#nonlinear-c2st-and-mle-gpu-support) and
[execution options](../../reference/configuration.md#set-execution-options).

## Task and outputs

Declare the target categorical for classification or numeric for regression.
If the target is omitted from both feature lists, its real-data dtype determines
the task: numeric, except Boolean, means regression; other dtypes mean
classification. The target is always excluded from predictors.

| Result | Classification | Regression |
| --- | --- | --- |
| `mle/score` | Raw held-out AUROC | Raw held-out MAE |
| `mle/ratio` | Generated AUROC / real AUROC | Generated MAE / real MAE |
| `mle/prediction_agreement` | Fraction of shared-test hard predictions that agree | Not applicable |
| `mle/prediction_mse` | Not applicable | Mean squared difference between shared-test predictions |

For `mle/score`, `value` stores the generated-trained score and `ideal` stores
the real-trained benchmark. With repeated evaluation, these are the finite means
of the scores already computed in the repetitions; no extra point fit or scoring
pass is run. Ratios and prediction comparisons
have no `ideal`. The inactive prediction statistic is retained with `NOT_APPLICABLE`.

Classification uses binary AUROC for two classes and unweighted macro
one-versus-rest AUROC for multiple classes. AUROC stays on its raw $[0,1]$
scale, including below-chance values. Regression uses mean absolute error in
the target's units:

$$
\operatorname{MAE}(y,\widehat y)=\frac{1}{n}\sum_{i=1}^{n}|y_i-\widehat y_i|.
$$

Zero MAE means exact predictions. Constant scoring targets and a single
observed scoring row are valid.

:::{note}
Some sources use $R^2$ to assess regression utility. We prefer MAE because it is
less sensitive to large prediction errors than squared-error measures and
expresses error in the target's original units.
:::

## Raw-score ratio

For generated-trained and real-trained raw scores $s_G$ and $s_R$,

$$
\operatorname{ratio}=\frac{s_G}{s_R}.
$$

Each repeated comparison calculates its own ratio. The distribution's mean
averages the finite ratios, which can differ from dividing the averaged
generated and real scores. This mean is the reported `mle/ratio.value`.
The following rules apply to each comparison:

| Situation | Retained ratio |
| --- | --- |
| Positive real score | Generated score / real score |
| Positive generated score and zero real score | Positive infinity |
| Both scores zero | Undefined (`NaN`) |
| Unsupported or nonfinite score | Undefined, with a diagnostic reason |

A perfect real regression model has MAE zero, so any positive generated MAE
produces an infinite ratio.

## Shared-test prediction comparisons

Classification agreement is the fraction of shared-test rows on which the two
models predict the same class label.
Regression prediction MSE is

$$
\frac{1}{n}\sum_{i=1}^{n}
\left(\widehat y_{G,i}-\widehat y_{R,i}\right)^2.
$$

It measures differences between models' predictions in squared target units.
MAE and AUROC evaluate predictions against observed targets. High agreement or
small prediction MSE can occur between two similarly weak models; read them
alongside the raw predictive scores.

(sampling-and-missing-values)=
## Sampling

MLE chooses equal-sized generated-training, real-training and real-scoring
samples without replacement. The two real samples are disjoint within each
comparison, and both models predict the same real-scoring sample. All eligible rows become
available again for the next comparison, so a real observation can have a
different role in a later draw.
The size of all three roles follows the
{ref}`same sampling rules <comparison-sample-size>` for the selected uncertainty mode.

## Missing values

Records with missing targets are excluded from each pool before sampling.
Rows with missing predictors remain eligible, and XGBoost handles their missing
values. `report.row_usage()` records the exclusions.

## Unavailable results

Classification training needs at least two classes. An AUROC is undefined if
the scoring rows omit a class known to that model or contain an unseen target
label. Prediction comparisons may remain computable when a raw AUROC is
undefined. Failed fits and unavailable scores retain their diagnostic reasons.

## Repeated evaluation

| `uncertainty` | Model fitting and scoring |
| --- | --- |
| `None` | Fit each model once and retain point statistics |
| `{"mode": "reference"}` | Keep the generated model fixed; refit the real model on fresh real training draws and score both on each shared real scoring draw |
| `{"mode": "candidate"}` | Use the same model fits and draws as `"both"`, retaining generated score, ratio and prediction-comparison bands without a real-trained score band |
| `{"mode": "both"}` | Refit both models on fresh matched training draws and score them on the same real scoring draw |

`n_replicates` sets the repetition count, defaulting to 10. Each repetition
draws fresh real training and scoring samples without replacement. Candidate
and both modes also redraw generated training rows. Each repetition retains
the raw scores, their ratio and the prediction comparison.

Reference mode rescores the single generated-trained point model on each fresh
real-scoring sample. Both raw scores, their ratio and prediction comparisons
can therefore vary in this mode.

The 2.5th and 97.5th percentiles of finite repeated scores form a descriptive
score band. It describes variation across the sampled comparisons; it is not a
confidence interval for the population mean score. The sample mean is the
reported score and is also retained in the corresponding distribution.

Switching to `"candidate"` or `"both"` can change the selected resolution.
Changing the repetition count can change the reported means and bands; existing
repetitions keep their source draws for the same seed and resolution.
See [uncertainty options](../../reference/configuration.md#choose-uncertainty)
and [score-band interpretation](../../methodology.md#seeing-resampling-at-work).

## Test-only evaluation

With only a real `test` pool, MLE fits a predictor and scores disjoint real rows
in each comparison. Its sole statistic is raw MAE or AUROC under `score`:

```python
from tabgen_eval import EvalData

reference = evaluate(
    EvalData(test=data.test, schema=data.schema),
    metrics=["mle"],
    uncertainty={"mode": "reference"},
)
raw_scores = reference["mle"].reference_distribution.samples
mean = reference["mle"].reference_distribution.mean
```

Without uncertainty, `value` contains one real-trained score. With repeated
evaluation, it contains the mean of the finite real-trained scores. The
individual scores are retained in `reference_distribution`, with their
percentile band in `ideal_ci_low` and `ideal_ci_high`.

## References

1. Chen, T., and Guestrin, C. (2016).
   [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining*.
2. [XGBoost Python Package](https://xgboost.readthedocs.io/en/stable/python/index.html).
   Official package documentation.
