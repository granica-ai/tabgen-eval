# The Sharp Bits

This guide reviews some of the nuanced aspects of evaluation using **TabGenEval**.
We explain choices that can change the question being asked, or make the answer
easy to misinterpret. For unexpected output and practical remedies, see
[Troubleshooting](troubleshooting.md).

(a-reference-comparison-supplies-context)=
## What the real reference tells you

The real-real score gives you a yardstick at the same sample size as your
generated-real comparison. Matching that reference does not establish that the
generated and real distributions are identical. See the
[statistical methodology](methodology.md#the-finite-sample-yardstick).

A generator can also score well while copying its training records. Use
[memorization metrics](metrics/memorization/index.md) to check for copying.

## Scores depend on comparison size

Expected scores can change with sample size even when the underlying
distributions remain the same. Compare generators at a consistent resolution,
and inspect each statistic's `n_eff`, for example `report["shape"].n_eff`, to see
how many rows were selected per comparison role.

Available pool sizes and the uncertainty mode can change that resolution.
Changing the uncertainty mode can therefore change the point scores as well as
the reported bands. The
[statistical methodology](methodology.md#the-finite-sample-yardstick) explains
the finite-sample reference.

(bootstrap-intervals-depend-on-the-supplied-pools)=

(repeated-scores-describe-variation-within-the-supplied-pools)=
## Repeated evaluations reuse the same data

Increasing the repetition count makes the reported mean and bands more stable.
Every repetition still draws from the same supplied rows.

The generator is not retrained, so the bands do not show how results might vary
between different training runs. See the
{ref}`statistical methodology <resampling>` for how the bands are constructed
and interpreted.

## Aggregates can hide important components

- **Averages can hide a weak component.** For example, 99 perfect Shape column
  scores and one zero score average to $0.99$.
- **Undefined components can be left out.** Check which components contributed
  to the score, since omitted components change what the average describes.

Use `details=True` to inspect individual columns, pairs and slices; see
[component tables](reference/reports.md#component-tables).

(data-roles-and-sample-resolution-matter)=
## Which real data you use matters

Comparing generated data with the records used to train the generator can make
fidelity scores look better than they would against unseen real data.

For memorization checks, use the generator's actual training records. Using a
different real dataset changes what the scores tell you about copying.

See [data roles](reference/data.md#assign-data-roles) for how to supply these tables.

## Preparation changes what is being compared

- **Small differences in category labels count.** `" A "` and `"A"`, or `1` and
  `"1"`, are treated as different categories. Check for unintended changes in
  labels or types.

- **A good score can hide different missing-value rates.** Shape compares the
  values present in each column. Two columns can score well even when one has
  many more missing entries. Check [missing-value counts](data-diagnostics.md#what-to-look-for)
  alongside the scores.

- **Different metrics can use different rows.** Distance metrics exclude
  incomplete records; MLE excludes records with missing targets. Use
  `report.row_usage()` to check exclusions.

- **Distance cutoffs depend on the scale.** Cutoffs apply after scaling numeric
  columns and encoding categories. Changing column roles or real-data ranges
  can change which records count as close. See the
  [DCR distance space](metrics/memorization/dcr.md#distance-space).

## Numerical precision affects results

Rounding can change decisions near a boundary. For example,
[PRDC](metrics/geometric-fidelity/prdc.md#definition) uses strict neighborhood
inequalities, so ties and precision affect which points count as inside. Keep
the numerical precision consistent when comparing results.

## Convergence diagnostics describe fitting

- **Linear C2ST:** a finite score remains available when fitting misses its
  stopping criterion. Inspect the
  [termination diagnostics](metrics/detection/linear-c2st.md#interpreting-the-result).
- **XGBoost:** `converged` records completion of the fixed boosting budget. See
  [Nonlinear C2ST](metrics/detection/nonlinear-c2st.md) and
  [MLE](metrics/utility/mle.md) for fit diagnostics.

## XGBoost scores can differ between CPU and GPU

[Nonlinear C2ST](metrics/detection/nonlinear-c2st.md) and
[MLE](metrics/utility/mle.md) fit XGBoost histogram models. CPU and GPU training
can learn different trees from the same observations and seed, changing C2ST
AUC, MLE scores and ratios. XGBoost uses single-precision features internally,
including when `dtype="float64"` is selected.

The table shows the largest observed absolute CPU/GPU difference for each
dataset, across four sample sources and fixed generated-real and real-reference
results where available.

| Metric / statistic | Video | Hotel | Allstate | Covertype | Mercari |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear C2ST AUC | 0.00588194 | 0.00689792 | 0.00196484 | 0.00395355 | 0.00870624 |
| MLE score (AUROC / MAE) | 0.00320756 | 0.00438361 | 0.00203622 | 0.000833276 | 1.07626e-07 |
| MLE ratio | 0.0495343 | 0.00294657 | 0.0030684 | 0.000812203 | 1.79422e-07 |
| MLE prediction agreement | N/A | 0.006141 | N/A | 0.00187273 | N/A |
| MLE prediction MSE | 0.000740841 | N/A | 0.00339434 | N/A | 9.31323e-09 |

Video is Video Transcoding Time Prediction; Hotel is Hotel Booking Demand;
Allstate is Allstate Claims Severity; Mercari is Mercari Price Suggestion.
MLE score is AUROC for Hotel and Covertype and MAE for the regression datasets.
Prediction MSE has squared target units. N/A marks a statistic that does not
apply to that prediction task. Values with different units should not be pooled.

Use the same device and model settings when comparing generators. With
`device="auto"`, check `report.provenance["metric_execution"]` for the device
actually used.
