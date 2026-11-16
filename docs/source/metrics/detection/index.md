# Detection

Classifier two-sample tests ask whether a fitted model can distinguish real
from generated records on held-out observations. Source membership supplies
the labels; a supervised target is unnecessary.

| Detector | What it can reveal |
| --- | --- |
| [Linear C2ST](linear-c2st.md) | Differences captured by a weighted combination of prepared features |
| [Nonlinear C2ST](nonlinear-c2st.md) | Differences captured by nonlinear relationships and interactions |

## Scoring convention

Both detectors use three scoring folds, stratified by source label.
Each fit learns its transformations and model on the other folds,
then reports ROC AUC on the
held-out rows. Fold AUCs are clipped below at $0.5$ before averaging. The final
score ranges from $0.5$ to $1$; lower means closer to chance discrimination.

Compare with the real-real reference at the same sample resolution:
finite samples and fold clipping can put that reference above $0.5$. The
reported `n_eff` counts sampled rows per source before cross-validation.
[Methodology](../../methodology.md) explains the reference and optional
percentile bands describing repeated score variability.

A chance-level score means this fitted detector found little separation at the
selected sample size. Interpret it alongside the other fidelity and
memorization metrics.

```{toctree}
:maxdepth: 1

linear-c2st
nonlinear-c2st
```
