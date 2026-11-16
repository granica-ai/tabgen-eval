# Predictive utility

[Machine-learning efficacy (MLE)](mle.md) asks whether generated data train a
useful predictor for a declared target. It compares a generated-trained model
with a fresh real-trained model on the same real scoring rows.

MLE reports raw AUROC for classification or MAE for regression, their
generated-to-real ratio, and a comparison of the models' predictions. AUROC and
its ratio prefer higher values; MAE and its ratio prefer lower values. See the
metric guide for examples, target requirements, and repeated model evaluation.

```{toctree}
:maxdepth: 1

mle
```
