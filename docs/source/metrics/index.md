# Metrics

TabGenEval consists of three metric pillars, each targeting different questions.
Fidelity metrics compare generated records with unseen real observations.
Memorization metrics measure proximity to the generator's training records.
Utility metrics ask how effective the generated data is relative to real data
for downstream machine learning prediction.

| Question | Metric guides |
|---|---|
| Are individual columns and their associations preserved? | [Shape](fidelity/shape.md), [Trend](fidelity/trend.md), [Conditional Fidelity](fidelity/conditional.md) |
| Is local geometric structure preserved? | [PRDC](geometric-fidelity/prdc.md) |
| Can a classifier distinguish the sources? | [Linear C2ST](detection/linear-c2st.md), [Nonlinear C2ST](detection/nonlinear-c2st.md) |
| Does the generator over-concentrate its observations? | [Concentration](concentration.md) |
| Does it reproduce training records? | [DCR](memorization/dcr.md), [Near-match Rate](memorization/near-match-rate.md), [Replication Ratio](memorization/replication-ratio.md) |
| How do generated-trained and real-trained predictors perform and agree? | [MLE](utility/mle.md) |

Each metric guide explains its definition, outputs, missing-data handling,
and interpretation.

## Pillars and selector groups

The three pillars describe what you are evaluating. Selector groups choose
which metrics to compute. The Fidelity pillar spans five selector groups;
`group=["fidelity"]` selects only Shape and Trend.

| Pillar | Selector group | Metrics selected | Data for evaluating generated output |
| --- | --- | --- | --- |
| Fidelity | `fidelity` | `shape`, `trend` | `test`, `generated` |
| Fidelity | `conditional_fidelity` | `cond_shape`, `cond_trend`, `recovery` | `test`, `generated` |
| Fidelity | `geometric_fidelity` | `precision`, `recall`, `density`, `coverage` | `test`, `generated` |
| Fidelity | `detection` | `linear_c2st`, `nonlinear_c2st` | `test`, `generated` |
| Fidelity | `concentration` | `concentration` | `test`, `generated` |
| Memorization | `memorization` | `dcr_ratio`, `near_match_rate`, `replication_ratio` | `train`, `test`, `generated` |
| Utility | `utility` | `mle` | `test`, `generated`, plus `Schema.target` |

Fidelity and utility also support real-only evaluation with `test`; MLE still
requires `Schema.target`. When selected individually, `near_match_rate` and
`replication_ratio` need only `train` and `generated`. Selecting the entire
`memorization` group also selects `dcr_ratio`, which requires `test`.
See [data roles](../reference/data.md#assign-data-roles) for preparing these pools.

To evaluate the full Fidelity pillar:

```python
from tabgen_eval import evaluate

report = evaluate(
    data,
    group=[
        "fidelity",
        "conditional_fidelity",
        "geometric_fidelity",
        "detection",
        "concentration",
    ],
)
```

`report.fidelity` presents the fidelity-related results already computed in your
report. Accessing it does not select or calculate additional metrics. See
[family views](../reference/reports.md#family-views) for the report interface and
[metric selection](../reference/evaluation.md#choose-metrics) for selection rules.

```{toctree}
:hidden:
:maxdepth: 2

fidelity/index
geometric-fidelity/index
detection/index
concentration
memorization/index
utility/index
```
