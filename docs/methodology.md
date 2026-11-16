# Statistical methodology

The illustrated high-level explanation lives in the
[public methodology guide](source/methodology.md). Per-metric definitions and
limitations live in the [metric catalogue](source/metrics/index.md).

Paired evaluation compares generated–real and disjoint real–real observations at
matched resolution. With no UQ or `"reference"`, use
`min(eligible_generated, eligible_test // 2)` rows per role. `"both"` uses
`eligible_test // 2` when the generated pool is larger than that size, and
`eligible_generated // 2` otherwise, including equality. Point estimates and
repetitions share that resolution. Changing the mode can change point scores;
changing the repetition count preserves the fixed points.
Distance procedures first retain complete declared features;
MLE retains observed targets. Fidelity retains full-pool draws and its existing
observed-value rules. Source exclusions and percentages are recorded separately
from finite score replicates. All retained source rows remain eligible later. Point estimates
and repeated draws use independent seeded streams.

For ordinary metrics, fixed evaluation retains one generated-real score and a
matched real-real point. Uncertainty requests fresh comparisons sampled without
replacement from the same eligible pools. Each real-real comparison uses
disjoint source positions, and all rows become available again for later draws.
`n_replicates` sets the repetition count, defaulting to 10. Reports retain raw
score vectors and finite summaries, including their means and 2.5th/97.5th
percentiles. The percentile bands describe variation across comparisons; they
are not confidence intervals for $\mu_n(P)$ or $\nu_n(P,Q)$.
Nonfinite outcomes retain their positions, counts and reasons. Reports attach
no automatic comparison labels or significance decisions.

MLE draws real training and scoring samples that are disjoint within each
comparison. Both memberships can change across repetitions.
Its generated-trained and real-trained models predict the same scoring rows.
Each comparison retains raw generated and real-trained scores and their ratio.
Reference mode retains the single generated-trained point model, while candidate
and both modes refit it. The repeated-score mean ratio can differ from the
quotient of the raw-score means.
MAE and its ratio are lower-is-better;
AUROC and its ratio are higher-is-better. Positive scores over zero give an
infinite ratio; zero over zero is undefined. Constant regression truth and a
single scoring row remain valid for MAE.
Classification prediction agreement and regression prediction MSE reuse those
fits and scoring rows. Test-only evaluation stores raw MAE/AUROC only. See the
[MLE protocol](source/metrics/utility/mle.md).

Memorization uses actual generator-training data and remains exact and point-only.
[DCR](source/metrics/memorization/dcr.md) retains generated and independent holdout
percentiles in the same training-fitted geometry. [Near-match rate](source/metrics/memorization/near-match-rate.md)
and [replication ratio](source/metrics/memorization/replication-ratio.md) use the
fixed inclusive Euclidean threshold `1e-3`, with generated and training denominators
respectively. They cannot alone establish privacy.

[Concentration](source/metrics/concentration.md) reports smoothed candidate/reference
local-mass ratios with fixed `epsilon=1e-8`. Fixed-radius keys identify actual
values, and fixed radius `1e-3` reports only the mean. Raw captured profiles and
counts make the aggregate ratios inspectable without inventing component bands.
