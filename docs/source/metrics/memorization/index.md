# Memorization

These metrics compare generated records with the actual data used to train the
generator. They measure different aspects of proximity:

| Metric | Query direction | Question |
| --- | --- | --- |
| [DCR ratio](dcr.md) | Generated and test → training | Are generated records closer to training than independent real records? |
| [Near-match rate](near-match-rate.md) | Generated → training | What fraction of generated rows are at a small distance from the training set? |
| [Replication ratio](replication-ratio.md) | Training → generated | What fraction of training rows are at a small distance from the generated set? |

All three use exact distances and return point-only results. DCR ratio requires
independent `test` data for its denominator and includes distance percentiles.
The two rates need only `train` and `generated`; their geometry and values
depend on these two pools.

Near-match and replication use the same fixed inclusive Euclidean threshold in
the [training-fitted distance space](dcr.md#distance-space). Repeated generated
copies each contribute to near-match rate, while a training row counts only
once toward replication ratio. Dense or discrete real populations can naturally
produce close records, so these measurements need population context.

```{toctree}
:maxdepth: 1

dcr
near-match-rate
replication-ratio
```
