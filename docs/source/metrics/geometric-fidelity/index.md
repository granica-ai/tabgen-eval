# Geometric fidelity

Geometric fidelity compares local neighborhoods in a distance space fitted on
sampled real-reference rows.
PRDC reports four complementary summaries: precision, recall, density and
coverage. Precision asks whether generated records fall in real neighborhoods;
recall asks whether generated neighborhoods reach the real records. Density
counts overlapping real neighborhoods, and coverage asks which real
neighborhoods contain generated records.

Compare these scores with their matched real-real reference. For repeated
records and tightly clustered output, also inspect
[Memorization](../memorization/index.md) and [Concentration](../concentration.md).

```{toctree}
:maxdepth: 1

prdc
```
