# Data diagnostics

Use `diagnose` to inspect missingness, duplication and support differences
between generated data and the **full data used to train the generator**.
It returns a `DiagnosticReport` with observations you can inspect and export.

## Run diagnostics

```python
from tabgen_eval import diagnose

diagnostics = diagnose(train=training, generated=generated, schema=schema)
print(diagnostics.summary())
```

Both inputs must be pandas DataFrames. You can supply a
[Schema](reference/data.md#describe-your-columns) to declare column meanings.
If omitted, it is inferred from training; inspect `diagnostics.schema` to check
those inferred roles. Diagnostics leave the supplied rows and values unchanged.

## What to look for

- **Input profiles:** row and column counts, missing cells and records,
  duplicate records, and per-column dtypes and missing counts.
- **Structure differences:** duplicate column names, missing or extra generated
  columns, and different column ordering.
- **Categorical support:** generated labels absent from training, including
  counts of affected cells and distinct unseen labels.
- **Numeric support:** training and generated ranges, generated values outside
  the training range, nonfinite values, and cells that cannot be interpreted
  numerically.

Use actual training data for these support checks. A smaller real test pool can
omit a category that the model saw during training; generated occurrences of
that category should not be treated as unseen by the model.

## Inspect and export

`summary()` gives a brief overview. Use `to_dataframe()` to inspect individual
observations, including support differences for each column:

```python
observations = diagnostics.to_dataframe()
print(observations)
diagnostics.to_json("data-diagnostics.json")
```

`to_dict()` provides the observations as a dictionary. `to_json()` returns JSON
and writes it to a file when a path is supplied.
