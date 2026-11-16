# Command-line evaluation

Use `tabgen-eval` to evaluate CSV or Parquet files. Supply the input roles and
save your column declarations in a schema JSON file, for example `schema.json`:

```json
{"numeric": ["age", "income"], "categorical": ["region"], "target": null}
```

See [Prepare evaluation data](data.md#describe-your-columns) for column meanings
and input roles.

## Evaluate files

Compute fixed fidelity scores from CSV files:

```bash
tabgen-eval --test real_test.csv --generated generated.csv --schema schema.json \
  --metrics shape trend
```

Request repeated scores for both comparisons from Parquet files and save them:

```bash
tabgen-eval --test real_test.parquet --generated generated.parquet --schema schema.json \
  --metrics shape trend --uncertainty both --n-replicates 25 \
  --include-replicates --out evaluation.json
```

Parquet input uses pandas with PyArrow, included in the base installation.

`--metrics` selects named metrics; `--group` selects whole families. A metric
selection overrides the group selection. Without either filter, all compatible
metrics are selected. With test data, these include MLE, which requires a
declared target. See
[Evaluate metrics](evaluation.md) for metric names and required data roles.

Add `--train training.csv` to include the generator's actual training data.
Test-only evaluation supplies only `--test`. `--uncertainty` accepts `none`
(the default), `reference`, `candidate`, or `both`. `--n-replicates` sets the
number of repeated comparisons and defaults to 10; it must be a positive integer.
Each comparison samples rows without replacement. The resulting percentile
bands describe repeated score variability and are not confidence intervals for
the population expected score. MLE follows its
[training and scoring protocol](../metrics/utility/mle.md#repeated-evaluation).
See [uncertainty options](configuration.md#choose-uncertainty) for sampling behavior.
Use `--uncertainty candidate` for generated-real intervals without reference intervals.

## Read and save results

The default output is a printed summary. Use `--format markdown` or
`--format json` to send those formats to stdout. `--out evaluation.json` writes
JSON; `--out evaluation.md` writes Markdown. Add `--include-replicates` to retain
each distribution's ordered scores in JSON output. Failure diagnostics remain
in provenance when available.

Run `tabgen-eval --help` for the full option list, including execution controls.
