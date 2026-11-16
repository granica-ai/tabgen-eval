"""Command-line access to unified evaluation with explicit input roles.

tabgen-eval --test a.csv --generated b.csv --schema s.json --metrics shape trend
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from os import PathLike
from typing import Literal

import pandas as pd

from ._options import UncertaintyOptions, resolve_options
from .data import EvalData
from .evaluate import evaluate
from .schema import Schema


def _read_table(
    path: str | PathLike[str], fmt: Literal["auto", "csv", "parquet"] = "auto"
) -> pd.DataFrame:
    """Read a table with pandas' ordinary parsing and missing-value recognition.

    ``auto`` selects Parquet for ``.parquet``/``.pq`` suffixes and CSV otherwise.
    No columns are converted after parsing. Call pandas directly when custom CSV
    dtypes, missing tokens, or parser options are needed to preserve source values.
    Reader and optional Parquet dependency errors propagate to the caller.
    """
    if fmt == "auto":
        fmt = "parquet" if str(path).endswith((".parquet", ".pq")) else "csv"
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)


def _positive_int(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("n_replicates must be a positive integer")
    return count


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tabgen-eval", description=__doc__)
    p.add_argument("--train", help="actual generator training CSV/Parquet")
    p.add_argument("--test", "--holdout", dest="test", help="independent real test CSV/Parquet")
    p.add_argument("--generated", help="generated (synthetic) CSV/Parquet")
    p.add_argument("--schema", required=True, help="schema JSON (numeric/categorical/target)")
    p.add_argument("--metrics", nargs="*", help="registered metric names; overrides --group")
    p.add_argument("--group", nargs="*", help="registered groups (default: all compatible metrics)")
    p.add_argument("--details", action="store_true", help="retain point comparison detail tables")
    p.add_argument("--out", help="write report to this path (.md / .json by extension)")
    p.add_argument("--format", choices=["summary", "markdown", "json"], default="summary")
    p.add_argument(
        "--include-replicates",
        action="store_true",
        help="include ordered raw score samples in JSON output",
    )
    defaults = resolve_options()
    p.add_argument(
        "--uncertainty",
        choices=["none", "reference", "candidate", "both"],
        default=defaults["uncertainty"],
        help="fixed paired scores, reference variability, candidate variability, or both",
    )
    p.add_argument(
        "--n-replicates",
        type=_positive_int,
        default=defaults["n_replicates"],
        help="replicate count for uncertainty modes (default: 10)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=defaults["seed"],
        help="nonnegative random seed for reproducible sampling",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=defaults["n_jobs"],
        help="CPU worker budget (default: 8): 1 serial, -1 all cores",
    )
    p.add_argument(
        "--device", default=defaults["device"], help="torch device: auto, cpu, or cuda[:index]"
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=defaults["dtype"],
        help="numeric precision for tensor metrics (default: float64)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    with open(args.schema) as fh:
        schema = Schema.from_dict(json.load(fh))

    # Keep the constructor's metric discovery summary separate from report
    # output, especially when stdout is consumed as JSON or Markdown.
    with redirect_stdout(sys.stderr):
        data = EvalData(
            train=_read_table(args.train) if args.train else None,
            test=_read_table(args.test) if args.test else None,
            generated=_read_table(args.generated) if args.generated else None,
            schema=schema,
        )

    uncertainty: UncertaintyOptions | None = (
        None
        if args.uncertainty == "none"
        else {"mode": args.uncertainty, "n_replicates": args.n_replicates}
    )
    report = evaluate(
        data,
        group=args.group,
        metrics=args.metrics,
        uncertainty=uncertainty,
        seed=args.seed,
        n_jobs=args.n_jobs,
        device=args.device,
        dtype=args.dtype,
        details=args.details,
    )

    if args.out:
        if args.out.endswith(".json"):
            report.to_json(args.out, include_replicates=args.include_replicates)
        else:
            report.to_markdown(args.out)
        print(f"wrote {args.out}")
    else:
        if args.format == "markdown":
            print(report.to_markdown())
        elif args.format == "json":
            print(report.to_json(include_replicates=args.include_replicates))
        else:
            print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
