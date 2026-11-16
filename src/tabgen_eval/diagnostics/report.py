"""A standalone descriptive report, separate from evaluation results."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..schema import Schema
from .profiling import CategoricalSupport, InputProfile, NumericSupport, StructureObservations


@dataclass
class DiagnosticReport:
    """Raw input profiles and support observations relative to full training data.

    Counts and rates describe observations. No overall quality score, preferred
    input-size ratio, uncertainty estimate, or evaluator metric is implied.
    """

    schema: Schema
    inputs: dict[str, InputProfile]
    structure: StructureObservations
    categorical_support: dict[str, CategoricalSupport]
    numeric_support: dict[str, NumericSupport]

    def to_dict(self) -> dict[str, object]:
        """Return independent JSON-safe report data."""
        return copy.deepcopy(
            {
                "schema_version": 1,
                "report_type": "data_diagnostics",
                "schema": self.schema.to_dict(),
                "inputs": self.inputs,
                "structure": self.structure,
                "categorical_support": self.categorical_support,
                "numeric_support": self.numeric_support,
            }
        )

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize observations, representing unavailable rates as null."""
        output = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(output, encoding="utf-8")
        return output

    def to_dataframe(self) -> pd.DataFrame:
        """Return profiles and support observations in a tidy, inspectable table."""
        rows = []
        for role, observations in self.inputs.items():
            for check, value in observations.items():
                if check != "columns":
                    rows.append(
                        {
                            "section": "inputs",
                            "input": role,
                            "column": None,
                            "check": check,
                            "value": value,
                        }
                    )
            for column in observations["columns"]:
                for check in ("dtype", "n_missing"):
                    rows.append(
                        {
                            "section": "inputs",
                            "input": role,
                            "column": column["name"],
                            "check": check,
                            "value": column[check],
                        }
                    )
        for check, value in self.structure.items():
            rows.append(
                {
                    "section": "structure",
                    "input": None,
                    "column": None,
                    "check": check,
                    "value": value,
                }
            )
        for section in ("categorical_support", "numeric_support"):
            for column, observations in getattr(self, section).items():
                for check, value in observations.items():
                    rows.append(
                        {
                            "section": section,
                            "input": "generated",
                            "column": column,
                            "check": check,
                            "value": value,
                        }
                    )
        return pd.DataFrame(rows, columns=["section", "input", "column", "check", "value"])

    def summary(self) -> str:
        """Summarize factual counts without assigning a data-quality judgment."""
        lines = ["Data diagnostics (support compared with full training data)"]
        for role, profile in self.inputs.items():
            lines.append(
                f"{role}: {profile['n_rows']} rows, {profile['n_columns']} columns; "
                f"{profile['n_missing_cells']} missing cells, "
                f"{profile['n_duplicate_rows']} duplicate rows"
            )
        unseen = sum(column["n_unseen_cells"] for column in self.categorical_support.values())
        lines.append(f"Generated categorical cells absent from training support: {unseen}")
        return "\n".join(lines)
