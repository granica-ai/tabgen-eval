"""Host-visible descriptive tables collected during one metric evaluation.

Fidelity backends return array tables with positional feature and slice IDs.
Report adapters attach names and format labels; execution never constructs
DataFrames, assigns quality flags, or estimates uncertainty for these tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class DetailTable:
    """Equal-length named column arrays and table-specific descriptive metadata.

    Arrays belong to the returned outcome, rather than numerical scratch storage.
    Feature indices use numeric columns followed by categorical columns. Category
    codes address the retained typed vocabulary for their conditioning column.
    """

    columns: dict[str, NDArray[np.generic]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = set()
        for name, values in self.columns.items():
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(f"Detail column {name!r} must be one-dimensional")
            lengths.add(len(array))
            self.columns[name] = array
        if len(lengths) > 1:
            raise ValueError("Detail columns must have equal lengths")


@dataclass
class FidelityDetails:
    """Descriptive tables for one fixed Shape, Trend, or Conditional comparison.

    Table keys are ``columns`` for column scores, ``pairs`` for global Trend,
    and ``slices`` for conditional subpopulations. Metadata retains feature names,
    typed category vocabularies and the scope of the evaluated point comparison.
    No table contains bootstrap draws or per-entry uncertainty estimates.
    """

    tables: dict[str, DetailTable] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ConcentrationDetails:
    """Raw local-mass summaries at shared radii for one point comparison.

    The ``radii`` table contains reference/candidate mean and upper-quantile
    masses, signed gaps and complete-row counts. It contains no slice intervals.
    """

    tables: dict[str, DetailTable] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class DetailedOutcome:
    """Aggregate values and their breakdown from the same numerical execution."""

    values: dict[str, float]
    details: FidelityDetails | ConcentrationDetails
