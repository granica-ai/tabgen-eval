"""Interactive Plotly figures for stored reports and explicitly supplied tables.

``plot_report`` never evaluates a metric. ``plot_column`` and
``plot_column_pair`` are separate exploratory utilities taking source tables;
reports do not retain those tables. All functions return figures without
opening a browser or selecting a notebook renderer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from plotly.graph_objects import Figure

    from tabgen_eval.report import Report
    from tabgen_eval.schema import Schema

__all__ = ["plot_report", "plot_column", "plot_column_pair"]


def plot_report(report: Report, *, kind: str = "scores", **kwargs: object) -> Figure:
    """Build a figure from scores and details already present in ``report``.

    Supported kinds:

    * ``scores``: independent panels with overlaid comparisons and requested
      bands; optionally select ``metrics`` by name or list of names. Set
      ``summary='mean'`` for retained repetition means instead of stored points.
    * ``columns``: stored column scores; accepts ``metric='shape'``, ``stat``
      and optional ``top_n`` (lowest finite scores first).
    * ``pairs``: Trend heatmap; accepts ``metric='trend'`` and an optional
      ``columns`` subset containing at least two stored column names.
    * ``slices``: Conditional scores; requires conditioning ``column`` and
      accepts ``stat='shape'`` (or ``trend`` / ``recovery``), ``top_n`` and
      ``metric='cond_shape'`` (or the corresponding selected component).
    * ``concentration``: real/generated point profiles at stored radii.
    * ``utility``: stored raw predictive scores alongside generated-to-real
      raw-score ratios; accepts ``metrics``.

    Fidelity breakdowns require evaluation with ``details=True``. Missing
    tables raise an informative error; plotting does not rerun computation.
    Returned Plotly figures can be customized or saved with ``write_html``.
    """
    from .concentration import concentration
    from .fidelity import columns, pairs, slices
    from .scores import scores
    from .utility import utility

    builders = {
        "scores": scores,
        "columns": columns,
        "pairs": pairs,
        "slices": slices,
        "concentration": concentration,
        "utility": utility,
    }
    if kind not in builders:
        raise ValueError(f"Unknown plot kind {kind!r}; choose from {list(builders)}")
    return builders[kind](report, **kwargs)


def plot_column(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame,
    column: str,
    *,
    schema: Schema,
    max_points: int = 5000,
    max_categories: int = 25,
    seed: int = 0,
) -> Figure:
    """Compare one column's ECDFs or categorical proportions.

    Roles must be declared by ``schema``; ``Schema.infer`` is available as an
    explicit helper. Input values and typed categorical labels are preserved.
    Numeric ECDFs use at most ``max_points`` sampled observed values per source;
    categorical proportions aggregate all rows and group less frequent labels
    beyond ``max_categories``. Missing counts and sampling sizes are retained
    in figure metadata. Neither input table is modified.
    """
    from .distributions import column_figure

    return column_figure(
        real_data,
        generated_data,
        column,
        schema=schema,
        max_points=max_points,
        max_categories=max_categories,
        seed=seed,
    )


def plot_column_pair(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame,
    column_1: str,
    column_2: str,
    *,
    schema: Schema,
    max_points: int = 3000,
    max_categories: int = 20,
    seed: int = 0,
) -> Figure:
    """Compare two columns using SVG scatter, mixed boxes or categorical heatmaps.

    Numeric and mixed plots deterministically sample at most ``max_points``
    complete rows per source. Categorical heatmaps aggregate all complete rows,
    use shared label order, and show joint proportions. Roles and validation follow
    ``plot_column``; category scalar families and exact string contents are
    preserved. Rare labels are grouped beyond ``max_categories``.
    """
    from .distributions import pair_figure

    return pair_figure(
        real_data,
        generated_data,
        column_1,
        column_2,
        schema=schema,
        max_points=max_points,
        max_categories=max_categories,
        seed=seed,
    )
