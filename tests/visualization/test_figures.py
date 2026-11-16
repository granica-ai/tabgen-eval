"""Presentation semantics and bounded, CPU-only exploratory figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabgen_eval import Schema
from tabgen_eval.execution.runtime.details import DetailTable, FidelityDetails
from tabgen_eval.metrics import Concentration
from tabgen_eval.metrics.base import Direction
from tabgen_eval.preprocessing.categories import MISSING_CATEGORY
from tabgen_eval.report import MetricResult, Report, StatResult
from tabgen_eval.statistics.constants import DEFAULT_CONFIDENCE_INTERVAL
from tabgen_eval.statistics.distributions import ScoreDistribution
from tabgen_eval.visualization import plot_column, plot_column_pair, plot_report
from tests.support.evaluation import paired_report as evaluate
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = pytest.mark.unit


def _stat(
    name, value, *, ideal=np.nan, generated=None, reference=None, direction=Direction.HIGHER_BETTER
):
    return StatResult(
        name,
        name,
        value,
        0.1,
        0.2,
        np.nan,
        ideal,
        0.3,
        0.4,
        np.nan,
        direction,
        None,
        100,
        100,
        50,
        generated_distribution=generated,
        reference_distribution=reference,
    )


def test_score_bands_require_retained_distributions_and_independent_axes():
    distribution = ScoreDistribution.from_samples(
        [0.8, 0.9, 1.0], confidence_interval=DEFAULT_CONFIDENCE_INTERVAL
    )
    stat = _stat("shape", 0.2, ideal=0.95, reference=distribution)
    density = _stat("density", 2.5, direction=Direction.HIGHER_BETTER)
    report = Report(
        [
            MetricResult("shape", "fidelity", {"shape": stat}),
            MetricResult("prdc", "geometric_fidelity", {"density": density}),
        ],
        {},
        {},
    )
    fig = plot_report(report)
    bands = [trace for trace in fig.data if "score band" in trace.name]
    assert len(bands) == 1
    assert list(bands[0].x) == [distribution.band_low, distribution.band_high]
    assert "95% score band" in bands[0].hovertemplate
    assert any(list(trace.x) == [0.2] for trace in fig.data)
    assert any(list(trace.x) == [2.5] for trace in fig.data)
    assert fig.layout.xaxis.matches is None and fig.layout.xaxis2.matches is None
    assert fig.layout.xaxis2.range is None  # density and ratios are not clipped
    assert "higher is better" in fig.layout.annotations[1].text


def test_nonfinite_scores_are_explicit_and_no_band_is_invented():
    report = Report([MetricResult("mle", "utility", {"ratio": _stat("ratio", np.inf)})], {}, {})
    fig = report.plot()
    assert all(trace.mode == "markers" for trace in fig.data)
    assert any("+∞" in annotation.text for annotation in fig.layout.annotations)
    assert list(fig.data[0].x) == [None]
    assert "plotly" in fig.to_html(include_plotlyjs=False)


def test_score_direction_labels_are_readable_and_keep_target_value():
    lower = _stat("auc", 0.6, direction=Direction.LOWER_BETTER)
    target = _stat("ratio", 1.2, direction=Direction.TARGET)
    target.target = 1.0
    report = Report(
        [
            MetricResult("linear_c2st", "detection", {"auc": lower}),
            MetricResult("dcr", "memorization", {"ratio": target}),
        ],
        {},
        {},
    )
    fig = report.plot()
    assert "lower is better" in fig.layout.annotations[0].text
    assert "target 1" in fig.layout.annotations[1].text
    assert lower.direction is Direction.LOWER_BETTER
    assert target.direction is Direction.TARGET and target.target == 1.0


@pytest.fixture(scope="module")
def detailed_report():
    rng = np.random.default_rng(7)
    real = pd.DataFrame(
        {"x": rng.normal(size=90), "y": rng.normal(size=90), "segment": ["a", "b", "a"] * 30}
    )
    generated = real.iloc[:45].copy()
    return evaluate(
        real,
        generated,
        schema=Schema(numeric=["x", "y"], categorical=["segment"]),
        metrics=["shape", "trend", "conditional_fidelity", "concentration"],
        details=True,
        options=make_options(device="cpu", uncertainty="none"),
    )


def test_report_plots_only_read_stored_details(detailed_report, monkeypatch):
    import tabgen_eval.execution.runtime.dispatch as dispatch

    def forbidden(*args, **kwargs):
        raise AssertionError("plotting must not evaluate metrics")

    monkeypatch.setattr(dispatch, "ExecutionSession", forbidden)
    for kind, kwargs in [
        ("scores", {}),
        ("columns", {}),
        ("pairs", {}),
        ("slices", {"column": "segment"}),
        ("concentration", {}),
    ]:
        fig = detailed_report.plot(kind=kind, **kwargs)
        assert fig.layout.meta["kind"] == kind
        assert len(fig.data) > 0
    with pytest.raises(ValueError, match="Unknown plot kind"):
        detailed_report.plot(kind="invented")
    with pytest.raises(KeyError, match="Metrics not present"):
        detailed_report.plot(metrics=["missing"])


def test_pair_undefined_entries_stay_missing_and_can_select_columns():
    details = FidelityDetails(
        {
            "pairs": DetailTable(
                {
                    "column_1_index": np.array([0, 0, 1]),
                    "column_2_index": np.array([1, 2, 2]),
                    "score": np.array([0.75, np.nan, 0.5]),
                }
            )
        },
        {"column_names": ["x", "y", "z"]},
    )
    report = Report([MetricResult("trend", "fidelity", {}, details=details)], {}, {})
    fig = report.plot(kind="pairs")
    values = np.asarray(fig.data[0].z, dtype=float)
    assert np.isnan(np.diag(values)).all()
    assert np.isnan(values[0, 2]) and values[0, 1] == 0.75
    assert len(report.plot(kind="pairs", columns=["z", "y"]).data[0].x) == 2


def test_conditional_recovery_not_capped_and_counts_retained(detailed_report):
    frame = detailed_report.details("conditional_fidelity", level="slice")
    fig = detailed_report.plot(kind="slices", column="segment", stat="recovery")
    assert fig.layout.xaxis.range is None
    assert all(len(row) == 4 for row in fig.data[0].customdata)
    assert set(row[0] for row in fig.data[0].customdata) == set(
        frame.loc[frame.column == "segment", "reference_count"]
    )
    with pytest.raises(ValueError, match="slice stat"):
        detailed_report.plot(kind="slices", column="segment", stat="imagined")


def test_concentration_uses_actual_stored_profiles(detailed_report):
    frame = detailed_report.details("concentration", level="radius").sort_values(
        "radius", kind="stable"
    )
    fig = detailed_report.plot(kind="concentration")
    assert list(fig.data[0].x) == frame.radius.tolist()
    assert list(fig.data[0].y) == frame.reference_mean.tolist()
    assert list(fig.data[1].y) == frame.candidate_mean.tolist()
    assert (
        list(fig.data[2].y)
        == frame.loc[frame.radius_id != "radii_1e-3", "reference_upper"].tolist()
    )
    assert not any(
        getattr(trace, "error_y", None) and trace.error_y.array is not None for trace in fig.data
    )


@pytest.mark.parametrize(
    "scoring,generated,reference,direction",
    [("auroc", 0.4, 0.8, "higher is better"), ("mae", 0.6, 0.2, "lower is better")],
)
def test_utility_uses_raw_task_scores_and_uncapped_ratios(scoring, generated, reference, direction):
    report = Report(
        [
            MetricResult(
                "mle",
                "utility",
                {},
                diagnostics={
                    "target": "y",
                    "scoring": scoring,
                    "generated_score": generated,
                    "reference_score": reference,
                    "ratio": generated / reference,
                },
            )
        ],
        {},
        {},
    )
    fig = report.plot(kind="utility")
    assert list(fig.data[0].x) == [generated]
    assert list(fig.data[1].x) == [reference]
    assert list(fig.data[2].x) == [generated / reference]
    assert fig.layout.xaxis.range is None
    assert fig.layout.xaxis.title.text == scoring.upper()
    assert all(direction in annotation.text for annotation in fig.layout.annotations)
    assert "1 = equal performance" in fig.layout.annotations[1].text.replace("<br>", " ")
    assert direction in fig.data[0].hovertemplate
    assert direction in fig.data[2].hovertemplate
    assert f"Generated / real {scoring.upper()}" in fig.data[2].hovertemplate
    report.results[0].diagnostics.update(reference_score=0.0, ratio=np.inf)
    fig = report.plot(kind="utility")
    assert any("Ratio: +∞" == annotation.text for annotation in fig.layout.annotations)
    report.results[0].diagnostics.update(generated_score=0.0, ratio=np.nan)
    fig = report.plot(kind="utility")
    assert any("Ratio: undefined" == annotation.text for annotation in fig.layout.annotations)
    assert list(fig.data[0].x) == [0.0] and list(fig.data[1].x) == [0.0]
    assert list(fig.data[2].x) == [None]


def test_typed_category_identity_whitespace_and_missing_are_preserved():
    real = pd.DataFrame(
        {"c": pd.Series([1, 1.0, "1", " A ", "A", None, MISSING_CATEGORY], dtype=object)}
    )
    generated = real.copy(deep=True)
    fig = plot_column(real, generated, "c", schema=Schema(categorical=["c"]))
    labels = list(fig.data[0].x)
    assert len(labels) == 6
    assert "integer: 1" in labels and "float: 1.0" in labels and "string: '1'" in labels
    assert "string: ' A '" in labels and "string: 'A'" in labels
    assert "〈missing〉" in labels
    assert fig.layout.meta["sources"]["reference"]["missing_or_nonfinite"] == 1
    assert real.equals(generated)


def test_category_tail_grouping_preserves_denominator():
    real = pd.DataFrame({"c": ["a"] * 6 + ["b"] * 3 + ["c"]})
    fig = plot_column(real, real, "c", schema=Schema(categorical=["c"]), max_categories=1)
    assert list(fig.data[0].x) == ["string: 'a'", "〈other categories〉"]
    assert list(fig.data[0].y) == [0.6, 0.4]


def test_numeric_samples_are_bounded_deterministic_and_svg_only():
    real = pd.DataFrame({"x": np.linspace(0, 1, 500), "y": np.linspace(1, 2, 500)})
    schema = Schema(numeric=["x", "y"])
    fig = plot_column_pair(real, real, "x", "y", schema=schema, max_points=31, seed=3)
    again = plot_column_pair(real, real, "x", "y", schema=schema, max_points=31, seed=3)
    assert all(trace.type == "scatter" and len(trace.x) == 31 for trace in fig.data)
    assert list(fig.data[0].x) == list(again.data[0].x)
    ecdf = plot_column(real, real, "x", schema=schema, max_points=17)
    assert len(ecdf.data[0].x) == 17 and ecdf.data[0].y[-1] == 1
    with pytest.raises(ValueError, match="max_points"):
        plot_column(real, real, "x", schema=schema, max_points=0)


def test_categorical_pairs_are_joint_normalized_and_share_labels():
    real = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["m", "m", "m", "n"]})
    generated = pd.DataFrame({"a": ["x", "z", "z", "z"], "b": ["m", "n", "n", "n"]})
    fig = plot_column_pair(
        real, generated, "a", "b", schema=Schema(categorical=["a", "b"]), max_categories=1
    )
    assert list(fig.data[0].x) == list(fig.data[1].x)
    assert list(fig.data[0].y) == list(fig.data[1].y)
    assert np.asarray(fig.data[0].z).shape == (2, 2)
    assert np.asarray(fig.data[0].z).sum() == 1
    assert np.asarray(fig.data[1].z).sum() == 1


def test_empty_and_missing_only_inputs_do_not_fabricate_observations():
    frame = pd.DataFrame({"x": [np.nan], "c": [None]})
    schema = Schema(numeric=["x"], categorical=["c"])
    numeric = plot_column(frame, frame, "x", schema=schema)
    categorical = plot_column(frame, frame, "c", schema=schema)
    pair = plot_column_pair(frame, frame, "x", "c", schema=schema)
    assert not len(numeric.data[0].x)
    assert not len(categorical.data[0].x)
    assert not len(pair.data[0].x)
    with pytest.raises(ValueError, match="distinct"):
        plot_column_pair(frame, frame, "x", "x", schema=schema)


def test_ineligible_slice_and_undefined_column_are_explicit(detailed_report):
    from copy import deepcopy

    report = deepcopy(detailed_report)
    conditional = next(result for result in report.results if result.name == "conditional_fidelity")
    table = conditional.details.tables["slices"].columns
    # The first categorical group is made ineligible without inventing a score.
    position = int(np.flatnonzero(table["column_index"] == 2)[0])
    table["eligible"][position] = False
    table["shape"][position] = np.nan
    figure = report.plot(kind="slices", column="segment")
    assert any("ineligible; real=" in annotation.text for annotation in figure.layout.annotations)
    assert None in figure.data[0].x
    shape = next(result for result in report.results if result.name == "shape")
    shape.details.tables["columns"].columns["score"][0] = np.nan
    figure = report.plot(kind="columns")
    assert any("undefined" in annotation.text for annotation in figure.layout.annotations)
    assert None in figure.data[0].x


@pytest.mark.parametrize(
    "name,value,scoring", [("prediction_agreement", 0.8, "auroc"), ("prediction_mse", 0.2, "mae")]
)
def test_nested_mle_diagnostics_and_prediction_comparisons_are_supported(name, value, scoring):
    record = {"generated_score": 0.2, "reference_score": 0.4, "ratio": 0.5, name: value}
    result = MetricResult(
        "mle",
        "utility",
        {},
        diagnostics={"point": dict(record, diagnostics={"target": "target", "scoring": scoring})},
    )
    figure = plot_report(Report([result], {}, {}), kind="utility")
    assert len(figure.data) == 4
    assert list(figure.data[0].x) == [0.2]
    assert list(figure.data[2].x) == [0.5]
    assert list(figure.data[3].x) == [value]


def test_mle_raw_reference_is_labelled_real_trained():
    report = Report(
        [
            MetricResult(
                "mle",
                "utility",
                {"score": _stat("score", 0.2, ideal=0.4, direction=Direction.LOWER_BETTER)},
            )
        ],
        {},
        {},
    )
    figure = report.plot()
    assert {trace.name for trace in figure.data} == {
        "Generated-trained point",
        "Real-trained point",
    }
    assert not any("Real–real" in trace.name for trace in figure.data)


@pytest.mark.parametrize("scoring", ["auroc", "mae"])
def test_repeated_utility_plot_reads_statistics_and_skips_failed_fit_metadata(scoring):
    prediction = "prediction_agreement" if scoring == "auroc" else "prediction_mse"
    direction = Direction.HIGHER_BETTER if scoring == "auroc" else Direction.LOWER_BETTER
    score = _stat("score", 0.4, ideal=0.8, direction=direction)
    score.value_estimate = score.ideal_estimate = "mean"
    result = MetricResult(
        "mle",
        "utility",
        {
            "score": score,
            "ratio": _stat("ratio", 0.6),
            prediction: _stat(prediction, 0.3),
        },
        diagnostics={
            "point": {},
            "replicates": [
                {"diagnostics": {"reason": "failed first fit"}},
                {
                    "generated_score": 0.2,
                    "reference_score": 0.5,
                    "ratio": 0.4,
                    "diagnostics": {"target": "y", "scoring": scoring},
                },
            ],
        },
    )
    figure = Report([result], {"target": "y"}, {}).utility.plot(kind="utility")
    assert [list(trace.x) for trace in figure.data] == [[0.4], [0.8], [0.6], [0.3]]
    assert scoring.upper() in figure.layout.annotations[0].text
    assert "mle · y" in figure.layout.annotations[0].text


def test_fixed_small_concentration_radius_has_only_mean_plot():
    frame = pd.DataFrame({"x": [0.0, 0.0, 1.0, 2.0]})
    options = make_options()
    report = evaluate(
        frame,
        frame.copy(),
        Schema(numeric=["x"]),
        metrics=[configured_metric(Concentration, options, radius_quantiles=[])],
        options=options,
    )
    figure = report.plot(kind="concentration")
    assert len(figure.data) == 2
    assert [annotation.text for annotation in figure.layout.annotations] == ["Mean local mass"]
    assert all(list(trace.x) == [1e-3] for trace in figure.data)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_mle_plots_reuse_stored_model_scores_and_shared_predictions(task, monkeypatch):
    pytest.importorskip("xgboost", minversion="2.1")
    from tabgen_eval.models import XGBoostModel

    x = np.linspace(-3, 3, 120)
    y = np.where(x > 0, "positive", "negative") if task == "classification" else 2 * x
    real = pd.DataFrame({"x": x, "target": y})
    schema = Schema(
        numeric=["x"] if task == "classification" else ["x", "target"],
        categorical=["target"] if task == "classification" else [],
        target="target",
    )
    report = evaluate(
        real,
        real.iloc[::2].copy(),
        schema,
        metrics=["mle"],
        options=make_options(device="cpu", uncertainty="none"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Stored MLE plots must not fit models or evaluate metrics")

    monkeypatch.setattr(XGBoostModel, "fit", forbidden)
    import tabgen_eval.execution.runtime.dispatch as dispatch

    monkeypatch.setattr(dispatch, "ExecutionSession", forbidden)
    figure = report.utility.plot(kind="utility")
    assert list(figure.data[0].x) == [report["mle/score"].value]
    assert list(figure.data[1].x) == [report["mle/score"].ideal]
    assert list(figure.data[2].x) == [report["mle/ratio"].value]
    name = "prediction_agreement" if task == "classification" else "prediction_mse"
    assert list(figure.data[3].x) == [report[f"mle/{name}"].value]
    direction = "higher is better" if task == "classification" else "lower is better"
    scoring = "AUROC" if task == "classification" else "MAE"
    assert scoring in figure.layout.annotations[0].text
    assert direction in figure.layout.annotations[0].text
    assert direction in figure.layout.annotations[1].text
    assert direction in figure.data[2].hovertemplate
    assert report["mle/ratio"].value == pytest.approx(
        report["mle/score"].value / report["mle/score"].ideal
    )
    assert any(trace.name == "Real-trained point" for trace in report.utility.plot().data)


def test_plotting_preserves_values_and_filters_observed_pairs():
    real = pd.DataFrame({"number": [1.0, 2.0, 3.0], "label": [" A ", None, "A"]})
    generated = real.copy(deep=True)
    before = real.copy(deep=True)
    schema = Schema(numeric=["number"], categorical=["label"])
    fig = plot_column_pair(real, generated, "number", "label", schema=schema)
    assert real.equals(before) and generated.equals(before)
    assert fig.layout.meta["sources"]["reference"]["available"] == 2
    from tabgen_eval.visualization._style import category_label

    assert set(fig.data[0].x) == {category_label(" A "), category_label("A")}


def test_legend_spacing_is_pixel_stable_for_short_and_tall_figures():
    import plotly.graph_objects as go

    from tabgen_eval.visualization._style import finish

    for height in (350, 1200):
        figure = finish(
            go.Figure(go.Scatter(x=[0], y=[0], name="Generated")), "Example", height=height
        )
        plot_height = height - figure.layout.margin.t - figure.layout.margin.b
        assert figure.layout.legend.yanchor == "top"
        assert -figure.layout.legend.y * plot_height == pytest.approx(60)
        assert figure.layout.margin.b >= 120


@pytest.mark.parametrize("count", [4, 18])
def test_score_rows_reserve_pixel_space_for_multiline_headings_and_axis_labels(count):
    results = []
    for index in range(count):
        stat = _stat("score", 0.5, ideal=0.8)
        stat.display = "Upper-quantile concentration ratio (radii_1e-3)"
        results.append(MetricResult(f"metric_{index}", "concentration", {"score": stat}))
    figure = plot_report(Report(results, {}, {}))
    plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
    axes = [figure.layout["yaxis" if index == 0 else f"yaxis{index + 1}"] for index in range(count)]
    for index, axis in enumerate(axes):
        assert (axis.domain[1] - axis.domain[0]) * plot_height >= 100
        xaxis = figure.layout["xaxis" if index == 0 else f"xaxis{index + 1}"]
        assert xaxis.anchor == "free"
        assert xaxis.position == pytest.approx(sum(axis.domain) / 2)
    row_gaps = [
        (axes[index].domain[0] - axes[index + 2].domain[1]) * plot_height
        for index in range(0, count - 2, 2)
    ]
    assert min(row_gaps) >= 120
    headings = [
        annotation for annotation in figure.layout.annotations if annotation.yref == "paper"
    ]
    assert len(headings) == count
    for index, annotation in enumerate(headings):
        axis = figure.layout["xaxis" if index == 0 else f"xaxis{index + 1}"]
        assert annotation.x == axis.domain[0]
        assert annotation.xanchor == "left"
        assert "<br>" in annotation.text
        assert "higher is better" in annotation.text


def test_score_headings_remove_repeated_metric_name_and_wrap_role_labels():
    stat = _stat("ratio", 1.0, direction=Direction.TARGET)
    stat.display = "DCR ratio <generated & test>"
    stat.target = 1.0
    stat.score_role = "distance_ratio"
    figure = plot_report(
        Report([MetricResult("dcr_ratio", "memorization", {"ratio": stat})], {}, {})
    )
    heading = figure.layout.annotations[0].text
    assert "dcr_ratio ·" not in heading
    assert "&lt;generated &amp; test&gt;" in heading.replace("<br>", " ")
    assert "target 1" in heading
    assert list(figure.data[0].y) == [0]
    assert not figure.layout.yaxis.visible
    role_label = figure.layout.annotations[1].text
    assert "Generated / test distance" in role_label.replace("<br>", " ")
    assert "<br>" in role_label


def test_utility_prediction_panel_uses_second_row_to_leave_room_for_labels():
    result = MetricResult(
        "mle",
        "utility",
        {},
        diagnostics={
            "target": "customer_account_monthly_spending",
            "scoring": "mae",
            "generated_score": 0.2,
            "reference_score": 0.8,
            "ratio": 0.25,
            "prediction_mse": 0.7,
        },
    )
    figure = plot_report(Report([result], {}, {}), kind="utility")
    assert [trace.xaxis for trace in figure.data] == ["x", "x", "x2", "x3"]
    assert figure.layout.yaxis3.domain[1] < figure.layout.yaxis.domain[0]
    assert "<br>" in figure.layout.annotations[0].text
    assert list(figure.layout.yaxis3.tickvals) == ["Prediction MSE"]
    assert figure.layout.xaxis3.rangemode == "tozero"


def test_single_large_score_uses_zero_context_without_forcing_distinct_score_ranges():
    report = Report(
        [MetricResult("mle", "utility", {"prediction_mse": _stat("prediction_mse", 74_900_000)})],
        {},
        {},
    )
    figure = plot_report(report)
    assert figure.layout.xaxis.rangemode == "tozero"
    assert figure.layout.xaxis.tickangle == 0
    report.results[0].stats["prediction_mse"].ideal = 73_000_000
    assert plot_report(report).layout.xaxis.rangemode is None


def test_fidelity_score_style_and_optional_means_preserve_stored_results():
    generated = ScoreDistribution.from_samples([0.6, 0.7, 0.8], confidence_interval=0.95)
    reference = ScoreDistribution.from_samples([0.85, 0.9, 0.95], confidence_interval=0.95)
    stat = _stat("shape", 0.4, ideal=0.99, generated=generated, reference=reference)
    report = Report([MetricResult("shape", "fidelity", {"shape": stat})], {}, {})
    points = report.fidelity.plot()
    means = report.fidelity.plot(summary="mean")
    assert list(points.data[1].x) == [stat.value]
    assert list(points.data[3].x) == [stat.ideal]
    assert list(means.data[1].x) == [generated.mean]
    assert list(means.data[3].x) == [reference.mean]
    assert stat.value == 0.4 and stat.ideal == 0.99
    for figure in (points, means):
        assert all(set(trace.y) == {0} for trace in figure.data)
        assert all(trace.showlegend is False for trace in figure.data)
        assert figure.data[0].line.color == "#2b70b5"
        assert figure.data[2].line.color == "#d27622"
        assert figure.data[0].marker.symbol == "circle"
        assert figure.data[0].opacity == 0.22
        assert figure.data[1].marker.line.color == "white"
        assert not figure.layout.xaxis.showgrid
        assert figure.layout.xaxis.anchor == "free"
        assert figure.layout.xaxis.position == 0.5
        assert not figure.layout.yaxis.visible
        assert list(figure.data[0].x) == [generated.band_low, generated.band_high]
        assert list(figure.data[2].x) == [reference.band_low, reference.band_high]
        labels = [a.text for a in figure.layout.annotations]
        assert "<b>synthetic vs. real</b>" in labels
        assert "<b>real vs. real</b>" in labels
    assert "mean" in means.data[1].hovertemplate
    assert "point" in points.data[1].hovertemplate


def test_score_means_fall_back_to_points_and_handle_reference_only_reports():
    reference = ScoreDistribution.from_samples([0.8, 0.9, 1.0], confidence_interval=0.95)
    stat = _stat("shape", 0.3, ideal=0.8, reference=reference)
    report = Report([MetricResult("shape", "fidelity", {"shape": stat})], {}, {})
    figure = report.fidelity.plot(summary="mean")
    assert list(figure.data[0].x) == [0.3]
    assert figure.data[0].name.endswith("point")
    assert list(figure.data[-1].x) == [reference.mean]
    stat.score_role = "test_test"
    stat.point_status = "absent"
    stat.value = np.nan
    point_figure = report.fidelity.plot()
    assert len(point_figure.data) == 1  # A retained band does not invent a point.
    assert any("Test–test" in annotation.text for annotation in point_figure.layout.annotations)
    figure = report.fidelity.plot(summary="mean")
    assert len(figure.data) == 2
    assert list(figure.data[-1].x) == [reference.mean]
    assert figure.data[-1].name == "Test–test mean"
    with pytest.raises(ValueError, match="summary"):
        report.fidelity.plot(summary="median")


def test_overlapping_scores_stagger_only_comparison_labels():
    stat = _stat("shape", 0.9, ideal=0.9)
    figure = Report([MetricResult("shape", "fidelity", {"shape": stat})], {}, {}).fidelity.plot()
    labels = [
        a for a in figure.layout.annotations if a.yref != "paper" and a.text.startswith("<b>")
    ]
    values = [
        a for a in figure.layout.annotations if a.yref != "paper" and not a.text.startswith("<b>")
    ]
    assert len(labels) == 2
    assert labels[1].yshift > labels[0].yshift
    assert all(a.yshift == -44 for a in values)
    assert all(set(trace.y) == {0} for trace in figure.data)


@pytest.mark.parametrize("reverse", [False, True])
def test_close_comparison_text_moves_without_moving_dots_bands_or_values(reverse):
    generated = ScoreDistribution.from_samples([0.6, 0.7, 0.8], confidence_interval=0.95)
    reference = ScoreDistribution.from_samples([0.61, 0.71, 0.81], confidence_interval=0.95)
    if reverse:
        generated, reference = reference, generated
    stat = _stat(
        "shape", generated.mean, ideal=reference.mean, generated=generated, reference=reference
    )
    report = Report([MetricResult("shape", "fidelity", {"shape": stat})], {}, {})
    figure = report.fidelity.plot()
    assert all(set(trace.y) == {0} for trace in figure.data)
    assert list(figure.data[1].x) == [generated.mean]
    assert list(figure.data[3].x) == [reference.mean]
    assert list(figure.data[0].x) == [generated.band_low, generated.band_high]
    assert list(figure.data[2].x) == [reference.band_low, reference.band_high]
    labels = [a for a in figure.layout.annotations if a.yref != "paper"]
    assert labels[2].yshift > labels[0].yshift
    assert labels[1].yshift == labels[3].yshift == -44
    assert all(a.y == 0 for a in labels)
    assert figure.layout.xaxis.position == 0.5
