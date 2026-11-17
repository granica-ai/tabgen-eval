"""Build public documentation and small CPU-only report examples."""

import runpy
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "_ext"))

project = "TabGenEval"
author = "Granica Research Team"
copyright = "2026 Granica Computing, Inc."
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "landing",
    "methodology_figures",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist"]
myst_heading_anchors = 5
myst_fence_as_directive = ["mermaid"]
# The landing page has no H1 by design; its sections start at H2.
suppress_warnings = ["myst.header"]
html_theme = "sphinx_book_theme"
html_title = "TabGenEval documentation"
html_logo = "_static/tabgen-eval-table.svg"
html_favicon = "_static/tabgen-eval-table.svg"
html_static_path = ["../../assets", "_static"]
html_css_files = ["branding.css"]
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "tabgen-brand.html",
        "icon-links.html",
        "sbt-sidebar-nav.html",
    ],
}
html_theme_options = {
    "article_header_end": ["toggle-secondary-sidebar.html"],
    "navbar_start": [],
    "navbar_center": [],
    "navbar_end": [],
    "navbar_persistent": [],
    "repository_url": "https://github.com/granica-ai/tabgen-eval",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "home_page_in_toc": False,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "logo": {
        "image_light": "_static/tabgen-eval-table.svg",
        "image_dark": "_static/tabgen-eval-table.svg",
        "text": "TabGenEval",
    },
}


def generate_report_examples(app):
    """Render real report views from the same offline example users can run."""
    from tabgen_eval import evaluate

    example = runpy.run_path(str(ROOT / "examples" / "report_walkthrough.py"))
    point = example["report"]
    reference = evaluate(
        example["eval_data"],
        metrics=["shape", "trend"],
        device="cpu",
        seed=42,
        uncertainty={"mode": "reference", "n_replicates": 10},
        details=True,
    )
    output = Path(app.srcdir) / "_generated"
    output.mkdir(exist_ok=True)
    (output / "point-summary.txt").write_text(point.summary() + "\n")
    table = point.to_dataframe()[
        [
            "metric",
            "stat",
            "value",
            "ci_low",
            "ci_high",
            "ideal",
            "ideal_ci_low",
            "ideal_ci_high",
            "n_eff",
        ]
    ].round(4)
    (output / "point-table.html").write_text(table.to_html(index=False, border=0))
    bands = reference.to_dataframe()[
        ["metric", "value", "ideal", "ideal_ci_low", "ideal_ci_high", "n_eff"]
    ].round(4)
    (output / "reference-table.html").write_text(bands.to_html(index=False, border=0))
    (output / "shape-details.txt").write_text(point.details("shape").to_string(index=False) + "\n")
    (output / "example-report.json").write_text(reference.to_json() + "\n")
    figure = point.fidelity.plot(kind="scores", metrics=["shape", "trend"])
    (output / "point-scores.html").write_text(
        '<div class="ltm-plot" role="group" aria-label="Quickstart metric scores">'
        + figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="ltm-quickstart-scores",
            default_width="100%",
            config={"responsive": True, "displaylogo": False, "scrollZoom": False},
        )
        + "</div>\n"
    )


def generate_visualization_examples(app):
    """Build interactive figures and their readable source tables on the CPU."""
    from html import escape

    from plotly.offline import get_plotlyjs

    example = runpy.run_path(str(ROOT / "examples" / "visualization_walkthrough.py"))
    holdout, generated = example["make_data"]()
    report = example["make_report"](holdout, generated)
    figures = example["make_figures"](report, holdout, generated)
    app._ltm_visualization_example = (example, report, holdout)
    output = Path(app.srcdir) / "_generated" / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    assets = Path(app.srcdir) / "_generated" / "visualization-assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "plotly.min.js").write_text(get_plotlyjs())
    app.config.html_static_path = [*app.config.html_static_path, str(assets)]

    scores = report.to_dataframe()
    score_columns = ["metric", "stat", "value", "ideal", "ideal_ci_low", "ideal_ci_high"]
    tables = {
        "scores": scores.loc[scores["metric"].isin(["shape", "trend"]), score_columns],
        "columns": report.details("shape"),
        "pairs": report.details("trend"),
        "slices": report.details("cond_shape", level="slice").query("column == 'segment'"),
        "concentration": report.details("concentration"),
        "prdc": scores.loc[
            scores["metric"].isin(["precision", "recall", "density", "coverage"]), score_columns
        ],
        "association": (
            "Real input: 600 rows. Generated input: 300 rows. "
            "The generated spend values were permuted before repeating 75 records."
        ),
    }
    for name, figure in figures.items():
        chart = figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=f"ltm-demo-{name}",
            default_width="100%",
            config={"responsive": True, "displaylogo": False, "scrollZoom": False},
        )
        table = tables[name]
        table_html = (
            f"<p>{escape(table)}</p>"
            if isinstance(table, str)
            else table.round(4).to_html(index=False, border=0)
        )
        (output / f"{name}.html").write_text(
            f'<div class="ltm-plot" role="group" aria-label="{escape(name)} example">'
            f"{chart}</div>\n"
            '<details class="ltm-plot-data"><summary>View the source values</summary>'
            f"{table_html}</details>\n"
        )
    report.to_json(str(output / "report.json"))


def add_visualization_assets(app, pagename, templatename, context, doctree):
    """Load the shared offline plotting bundle only where a figure is embedded."""
    if pagename in {
        "index",
        "methodology",
        "getting-started/visualization",
        "getting-started/quickstart",
        "metrics/fidelity/shape",
        "metrics/fidelity/trend",
        "metrics/fidelity/conditional",
        "metrics/geometric-fidelity/prdc",
        "metrics/concentration",
    }:
        app.add_js_file("plotly.min.js")
        app.add_js_file("visualization.js")
        app.add_css_file("visualization.css")


def setup(app):
    app.connect("builder-inited", generate_report_examples)
    app.connect("builder-inited", generate_visualization_examples)
    app.connect("html-page-context", add_visualization_assets)
