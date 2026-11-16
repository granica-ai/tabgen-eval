"""Shared offline documentation figures with accessible source-value tables."""

from html import escape


def write_figure(
    output,
    name,
    figure,
    description,
    table,
    *,
    figure_id,
    table_caption=None,
    show_values=True,
    static_plot=False,
):
    """Embed an offline figure with an accessible description and source values."""
    chart = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=figure_id,
        default_width="100%",
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": False,
            "staticPlot": static_plot,
        },
    )
    table_html = table.to_html(index=False, border=0)
    if not show_values:
        values = ""
    elif table_caption is None:
        values = (
            '<details class="ltm-plot-data"><summary>View the source values</summary>'
            f"{table_html}</details>\n"
        )
    else:
        opening, content = table_html.split("\n", 1)
        values = (
            '<div class="ltm-plot-data">'
            f"{opening}\n<caption>{escape(table_caption)}</caption>\n{content}</div>\n"
        )
    (output / f"{name}.html").write_text(
        f'<div class="ltm-plot" role="group" aria-label="{escape(description)}">'
        f"{chart}</div>\n"
        f"{values}"
    )
