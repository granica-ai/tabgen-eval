/* Keep locally rendered figures legible as the documentation theme changes. */
(() => {
  const comparisonColors = new Set(["#2b70b5", "#d27622"]);

  function updateTheme() {
    const dark = document.documentElement.dataset.theme === "dark";
    const foreground = dark ? "#e7eaf0" : "#26344a";
    const background = dark ? "#1d2634" : "#ffffff";
    const grid = dark ? "#38465b" : "#e4e9f0";
    document.querySelectorAll(".ltm-plot .js-plotly-plot").forEach((chart) => {
      if (!chart.layout || !window.Plotly) return;
      const updates = {
        paper_bgcolor: background,
        plot_bgcolor: background,
        "font.color": foreground,
        "hoverlabel.bgcolor": background,
        "hoverlabel.font.color": foreground,
      };
      Object.keys(chart.layout).forEach((key) => {
        if (/^[xy]axis\d*$/.test(key)) {
          updates[`${key}.gridcolor`] = grid;
          updates[`${key}.zerolinecolor`] = grid;
        }
      });
      (chart.layout.annotations || []).forEach((annotation, index) => {
        const color = annotation.font?.color;
        const comparisonLabel = chart.layout.meta?.kind === "scores"
          && typeof color === "string"
          && comparisonColors.has(color.toLowerCase());
        if (!comparisonLabel) {
          updates[`annotations[${index}].font.color`] = foreground;
        }
      });
      window.Plotly.relayout(chart, updates);
      const intervals = chart.data.flatMap((trace, index) =>
        trace.meta?.theme_role === "reference-interval" ? [index] : []
      );
      if (intervals.length) {
        window.Plotly.restyle(chart, {
          fillcolor: dark ? "rgba(167, 139, 250, 0.28)" : "rgba(105, 114, 174, 0.10)",
          "line.color": dark ? "#c4b5fd" : "rgba(105, 114, 174, 0.6)",
        }, intervals);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", updateTheme);
  new MutationObserver(updateTheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
})();
