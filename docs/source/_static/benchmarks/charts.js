// Replaces the static SVGs of the performance page with interactive Vega-Lite
// charts (tooltips; click a legend entry to highlight a framework), in the
// docs' current colour theme. The SVGs stay when the scripts or specs cannot load.
(function () {
  const LIBS = [
    "https://cdn.jsdelivr.net/npm/vega@5.30.0/build/vega.min.js",
    "https://cdn.jsdelivr.net/npm/vega-lite@5.21.0/build/vega-lite.min.js",
    "https://cdn.jsdelivr.net/npm/vega-embed@6.26.0/build/vega-embed.min.js",
  ];
  const load = (src) => new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
  const mode = () => {
    const t = document.body.dataset.theme;  // furo: "light", "dark" or "auto"
    if (t === "light" || t === "dark") return t;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  };

  async function render() {
    const m = mode();
    for (const el of document.querySelectorAll(".bench-chart[data-chart]")) {
      if (el.dataset.mode === m) continue;
      try {
        const res = await fetch(el.dataset.base + el.dataset.chart + "-" + m + ".vl.json");
        if (!res.ok) continue;
        const spec = await res.json();
        const live = document.createElement("div");
        await vegaEmbed(live, spec, { actions: false, renderer: "svg", tooltip: { theme: m } });
        el.replaceChildren(live);
        el.dataset.mode = m;
      } catch (e) {
        // keep the static SVG
      }
    }
  }

  document.addEventListener("DOMContentLoaded", async () => {
    if (!document.querySelector(".bench-chart[data-chart]")) return;
    try {
      for (const src of LIBS) await load(src);
    } catch (e) {
      return;
    }
    await render();
    new MutationObserver(render).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  });
})();
