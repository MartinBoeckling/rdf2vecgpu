"""Charts and tables for the docs' performance page, from benchmark runs.jsonl files.

    python performance/benchmark_charts.py RUNS.jsonl [RUNS.jsonl ...] \
        --out docs/source/_static/benchmarks --table docs/source/_generated/benchmark_tables.inc

Every chart is written as a Vega-Lite spec per colour theme (``<name>-light.vl.json``,
``<name>-dark.vl.json``), which the docs page renders interactively, and as the same
spec rendered to SVG (``<name>-light.svg``, ``<name>-dark.svg``), shown when JavaScript
or the CDN is unavailable and usable in the README. Rendering the SVGs needs
``pip install vl-convert-python``.

A run is identified by (framework, dataset, walks, depth); when several files hold the
same run, the later file on the command line wins, so a newer rdf2vecgpu run can be
combined with older reference runs. All runs must come from one node unless
``--allow-mixed-nodes`` is given, as wall times differ between nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# framework id in runs.jsonl -> label; the order is the colour order
FRAMEWORKS = {
    "gpu-random-pytorch": "rdf2vecgpu (PyTorch)",
    "gpu-random-gensim": "rdf2vecgpu (gensim)",
    "jrdf2vec": "jRDF2Vec",
    "pyrdf2vec": "pyRDF2Vec",
}
# categorical slots 1-4 of the dataviz reference palette, checked for colour-vision
# deficiency separation on the light and dark surfaces
THEMES = {
    "light": {
        "colors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
    },
    "dark": {
        "colors": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#33332f",
    },
}
SHAPES = ["circle", "square", "triangle-up", "diamond"]
EXCLUDED_DATASETS = {"wikidata5m"}
FAMILIES = {
    "barabasi_graph": "Barabási–Albert",
    "erdos_renyi_graph": "Erdős–Rényi",
    "random_graph": "Growing random",
}
FAILED = {"timeout": "timed out", "oom": "out of memory", "failed": "failed"}


def load_runs(paths: list[Path]) -> dict[tuple, list[dict]]:
    """Runs per (framework, dataset, walks, depth); a later file replaces a key."""
    runs: dict[tuple, list[dict]] = {}
    for path in paths:
        here: dict[tuple, list[dict]] = defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["framework"] not in FRAMEWORKS or r["dataset"] in EXCLUDED_DATASETS:
                continue
            if r["status"] == "skipped":
                continue
            here[(r["framework"], r["dataset"], r["walks"], r["depth"])].append(r)
        runs.update(here)
    return runs


def summarise(runs: dict[tuple, list[dict]]) -> list[dict]:
    """One record per run key: median wall time of the successful repeats, or the
    failure (with the time limit for a timeout)."""
    out = []
    for (fw, dataset, walks, depth), rs in sorted(runs.items(), key=str):
        ok = [r["wall_seconds"] for r in rs if r["status"] == "success"]
        rec = {
            "framework": FRAMEWORKS[fw],
            "dataset": dataset,
            "walks": walks,
            "depth": depth,
            "runs": len(ok),
            "node": sorted({r.get("node", "").split(".")[0] for r in rs}),
        }
        if ok:
            good = [r for r in rs if r["status"] == "success"]
            rec.update(status="success", seconds=statistics.median(ok), min=min(ok), max=max(ok),
                       # highest peak over the repeats; sampled once per second by benchmark.py
                       gpu_gb=max(r.get("peak_gpu_gb") or 0.0 for r in good),
                       ram_gb=max(r.get("peak_rss_gb") or 0.0 for r in good))
        else:
            r = rs[0]
            rec.update(status=r["status"], seconds=r["wall_seconds"] if r["status"] == "timeout" else None)
        out.append(rec)
    return out


def _duration(s: float) -> str:
    if s < 100:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _points(records: list[dict], keep, x_of, facet_of) -> list[dict]:
    pts = []
    for r in records:
        if not keep(r) or r["seconds"] is None:
            continue
        ok = r["status"] == "success"
        pts.append({
            "framework": r["framework"],
            "x": x_of(r),
            "facet": facet_of(r),
            "seconds": r["seconds"],
            "ok": ok,
            "time": _duration(r["seconds"]) if ok else f"> {_duration(r['seconds'])} ({FAILED[r['status']]})",
            "range": f"{_duration(r['min'])} – {_duration(r['max'])}, {r['runs']} runs" if ok else "",
        })
    return pts


def chart_spec(points, theme, *, title, subtitle, x_title, x_sort, facet_title, facet_sort):
    t = THEMES[theme]
    labels = list(FRAMEWORKS.values())
    present = [l for l in labels if any(p["framework"] == l for p in points)]
    color = {"field": "framework", "type": "nominal", "title": None,
             "scale": {"domain": present, "range": [t["colors"][labels.index(l)] for l in present]},
             "legend": {"orient": "top", "direction": "horizontal", "symbolStrokeWidth": 2}}
    shape = {"field": "framework", "type": "nominal", "title": None,
             "scale": {"domain": present, "range": [SHAPES[labels.index(l)] for l in present]}}
    x = {"field": "x", "type": "ordinal", "title": x_title, "sort": x_sort,
         "axis": {"labelAngle": 0}}
    lo = min((p["seconds"] for p in points), default=1)
    hi = max((p["seconds"] for p in points), default=10)
    decades = [10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]
    y = {"field": "seconds", "type": "quantitative", "title": "Wall time, end to end (s, log scale)",
         "scale": {"type": "log", "domain": [decades[0], decades[-1]]},
         "axis": {"values": decades, "format": ",.0f"}}
    tooltip = [
        {"field": "framework", "title": "Framework"},
        {"field": "facet", "title": facet_title},
        {"field": "x", "title": x_title},
        {"field": "time", "title": "Median"},
        {"field": "range", "title": "Range"},
    ]
    pick = {"name": "pick", "select": {"type": "point", "fields": ["framework"]}, "bind": "legend"}
    fade = {"condition": {"param": "pick", "empty": True, "value": 1}, "value": 0.15}
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": title, "subtitle": subtitle, "anchor": "start"},
        "data": {"values": points},
        "facet": {"column": {"field": "facet", "type": "ordinal", "title": None,
                             "sort": facet_sort, "header": {"labelFontSize": 12}}},
        "spec": {
            "width": 180,
            "height": 230,
            "layer": [
                {"transform": [{"filter": "datum.ok"}],
                 "mark": {"type": "line", "strokeWidth": 2, "strokeCap": "round", "strokeJoin": "round"},
                 "encoding": {"x": x, "y": y, "color": color, "detail": {"field": "framework"},
                              "opacity": fade}},
                {"params": [pick],
                 "mark": {"type": "point", "size": 70, "strokeWidth": 2, "filled": True},
                 "encoding": {"x": x, "y": y, "color": color, "stroke": color, "shape": shape, "opacity": fade,
                              "fillOpacity": {"condition": {"test": "datum.ok", "value": 1}, "value": 0},
                              "tooltip": tooltip}},
            ],
        },
        "resolve": {"scale": {"y": "shared"}},
        "config": _config(t),
    }


def _config(t):
    return {
        "background": None,
        "font": "system-ui, -apple-system, Segoe UI, sans-serif",
        "view": {"stroke": None},
        "title": {"color": t["text"], "subtitleColor": t["muted"], "fontSize": 14, "fontWeight": 600},
        "axis": {"labelColor": t["muted"], "titleColor": t["muted"], "gridColor": t["grid"],
                 "domainColor": t["grid"], "tickColor": t["grid"], "gridWidth": 1,
                 "titleFontWeight": 500},
        "header": {"labelColor": t["text"], "titleColor": t["muted"], "titleFontWeight": 500},
        "legend": {"labelColor": t["text"], "labelFontSize": 12},
    }


def _family(dataset):
    return next(v for k, v in FAMILIES.items() if dataset.startswith(k + "_"))


def _row_keys(records):
    """(group, row label, dataset, walks, depth) in page order: FB15k-237, then synthetic."""
    keys = {(r["dataset"], r["walks"], r["depth"]) for r in records}
    fb = sorted((k for k in keys if k[0] == "fb15k-237"), key=lambda k: (k[1], k[2]))
    syn = sorted((k for k in keys if k[0] != "fb15k-237" and any(k[0].startswith(f + "_") for f in FAMILIES)),
                 key=lambda k: (list(FAMILIES).index(k[0].rsplit("_", 1)[0]), int(k[0].rsplit("_", 1)[1])))
    rows = [("FB15k-237", f"{w} walks, depth {d}", ds, w, d) for ds, w, d in fb]
    rows += [("Synthetic graphs", f"{_family(ds)}, {int(ds.rsplit('_', 1)[1]):,}", ds, w, d) for ds, w, d in syn]
    return rows


def _times(x: float) -> str:
    return f"{x:.2f}×" if x < 1 else (f"{x:.1f}×" if x < 100 else f"{x:.0f}×")


def speedup_points(records):
    """rdf2vecgpu (PyTorch) against each CPU reference: reference time / rdf2vecgpu time.
    A reference that timed out gives a lower bound."""
    by = {(r["framework"], r["dataset"], r["walks"], r["depth"]): r for r in records}
    ours = FRAMEWORKS["gpu-random-pytorch"]
    pts = []
    for group, row, ds, w, d in _row_keys(records):
        a = by.get((ours, ds, w, d))
        if not a or a["status"] != "success":
            continue
        for ref in (FRAMEWORKS["jrdf2vec"], FRAMEWORKS["pyrdf2vec"]):
            b = by.get((ref, ds, w, d))
            if not b or b["seconds"] is None:
                continue
            x = b["seconds"] / a["seconds"]
            exact = b["status"] == "success"
            pts.append({
                "group": group, "row": row, "framework": ref, "speedup": x, "ok": exact,
                "label": ("" if exact else "> ") + _times(x),
                "times": f"{_duration(a['seconds'])} vs " + (_duration(b["seconds"]) if exact else
                                                            f"> {_duration(b['seconds'])} ({FAILED[b['status']]})"),
            })
    return pts


def speedup_spec(points, theme, *, title, subtitle):
    t = THEMES[theme]
    labels = list(FRAMEWORKS.values())
    present = [l for l in labels if any(p["framework"] == l for p in points)]
    color = {"field": "framework", "type": "nominal", "title": None,
             "scale": {"domain": present, "range": [t["colors"][labels.index(l)] for l in present]},
             "legend": {"orient": "top", "direction": "horizontal", "symbolStrokeWidth": 2,
                        "labelExpr": "'vs ' + datum.label"}}
    shape = {"field": "framework", "type": "nominal", "title": None,
             "scale": {"domain": present, "range": [SHAPES[labels.index(l)] for l in present]}}
    lo = min([p["speedup"] for p in points] + [1.0])
    hi = max([p["speedup"] for p in points] + [1.0])
    decades = [10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)]
    if len(decades) == 1:
        decades.append(decades[0] * 10)
    x = {"field": "speedup", "type": "quantitative",
         "title": "← rdf2vecgpu slower\u2003\u2003reference time ÷ rdf2vecgpu time (log scale)\u2003\u2003rdf2vecgpu faster →",
         "scale": {"type": "log", "domain": [decades[0], decades[-1]]},
         "axis": {"values": decades, "grid": True, "labelExpr": "format(datum.value, '~g') + '×'"}}
    rows = []
    for p in points:
        if p["row"] not in rows:
            rows.append(p["row"])
    y = {"field": "row", "type": "ordinal", "title": None, "sort": rows}
    pick = {"name": "pick", "select": {"type": "point", "fields": ["framework"]}, "bind": "legend"}
    fade = {"condition": {"param": "pick", "empty": True, "value": 1}, "value": 0.15}
    groups = [g for g in ("FB15k-237", "Synthetic graphs") if any(p["group"] == g for p in points)]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {"text": title, "subtitle": subtitle, "anchor": "start"},
        "data": {"values": points},
        "facet": {"row": {"field": "group", "type": "nominal", "title": None, "sort": groups,
                          "header": {"labelAngle": 0, "labelAlign": "left", "labelAnchor": "start",
                                     "labelOrient": "top", "labelFontSize": 12, "labelFontWeight": 600}}},
        "spec": {
            "width": 400,
            "height": {"step": 20},
            "layer": [
                {"mark": {"type": "rule", "strokeWidth": 1.5, "color": t["muted"]},
                 "encoding": {"x": {"datum": 1}}},
                {"params": [pick],
                 "mark": {"type": "point", "size": 70, "strokeWidth": 2, "filled": True},
                 "encoding": {"x": x, "y": y, "color": color, "stroke": color, "shape": shape, "opacity": fade,
                              "fillOpacity": {"condition": {"test": "datum.ok", "value": 1}, "value": 0},
                              "tooltip": [{"field": "row", "title": "Run"},
                                          {"field": "framework", "title": "Against"},
                                          {"field": "label", "title": "Speed-up"},
                                          {"field": "times", "title": "rdf2vecgpu vs reference"}]}},
            ],
        },
        "resolve": {"scale": {"y": "independent"}},
        "config": _config(t),
    }


def fb15k_points(records):
    return _points(records, lambda r: r["dataset"] == "fb15k-237",
                   lambda r: str(r["depth"]), lambda r: f"{r['walks']} walks")


def synthetic_points(records):
    def family(r):
        return next(v for k, v in FAMILIES.items() if r["dataset"].startswith(k + "_"))

    def size(r):
        n = int(r["dataset"].rsplit("_", 1)[1])
        return f"{n:,}"

    return _points(records, lambda r: any(r["dataset"].startswith(k + "_") for k in FAMILIES),
                   size, family)


def write_table(records, path: Path, caption_note: str):
    by = {(r["framework"], r["dataset"], r["walks"], r["depth"]): r for r in records}
    labels = list(FRAMEWORKS.values())

    def cell(fw, ds, w, d):
        r = by.get((fw, ds, w, d))
        if r is None:
            return "–"
        if r["status"] != "success":
            return f"> {_duration(r['seconds'])}" if r["seconds"] else FAILED[r["status"]]
        return _duration(r["seconds"])

    def speedup(ds, w, d, ref):
        a, b = by.get((labels[0], ds, w, d)), by.get((ref, ds, w, d))
        if not a or a["status"] != "success" or not b or b["seconds"] is None:
            return "–"
        x = b["seconds"] / a["seconds"]
        return (">" if b["status"] != "success" else "") + (f"{x:.1f}×" if x < 100 else f"{x:.0f}×")

    def table(title, rows):
        head = ["Graph", "Walks", "Depth", *labels, "vs jRDF2Vec", "vs pyRDF2Vec"]
        lines = [f".. list-table:: {title}", "   :header-rows: 1", "   :widths: auto", ""]
        lines.append("   * - " + "\n     - ".join(head))
        for ds, name, w, d in rows:
            vals = [name, str(w), str(d), *(cell(l, ds, w, d) for l in labels),
                    speedup(ds, w, d, "jRDF2Vec"), speedup(ds, w, d, "pyRDF2Vec")]
            lines.append("   * - " + "\n     - ".join(vals))
        return "\n".join(lines) + "\n"

    # rdf2vecgpu uses the GPU and host memory; the CPU frameworks only host memory
    mem_cols = [(l, kind) for l in labels for kind in (("gpu_gb", "ram_gb") if l.startswith("rdf2vecgpu") else ("ram_gb",))]

    def mem(fw, ds, w, d, kind):
        r = by.get((fw, ds, w, d))
        if r is None:
            return "–"
        if r["status"] != "success":
            return FAILED[r["status"]] if r["status"] == "oom" else "–"
        return f"{r[kind]:.1f}"

    def memory_table(title, rows):
        head = ["Graph", "Walks", "Depth",
                *(f"{l.replace('rdf2vecgpu (', '').rstrip(')')} {'GPU' if k == 'gpu_gb' else 'RAM'}" for l, k in mem_cols)]
        lines = [f".. list-table:: {title}", "   :header-rows: 1", "   :widths: auto", ""]
        lines.append("   * - " + "\n     - ".join(head))
        for ds, name, w, d in rows:
            vals = [name, str(w), str(d), *(mem(l, ds, w, d, k) for l, k in mem_cols)]
            lines.append("   * - " + "\n     - ".join(vals))
        return "\n".join(lines) + "\n"

    rows = _row_keys(records)
    fb = [(ds, "FB15k-237", w, d) for g, _, ds, w, d in rows if g == "FB15k-237"]
    syn = [(ds, f"{_family(ds)}, {int(ds.rsplit('_', 1)[1]):,} vertices", w, d)
           for g, _, ds, w, d in rows if g != "FB15k-237"]
    text = [
        ".. This file is generated by performance/benchmark_charts.py; do not edit.",
        "",
        table("FB15k-237, median wall time end to end", fb),
        table("Synthetic graphs, 100 walks of depth 4", syn),
        memory_table("FB15k-237, peak memory (GB)", fb),
        memory_table("Synthetic graphs, peak memory (GB)", syn),
    ]
    if caption_note:
        text.append(caption_note + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="runs.jsonl files; later files win")
    ap.add_argument("--out", type=Path, required=True, help="directory for the specs and SVGs")
    ap.add_argument("--table", type=Path, help="reStructuredText include file for the tables")
    ap.add_argument("--subtitle", default="", help="hardware / version line under each chart title")
    ap.add_argument("--allow-mixed-nodes", action="store_true")
    ap.add_argument("--no-svg", action="store_true", help="write only the Vega-Lite specs")
    args = ap.parse_args()

    records = summarise(load_runs(args.runs))
    nodes = sorted({n for r in records for n in r["node"]})
    if len(nodes) > 1 and not args.allow_mixed_nodes:
        raise SystemExit(f"runs come from several nodes {nodes}; wall times are not comparable "
                         "(pass --allow-mixed-nodes for a draft)")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = [{k: v for k, v in r.items() if k != "node"} for r in records]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    charts = {
        "speedup": (speedup_points(records), dict(
            title="Speed-up of rdf2vecgpu (PyTorch) over the CPU implementations")),
        "fb15k": (fb15k_points(records), dict(
            title="FB15k-237: time by walk depth", x_title="Walk depth", x_sort=["4", "8", "12"],
            facet_title="Walks per vertex", facet_sort=["100 walks", "500 walks", "1000 walks"])),
        "synthetic": (synthetic_points(records), dict(
            title="Synthetic graphs: time by graph size (100 walks, depth 4)", x_title="Vertices",
            x_sort=["100", "1,000", "10,000"], facet_title="Graph model",
            facet_sort=list(FAMILIES.values()))),
    }
    for name, (points, kw) in charts.items():
        for theme in THEMES:
            build = speedup_spec if name == "speedup" else chart_spec
            spec = build(points, theme, subtitle=args.subtitle, **kw)
            (args.out / f"{name}-{theme}.vl.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
            if not args.no_svg:
                import vl_convert

                (args.out / f"{name}-{theme}.svg").write_text(vl_convert.vegalite_to_svg(spec, vl_version="5.21"))
    if args.table:
        note = "" if len(nodes) <= 1 else f".. note:: Draft: these runs come from different nodes ({', '.join(nodes)})."
        write_table(records, args.table, note)
    print(f"{len(records)} runs from {', '.join(nodes)} -> {args.out}")


if __name__ == "__main__":
    main()
