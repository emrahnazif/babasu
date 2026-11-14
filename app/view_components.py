"""The components a view is built from (app/views.py), and their reference.

Text and layout are plain HTML with our CSS classes (headings, `card`, `grid`, `muted`): the
reference shows them. What draws data is here, as functions a template calls on `ui`
(`{{ ui.table(rows) }}`), plus a few filters for shaping API answers. Each builds its HTML from
values it escapes, and returns it as safe HTML.

COMPONENTS is the reference: each component with an example the reference page (/components)
renders; how to call it and what it does come from the function itself. The agent reads the same
as text.
"""

import inspect
import json
import math
import re
from collections import Counter
from functools import cache

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape

from app import config



@cache
def _annotated():
    """The app's own macro for an image with its annotations (projects/annotated.html), the one its
    pages use. Its file needs only Jinja's built-in filters; templating's environment would import
    the agent, which imports views, which imports this."""
    env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR), autoescape=True, auto_reload=False)
    return env.get_template("projects/annotated.html").module.annotated


def _value(row, key: str):
    """A row's value at `key`; dots go into nested objects ("project.name")."""
    for part in key.split("."):
        row = row.get(part) if isinstance(row, dict) else None
    return row


def _number(value) -> str:
    if isinstance(value, float):
        return f"{value:,.3g}" if abs(value) < 1000 else f"{value:,.0f}"
    return f"{value:,}" if isinstance(value, int) else escape(value if value is not None else "")


# --- Numbers and tables ---


def stat(value, label: str, note: str | None = None) -> Markup:
    """A number to notice, with what it counts under it."""
    extra = Markup(f'<span class="muted">{escape(note)}</span>') if note else ""
    return Markup(f'<div class="stat"><strong>{_number(value)}</strong><span>{escape(label)}</span>{extra}</div>')


def stats(items: list) -> Markup:
    """A row of stats: [[value, label], …] (a third item is a note)."""
    cells = [stat(*i) for i in items]
    return Markup('<div class="stats">' + "".join(cells) + "</div>")


def _cell(row, column: dict) -> Markup:
    value = _value(row, column["key"])
    kind = column.get("type", "text")
    if kind == "image" and value:
        return Markup(f'<img src="{escape(value)}" alt="" loading="lazy">')
    if kind == "badge" and value is not None:
        return Markup(f'<span class="badge plain">{escape(value)}</span>')
    if kind == "link" and value is not None:
        href = re.sub(r"\{([\w.]+)\}", lambda m: str(_value(row, m[1]) or ""), column.get("href", "#"))
        return Markup(f'<a href="{escape(href)}">{escape(value)}</a>')
    return Markup(_number(value)) if kind == "number" else escape(value if value is not None else "")


def table(rows: list, columns: list | None = None) -> Markup:
    """A table of rows (dicts). columns: keys ("name", "project.name") or {"key", "label", "type"}
    with type text, number, image (a URL), badge or link ("href": "/projects/{id}", from the
    row); default: every key of the first row."""
    columns = columns or (list(rows[0]) if rows else [])
    columns = [c if isinstance(c, dict) else {"key": c} for c in columns]
    head = "".join(f"<th>{escape(c.get('label') or c['key'])}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{_cell(r, c)}</td>" for c in columns) + "</tr>" for r in rows)
    return Markup(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


# --- Charts (SVG, drawn here, in the app's colours) ---

EMPTY = Markup('<p class="muted">Nothing to show.</p>')  # a chart with no data
WIDTH = 520  # a chart's drawing width; CSS scales it to the page
DOTS = 60  # a line marks its points when it has at most this many


def _svg(height: float, body: str) -> Markup:
    return Markup(f'<svg class="chart" viewBox="0 0 {WIDTH} {height:.0f}" role="img">{body}</svg>')


def _ends(left, right, y: float, pad: float) -> str:
    """The values at the two ends of an axis, under it."""
    return (f'<text x="{pad}" y="{y}">{escape(left)}</text>'
            f'<text x="{WIDTH - pad}" y="{y}" text-anchor="end">{escape(right)}</text>')


def _pairs(data, label_key: str, value_key: str) -> list[tuple[str, float]]:
    """[label, value] pairs from a dict, a list of pairs, or a list of dicts."""
    if isinstance(data, dict):
        return [(str(k), float(v or 0)) for k, v in data.items()]
    return [(str(_value(d, label_key)), float(_value(d, value_key) or 0)) if isinstance(d, dict)
            else (str(d[0]), float(d[1] or 0)) for d in data]


def bar(data, label_key: str = "label", value_key: str = "value", unit: str = "") -> Markup:
    """Horizontal bars: {label: value}, [[label, value], …], or a list of dicts named by label_key and
    value_key, e.g. the class stats as they come: ui.bar(stats.counts, label_key="class", value_key="objects")."""
    pairs = _pairs(data, label_key, value_key)
    if not pairs:
        return EMPTY
    top = max(v for _, v in pairs) or 1
    row, label_w = 24, 120
    bars = []
    for i, (label, value) in enumerate(pairs):
        y, w = i * row, (WIDTH - label_w - 70) * value / top
        bars.append(
            f'<text x="{label_w - 8}" y="{y + 16}" text-anchor="end">{escape(label[:18])}</text>'
            f'<rect x="{label_w}" y="{y + 4}" width="{w:.1f}" height="{row - 8}" rx="2" class="chart-bar"/>'
            f'<text x="{label_w + w + 6:.1f}" y="{y + 16}">{_number(value)}{escape(unit)}</text>')
    return _svg(len(pairs) * row, "".join(bars))


def line(points, x_key: str = "x", y_key: str = "y") -> Markup:
    """A line through points in order: [[x, y], …], {x: y}, or a list of dicts (x_key, y_key); x is
    shown as given (numbers, dates…)."""
    pairs = _pairs(points, x_key, y_key)
    if not pairs:
        return EMPTY
    height, pad = 200, 32
    low, high = min(v for _, v in pairs), max(v for _, v in pairs)
    step = (WIDTH - 2 * pad) / max(len(pairs) - 1, 1)
    xy = [(pad + i * step, height - pad - (v - low) / ((high - low) or 1) * (height - 2 * pad))
          for i, (_, v) in enumerate(pairs)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="chart-dot"/>' for x, y in xy) if len(xy) <= DOTS else ""
    scale = f'<text x="4" y="{pad}">{_number(high)}</text><text x="4" y="{height - pad}">{_number(low)}</text>'
    return _svg(height, f'<polyline points="{path}" class="chart-line"/>{dots}{scale}'
                        f"{_ends(pairs[0][0], pairs[-1][0], height - 8, pad)}")


def histogram(values, bins: int = 10) -> Markup:
    """How numbers are spread: a list of numbers (in `bins` equal bins), or the bins
    /api/projects/{id}/stats gives ([{"from", "to", "objects"}])."""
    if values and isinstance(values[0], dict):
        counts = [(b["from"], b["to"], b["objects"]) for b in values]
    else:
        numbers = [float(v) for v in values if v is not None]
        low, high = min(numbers, default=0), max(numbers, default=0)
        size = (high - low) / bins or 1
        tally = Counter(min(int((v - low) / size), bins - 1) for v in numbers)
        counts = [(low + i * size, low + (i + 1) * size, tally.get(i, 0)) for i in range(bins)]
    top = max((c for *_, c in counts), default=0)
    if not top:
        return EMPTY
    height, pad = 180, 24
    w = (WIDTH - 2 * pad) / len(counts)
    rects = "".join(
        f'<rect x="{pad + i * w + 1:.1f}" y="{height - pad - c / top * (height - 2 * pad):.1f}" width="{w - 2:.1f}" '
        f'height="{c / top * (height - 2 * pad):.1f}" class="chart-bar"><title>{_number(a)}–{_number(b)}: {c}</title></rect>'
        for i, (a, b, c) in enumerate(counts))
    return _svg(height, rects + _ends(_number(counts[0][0]), _number(counts[-1][1]), height - 6, pad))


# --- Images ---


def image(item: dict, classes: list) -> Markup:
    """An image with its annotations drawn (an image from the API: width, height, annotations,
    thumb_url or url; a project image, or one result of /api/runs/{id}/annotations); classes: the
    project's, from /api/projects/{id}."""
    return Markup(_annotated()(item | {"filename": item.get("filename", "")}, classes, item.get("thumb_url") or item.get("url")))


def images(items: list, classes: list, caption: str = "filename") -> Markup:
    """Images with their annotations in a grid, each captioned by its `caption` field."""
    tiles = "".join(f'<figure class="thumb">{image(i, classes)}<figcaption>{escape(_value(i, caption) or "")}</figcaption></figure>'
                    for i in items)
    return Markup(f'<div class="grid">{tiles}</div>')


def compare(results: list, classes: list) -> Markup:
    """The same images side by side, a column per result label (e.g. two models in one workflow):
    the `results` of /api/runs/{id}/annotations. Each says how many objects it found."""
    labels = list(dict.fromkeys(r["label"] for r in results))
    by_image: dict[int, dict] = {}
    for r in results:
        by_image.setdefault(r["media_id"], {})[r["label"]] = r
    head = "".join(f"<strong>{escape(label)}</strong>" for label in labels)
    cells = []
    for found in by_image.values():
        for label in labels:
            r = found.get(label)
            n = len(r["annotations"]) if r else 0
            cells.append(f'<figure class="thumb">{image(r, classes)}<figcaption>{n} object{"" if n == 1 else "s"} · '
                         f'{escape(r["filename"])}</figcaption></figure>' if r else "<div></div>")
    return Markup(f'<div class="compare" style="--columns: {len(labels)}">{head}{"".join(cells)}</div>')


# --- Studio: running a workflow from the page (js/studio.js) ---


def dropzone(workflow_id: str, prompts: list | None = None, ask_prompts: bool = False) -> Markup:
    """Drop or pick images: they're stored and run through the workflow, the progress is shown, and
    the page shows again when the run ends (read its results with
    api("/api/runs?workflow_id=…&limit=1") and /api/runs/{id}/annotations). prompts: what to look
    for; ask_prompts: a field for them on the page (comma-separated), starting with `prompts`."""
    run = escape(json.dumps({"workflow_id": workflow_id, "prompts": prompts}))
    field = (Markup(f'<label class="field">What to look for<input name="prompts" value="{escape(", ".join(prompts or []))}" '
                    'placeholder="car, person"></label>') if ask_prompts else "")
    return Markup(f'<div class="dropzone" data-studio-drop data-run="{run}" role="button" tabindex="0">'
                  f'{field}<p>Drop images here, or click to choose</p><input type="file" accept="image/*" multiple hidden>'
                  '<p class="muted" data-status></p></div>')


def run_button(label: str, workflow_id: str, project_id: int | None = None, where: str = "",
               count: int | None = None, prompts: list | None = None) -> Markup:
    """A button that runs the workflow on a project's images (its filter `where`; `count` of them
    at random) with `prompts`, shows the progress, and shows the page again when the run ends."""
    run = escape(json.dumps({"workflow_id": workflow_id, "project_id": project_id, "where": where, "count": count,
                             "prompts": prompts}))
    return Markup(f'<div class="actions"><button class="button" type="button" data-studio-run data-run="{run}">'
                  f'{escape(label)}</button><span class="muted" data-status></span></div>')


# --- Filters, for shaping API answers ---


def count_by(rows: list, key: str) -> list:
    """How many rows have each value of `key`, most first: [[value, count], …]."""
    return [list(p) for p in Counter(_value(r, key) for r in rows).most_common()]


def mean(values) -> float | None:
    """The average of the numbers (None ignored); None if there are none."""
    numbers = [v for v in values if isinstance(v, int | float) and not math.isnan(v)]
    return sum(numbers) / len(numbers) if numbers else None


FILTERS = {"count_by": count_by, "mean": mean}


# --- The reference ---

LAYOUT = """Text and layout are HTML with the app's classes:
- `<h1>Title</h1>` then `<p class="muted">a line saying what the page shows</p>`
- `<section><h2>Part</h2>…</section>`; `<hr>` between parts
- `<div class="card"><h2>Box</h2><p>…</p></div>`: a bordered box
- `<div class="grid">…</div>`: as many columns as fit (put cards, stats or images in it)
- `<span class="badge plain">label</span>`, `<span class="muted">quiet text</span>`"""

_SAMPLE = """{% set rows = [{"name": "Street", "task": "detection", "image_count": 1240, "id": 1},
                {"name": "Cells", "task": "segmentation", "image_count": 310, "id": 2}] %}"""
_IMAGE = """{% set classes = [{"id": 1, "name": "car", "color": "#e11d48"}] %}
{% set item = {"width": 256, "height": 256, "url": "/static/brand/app-icon.png", "filename": "example.png",
               "annotations": [{"class_id": 1, "bbox": [40, 60, 120, 90]}]} %}"""

# Each component and an example of it (rendered on the reference page).
COMPONENTS = [
    (stat, '{{ ui.stat(1240, "images", "in 2 projects") }}'),
    (stats, '{{ ui.stats([[1240, "images"], [87, "objects"], [0.91, "average score"]]) }}'),
    (table, _SAMPLE + '\n{{ ui.table(rows, [{"key": "name", "type": "link", "href": "/projects/{id}"}, '
                      '{"key": "task", "type": "badge"}, {"key": "image_count", "label": "Images", "type": "number"}]) }}'),
    (bar, '{% set stats = {"counts": [{"class": "car", "objects": 412}, {"class": "bus", "objects": 57}]} %}\n'
          '{{ ui.bar(stats.counts, label_key="class", value_key="objects") }}'),
    (line, '{{ ui.line([["Mon", 3], ["Tue", 7], ["Wed", 4], ["Thu", 9]]) }}'),
    (histogram, "{{ ui.histogram([12, 15, 15, 18, 22, 22, 22, 30, 31, 45]) }}"),
    (image, _IMAGE + '\n<div style="max-width: 12rem">{{ ui.image(item, classes) }}</div>'),
    (images, _IMAGE + "\n{{ ui.images([item, item], classes) }}"),
    (dropzone, '{{ ui.dropzone("sam3", ["car"], ask_prompts=true) }}'),
    (run_button, '{{ ui.run_button("Run on 5 images", "sam3", project_id=1, count=5, prompts=["car"]) }}'),
    (compare, _IMAGE + '\n{% set sam = {"media_id": 1, "label": "SAM 3", "filename": "example.png", "width": 256, '
              '"height": 256, "url": item.url, "annotations": item.annotations} %}\n'
              '{{ ui.compare([sam, {"media_id": 1, "label": "DINO → EdgeTAM", "filename": "example.png", "width": 256, '
              '"height": 256, "url": item.url, "annotations": []}], classes) }}'),
]

# What a template calls as `ui`.
UI = {fn.__name__: fn for fn, _ in COMPONENTS}


def _about(fn) -> str:
    return " ".join(inspect.getdoc(fn).split())


def docs() -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]]]:
    """(name, call, what it does, example) per component; (call, what it does) per filter."""
    def call(fn, prefix: str) -> str:
        params = inspect.signature(fn).parameters.values()
        args = ", ".join(p.name if p.default is p.empty else f"{p.name}={p.default!r}" for p in params)
        return f"{prefix}{fn.__name__}({args})"

    components = [(fn.__name__, call(fn, "ui."), _about(fn), example) for fn, example in COMPONENTS]
    filters = [(f"… | {name}" + ("(key)" if name == "count_by" else ""), _about(fn)) for name, fn in FILTERS.items()]
    return components, filters


def reference() -> str:
    """The components as text, for the agent."""
    components, filters = docs()
    parts = ["## Layout and text", "", LAYOUT, "", "## Components (`ui`)", ""]
    parts += [f"- `{c}`: {about}" for _, c, about, _ in components]
    parts += ["", "## Filters", ""] + [f"- `{c}`: {about}" for c, about in filters]
    return "\n".join(parts) + "\n"
