# Building a view

A view is a page you write: a Jinja template, shown in the app's main area (the app's header and chat
stay around it). Build it from the components below, the same ones the app's own pages are made of;
write your own HTML, CSS and JS only where none fits.

## Rules

- Data comes from `api("/api/…")`: the app's own API (GET only, the endpoints listed at the end), the
  same JSON a browser gets. E.g. `{% set runs = api("/api/runs") %}`. A refused call stops the page
  with the reason, so ask only for what exists.
- Look at an answer with the `call_api` tool (GET) before using its fields, so the template uses the
  names it really has.
- The template sees only `api`, `ui` (the components below), `view` (its id and title) and the filters
  below. Jinja's usual tags and filters work (`for`, `if`, `set`, `length`, `round`, `sum`, `groupby`,
  `selectattr`, `map`…).
- A component is called with its arguments: `{{ ui.heading("Cars", "Counted per project") }}`,
  `{{ ui.table(rows, columns=["Project", {"label": "Cars", "number": true}]) }}`. A container's content is
  its `children` (a component, text, or a list of them: `children=[ui.text("…"), ui.button("Open",
  href="/projects/3")]`), or what's between `{% call ui.card("Title") %}` and `{% endcall %}`.
- A component shows what it's given: to leave a part out, don't give it (an empty text or list counts as
  not given; 0 is a value). An argument it doesn't take is an error.
- A `{% set %}` inside a `{% for %}` is gone after the loop (Jinja). To build a list in a loop (a
  table's rows, a gallery's images), use a namespace: `{% set ns = namespace(rows=[]) %}`, then
  `{% set ns.rows = ns.rows + [[p.name, p.image_count]] %}` in the loop, and `ns.rows` after it.
- Output is escaped; mark trusted HTML with `|safe` only when you built it yourself.
- Your page sits inside `<div class="view">`. Keep your own CSS inside it (`.view .my-chart {…}`)
  so the rest of the app keeps its look. Use the app's colours: `var(--color-text)`,
  `var(--color-text-muted)`, `var(--color-border)`, `var(--color-bg)`.
- JS is allowed (`<script>`); it may call the API with `fetch("/api/…")`.
- Images: `/media/…` URLs from the API (`thumb_url` for a small one).
- A view keeps what it showed: every api() answer is recorded when it's written, and the page shows
  those, not today's data. To bring it up to date, write it again.
- Writing checks the page first: if it doesn't render, you get the line and why, and nothing is saved.
  Every version is kept; an older one can be restored.
