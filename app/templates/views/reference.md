# Building a view

A view is a page you write: a Jinja template, shown in the app's main area (the app's header and chat
stay around it). Write HTML; use the components below where they fit, and your own HTML, CSS and JS
where they don't.

## Rules

- Data comes from `api("/api/…")`: the app's own API (GET only, the endpoints listed at the end), the
  same JSON a browser gets. E.g. `{% set runs = api("/api/runs") %}`. A refused call stops the page
  with the reason, so ask only for what exists.
- Look at an answer with the `read_api` tool before using its fields, so the template uses the
  names it really has.
- The template sees only `api`, `ui` (the components below) and `view` (its id, title, kind, workflow). Jinja's usual tags and
  filters work (`for`, `if`, `set`, `length`, `round`, `sum`, `groupby`, `selectattr`, `map`…).
- A `{% set %}` inside a `{% for %}` is gone after the loop (Jinja). Shape data with filters
  instead (`map`, `selectattr`, `sum`, `groupby`, and `count_by` below), or pass the API's list
  straight to a component that names its keys.
- Output is escaped; mark trusted HTML with `|safe` only when you built it yourself.
- Your page sits inside `<div class="view">`. Keep your own CSS inside it (`.view .my-chart {…}`)
  so the rest of the app keeps its look. Use the app's colours: `var(--color-text)`,
  `var(--color-text-muted)`, `var(--color-border)`, `var(--color-bg)`.
- JS is allowed (`<script>`); it may call the API with `fetch("/api/…")`.
- Images: `/media/…` URLs from the API (`thumb_url` for a small one).
- Kinds: `report` (the default; what it shows is kept as it was when written) or `studio` (a live page
  for one workflow: give `workflow`).
- Writing checks the page first: if it doesn't render, you get the line and why, and nothing is saved.
  Every version is kept; an older one can be restored.
