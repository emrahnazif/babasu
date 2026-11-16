"""Views: pages the agent writes (product.md "The agent"), shown in the main area inside babasu's
shell. A view is a Jinja template: HTML, the components the app's pages are built from
(components.html, as `ui`), and any JS.

Each view is a folder in the workspace (config.VIEWS_DIR/<id>): `view.json` (its title) and one
file per version (`1.html`, `2.html`…). Writing adds a version; restoring an old one adds it again
as the newest, so nothing is ever lost.

A template is rendered in Jinja's sandbox: it sees only what it's given. Its data comes from
`api("/api/…")`: babasu's own API, the same GET routes and JSON as the browser gets, called in the
same process; and from `data`, what the agent's code worked out (a version written from code may
come with it), so no number the code found passes through the agent's words. The page's JS may
call the API too.

A view keeps what it showed: when a version is saved, every api() answer is recorded with it, and
its data kept (`<n>.json`); the page replays them. The images they name count as used while the
view exists (the view_media table), so the media cleanup keeps them.
"""

import html
import inspect
import json
import re
import traceback
from contextlib import closing
from datetime import UTC, datetime
from functools import cache, lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from jinja2 import FileSystemLoader, Template, TemplateError, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from markupsafe import Markup

from app import config, db, formatting, media, settings
from app.history import trash  # views.history() is a view's versions

TEMPLATE_NAME = "<template>"  # what Jinja calls a template made from a string, in its tracebacks


class ViewError(ValueError):
    """A view that can't be written or shown: why, and at which line of its template when known."""

    def __init__(self, message: str, line: int | None = None):
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line


class ApiError(Exception):
    """An api() call the template made that babasu refused."""


def _dir(view_id: str) -> Path:
    if not settings.SLUG.fullmatch(view_id):
        raise ViewError("A view's id is lowercase letters, digits and dashes, e.g. model-comparison")
    return config.VIEWS_DIR / view_id


def _versions(folder: Path) -> list[int]:
    """The version numbers, oldest first; empty if there's no such view."""
    if not (folder / "view.json").is_file():
        return []
    return sorted(int(p.stem) for p in folder.glob("*.html") if p.stem.isdigit())


def _when(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime(db.TIME)


def _title(folder: Path) -> str:
    return json.loads((folder / "view.json").read_text())["title"]


def _entry(view_id: str, folder: Path, versions: list[int]) -> dict:
    """A view as lists show it: its title, its newest version and when that was written."""
    return {"id": view_id, "title": _title(folder), "version": versions[-1],
            "updated_at": _when(folder / f"{versions[-1]}.html")}


def list_all() -> list[dict]:
    """Every view: id, title, its newest version and when it was written; newest change first."""
    if not config.VIEWS_DIR.is_dir():
        return []
    found = [(f, _versions(f)) for f in config.VIEWS_DIR.iterdir()]
    return sorted((_entry(f.name, f, v) for f, v in found if v), key=lambda v: v["updated_at"], reverse=True)


def get(view_id: str, version: int | None = None) -> dict | None:
    """A view with a version's template (the newest unless given), the API answers it recorded and
    its data; None if there's no such view or version."""
    folder = _dir(view_id)
    versions = _versions(folder)
    version = version or (versions[-1] if versions else None)
    if version not in versions:
        return None
    return _entry(view_id, folder, versions) | {
        "shown": version, "template": (folder / f"{version}.html").read_text(),
        **json.loads((folder / f"{version}.json").read_text())}


def history(view_id: str) -> list[dict] | None:
    """A view's versions, newest first: number, when it was written, and its size in bytes."""
    folder = _dir(view_id)
    versions = _versions(folder)
    if not versions:
        return None
    return [{"version": n, "written_at": _when(folder / f"{n}.html"), "size": (folder / f"{n}.html").stat().st_size}
            for n in reversed(versions)]


def save(view_id: str, template: str, title: str | None = None, data=None) -> dict:
    """Add a version of a view (its template, and the data it reads, if any), checked first by
    rendering it (ViewError with the line), and return the view. A new view needs a title; the
    title and the data change only when given."""
    folder = _dir(view_id)
    versions = _versions(folder)
    title = title or (_title(folder) if versions else None)
    if not title:
        raise ViewError("A new view needs a title")
    if data is None and versions:
        data = get(view_id)["data"]
    answers = {}
    render(template, {"id": view_id, "title": title}, answers, data)
    version = (versions[-1] if versions else 0) + 1
    _write_version(view_id, folder, version, template, answers, data)
    settings.write_file(folder / "view.json", json.dumps({"title": title}, indent=2))
    return get(view_id)


def _write_version(view_id: str, folder: Path, version: int, template: str, answers: dict, data) -> None:
    """A version's files, and the images its answers and data show, kept as used."""
    settings.write_file(folder / f"{version}.json", json.dumps({"answers": answers, "data": data}))
    with closing(db.connect()) as conn, conn:
        media.keep(conn, view_id, _media_ids([answers, data]))
    settings.write_file(folder / f"{version}.html", template)


def _media_ids(answer) -> set[int]:
    """The images JSON names: every `media_id` in it, at any depth."""
    if isinstance(answer, dict):
        found = {answer["media_id"]} if isinstance(answer.get("media_id"), int) else set()
        return found.union(*(_media_ids(v) for v in answer.values()))
    if isinstance(answer, list):
        return set().union(*(_media_ids(v) for v in answer))
    return set()


def restore(view_id: str, version: int) -> dict | None:
    """Make an older version the newest again (as a new version, with the answers it recorded
    then and its data); None if there's no such one."""
    old = get(view_id, version)
    if old is None:
        return None
    folder = _dir(view_id)
    _write_version(view_id, folder, old["version"] + 1, old["template"], old["answers"], old["data"])
    return get(view_id)


def delete(view_id: str) -> bool:
    """Delete a view and all its versions; False if there's no such view. (Its link in the
    navigation is the caller's: settings.remove_nav.)"""
    folder = _dir(view_id)
    if not _versions(folder):
        return False
    with closing(db.connect()) as conn, conn:
        media.release(conn, view_id)
        trash(conn, folder)
    return True


# --- The reference the agent reads ---

REFERENCE = config.TEMPLATES_DIR / "views" / "reference.md"
COMPONENTS = config.TEMPLATES_DIR / "components.html"
# A macro and the comment right before it, which says what it draws: components.html's convention.
_MACRO = re.compile(r"\{#((?:(?!#\}).)*)#\}\s*\{%-?\s*macro (\w+)\((.*?)\)\s*-?%\}", re.DOTALL)
# The API parts a view reads: data, not the app's settings or the chat.
API_FOR_VIEWS = ("/api/projects", "/api/jobs", "/api/vision", "/api/workflows", "/api/runs")


def _words(text: str) -> str:
    return " ".join(text.split())


def reference() -> str:
    """How to build a view (reference.md); the components, each macro's call and its comment
    (components.html); the filters (formatting.py) and icons; and the API a view can read: each GET
    endpoint with its query parameters and what it returns, from the app's own OpenAPI description."""
    components = [f"- `ui.{name}({_words(args)})`: {_words(about)}"
                  for about, name, args in _MACRO.findall(COMPONENTS.read_text()) if not name.startswith("_")]
    filters = []
    for name, fn in formatting.FILTERS.items():
        args = list(inspect.signature(fn).parameters)[1:]
        filters.append(f"- `| {name}{'(' + ', '.join(args) + ')' if args else ''}`: {_words(fn.__doc__)}")
    api = []
    for path, ops in openapi()["paths"].items():
        get = ops.get("get")
        if not get or not path.startswith(API_FOR_VIEWS):
            continue
        query = [p["name"] for p in get.get("parameters", []) if p["in"] == "query"]
        api.append(f"- `GET {path}`{' (?' + ', '.join(query) + ')' if query else ''}: {_words(get.get('description') or get['summary'])}")
    icons = ", ".join(sorted(p.stem for p in formatting.ICONS_DIR.glob("*.svg")))
    return "\n".join([REFERENCE.read_text(), "## Components (`ui`)", "", *components, "", "## Filters", "", *filters,
                      "", "## Icons", "", f"The names a component's `icon` takes: {icons}.", "", "## The API", "", *api, ""])


# --- Rendering ---

_app: FastAPI | None = None


def setup(app: FastAPI) -> None:
    """Give views the app whose API api() calls (the server does, at start)."""
    global _app
    _app = app


@cache
def client() -> TestClient:
    """The app's API in the same process, without starting it again (no lifespan): requests only,
    for views and the agent."""
    return TestClient(_app, base_url="http://babasu", raise_server_exceptions=True)


def openapi() -> dict:
    """The app's OpenAPI spec: every operation, its description, parameters and schemas."""
    return _app.openapi()


def api(path: str):
    """babasu's own API, GET only: the same JSON the browser gets. ApiError if it refuses."""
    if not isinstance(path, str) or not path.startswith("/api/"):
        raise ApiError(f"api() reads babasu's API, a path starting with /api/ (not {path!r})")
    response = client().get(path)
    if response.status_code >= 400:
        raise ApiError(f"api({path!r}): {refusal(response)}")
    return response.json()


def refusal(response) -> str:
    """Why the API refused a request, in words: its status and `detail` (a refused body or
    parameter, field by field; for a method the path doesn't take, the ones it does)."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg')}" for d in detail)
    if response.status_code == 405:
        detail = f"{response.request.url.path} takes {response.headers.get('allow', 'another method')}, not {response.request.method}"
    return f"{response.status_code}: {detail or response.reason_phrase}"


def page_title(path: str) -> str:
    """The title of babasu's own page at `path`, rendered as the browser would get it: checks it
    shows (the agent opens it). ApiError if there's no such page or it fails."""
    if not isinstance(path, str) or not path.startswith("/") or path.startswith(("//", "/api/", "/static/")):
        raise ApiError(f"{path!r} isn't a page of the app: give its path, e.g. /projects/3")
    response = client().get(path, follow_redirects=False)
    if response.status_code != 200 or not response.headers.get("content-type", "").startswith("text/html"):
        raise ApiError(f"{path}: no page there ({response.status_code} {response.reason_phrase})")
    found = re.search(r"<title>(.*?)</title>", response.text, re.DOTALL)
    return html.unescape(found[1]).strip() if found else path


_env = SandboxedEnvironment(loader=FileSystemLoader(config.TEMPLATES_DIR), autoescape=True)
_env.filters.update(formatting.FILTERS)
_env.globals.update(formatting.GLOBALS)


@lru_cache(maxsize=64)
def _compiled(template: str) -> Template:
    """A template compiled once: a saved version never changes, so a page view doesn't compile it."""
    return _env.from_string(template)


def _line(error: BaseException) -> int | None:
    """The template's line an error happened at, from Jinja's rewritten traceback."""
    if isinstance(error, TemplateSyntaxError):
        return error.lineno
    frames = [f for f in traceback.extract_tb(error.__traceback__) if f.filename == TEMPLATE_NAME]
    return frames[-1].lineno if frames else None


def page(view: dict) -> tuple[Markup | None, str | None]:
    """A view as its page shows it, replaying what it recorded: (HTML, None), or (None, why it
    can't be shown)."""
    try:
        return render(view["template"], view, view["answers"], view["data"], replay=True), None
    except ViewError as e:
        return None, str(e)


def render(template: str, view: dict, answers: dict, data, replay: bool = False) -> Markup:
    """A view's template as HTML, reading `data` (None if it has none). ViewError (with the line)
    if it can't be rendered. replay=False records each api() answer in `answers` (path -> JSON);
    replay=True answers from it instead of the API (showing what it recorded)."""
    def recorded(path: str):
        if path not in answers:
            raise ApiError(f"api({path!r}) wasn't asked for when this version was written")
        return answers[path]

    def recording(path: str):
        answers[path] = api(path)
        return answers[path]

    call = recorded if replay else recording
    try:
        ui = _env.get_template("components.html").module  # reloaded when the file changes, like the pages'
        return Markup(_compiled(template).render(api=call, data=data, view=view, ui=ui))
    except (TemplateError, ApiError, ArithmeticError, LookupError, TypeError, ValueError) as error:  # the template's
        raise ViewError(str(error), _line(error)) from None
