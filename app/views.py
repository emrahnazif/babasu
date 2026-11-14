"""Views: pages the agent writes (product.md "The agent"), shown in the main area inside babasu's
shell. A view is a Jinja template: HTML, our components, and any JS. Kinds: a report (what it
shows is kept as it was when written) and a Studio page (live, for one workflow).

Each view is a folder in the workspace (config.VIEWS_DIR/<id>): `view.json` (title, kind,
workflow) and one file per version (`1.html`, `2.html`…). Writing adds a version; restoring an old
one adds it again as the newest, so nothing is ever lost.

A template is rendered in Jinja's sandbox: it sees only what it's given. Its data comes from
`api("/api/…")`: babasu's own API, the same GET routes and JSON as the browser gets, called in the
same process. The page's JS may call the API too.

A report keeps what it showed: when a version is saved, every api() answer is recorded with it
(`<n>.json`), and the page replays them. The images in those answers count as used while the
report exists (the view_media table), so the media cleanup keeps them.
"""

import html
import json
import re
import traceback
from contextlib import closing
from datetime import UTC, datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.testclient import TestClient
from jinja2 import Template, TemplateError, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from markupsafe import Markup

from app import config, db, media, settings, view_components
from app.history import trash  # views.history() is a view's versions

Kind = Literal["report", "studio"]
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
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")


def _entry(view_id: str, folder: Path, versions: list[int]) -> dict:
    """A view as lists show it: its own fields (view.json), its newest version, when that was
    written, its page."""
    return {"id": view_id} | json.loads((folder / "view.json").read_text()) | {
        "version": versions[-1], "updated_at": _when(folder / f"{versions[-1]}.html"), "url": f"/views/{view_id}"}


def studio_for(workflow_id: str) -> dict | None:
    """The newest Studio page built for a workflow, as get() gives it; None if there's none."""
    found = [v for v in list_all() if v["kind"] == "studio" and v["workflow"] == workflow_id]
    return found and get(found[0]["id"])


DEFAULT_STUDIO = config.TEMPLATES_DIR / "studio" / "default.html"  # a workflow's page when nobody built one


def studio(workflow_id: str, name: str) -> dict:
    """A workflow's Studio page, as get() gives a view: the newest one built for it (kind studio),
    else the plain one (drop images in, see what it keeps)."""
    return studio_for(workflow_id) or {
        "id": None, "title": name, "kind": "studio", "workflow": workflow_id, "shown": 0, "version": 0,
        "answers": None, "template": DEFAULT_STUDIO.read_text()}


def list_all() -> list[dict]:
    """Every view: id, title, kind, workflow, its newest version and its page; newest change first."""
    if not config.VIEWS_DIR.is_dir():
        return []
    found = [(f, _versions(f)) for f in config.VIEWS_DIR.iterdir()]
    return sorted((_entry(f.name, f, v) for f, v in found if v), key=lambda v: v["updated_at"], reverse=True)


def get(view_id: str, version: int | None = None) -> dict | None:
    """A view with a version's template (the newest unless given); None if there's no such view
    or version."""
    folder = _dir(view_id)
    versions = _versions(folder)
    version = version or (versions[-1] if versions else None)
    if version not in versions:
        return None
    answers = folder / f"{version}.json"
    return _entry(view_id, folder, versions) | {
        "shown": version, "template": (folder / f"{version}.html").read_text(),
        "answers": json.loads(answers.read_text()) if answers.is_file() else None}


def history(view_id: str) -> list[dict] | None:
    """A view's versions, newest first: number, when it was written, and its size in bytes."""
    folder = _dir(view_id)
    versions = _versions(folder)
    if not versions:
        return None
    return [{"version": n, "written_at": _when(folder / f"{n}.html"), "size": (folder / f"{n}.html").stat().st_size}
            for n in reversed(versions)]


def save(view_id: str, template: str, title: str | None = None, kind: Kind | None = None,
         workflow: str | None = None) -> dict:
    """Add a version of a view, checked first by rendering it (ViewError with the line), and
    return the view. A new view needs a title; title, kind and workflow change only when given."""
    folder = _dir(view_id)
    versions = _versions(folder)
    meta = json.loads((folder / "view.json").read_text()) if versions else {"title": None, "kind": "report", "workflow": None}
    meta |= {k: v for k, v in {"title": title, "kind": kind, "workflow": workflow}.items() if v is not None}
    if not meta["title"]:
        raise ViewError("A new view needs a title")
    answers = {} if meta["kind"] == "report" else None
    render(template, meta, answers)
    version = (versions[-1] if versions else 0) + 1
    _write_version(view_id, folder, version, template, answers)
    settings.write_file(folder / "view.json", json.dumps(meta, indent=2))
    return get(view_id)


def _write_version(view_id: str, folder: Path, version: int, template: str, answers: dict | None) -> None:
    """A version's files, and (a report's) the images its answers show, kept as used."""
    if answers is not None:
        settings.write_file(folder / f"{version}.json", json.dumps(answers))
        with closing(db.connect()) as conn, conn:
            media.keep(conn, view_id, _media_ids(answers))
    settings.write_file(folder / f"{version}.html", template)


def _media_ids(answer) -> set[int]:
    """The images an API answer names: every `media_id` in it, at any depth."""
    if isinstance(answer, dict):
        found = {answer["media_id"]} if isinstance(answer.get("media_id"), int) else set()
        return found.union(*(_media_ids(v) for v in answer.values()))
    if isinstance(answer, list):
        return set().union(*(_media_ids(v) for v in answer))
    return set()


def restore(view_id: str, version: int) -> dict | None:
    """Make an older version the newest again (as a new version, with the answers a report
    recorded then); None if there's no such one."""
    old = get(view_id, version)
    if old is None:
        return None
    folder = _dir(view_id)
    _write_version(view_id, folder, old["version"] + 1, old["template"], old["answers"])
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
# The API parts a view reads: data, not the app's settings or the chat.
API_FOR_VIEWS = ("/api/projects", "/api/jobs", "/api/vision", "/api/workflows", "/api/runs")


def reference() -> str:
    """How to build a view (reference.md), and the API a view can read: each GET endpoint with its
    query parameters and what it returns, from the app's own OpenAPI description."""
    lines = []
    for path, ops in _app.openapi()["paths"].items():
        get = ops.get("get")
        if not get or not path.startswith(API_FOR_VIEWS):
            continue
        query = [p["name"] for p in get.get("parameters", []) if p["in"] == "query"]
        about = " ".join((get.get("description") or get["summary"]).split())
        lines.append(f"- `GET {path}`{' (?' + ', '.join(query) + ')' if query else ''}: {about}")
    return (REFERENCE.read_text() + "\n" + view_components.reference()
            + "\n## The API\n\n" + "\n".join(lines) + "\n")


# --- Rendering ---

_app: FastAPI | None = None


def setup(app: FastAPI) -> None:
    """Give views the app whose API api() calls (the server does, at start)."""
    global _app
    _app = app


@cache
def _client() -> TestClient:
    # The app in the same process, without starting it again (no lifespan): requests only.
    return TestClient(_app, base_url="http://babasu", raise_server_exceptions=True)


def api(path: str):
    """babasu's own API, GET only: the same JSON the browser gets. ApiError if it refuses."""
    if not isinstance(path, str) or not path.startswith("/api/"):
        raise ApiError(f"api() reads babasu's API, a path starting with /api/ (not {path!r})")
    response = _client().get(path)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        raise ApiError(f"api({path!r}): {response.status_code} {detail or response.reason_phrase}")
    return response.json()


def page_title(path: str) -> str:
    """The title of babasu's own page at `path`, rendered as the browser would get it: checks it
    shows (the agent opens it). ApiError if there's no such page or it fails."""
    if not isinstance(path, str) or not path.startswith("/") or path.startswith(("//", "/api/", "/static/")):
        raise ApiError(f"{path!r} isn't a page of the app: give its path, e.g. /projects/3")
    response = _client().get(path, follow_redirects=False)
    if response.status_code != 200 or not response.headers.get("content-type", "").startswith("text/html"):
        raise ApiError(f"{path}: no page there ({response.status_code} {response.reason_phrase})")
    found = re.search(r"<title>(.*?)</title>", response.text, re.DOTALL)
    return html.unescape(found[1]).strip() if found else path


_env = SandboxedEnvironment(autoescape=True)
_env.filters.update(view_components.FILTERS)
_env.globals["ui"] = view_components.UI


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
    """A view as its page shows it (a report replaying what it recorded): (HTML, None), or
    (None, why it can't be shown, e.g. the data it reads is gone)."""
    try:
        return render(view["template"], view, view["answers"], replay=True), None
    except ViewError as e:
        return None, str(e)


def render(template: str, view: dict, answers: dict | None = None, replay: bool = False) -> Markup:
    """A view's template as HTML. ViewError (with the line) if it can't be rendered. With
    `answers`: replay=False records each api() answer in it (path -> JSON); replay=True answers
    from it instead of the API (a report showing what it recorded)."""
    def recorded(path: str):
        if path not in answers:
            raise ApiError(f"api({path!r}) wasn't asked for when this report was written")
        return answers[path]

    def recording(path: str):
        answers[path] = api(path)
        return answers[path]

    call = api if answers is None else recorded if replay else recording
    try:
        return Markup(_compiled(template).render(api=call, view=view))
    except (TemplateError, ApiError, ArithmeticError, LookupError, TypeError, ValueError) as error:  # the template's
        raise ViewError(str(error), _line(error)) from None
