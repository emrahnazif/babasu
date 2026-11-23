"""Tools the agent can call: babasu's own functions, the same ones the UI and the API use.

Each tool's arguments are a pydantic model, so the JSON schema the model sees is generated, never
written by hand. Tools that change things don't ask for confirmation: what they change in the
database, and the workflows they delete, can be undone for 30 days (app/history.py).
The tools are what the agent has to do: read anything (any GET of the API) and change anything (any
write; a change to many images at once previews first), find the API's paths, look at images (and
at what a run found), show the person a page, wait for a job, run code for what nothing else does,
and read the docs. Everything else is the API's: search, workflow nodes, runs, bulk changes.
A tool registered with opens=True returns {"open": "/a/page", …}: that page is shown to the person
once the reply is done (app/agent.py, run_turn); after a call that changed something through the
API (change, code), the page they're on reloads then instead (unless one opens).

What the person sees of a call is one line: its label, what it's doing (the `step` the model gives,
or the tool's own label), and its note, what came out (written from the result by the tool, so it's
always true), with the images it found. An earlier turn's result reaches the model as its note.
"""

import html
import importlib.resources
import inspect
import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import code as code_mode  # `code` is run_code's argument
from app import config, jobs, media, projects
from app.formatting import plural
from app.routes import IMAGES, UI_ONLY, fields_at


class Args(BaseModel):
    """A tool's arguments. An argument it doesn't take is refused, not dropped: the model learns
    its mistake (a misspelt field) instead of being told a change it asked for was made."""

    model_config = ConfigDict(extra="forbid")
    step: str = Field("", description="What this does, for the person: Counting cars in Street cars")


@dataclass(frozen=True)
class Tool:
    description: str
    args: type[Args]
    func: Callable[[sqlite3.Connection, Args], object]
    label: str  # what it does, for the person, when the model gave no `step`
    note: Callable[[Args, object], str]  # what came out, in a few words, from its arguments and result
    # The images it found, for the person to see in its row: a project's id and a few media ids.
    shows: Callable[[Args, object], tuple[int, list[int]]] | None = None
    opens: bool = False  # it returns {"open": "/a/page", …}: that page is shown to the person after the reply
    vision: bool = False  # it gives the model a picture (Seen): offered only to a model that sees images


TOOLS: dict[str, Tool] = {}


class ToolError(Exception):
    """The model asked for a tool that doesn't exist, or gave arguments that don't fit."""


def tool(description: str, args: type[Args], label: str, note: Callable[[Args, object], str] = lambda args, result: "",
         shows: Callable[[Args, object], tuple[int, list[int]]] | None = None, opens: bool = False,
         vision: bool = False):
    """Register a function as a tool, under its own name."""

    def register(func):
        TOOLS[func.__name__] = Tool(description, args, func, label, note, shows, opens, vision)
        return func

    return register


def schemas(sees_images: bool) -> list[dict]:
    """The tool list, in the chat completions API's format; those that give the model a picture
    only for a model that sees images."""
    return [
        {"type": "function", "function": {"name": name, "description": t.description, "parameters": t.args.model_json_schema()}}
        for name, t in TOOLS.items() if sees_images or not t.vision
    ]


@dataclass(frozen=True)
class Seen:
    """What a tool that looks gives back: its result, and the picture the model looks at (a JPEG)."""

    result: dict
    picture: bytes


@dataclass
class _Turn:
    conversation_id: int  # run_code keeps one code process per conversation, in its folder
    stop: threading.Event  # set when the person stops the turn: a tool that takes long ends at once
    changed: bool = False  # a call changed something through the API (change, code): pages show it after a reload


_turn: ContextVar[_Turn | None] = ContextVar("turn", default=None)


@contextmanager
def turn_of(conversation_id: int, stop: threading.Event) -> Iterator[_Turn]:
    """Tools called inside this are a turn of this conversation (agent.run_turn); it gives the turn."""
    turn = _Turn(conversation_id, stop)
    token = _turn.set(turn)
    try:
        yield turn
    finally:
        _turn.reset(token)


def label(name: str, arguments: str) -> str:
    """What a call does, for the person: the `step` the model gave, or the tool's own label."""
    found = TOOLS.get(name)
    if found is None:
        return name
    try:
        step = json.loads(arguments or "{}").get("step")
    except (ValueError, AttributeError):
        step = None
    return step.strip()[:LABEL_SIZE] if isinstance(step, str) and step.strip() else found.label


LABEL_SIZE = 100  # characters of a step's label: what the person reads at a glance


@dataclass(frozen=True)
class Outcome:
    """A call that ran (or failed): its result for the model (JSON text), and what the person sees
    of it: its note, the images it found (their media ids) and the picture it gave the model."""

    result: str
    note: str
    images: list[int] | None = None
    picture: bytes | None = None


def call(conn: sqlite3.Connection, name: str, arguments: str, sees_images: bool) -> Iterator[dict]:
    """Run a tool with the model's JSON arguments (one schemas(sees_images) offers it). Yields what
    it shows while it works (a tool that's a generator: {"type": "output", "text"}, {"type":
    "progress", "done", "total"}); returns its Outcome. Raises ToolError."""
    if name not in TOOLS or TOOLS[name].vision and not sees_images:
        raise ToolError(f"There is no tool named {name!r}.")
    found = TOOLS[name]
    try:
        args = found.args.model_validate_json(arguments or "{}")
    except ValidationError as exc:
        problems = "; ".join(_problem(found.args, e) for e in exc.errors(include_url=False))
        raise ToolError(f"Invalid arguments for {name}: {problems}") from None
    result = found.func(conn, args)
    if inspect.isgenerator(result):
        result = yield from result
    result, picture = (result.result, result.picture) if isinstance(result, Seen) else (result, None)
    shown = found.shows(args, result) if found.shows else None
    return Outcome(json.dumps(result, ensure_ascii=False), found.note(args, result), shown or None, picture)


def _problem(args: type[BaseModel], error: dict) -> str:
    """One argument error in words, where it is; an argument that isn't one says which there are."""
    where = ".".join(str(part) for part in error["loc"]) or "arguments"
    if error["type"] != "extra_forbidden":
        return f"{where}: {error['msg']}"
    return f"{where}: not an argument here; it takes {', '.join(fields_at(args, error['loc'][:-1]))}"


# --- The app's API: everything the pages can do ---

_client: TestClient | None = None  # the app's API in the same process, without starting it again (no lifespan)


def setup(app: FastAPI) -> None:
    """Give the tools the app whose API they call (the server does, at start)."""
    global _client
    _client = TestClient(app, base_url="http://babasu")


def _refusal(response) -> str:
    """Why the API refused a request, in words: its status and `detail` (a refused body or
    parameter, field by field)."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg')}" for d in detail)
    return f"{response.status_code}: {detail or response.reason_phrase}"


def _inline(schema, defs: dict):
    """A JSON schema with its $refs put in place and its titles left out, as the agent reads it.
    What stands beside a $ref (a field's own description) is kept over the definition's."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            return _inline(defs[schema["$ref"].split("/")[-1]], defs) | _inline(
                {k: v for k, v in schema.items() if k != "$ref"}, defs)
        return {k: _inline(v, defs) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_inline(v, defs) for v in schema]
    return schema


def _json(part: dict) -> dict | None:
    """The JSON schema of a request body or a response."""
    return part.get("content", {}).get("application/json", {}).get("schema")


@cache
def operations() -> list[dict]:
    """Every /api operation the agent may call (not those only the pages use: routes.UI_ONLY), from
    the app's OpenAPI spec: its name ("PATCH /api/projects/{project_id}"), description, parameters,
    body schema and, when it has a model (routes.Answer), the fields of its answer."""
    spec = _client.app.openapi()
    defs = spec.get("components", {}).get("schemas", {})
    found = []
    for path, item in spec["paths"].items():
        for method, op in item.items():
            name = f"{method.upper()} {path}"
            if not path.startswith("/api/") or UI_ONLY in op.get("tags", []):
                continue
            params = [{"name": p["name"], "in": p["in"], "required": p.get("required", False),
                       **_inline({k: v for k, v in p["schema"].items() if k in ("type", "enum", "anyOf", "default")}, defs),
                       **({"description": p["description"]} if p.get("description") else {})}
                      for p in op.get("parameters", [])]
            body = _json(op.get("requestBody", {}))
            answer = _json(next((r for code, r in op.get("responses", {}).items() if code.startswith("2")), {}))
            fields = _fields(_inline(answer, defs)) if answer else ""
            found.append({"operation": name, "description": op.get("description", ""), "parameters": params,
                          **({"body": _inline(body, defs)} if body else {}), **({"answer": fields} if fields else {})})
    return found


def _kind(schema: dict) -> tuple[str, list[dict]]:
    """A schema in a few words ("integer or null", "list of", "map of") and the objects (with
    properties) it holds, whose fields go below it (more than one: the answer takes one of them)."""
    if "anyOf" in schema:
        options = [_kind(o) for o in schema["anyOf"] if o.get("type") != "null"]
        words = " or ".join(dict.fromkeys(w for w, _ in options)) or "null"
        nullable = len(options) < len(schema["anyOf"])
        return words + (" or null" if nullable and options else ""), [i for _, inner in options for i in inner]
    if "prefixItems" in schema:
        return "[" + ", ".join(_kind(item)[0] for item in schema["prefixItems"]) + "]", []
    if schema.get("type") == "array":
        words, inner = _kind(schema.get("items", {}))
        return f"list of {words}", inner
    if "properties" in schema:
        return "object", [schema]
    if schema.get("type") == "object" and isinstance(schema.get("additionalProperties"), dict):
        words, inner = _kind(schema["additionalProperties"])
        return f"map of {words}", inner
    if "enum" in schema:
        return " | ".join(map(str, schema["enum"])), []
    return schema.get("type", "any"), []


def _fields(schema: dict) -> str:
    """An answer's fields, one a line, each with its type and meaning; an object's fields indented
    under it: what find_api shows so code knows what it gets. Empty for an answer without a model."""
    words, inner = _kind(schema)
    if not inner:
        return ""
    if words == "object":
        return _objects(inner, 0)
    return f"{words}{', each' if words.startswith(('list', 'map')) else ''}:\n" + _objects(inner, 1)


def _objects(objects: list[dict], depth: int) -> str:
    """The fields of one object, or of each of several it may be ("or:" between them)."""
    parts = []
    for schema in objects:
        lines = []
        for name, field in schema.get("properties", {}).items():
            kind, sub = _kind(field)
            meaning = field.get("description", "")
            lines.append("  " * depth + f"{name} ({kind})" + (f": {meaning}" if meaning else ""))
            if sub:
                lines.append(_objects(sub, depth + 1))
        parts.append("\n".join(lines))
    return ("\n" + "  " * depth + "or:\n").join(parts)


@cache
def _search() -> list[tuple[dict, str, re.Pattern]]:
    """Each operation with its text to search (find_api) and the paths it answers (_request)."""
    def answers(operation: str) -> re.Pattern:
        path = re.escape(operation.split(" ", 1)[1])
        return re.compile(re.sub(r"\\\{\w+\\\}", "[^/]+", path) + "$")  # /api/projects/{project_id} -> /api/projects/[^/]+
    def text(o: dict) -> str:  # what it does and takes; its answer's fields would match too much
        return json.dumps({k: v for k, v in o.items() if k != "answer"}).lower()
    return [(o, text(o), answers(o["operation"])) for o in operations()]


class FindApiArgs(Args):
    query: str = Field(min_length=1, description='An operation ("PATCH /api/projects/{project_id}"), or words ("annotations", '
                       '"cancel job").')


@tool(
    "The details of the app's API operations: what one does, its parameters, the JSON body it takes and the fields "
    "of what it answers (what your code gets). Give an operation (\"PATCH /api/projects/{project_id}\"), or "
    "words to find up to three.",
    FindApiArgs,
    label="Looking up the app's API",
    note=lambda args, found: f"{len(found)} found",
)
def find_api(conn: sqlite3.Connection, args: FindApiArgs) -> list[dict]:
    query = args.query.strip().lower()
    exact = [o for o in operations() if o["operation"].lower() == query]
    if exact:
        return exact
    words = re.findall(r"\w+", query)
    scored = sorted(((sum(w in text for w in words), o) for o, text, _ in _search()), key=lambda t: -t[0])
    found = [o for score, o in scored[:3] if score]
    if not found:
        raise ToolError("No operation matches those words; try others.")
    return found


ANSWER_SIZE = 8_000  # characters of an answer the agent reads; a longer one is refused, with how to ask for less


def _answer(method: str, path: str, body=None) -> object:
    """An answer of the API for the agent: whole, or refused when it's too long to read, saying
    the operation's own ways to ask for less (a cut answer reads as a wrong one). As code gets it
    (api.get), so what the agent learns reading holds in its code."""
    path = _api_path(path)
    response = _request(method, path, body)
    if not response.content:
        return {"done": True}
    answer = response.json()
    if len(response.text) > ANSWER_SIZE:
        shape = f"its fields: {', '.join(answer)}" if isinstance(answer, dict) else f"a list of {len(answer):,}"
        asks = [f"?{p['name']}=" for o in _operations_at(path) if o["operation"].startswith(f"{method} ")
                for p in o["parameters"] if p["in"] == "query"]
        less = f"Ask for less ({', '.join(asks)}), or go through it in code (run_code)." if asks else \
            "It can't be asked for in parts: go through it in code (run_code)."
        raise ToolError(f"That answer is {len(response.text):,} characters, too long to read here ({shape}). {less}")
    return answer


SHOWN = 4  # images a step draws at most


def _shown(args, answer) -> list[int] | None:
    """The images an answer is about, drawn in its step: a page of images, one image, or a change's
    examples."""
    if not isinstance(answer, dict):
        return None
    path = _api_path(args.path)
    if re.match(r"/api/(projects|images)\b", path):
        images = answer.get("images")  # a page of images; in some answers, how many (a Try's)
        ids = answer.get("examples") or ([i["media_id"] for i in images] if isinstance(images, list) else []) or \
            ([answer["media_id"]] if "media_id" in answer else [])
        return ids[:SHOWN]
    return None


class ReadArgs(Args):
    path: str = Field(description="An API path with its ids and query, e.g. /api/projects/3?where=car.")


def _read_note(args: ReadArgs, answer) -> str:
    if isinstance(answer, list):
        return f"{len(answer)} found"
    if not isinstance(answer, dict):
        return ""
    if "image_count" in answer and "match_count" in answer:  # a project's images (an image's own answer has its place)
        return f"{answer['match_count']:,} of {plural(answer['image_count'], 'image')}"
    return ", ".join(f"{len(found)} {name}" for name, found in answer.items() if isinstance(found, list) and found)


@tool(
    "Read anything in the app: any GET of its API. Answers are short (one too long is refused, saying how to ask "
    "for less). The main ones: /api/projects (every project, with its counts); /api/projects/ID (one, with a page "
    "of its images; ?where=FILTER for some, ?images=0 for the project alone); /api/projects/ID/stats?field=class "
    "(objects per class); /api/images/MEDIA_ID (one image and its objects); /api/images?where=FILTER (every project's "
    "images matching it, or any image set: how many, the first with their ids and projects); /api/workflows; /api/workflows/nodes; /api/jobs; /api/vision/models; /api/search?q=NAME; "
    "/api/history. A page's path works too (/projects/3 reads /api/projects/3); find_api finds any other.",
    ReadArgs,
    label="Reading the app's data",
    note=_read_note,
    shows=_shown,
)
def read(conn: sqlite3.Connection, args: ReadArgs) -> object:
    return _answer("GET", args.path)


class ChangeArgs(Args):
    method: Literal["POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(description="The operation's path with its ids, e.g. /api/projects/3/changes.")
    body: dict[str, Any] | list[Any] | None = Field(
        None, description="The JSON body (an object or a list), for an operation that takes one (find_api shows it).")

    @field_validator("body", mode="before")
    @classmethod
    def _not_text(cls, value):
        if isinstance(value, str):
            raise ValueError("send the body as JSON itself (an object or a list), not as text holding JSON")
        return value


def _change_note(args: ChangeArgs, answer) -> str:
    """What the write says it did (every write's answer has `did`)."""
    return answer.get("did", "Done") if isinstance(answer, dict) else "Done"


@tool(
    "Change anything the app's pages can: any write of its API, checked as the pages' changes are (a refusal says "
    "why); its answer says what it did (did), and History can undo it. The main ones: POST /api/projects (a new "
    "project, nothing required); POST /api/projects/ID/images (copy images into it, named as run takes them); PATCH "
    "/api/projects/ID (name, type, annotator); PATCH /api/projects/ID/classes (add, hide, recolour, remove unused "
    "classes, e.g. {\"add\": [{\"name\": \"truck\"}]}); POST /api/projects/ID/classes/rename and …/clear (a class on "
    "the images too); POST /api/annotate (a workflow over images, e.g. a project's: what it finds becomes their "
    "annotations; the find tool does it and waits); "
    "PUT /api/images/MEDIA_ID/annotations (one image's objects); POST /api/projects/ID/changes (many "
    "images at once: delete objects, change their class, remove, copy or move images; only a preview unless "
    "dry_run is false); POST /api/projects/ID/import (images from files and folders); POST /api/workflows (save "
    "one). find_api has what each takes, and any other.",
    ChangeArgs,
    label="Changing the app's data",
    note=_change_note,
    shows=_shown,
)
def change(conn: sqlite3.Connection, args: ChangeArgs) -> object:
    return _answer(args.method, args.path, args.body)


def _operations_at(path: str) -> list[dict]:
    """The operations that answer at a path (with its ids; its query left out), one a method."""
    target = path.strip().split("?")[0]
    return [o for o, _, answers in _search() if answers.match(target)]


def _api_path(path: str) -> str:
    """The API path the agent means: as given, or a page's path, whose data is at the same path under
    /api (/projects/3 is /api/projects/3); a missing first / added."""
    path = "/" + path.strip().lstrip("/")
    return path if path.startswith("/api/") else "/api" + path


def _request(method: str, path: str, body=None):
    """One of the agent's API operations (_api_path), made in this turn (its writes are the turn's
    action). Raises ToolError: an unknown operation, a method the path doesn't take, the API's refusal."""
    path = _api_path(path)
    target = path.split("?")[0]
    operations_here = _operations_at(path)
    methods = [o["operation"].split(" ")[0] for o in operations_here]
    if not methods:
        raise ToolError(f"There's no operation at {target}; find_api finds them.")
    if method not in methods:  # what it does take, in full, so the next call can be right
        raise ToolError(f"{target} takes {', '.join(methods)}, not {method}:\n{json.dumps(operations_here, ensure_ascii=False)}")
    response = _client.request(method, path, json=body)
    if response.status_code >= 400:
        raise ToolError(_refusal(response))
    if method != "GET":
        _turn.get().changed = True
    return response


def _for_code(method: str, path: str, body) -> object:
    """An API call the agent's code makes (code_mode.Call): the whole answer, never cut."""
    try:
        response = _request(method, path, body)
    except ToolError as e:
        raise code_mode.Refused(str(e)) from None
    return response.json() if response.content else None


def _file_for_code(conn: sqlite3.Connection, media_id: int) -> dict:
    """Where an image's file is, for the agent's code (code_mode.Locate)."""
    found = projects.media_file(conn, media_id)
    if found is None:
        raise code_mode.Refused(f"There's no image {media_id}.")
    return found


# --- The conversation's folder: the agent's code and its files ---


def folder() -> Path:
    """The working folder of the conversation whose turn this is (made when first used)."""
    turn = _turn.get()
    if turn is None:
        raise ToolError("There's no conversation here, so no folder.")
    found = config.CONVERSATIONS_DIR / str(turn.conversation_id)
    found.mkdir(parents=True, exist_ok=True)
    return found.resolve()


class CodeArgs(Args):
    code: str = Field(description="Python to run. `babasu` has the projects: babasu.project(id).images() are its "
                             "images, each with its objects and `file` (read it with pixelflow: pf.read_image, "
                             "pf.VideoReader); project.add(paths) puts files you made in as new images. `api` is "
                             "the app's API (api.get, post, put, patch, delete). print() what you need, and the "
                             "counts that check it.")
    seconds: int = Field(code_mode.TIMEOUT, ge=1, le=600, description="How long it may take at most, in seconds.")


def _last_line(text: str) -> str:
    """The last line printed, what a script usually ends with: what it found."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:120] if lines else ""


@tool(
    "Run Python for what no single operation does: going through many images, measuring, comparing, working on "
    "their pixels (making edited copies). One process "
    "for this chat (variables stay for the next run), in its own folder (files stay). read_docs code first. "
    "Returns what it printed.",
    CodeArgs,
    label="Running code",
    note=lambda args, ran: _last_line(ran["output"]),
)
def run_code(conn: sqlite3.Connection, args: CodeArgs) -> Iterator[dict]:
    result = yield from code_mode.session(folder(), frozenset(TOOLS)).run(
        args.code, _for_code, partial(_file_for_code, conn), _turn.get().stop, args.seconds)
    if result["error"]:
        raise ToolError(result["error"] + (f"\nPrinted before it:\n{result['output']}" if result["output"] else ""))
    return {"output": result["output"]}


# --- The tools that show and wait ---


class PageArgs(Args):
    path: str = Field(description="The page's path, with its ids and query, e.g. /images/42.")


@tool(
    "Open a page of the app for the person, once you've answered: when they ask to see or open something, or "
    "it's what you're building (the workflow you're changing); anything else, link in your reply. Its path, e.g. "
    "/projects/3?where=car (the images matching a filter), /images/42, /workflows/ID/edit.",
    PageArgs,
    label="Opening a page",
    opens=True,
)
def show(conn: sqlite3.Connection, args: PageArgs) -> dict:
    path = _page_path(args.path)
    return {"open": path, "title": _page_title(path)}


def _page_path(path: str) -> str:
    """The page the agent means: as given, or the one an API path's data is shown on (/api/projects/3
    is /projects/3, as _api_path reads a page's data); a missing first / added."""
    return ("/" + path.strip().lstrip("/")).removeprefix("/api") or "/"


def _page_title(path: str) -> str:
    """The title of the app's page at `path`, rendered as the browser would get it: checks it shows.
    ToolError if there's no such page."""
    if path.startswith(("//", "/static/")):
        raise ToolError(f"{path!r} isn't a page of the app: give its path, e.g. /projects/3")
    response = _client.get(path, follow_redirects=False)
    if response.status_code != 200 or not response.headers.get("content-type", "").startswith("text/html"):
        raise ToolError(f"{path}: no page there ({response.status_code} {response.reason_phrase}). Look up what "
                        "you're opening first (GET /api/search?q=): its page is in what that answers.")
    return html.unescape(re.search(r"<title>(.*?)</title>", response.text, re.DOTALL)[1]).strip()


class WaitArgs(Args):
    job_id: int = Field(description="The job's id (what started it says it).")
    seconds: int = Field(60, ge=1, le=300, description="How long to wait at most, in seconds.")


@tool("Wait for a background job (a run, an import, a download) to end, or `seconds` to pass; its bar shows "
      "meanwhile. Answers as GET /api/jobs/{job_id}.", WaitArgs,
      label="Waiting for a job", note=lambda args, job: f"{job['title']} {job['status']}" + (f": {job['summary']}" if job["summary"] else ""))
def wait(conn: sqlite3.Connection, args: WaitArgs) -> Iterator[dict]:
    return (yield from _waited(conn, args.job_id, args.seconds))


def _waited(conn: sqlite3.Connection, job_id: int, seconds: int) -> Iterator[dict]:
    """Wait for a job to end, or `seconds` to pass (or the turn to stop), its bar showing meanwhile;
    the job as it is then."""
    job = None
    for job in jobs.watched(conn, job_id, _turn.get().stop, seconds):
        if job["status"] in jobs.ACTIVE and job["total"]:  # the chat shows the job's bar under the step
            yield {"type": "progress", "done": job["done"], "total": job["total"]}
    if job is None:
        raise ToolError(f"There is no job {job_id}.")
    return job


FIND_WAIT = 60  # seconds the find tool waits; a longer run goes on (wait for its job)


class FindArgs(Args):
    images: dict[str, Any] = Field(description=IMAGES)  # checked by the API (routes.Images), refusals naming its fields
    classes: list[str] | None = Field(None, description="What to find, by class name, e.g. [\"person\", \"car\"]; "
                                      "empty: what the workflow finds by itself.")
    workflow_id: str | None = Field(None, description="Which workflow (GET /api/workflows says what each does); empty: the default.")
    again: bool = Field(False, description="Also images already looked at for these classes (replacing what a model found then).")


def _found_note(found: dict) -> str:
    """What a find found, in a few words: "Found person 23, car 2 on 9 images"."""
    if found["status"] in jobs.ACTIVE:
        return f"Still running on {plural(found['looked_at_images'], 'image')} (job {found['job_id']})"
    if found["status"] != "succeeded":
        return f"The run {found['status']}" + (f": {found['error']}" if found.get("error") else "")
    counted = ", ".join(plural(n, f"{name} object") for name, n in found["objects_found"].items()) or "nothing"
    frames = f" ({plural(found['video_frames'], 'video frame')})" if found.get("video_frames") else ""
    return f"Found {counted}, looking at {plural(found['looked_at_images'], 'image')}{frames}"


@tool("Find things in images with a model: run a workflow over them (the default finds and outlines the classes "
      "named) and wait for it. What it finds is kept as the images' annotations, in their projects; it answers how "
      "many objects of each class it found, how many images it looked at, and the projects' pages to link. On a "
      "video, an object counts on each frame it's on, unless the workflow tracks it (SAM 3, tracked): then once. "
      "Before running a model, see whether the images are "
      "annotated already (GET /api/images?where=CLASS). A run not done in a minute goes on: wait for its job.",
      FindArgs, label="Finding objects", note=lambda args, found: found["did"])
def find(conn: sqlite3.Connection, args: FindArgs) -> Iterator[dict]:
    body = {"images": args.images, "classes": args.classes, "workflow_id": args.workflow_id, "again": args.again}
    started = _request("POST", "/api/annotate", {k: v for k, v in body.items() if v not in (None, False)}).json()
    job = yield from _waited(conn, started["job_id"], FIND_WAIT)
    where = {"where": "|".join(args.classes)} if args.classes else {}
    pages = [f"/projects/{p}" + (f"?{urlencode(where)}" if where else "") for p in dict.fromkeys(
        r[0] for r in conn.execute("SELECT project_id FROM media WHERE id IN (SELECT value FROM json_each(?))",
                                   (json.dumps(job["params"]["media_ids"]),)))]
    result = job["result"] or {}
    found = {"job_id": started["job_id"], "status": job["status"], "looked_at_images": started["images"],
             "objects_found": result.get("found", {}), "pages": pages, "error": job["error"]}
    if result.get("frames"):  # on a video, objects_found counts what's on each frame (a tracked one once)
        found["video_frames"] = result["frames"]
    return found | {"did": _found_note(found)}


# --- Looking ---

MARGIN = 0.15  # of an object's size, kept around its crop: what's next to it helps to judge it


class LookArgs(Args):
    media_id: int = Field(description="The image or video (its media_id).")
    object: int | None = Field(None, ge=0, description="#n: one of the image's objects (its annotations), to look at closely "
                                                     "(on a video, on its frame).")
    box: list[float] | None = Field(None, min_length=4, max_length=4, description="Or a part of it: [x, y, w, h] in pixels.")
    frame: int | None = Field(None, ge=0, description="On a video: which frame (its number, as annotations' frame says); "
                                                      "the first if left out.")


@tool(
    "Look at an image yourself (a video: one of its frames): all of it, one of its objects closely, or a part of "
    "it: whether an object really is what it's labelled, what a model missed or got wrong. The picture comes "
    "after the results.",
    LookArgs,
    label="Looking at an image",
    note=lambda args, seen: seen["looked_at"],
    vision=True,
)
def look(conn: sqlite3.Connection, args: LookArgs) -> Seen:
    found = projects.images(conn, [args.media_id])["images"]
    if not found:
        raise ToolError(f"There's no image {args.media_id}: an image's id is its media_id, as its page says "
                        "(/images/MEDIA_ID) and a project's answer lists them.")
    image, frame = found[0], args.frame
    name, objects = image["filename"], image["annotations"]
    what, part = name, None
    if args.object is not None:
        if args.object >= len(objects):
            raise ToolError(f"{name} has no object #{args.object}: " + (f"its objects are #0 to #{len(objects) - 1}."
                            if objects else "it has none."))
        shown = objects[args.object]
        part, frame = projects.corners(shown), shown.get("frame", frame)
        what = f"{shown['class']} (#{args.object}) on {name}"
    elif args.box:
        x, y, w, h = args.box
        what, part = f"part of {name}", (x, y, x + w, y + h)
    if video := image["video"]:
        frame = frame or 0
        if frame >= video["frame_count"]:
            raise ToolError(f"{name} has frames 0 to {video['frame_count'] - 1}.")
        what += f", frame {frame}"
    try:
        picture = Image.open(projects.image_file(conn, args.media_id, frame)["path"])
    except ValueError as e:  # a frame of an image
        raise ToolError(f"{name}: {e}") from None
    if part:
        x1, y1, x2, y2 = part
        dx, dy = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        part = (max(0, x1 - dx), max(0, y1 - dy), min(picture.width, x2 + dx), min(picture.height, y2 + dy))
    return Seen({"looked_at": what}, media.picture(picture, part))


# --- The docs ---


def _docs() -> dict[str, str]:
    """The docs, by topic: the app's by file name, and the docs pixelflow ships (run_code has it) as
    "pixelflow" (its index, llms.txt) and "pixelflow/PATH" (a page, by its path in the index)."""
    docs = {path.stem: path.read_text() for path in sorted(config.DOCS_DIR.glob("*.md"))}
    library = Path(str(importlib.resources.files("pixelflow") / "docs"))  # its package data
    docs["pixelflow"] = (library / "llms.txt").read_text()
    docs |= {f"pixelflow/{path.relative_to(library).with_suffix('').as_posix()}": path.read_text()
             for path in sorted(library.rglob("*.md"))}
    return docs


DOCS = _docs()
TOPICS = [topic for topic in DOCS if "/" not in topic]  # the pages of a library are listed in its index


class DocsArgs(Args):
    topic: str = Field(description="A topic, from the list in read_docs' description.")


@tool(
    "How the app works, and how to do something in it, on a topic: " + ", ".join(TOPICS) + ". Read it when "
    "you're unsure what something is or how it's done, or before explaining it. pixelflow (for run_code) lists "
    "its pages; read one as pixelflow/PATH, e.g. pixelflow/core-concepts/detections.",
    DocsArgs,
    label="Reading the docs",
    note=lambda args, doc: doc.partition("\n")[0].removeprefix("# "),
)
def read_docs(conn: sqlite3.Connection, args: DocsArgs) -> str:
    topic = args.topic.strip().lower().removesuffix(".md").removeprefix("/").removeprefix("docs/")
    if "/" in topic and f"pixelflow/{topic}" in DOCS:  # a link in a pixelflow page: /docs/tracking/zones.md
        topic = f"pixelflow/{topic}"
    if topic in DOCS:
        return DOCS[topic]
    if topic.startswith("pixelflow/"):
        raise ToolError(f"pixelflow has no page {args.topic!r}; read_docs pixelflow lists them.")
    found = _sections_about(topic)
    if not found:
        raise ToolError(f"There is no topic {args.topic!r}, and the docs don't mention it; the topics: {', '.join(TOPICS)}.")
    return (f"There is no topic {args.topic!r}; what the docs say about it (the topics: {', '.join(TOPICS)}):\n\n"
            + "\n\n".join(f"From {name}:\n{text}" for name, text in found))


def _words(text: str) -> set[str]:
    """Words to match, by their first five letters: classes is class, renaming is rename."""
    return {w[:5] for w in re.findall(r"[a-z]{3,}", text.lower())}


@cache
def _sections() -> list[tuple[str, str, set[str], set[str]]]:
    """The app's docs cut at their ## headings: topic, text, its heading's words, its words."""
    found = []
    for name in TOPICS:
        if name != "pixelflow":
            for part in re.split(r"\n(?=## )", DOCS[name]):
                heading = part.partition("\n")[0]
                found.append((name, part.strip(), _words(heading), _words(part)))
    return found


def _sections_about(query: str, most: int = 2) -> list[tuple[str, str]]:
    """The docs' sections that match the words of `query` best (a word in the heading counts more)."""
    wanted = _words(query)
    scored = [(sum(3 * (w in heading) + (w in words) for w in wanted), name, text)
              for name, text, heading, words in _sections()]
    scored = sorted((s for s in scored if s[0]), key=lambda s: -s[0])
    return [(name, text) for _, name, text in scored[:most]]
