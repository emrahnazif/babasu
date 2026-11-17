"""Tools the agent can call: babasu's own functions, the same ones the UI and the API use.

Each tool's arguments are a pydantic model, so the JSON schema the model sees is generated, never
written by hand. Tools that change things don't ask for confirmation: what they change in the
database, and the workflows they delete, can be undone for 30 days (app/history.py).
Everything the pages can do goes through call_api, one tool for the whole API (find_api finds an
operation and its details); the other tools do what one call can't: open a page, wait, search
across things, run code.
A tool registered with opens=True returns {"open": "/a/page", …}: that page is shown to the person
once the reply is done (app/agent.py, run_turn); after a tool that `writes`, the page they're on
reloads then instead (unless one opens).

What the person sees of a call is one line: its label, what it's doing (the `step` the model gives,
or the tool's own label), and its note, what came out (written from the result by the tool, so it's
always true), with the images it found. An earlier turn's result reaches the model as its note.
"""

import difflib
import html
import inspect
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mozo.workflow import catalogue
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app import code as code_mode  # `code` is run_code's argument
from app import config, filters, jobs, media, projects, runs, settings
from app.filters import Filter
from app.routes import UI_ONLY


class Args(BaseModel):
    """A tool's arguments. An argument it doesn't take is refused, not dropped: the model learns
    its mistake (a misspelt field) instead of being told a change it asked for was made."""

    model_config = ConfigDict(extra="forbid")
    step: str = Field("", max_length=100, description="What this does, for the person: Counting cars in Street cars")


class PageArgs(Args):
    path: str = Field(description="The page's path, with its ids and query, e.g. /projects/3/images/42.")


class SearchArgs(Args):
    text: str = Field(min_length=1, description="Part of the name, e.g. img_0042 or street.")


class ProjectArgs(Args):
    project_id: int = Field(description="The project's id (GET /api/projects lists them).")


class FilterArgs(ProjectArgs):
    where: str = Field("", description=filters.WHERE)


class JobArgs(Args):
    job_id: int = Field(description="The job's id (GET /api/jobs lists them).")


class WaitArgs(JobArgs):
    seconds: int = Field(60, ge=1, le=300, description="How long to wait at most, in seconds.")


class NodesArgs(Args):
    names: list[str] | None = Field(None, description="Nodes to show in full (inputs, outputs, parameters); empty for "
                                    "the list of all nodes, each with its category and what it does.")


class DocsArgs(Args):
    topic: str = Field(description="A topic, from the list in read_docs' description.")


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
    # It changes what pages show: the page the person is on reloads after the reply. A function of
    # the arguments for a tool that only sometimes does (call_api: not a GET).
    writes: bool | Callable[[BaseModel], bool] = False


TOOLS: dict[str, Tool] = {}


class ToolError(Exception):
    """The model asked for a tool that doesn't exist, or gave arguments that don't fit."""


def tool(description: str, args: type[Args], label: str, note: Callable[[Args, object], str] = lambda args, result: "",
         shows: Callable[[Args, object], tuple[int, list[int]]] | None = None, opens: bool = False,
         vision: bool = False, writes: bool | Callable[[Args], bool] = False):
    """Register a function as a tool, under its own name."""

    def register(func):
        TOOLS[func.__name__] = Tool(description, args, func, label, note, shows, opens, vision, writes)
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


@dataclass(frozen=True)
class _Turn:
    conversation_id: int  # run_code keeps one code process per conversation, in its folder
    stop: threading.Event  # set when the person stops the turn: a tool that takes long ends at once


_turn: ContextVar[_Turn | None] = ContextVar("turn", default=None)


@contextmanager
def turn_of(conversation_id: int, stop: threading.Event) -> Iterator[None]:
    """Tools called inside this are a turn of this conversation (agent.run_turn)."""
    token = _turn.set(_Turn(conversation_id, stop))
    try:
        yield
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
    return step.strip() if isinstance(step, str) and step.strip() else found.label


@dataclass(frozen=True)
class Outcome:
    """A call that ran (or failed): its result for the model (JSON text), and what the person sees
    of it: its note, the images it found ({"project_id", "media_ids"}) and the picture it gave the model."""

    result: str
    note: str
    images: dict | None = None
    picture: bytes | None = None


def call(conn: sqlite3.Connection, name: str, arguments: str) -> Iterator[dict]:
    """Run a tool with the model's JSON arguments. Yields what it shows while it works (a tool
    that's a generator: {"type": "output", "text"}, {"type": "progress", "done", "total"});
    returns its Outcome. Raises ToolError."""
    if name not in TOOLS:
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
    images = {"project_id": shown[0], "media_ids": shown[1]} if shown and shown[1] else None
    return Outcome(json.dumps(result, ensure_ascii=False), found.note(args, result), images, picture)


def writes(name: str, arguments: str) -> bool:
    """Whether this call of a tool (one that ran) changed what pages show."""
    found = TOOLS[name]
    return found.writes(found.args.model_validate_json(arguments or "{}")) if callable(found.writes) else found.writes


def _problem(args: type[BaseModel], error: dict) -> str:
    """One argument error in words, where it is; an argument that isn't one says which there are."""
    where = ".".join(str(part) for part in error["loc"]) or "arguments"
    if error["type"] != "extra_forbidden":
        return f"{where}: {error['msg']}"
    model = args
    for part in error["loc"][:-1]:  # down to the object holding the unknown name, through lists and Nones
        if isinstance(part, str):
            model = next(t for t in _types(model.model_fields[part].annotation)
                         if isinstance(t, type) and issubclass(t, BaseModel))
    return f"{where}: not an argument here; it takes {', '.join(model.model_fields)}"


def _types(annotation) -> list:
    """The types an annotation is made of: list[ClassEdit] | None -> [list, ClassEdit, None]."""
    return [annotation, *(t for arg in get_args(annotation) for t in _types(arg))]


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


ANSWER_SIZE = 8_000  # characters of an answer the agent reads; a longer one is refused, with how to ask for less


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
    """Each operation with its text to search (find_api) and the paths it answers (call_api)."""
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


class CallApiArgs(Args):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(description="The operation's path with its ids and query, e.g. /api/projects/3?images=0.")
    body: dict[str, Any] | list[Any] | None = Field(
        None, description="The JSON body (an object or a list), for an operation that takes one (find_api shows it).")

    @field_validator("body", mode="before")
    @classmethod
    def _not_text(cls, value):
        if isinstance(value, str):
            raise ValueError("send the body as JSON itself (an object or a list), not as text holding JSON")
        return value


@tool(
    "Do anything the app's pages do: call one of its API operations (find_api finds one and its details). Reads (GET) answer with the data; changes are checked as the pages' are, and a refusal says why.",
    CallApiArgs,
    label="Using the app's data",
    note=lambda args, answer: f"{len(answer)} found" if isinstance(answer, list) else "",
    writes=lambda args: args.method != "GET",
)
def call_api(conn: sqlite3.Connection, args: CallApiArgs) -> object:
    response = _request(args.method, args.path, args.body)
    if not response.content:
        return {"done": True}
    if len(response.text) > ANSWER_SIZE:
        answer = response.json()
        shape = f"its fields: {', '.join(answer)}" if isinstance(answer, dict) else f"a list of {len(answer):,}"
        asks = [f"?{p['name']}=" for o in _operations_at(args.path) if o["operation"].startswith(f"{args.method} ")
                for p in o["parameters"] if p["in"] == "query"]
        less = f"Ask for less ({', '.join(asks)}), or go through it in code (run_code)." if asks else \
            "It can't be asked for in parts: go through it in code (run_code)."
        raise ToolError(f"That answer is {len(response.text):,} characters, too long to read here ({shape}). {less}")
    return response.json()


def _operations_at(path: str) -> list[dict]:
    """The operations that answer at a path (with its ids; its query left out), one a method."""
    target = path.strip().split("?")[0]
    return [o for o, _, answers in _search() if answers.match(target)]


def _request(method: str, path: str, body=None):
    """One of the agent's API operations, made in this turn (its writes are the turn's action).
    Raises ToolError: an unknown operation, a method the path doesn't take, the API's refusal."""
    path = path.strip()
    target = path.split("?")[0]
    methods = [o["operation"].split(" ")[0] for o in _operations_at(path)]
    if not methods:
        raise ToolError(f"There's no operation at {target}; find_api finds them.")
    if method not in methods:
        raise ToolError(f"{target} takes {', '.join(methods)}, not {method}.")
    response = _client.request(method, path, json=body)
    if response.status_code >= 400:
        raise ToolError(_refusal(response))
    return response


def _for_code(method: str, path: str, body) -> object:
    """An API call the agent's code makes (code_mode.Call): the whole answer, never cut."""
    try:
        response = _request(method, path, body)
    except ToolError as e:
        raise code_mode.Refused(str(e)) from None
    return response.json() if response.content else None


# --- The conversation's folder: the agent's code, results and drafts ---

READ_SIZE = 8_000  # characters of a file read_file gives at once; the rest in parts
LISTED = 300  # files list_files names at most


def folder() -> Path:
    """The working folder of the conversation whose turn this is (made when first used)."""
    turn = _turn.get()
    if turn is None:
        raise ToolError("There's no conversation here, so no folder.")
    found = config.CONVERSATIONS_DIR / str(turn.conversation_id)
    found.mkdir(parents=True, exist_ok=True)
    return found.resolve()


def _path(relative: str, kind: Literal["file", "folder"] | None = None) -> Path:
    """A path in the folder (a file or folder that's there, if `kind` says so); ToolError for one
    outside it (absolute, .., a link out) or not there."""
    root = folder()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ToolError(f"{relative}: outside your folder; paths are relative to it, e.g. results/misses.json")
    if kind == "file" and not path.is_file() or kind == "folder" and not path.is_dir():
        raise ToolError(f"{relative}: no such {kind}")
    return path


class PathArgs(Args):
    path: str = Field(".", description="A path in your folder, e.g. steps; . for the folder itself.")


@tool(f"The files in your folder (or a folder in it), with their sizes in bytes (at most {LISTED}). Your folder is "
      "this chat's: your scripts, the results they save, your drafts. run_code works in it.", PathArgs,
      label="Listing files", note=lambda args, found: f"{len(found['files']) + found.get('more', 0)} files")
def list_files(conn: sqlite3.Connection, args: PathArgs) -> dict:
    root = folder()
    files = [p for p in _path(args.path, "folder").rglob("*") if p.is_file()]
    listed = [{"path": str(p.relative_to(root)), "size": p.stat().st_size} for p in sorted(files)[:LISTED]]
    return {"files": listed} | ({"more": len(files) - LISTED} if len(files) > LISTED else {})


class ReadArgs(Args):
    path: str = Field(description="A file in your folder.")
    offset: int = Field(0, ge=0, description="Where to start, in characters (for the next part of a big file).")


@tool(f"Read a file of your folder, {READ_SIZE:,} characters at a time (a bigger one says where the next part "
      "starts). For data, read what you need in code instead: a file is for you to check, not to copy from.",
      ReadArgs, label="Reading a file", note=lambda args, read: args.path)
def read_file(conn: sqlite3.Connection, args: ReadArgs) -> dict:
    path = _path(args.path, "file")
    with path.open(errors="replace") as f:
        f.read(args.offset)
        part = f.read(READ_SIZE + 1)
    if len(part) <= READ_SIZE:
        return {"text": part}
    return {"text": part[:READ_SIZE], "next": args.offset + READ_SIZE, "bytes": path.stat().st_size}


class WriteArgs(Args):
    path: str = Field(description="A file in your folder, e.g. steps/compare.py (folders are made as needed).")
    content: str = Field(description="All of the file.")


@tool("Write a file in your folder (all of it; a file that's there is replaced): a script to run with run_code, "
      "notes, a draft. Results your code makes, it saves itself. Your folder is your working copy: undoing a turn "
      "doesn't change its files.", WriteArgs, label="Writing a file", note=lambda args, written: args.path)
def write_file(conn: sqlite3.Connection, args: WriteArgs) -> dict:
    path = _path(args.path)
    settings.write_file(path, args.content)
    return {"written": args.path, "size": path.stat().st_size}


class EditArgs(Args):
    path: str = Field(description="A file in your folder.")
    old: str = Field(min_length=1, description="Text in the file to replace, exactly as it is there, once.")
    new: str = Field(description="What replaces it.")


@tool("Change part of a file in your folder: `old`, which must be in it exactly once, becomes `new`.", EditArgs,
      label="Editing a file", note=lambda args, edited: args.path)
def edit_file(conn: sqlite3.Connection, args: EditArgs) -> dict:
    """`old` is found once, give or take the spaces around it; not found, the closest lines say where to look."""
    path = _path(args.path, "file")
    text, old, new = path.read_text(), args.old, args.new
    if text.count(old) == 0 and old.strip() and text.count(old.strip()) == 1:
        old, new = old.strip(), new.strip()
    found = text.count(old)
    if found > 1:
        raise ToolError(f"That piece is in {args.path} {found} times; give more of it, so it's found once.")
    if not found:
        lines = text.splitlines()
        first = old.strip().splitlines()[0] if old.strip() else ""
        near = difflib.get_close_matches(first, lines, n=3, cutoff=0.5)
        hint = "".join(f"\n  line {lines.index(n) + 1}: {n!r}" for n in near)
        raise ToolError(f"That piece isn't in {args.path}; copy it exactly from read_file."
                        + (f" The closest lines:{hint}" if hint else ""))
    settings.write_file(path, text.replace(old, new))
    return {"edited": args.path}


class CodeArgs(Args):
    code: str | None = Field(None, description="Python to run. `api` is the app's API: api.get(path), api.post(path, body), "
                             "api.put, api.patch, api.delete: the JSON answer back; a refusal raises Refused with the "
                             "reason. api.images(project_id, where='') goes through a project's images: each has "
                             "media_id, filename and annotations (objects: class_id, bbox [x, y, w, h], …); write an "
                             "image's objects back with api.put(f'/api/projects/{id}/images/{media_id}/annotations', "
                             "objects). Class ids and names: api.get(f'/api/projects/{id}?images=0')['classes']. "
                             "print() what you need to see or report (and the checks: counts read, matched, left "
                             "out); progress(done, total) shows a bar while a long loop runs. pixelflow and numpy "
                             "are there.")
    file: str | None = Field(None, description="Instead of code: a .py file of your folder to run.")
    seconds: int = Field(code_mode.TIMEOUT, ge=1, le=600, description="How long it may take at most, in seconds.")

    @model_validator(mode="after")
    def _one(self):
        if (self.code is None) == (self.file is None):
            raise ValueError("give code or file (one of them)")
        return self


def _last_line(text: str) -> str:
    """The last line printed, what a script usually ends with: what it found."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:120] if lines else ""


@tool(
    "Run Python for what one API call can't do: going through many images or projects, filtering objects, "
    "changing many at once, combining projects, measuring, comparing. Read read_docs code first (the client, "
    "paging, examples). It runs in your folder (files open by relative paths; a .py file runs like code), in one "
    "process kept for this chat: variables stay for the next run, and files stay for good. Work in steps: save "
    "what a step finds (a json file), look at it, decide, go on with it instead of copying it. Its changes go "
    "through the API, so they're checked, and undone with your turn (files in your folder stay as they are). "
    "Returns what it printed.",
    CodeArgs,
    label="Running code",
    note=lambda args, ran: _last_line(ran["output"]),
    writes=True,
)
def run_code(conn: sqlite3.Connection, args: CodeArgs) -> Iterator[dict]:
    code = _path(args.file, "file").read_text() if args.file else args.code
    result = yield from code_mode.session(folder()).run(code, _for_code, _turn.get().stop, args.seconds, args.file)
    if result["error"]:
        raise ToolError(result["error"] + (f"\nPrinted before it:\n{result['output']}" if result["output"] else ""))
    return {"output": result["output"]}


# --- The tools ---


@tool(
    "Open a page of the app for the person, once you've answered: when they ask to see or open something, or it's "
    "what you're building (the workflow you're changing); anything else, link in your reply. Something named: find it first "
    "(search, find_images), its page is in what they return. The pages: /projects (all projects), "
    "/projects/new (create one), /projects/ID (its images; ?where=FILTER for some of them), "
    "/projects/ID/images/IMAGE_ID (one image with its objects), /projects/ID/edit (its settings: name, type, "
    "annotator, classes, delete), /workflows, /workflows/new, /workflows/ID/edit (the builder), /models (vision "
    "models), /jobs/ID (a job and its log), "
    "/settings/general, /settings/models and /settings/agents (the models and agents you run on).",
    PageArgs,
    label="Opening a page",
    opens=True,
)
def open_page(conn: sqlite3.Connection, args: PageArgs) -> dict:
    return {"open": args.path, "title": _page_title(args.path)}


def _page_title(path: str) -> str:
    """The title of the app's page at `path`, rendered as the browser would get it: checks it shows.
    ToolError if there's no such page."""
    if not path.startswith("/") or path.startswith(("//", "/api/", "/static/")):
        raise ToolError(f"{path!r} isn't a page of the app: give its path, e.g. /projects/3")
    response = _client.get(path, follow_redirects=False)
    if response.status_code != 200 or not response.headers.get("content-type", "").startswith("text/html"):
        raise ToolError(f"{path}: no page there ({response.status_code} {response.reason_phrase}). Look up what "
                        "you're opening first (search, find_images): its page is in what they return.")
    return html.unescape(re.search(r"<title>(.*?)</title>", response.text, re.DOTALL)[1]).strip()


@tool(
    "Find projects, images and workflows by name (part of it, ignoring case) anywhere in the app, each "
    "with its page. Use it when the person names something without saying where it is; if more than one "
    "thing could be what they mean, ask which.",
    SearchArgs,
    label="Searching by name",
    note=lambda args, found: ", ".join(f"{len(found[kind])} {kind}" for kind in ("projects", "images", "workflows")
                                       if found[kind]) or "nothing found",
)
def search(conn: sqlite3.Connection, args: SearchArgs) -> dict:
    found = projects.named(conn, args.text.strip())
    text = args.text.strip().lower()
    return {
        "projects": [p | {"page": f"/projects/{p['id']}"} for p in found["projects"]],
        "images": [{"id": i["media_id"], "filename": i["filename"], "project": i["project"],
                    "page": f"/projects/{i['project_id']}/images/{i['media_id']}"} for i in found["images"]],
        "image_count": found["image_count"],
        "workflows": [{"id": w["id"], "name": w["name"], "page": f"/workflows/{w['id']}/edit"}
                      for w in runs.workflows() if text in w["id"] or text in (w["name"] or "").lower()],
    }


def _image(project: dict, image: dict) -> dict:
    """An image as tools list it: its id, file name and page (stepping through the project's filter, if any)."""
    query = f"?{urlencode({'where': project['filter']})}" if project["filter"] else ""
    return {"id": image["media_id"], "filename": image["filename"],
            "page": f"/projects/{project['id']}/images/{image['media_id']}{query}"}


def _filtered(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = projects.detail(conn, args.project_id, filter=Filter.parse(args.where))
    if project is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return project


@tool(
    "Find a project's images matching a filter: how many match, and the first ones (id, file name, page) with how "
    "many objects of each class each has (and how many of those asked for). Use it to check a filter (and see it written back the way it was understood) before showing it.",
    FilterArgs,
    label="Finding images",
    note=lambda args, found: f"{found['match_count']:,} of {found['image_count']:,} images",
    shows=lambda args, found: (args.project_id, [image["id"] for image in found["first"][:4]]),
)
def find_images(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = _filtered(conn, args)
    first = [
        _image(project, i) | {"objects": i["objects"], **({"found": len(i["matches"])} if "matches" in i else {})}
        for i in project["images"][:20]
    ]
    return {"filter": project["filter"], "match_count": project["match_count"], "image_count": project["image_count"],
            "first": first}


@tool(
    "When they ask to see a project's images matching a filter: their project page opens with it once you've answered, "
    "where they see it as chips they can change. Say in a few words what you're showing.",
    FilterArgs,
    label="Showing images",
    note=lambda args, shown: f"{shown['match_count']:,} images",
    opens=True,
)
def show_images(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = _filtered(conn, args)  # checks the filter, so the page never opens with an error
    query = f"?{urlencode({'where': project['filter']})}" if project["filter"] else ""
    return {"open": f"/projects/{args.project_id}{query}", "match_count": project["match_count"]}


@tool("Wait for a background job to end (or for `seconds` to pass), then show it as GET /api/jobs/{job_id} does.", WaitArgs,
      label="Waiting for a job", note=lambda args, job: f"{job['title']} {job['status']}" + (f": {job['summary']}" if job["summary"] else ""))
def wait_for_job(conn: sqlite3.Connection, args: WaitArgs) -> Iterator[dict]:
    deadline, stop = time.monotonic() + args.seconds, _turn.get().stop
    while (job := jobs.get(conn, args.job_id)) and job["status"] in jobs.ACTIVE and time.monotonic() < deadline:
        if job["total"]:  # the chat shows the job's bar under the step
            yield {"type": "progress", "done": job["done"], "total": job["total"]}
        if stop.wait(1):
            break
    if job is None:
        raise ToolError(f"There is no job {args.job_id}.")
    return job


# --- Looking ---

MARGIN = 0.15  # of an object's size, kept around its crop: what's next to it helps to judge it


class LookArgs(ProjectArgs):
    media_id: int = Field(description="The image's id (media_id).")
    object: int | None = Field(None, ge=0, description="One of its objects to look at closely: its place (#0, #1… "
                               "in the image's annotations).")
    box: list[float] | None = Field(None, min_length=4, max_length=4, description="Or a part of it: [x, y, w, h] in pixels.")


@tool(
    "Look at an image of a project, or closely at one of its objects or a part of it, to check what's there: "
    "whether an object really is what it's labelled, what a model missed. The picture comes after the results.",
    LookArgs,
    label="Looking at an image",
    note=lambda args, seen: seen["looked_at"],
    vision=True,
)
def look_at(conn: sqlite3.Connection, args: LookArgs) -> Seen:
    found = projects.images(conn, args.project_id, [args.media_id])
    if not found or not found["images"]:
        raise ToolError(f"Project {args.project_id} has no image {args.media_id}: an image's id is its media_id, as "
                        "its page says (/projects/ID/images/MEDIA_ID) and find_images gives them.")
    image, part = found["images"][0], None
    what = image["filename"]
    if args.object is not None:
        objects = image["annotations"]
        if args.object >= len(objects):
            raise ToolError(f"{image['filename']} has no object #{args.object}: its objects are #0 to #{len(objects) - 1}, "
                            "as its page lists them." if objects else f"{image['filename']} has no objects.")
        annotation = objects[args.object]
        what, part = f"{projects.class_names(found)[annotation['class_id']]} (#{args.object}) on {image['filename']}", \
            projects.corners(annotation)
    elif args.box:
        x, y, w, h = args.box
        what, part = f"part of {image['filename']}", (x, y, x + w, y + h)
    if part:
        x1, y1, x2, y2 = part
        dx, dy = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        part = (max(0, x1 - dx), max(0, y1 - dy), min(image["width"], x2 + dx), min(image["height"], y2 + dy))
    path = projects.image_file(conn, args.project_id, args.media_id)["path"]
    return Seen({"looked_at": what}, media.picture(path, part))


# --- The docs ---


def _docs() -> dict[str, list[str]]:
    """The docs, by topic (the file's name): its lines. A doc starts with its title and a line saying what's in it."""
    return {path.stem: path.read_text().splitlines() for path in sorted(config.DOCS_DIR.glob("*.md"))}


DOCS = _docs()


@tool(
    "Read how the app works, on a topic: what things are, how they relate, what's there and what comes later. Read "
    "it when you're unsure what something is or does, or before explaining it. The topics: "
    + "; ".join(f"{topic} ({lines[1].rstrip('.')})" for topic, lines in DOCS.items()) + ".",
    DocsArgs,
    label="Reading the docs",
    note=lambda args, doc: doc.partition("\n")[0].removeprefix("# "),
)
def read_docs(conn: sqlite3.Connection, args: DocsArgs) -> str:
    topic = args.topic.strip().lower().removesuffix(".md")
    if topic not in DOCS:
        raise ToolError(f"There is no topic {args.topic!r}; the topics: {', '.join(DOCS)}.")
    return "\n".join(DOCS[topic])


# --- Building workflows ---


@tool(
    "List the nodes a workflow is built from (name, category, what it does); with `names`, those nodes in full: "
    "inputs and outputs (name: type) and parameters (name, kind, default, options; `prompts`: it takes the prompts). "
    "A run keeps what reaches a save_to_babasu node, under its label; read_media is where each image comes in.",
    NodesArgs,
    label="Looking at workflow nodes",
    note=lambda args, nodes: f"{len(nodes)} nodes",
)
def list_nodes(conn: sqlite3.Connection, args: NodesArgs) -> list[dict]:
    def ports(items: list[dict]) -> list[str]:
        return [f"{p['name']}: {p['type']}" for p in items]

    def parameter(p: dict) -> dict:
        return {k: v for k, v in p.items() if v not in (None, [], "", False) or k == "default"}

    nodes = catalogue()
    if args.names is None:
        return [{"name": n["name"], "category": n["category"], "description": n["description"]} for n in nodes]
    unknown = set(args.names) - {n["name"] for n in nodes}
    if unknown:
        raise ToolError(f"No such nodes: {sorted(unknown)}; list_nodes without names lists them.")
    return [{"name": n["name"], "category": n["category"], "description": n["description"],
             "inputs": ports(n["inputs"]), "outputs": ports(n["outputs"]),
             "parameters": [parameter(p) for p in n["parameters"]]}
            for n in nodes if n["name"] in args.names]

