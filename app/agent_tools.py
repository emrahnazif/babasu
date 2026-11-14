"""Tools the agent can call: babasu's own functions, the same ones the UI and the API use.

Each tool's arguments are a pydantic model, so the JSON schema the model sees is generated, never
written by hand. Tools that change things don't ask for confirmation: what they change in the
database, and the views and workflows they delete, can be undone for 30 days (app/history.py).
Everything the pages can do goes through call_api, one tool for the whole API (its operations are
listed in the agent's instructions, find_api gives the details); the other tools do what one call
can't: open a page, wait, search across things, write views.
A tool registered with opens=True returns {"open": "/a/page", …}: that page is shown to the person
once the reply is done (app/agent.py, run_turn); after a tool that `writes`, the page they're on
reloads then instead (unless one opens).
"""

import difflib
import json
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, get_args
from urllib.parse import urlencode

from mozo.workflow import catalogue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import config, filters, jobs, projects, runs, views
from app.filters import Filter
from app.routes import UI_ONLY


class Args(BaseModel):
    """A tool's arguments. An argument it doesn't take is refused, not dropped: the model learns
    its mistake (a misspelt field) instead of being told a change it asked for was made."""

    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


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


class ViewIdArgs(Args):
    view_id: str = Field(description="The view's id: lowercase letters, digits and dashes (e.g. model-comparison).")


class WriteViewArgs(ViewIdArgs):
    template: str = Field(description="The whole page: a Jinja template (see get_components).")
    title: str | None = Field(None, description="The page's title; needed for a new view.")
    kind: Literal["report", "studio"] | None = Field(
        None, description="report (the default; kept as it was when written) or studio (a live page for one workflow).")
    workflow: str | None = Field(None, description="For a studio page: the workflow's id, from GET /api/workflows.")


class EditViewArgs(ViewIdArgs):
    old: str = Field(description="A piece of the view's newest template, exactly as it is, found once in it.")
    new: str = Field(description="What replaces it (empty to remove it).")


@dataclass(frozen=True)
class Tool:
    description: str
    args: type[BaseModel]
    func: Callable[[sqlite3.Connection, BaseModel], object]
    opens: bool = False  # it returns {"open": "/a/page", …}: that page is shown to the person after the reply
    # It changes what pages show: the page the person is on reloads after the reply. A function of
    # the arguments for a tool that only sometimes does (call_api: not a GET).
    writes: bool | Callable[[BaseModel], bool] = False


TOOLS: dict[str, Tool] = {}


class ToolError(Exception):
    """The model asked for a tool that doesn't exist, or gave arguments that don't fit."""


def tool(description: str, args: type[BaseModel] = NoArgs, opens: bool = False,
         writes: bool | Callable[[BaseModel], bool] = False):
    """Register a function as a tool, under its own name."""

    def register(func):
        TOOLS[func.__name__] = Tool(description, args, func, opens, writes)
        return func

    return register


def schemas() -> list[dict]:
    """The tool list, in the chat completions API's format."""
    return [
        {"type": "function", "function": {"name": name, "description": t.description, "parameters": t.args.model_json_schema()}}
        for name, t in TOOLS.items()
    ]


def call(conn: sqlite3.Connection, name: str, arguments: str) -> str:
    """Run a tool with the model's JSON arguments; the result as JSON text. Raises ToolError."""
    if name not in TOOLS:
        raise ToolError(f"There is no tool named {name!r}.")
    found = TOOLS[name]
    try:
        args = found.args.model_validate_json(arguments or "{}")
    except ValidationError as exc:
        problems = "; ".join(_problem(found.args, e) for e in exc.errors(include_url=False))
        raise ToolError(f"Invalid arguments for {name}: {problems}") from None
    return json.dumps(found.func(conn, args), ensure_ascii=False)


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

ANSWER_SIZE = 8_000  # characters of an answer the agent gets; a longer one is cut


def _inline(schema, defs: dict):
    """A JSON schema with its $refs put in place and its titles left out, as the agent reads it."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            return _inline(defs[schema["$ref"].split("/")[-1]], defs)
        return {k: _inline(v, defs) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_inline(v, defs) for v in schema]
    return schema


@cache
def operations() -> list[dict]:
    """Every /api operation the agent may call (not those only the pages use: routes.UI_ONLY), from
    the app's OpenAPI spec: its name ("PATCH /api/projects/{project_id}"), description, parameters
    and body schema."""
    spec = views.openapi()
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
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            found.append({"operation": name, "description": op.get("description", ""), "parameters": params,
                          **({"body": _inline(body, defs)} if body else {})})
    return found


@cache
def _search() -> list[tuple[dict, str, re.Pattern]]:
    """Each operation with its text to search (find_api) and the paths it answers (call_api)."""
    def answers(operation: str) -> re.Pattern:
        path = re.escape(operation.split(" ", 1)[1])
        return re.compile(re.sub(r"\\\{\w+\\\}", "[^/]+", path) + "$")  # /api/projects/{project_id} -> /api/projects/[^/]+
    return [(o, json.dumps(o).lower(), answers(o["operation"])) for o in operations()]


@cache
def operation_list() -> str:
    """One line per operation, for the agent's instructions: "GET /api/projects: Every project…"."""
    def first(text: str) -> str:
        sentence = re.split(r"\.\s+(?=[A-Z])", " ".join(text.split()))[0].rstrip(".")  # not at "e.g. to"
        return sentence if len(sentence) <= 160 else sentence[:157] + "…"
    return "\n".join(f"{o['operation']}: {first(o['description'])}" for o in operations())


class FindApiArgs(Args):
    query: str = Field(min_length=1, description='An operation from the list ("PATCH /api/projects/{project_id}"), '
                       'or words ("annotations", "cancel job").')


@tool(
    "The details of the app's API operations: what one does, its parameters and the JSON body it takes. Give the "
    "operation from the list in your instructions, or words to find up to three.",
    FindApiArgs,
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
        raise ToolError("No operation matches those words; the list in your instructions has them all.")
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
    "Do anything the app's pages do: call one of its API operations (the list in your instructions; find_api for "
    "the details). Reads (GET) answer with the data; changes are checked as the pages' are, and a refusal says why.",
    CallApiArgs,
    writes=lambda args: args.method != "GET",
)
def call_api(conn: sqlite3.Connection, args: CallApiArgs) -> object:
    path = args.path.strip()
    target = path.split("?")[0]
    methods = [o["operation"].split(" ")[0] for o, _, answers in _search() if answers.match(target)]
    if not methods:
        raise ToolError(f"There's no operation at {target}; the list in your instructions has them all.")
    if args.method not in methods:
        raise ToolError(f"{target} takes {', '.join(methods)}, not {args.method}.")
    response = views.client().request(args.method, path, json=args.body)
    if response.status_code >= 400:
        raise ToolError(views.refusal(response))
    if not response.content:
        return {"done": True}
    if len(response.text) <= ANSWER_SIZE:
        return response.json()
    return {"cut": f"The answer is {len(response.text):,} characters; its start (ask for less: a filter, ?images=, "
            "?limit=):", "start": response.text[:ANSWER_SIZE]}


# --- The tools ---


@tool(
    "Open a page of the app for the person, once you've answered. When they ask to see, show, open or find "
    "something, open its page: it shows it better than a list in your reply. Something named: find it first "
    "(search, find_images), its page is in what they return. The pages: /projects (all projects), "
    "/projects/new (create one), /projects/ID (its images; ?where=FILTER for some of them), "
    "/projects/ID/images/IMAGE_ID (one image with its objects), /projects/ID/edit (its settings: name, type, "
    "annotator, classes, delete), /workflows, /workflows/new, /workflows/ID/edit (the builder), /studio/ID (try a "
    "workflow on images), /models (vision models), /jobs/ID (a job and its log), /views/ID (a view), "
    "/settings/general, /settings/models and /settings/agents (the models and agents you run on).",
    PageArgs,
    opens=True,
)
def open_page(conn: sqlite3.Connection, args: PageArgs) -> dict:
    try:
        return {"open": args.path, "title": views.page_title(args.path)}
    except views.ApiError as e:
        raise ToolError(f"{e}. Look up what you're opening first (search, find_images): its page is in what they "
                        "return.") from None


@tool(
    "Find projects, images, workflows and views by name (part of it, ignoring case) anywhere in the app, each "
    "with its page. Use it when the person names something without saying where it is; if more than one "
    "thing could be what they mean, ask which.",
    SearchArgs,
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
        "views": [{"id": v["id"], "title": v["title"], "page": v["url"]}
                  for v in views.list_all() if text in v["id"] or text in v["title"].lower()],
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
    "many of the objects asked for each has. Use it to check a filter (and see it written back the way it was understood) before showing it.",
    FilterArgs,
)
def find_images(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = _filtered(conn, args)
    first = [
        _image(project, i) | {"objects": len(i["annotations"]), **({"found": len(i["matches"])} if "matches" in i else {})}
        for i in project["images"][:20]
    ]
    return {"filter": project["filter"], "match_count": project["match_count"], "image_count": project["image_count"],
            "first": first}


@tool(
    "Show the person a project's images matching a filter: their project page opens with it once you've answered, "
    "where they see it as chips they can change. Say in a few words what you're showing.",
    FilterArgs,
    opens=True,
)
def show_images(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = _filtered(conn, args)  # checks the filter, so the page never opens with an error
    query = f"?{urlencode({'where': project['filter']})}" if project["filter"] else ""
    return {"open": f"/projects/{args.project_id}{query}", "match_count": project["match_count"]}


@tool("Wait for a background job to end (or for `seconds` to pass), then show it as GET /api/jobs/{job_id} does.", WaitArgs)
def wait_for_job(conn: sqlite3.Connection, args: WaitArgs) -> dict:
    deadline = time.monotonic() + args.seconds
    while (job := jobs.get(conn, args.job_id)) and job["status"] in jobs.ACTIVE and time.monotonic() < deadline:
        time.sleep(1)
    if job is None:
        raise ToolError(f"There is no job {args.job_id}.")
    return job


# --- Views ---


@tool("How to build a view (a page in the app you write) and the API a view can read. Read it before writing one.")
def get_components(conn: sqlite3.Connection, args: NoArgs) -> str:
    return views.reference()


def _docs() -> dict[str, list[str]]:
    """The docs, by topic (the file's name): its lines. A doc starts with its title and a line saying what's in it."""
    return {path.stem: path.read_text().splitlines() for path in sorted(config.DOCS_DIR.glob("*.md"))}


DOCS = _docs()


@tool(
    "Read how the app works, on a topic: what things are, how they relate, what's there and what comes later. Read "
    "it when you're unsure what something is or does, or before explaining it. The topics: "
    + "; ".join(f"{topic} ({lines[1].rstrip('.')})" for topic, lines in DOCS.items()) + ".",
    DocsArgs,
)
def read_docs(conn: sqlite3.Connection, args: DocsArgs) -> str:
    topic = args.topic.strip().lower().removesuffix(".md")
    if topic not in DOCS:
        raise ToolError(f"There is no topic {args.topic!r}; the topics: {', '.join(DOCS)}.")
    return "\n".join(DOCS[topic])


def _view(view_id: str, version: int | None = None) -> dict:
    view = views.get(view_id, version)
    if view is None:
        raise ToolError(f"There is no view {view_id!r}" + (f" version {version}." if version else "."))
    return view


def _opened(view: dict) -> dict:
    return {"open": view["url"], "id": view["id"], "version": view["version"]}


@tool(
    "Write a view: a new one, or a new version of one (its whole template). The page is checked first; if it "
    "doesn't render you get the line and why, and nothing is saved. It opens for the person once you've answered.",
    WriteViewArgs,
    opens=True,
)
def write_view(conn: sqlite3.Connection, args: WriteViewArgs) -> dict:
    return _opened(views.save(args.view_id, args.template, args.title, args.kind, args.workflow))


@tool(
    "Change part of a view: replace one exact piece of its newest template, saved as a new version (checked first, "
    "as write_view). Use it for small changes instead of writing the whole page again.",
    EditViewArgs,
    opens=True,
)
def edit_view(conn: sqlite3.Connection, args: EditViewArgs) -> dict:
    template = _view(args.view_id)["template"]
    old, new = args.old, args.new
    if template.count(old) == 0 and old.strip() and template.count(old.strip()) == 1:
        old, new = old.strip(), new.strip()  # the piece, give or take the spaces around it
    found = template.count(old)
    if found > 1:
        raise ToolError(f"That piece is in the template {found} times; give more of it, so it's found once.")
    if not found:
        lines = template.splitlines()
        near = difflib.get_close_matches(old.strip().splitlines()[0] if old.strip() else "", lines, n=3, cutoff=0.5)
        hint = "".join(f"\n  line {lines.index(n) + 1}: {n!r}" for n in near)
        raise ToolError("That piece isn't in the template; copy it exactly from GET /api/views/{view_id}."
                        + (f" The closest lines:{hint}" if hint else ""))
    return _opened(views.save(args.view_id, template.replace(old, new)))


# --- Building workflows ---


@tool(
    "List the nodes a workflow is built from (name, category, what it does); with `names`, those nodes in full: "
    "inputs and outputs (name: type) and parameters (name, kind, default, options; `prompts`: it takes the prompts). "
    "A run keeps what reaches a save_to_babasu node, under its label; read_media is where each image comes in.",
    NodesArgs,
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

