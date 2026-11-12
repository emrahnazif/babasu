"""Tools the agent can call: babasu's own functions, the same ones the UI and the API use.

Each tool's arguments are a pydantic model, so the JSON schema the model sees is generated, never
written by hand. Read-only for now; tools that change data will ask for confirmation first.
A tool registered with opens_page=True returns {"open": "/a/page", …}: that page is shown to the
person once the reply is done (app/agent.py, run_turn).
"""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field, ValidationError

from app import jobs, projects
from app.filters import Filter


class NoArgs(BaseModel):
    pass


class ProjectArgs(BaseModel):
    project_id: int = Field(description="The project's id, from list_projects.")


# What the model needs to write a filter: the text form of app/filters.py, in short.
WHERE = """Which images, as a filter; empty for all. Terms, all of which must hold:
labelled | unlabelled; name:TEXT (file name contains); objects:RANGE (how many objects the image has);
(…) the image has an object with ALL of these at once, -(…) it has none: class names (car|truck for
either; quote names with spaces), w:RANGE h:RANGE size:RANGE (pixels; size is both), area:RANGE,
aspect:RANGE (width/height), edge or edge:no (touches the image edge). `car` is short for (car),
`-person` for -(person). RANGE: 8..16, 8.., ..16, >8, <16, >=8, <=16 or 8.
Example: labelled (car size:8..16) -person. Use the project's own class names (get_project)."""


class FilterArgs(ProjectArgs):
    where: str = Field("", description=WHERE)


class StatsArgs(FilterArgs):
    field: Literal[projects.STAT_FIELDS] = Field(  # type: ignore[valid-type]
        description="class: objects per class; w, h, area, aspect: a histogram of the objects (those the filter "
        "asks for in brackets, or all); objects: how many images have how many objects."
    )


class JobArgs(BaseModel):
    job_id: int = Field(description="The job's id, from list_jobs.")


@dataclass(frozen=True)
class Tool:
    description: str
    args: type[BaseModel]
    func: Callable[[sqlite3.Connection, BaseModel], object]
    opens_page: bool = False


TOOLS: dict[str, Tool] = {}


class ToolError(Exception):
    """The model asked for a tool that doesn't exist, or gave arguments that don't fit."""


def tool(description: str, args: type[BaseModel] = NoArgs, opens_page: bool = False):
    """Register a function as a tool, under its own name."""

    def register(func):
        TOOLS[func.__name__] = Tool(description, args, func, opens_page)
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
        raise ToolError(f"Invalid arguments for {name}: {exc.errors(include_url=False)}") from None
    return json.dumps(found.func(conn, args), ensure_ascii=False)


# --- The tools ---


@tool("List all projects, with their type, image counts, how many images are labelled, and class names.")
def list_projects(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return [
        {"id": p["id"], "name": p["name"], "description": p["description"], "type": p["task"], "images": p["image_count"],
         "labelled": p["labelled_count"], "classes": [c["name"] for c in p["classes"]]}
        for p in projects.list_all(conn)
    ]


@tool("Show one project: its type and annotator, its classes (with the prompt sent to annotators and how many annotations each has), and its first image file names.", ProjectArgs)
def get_project(conn: sqlite3.Connection, args: ProjectArgs) -> dict:
    project = projects.detail(conn, args.project_id)
    if project is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return {
        "id": project["id"],
        "name": project["name"],
        "description": project["description"],
        "type": project["task"],
        "annotator": project["annotator"],
        "images": project["image_count"],
        "labelled": project["labelled_count"],
        "classes": [{"id": c["id"], "name": c["name"], "prompt": c["prompt"], "annotations": c["annotation_count"]}
                    for c in project["classes"]],
        "first_images": [image["filename"] for image in project["images"][:20]],
    }


def _filtered(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = projects.detail(conn, args.project_id, filter=Filter.parse(args.where))
    if project is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return project


@tool(
    "Find a project's images matching a filter: how many match, and the first ones with how many of the objects "
    "asked for each has. Use it to check a filter (and see it written back the way it was understood) before showing it.",
    FilterArgs,
)
def find_images(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = _filtered(conn, args)
    first = [
        {"file": i["filename"], "objects": len(i["annotations"]), **({"found": len(i["matches"])} if "matches" in i else {})}
        for i in project["images"][:20]
    ]
    return {"where": project["filter"], "matching_images": project["match_count"], "of": project["image_count"], "first": first}


@tool("Measure a project's images matching a filter: objects per class, the spread of object sizes or aspect, or of objects per image.", StatsArgs)
def describe(conn: sqlite3.Connection, args: StatsArgs) -> dict:
    result = projects.stats(conn, args.project_id, args.field, Filter.parse(args.where))
    if result is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return result


@tool(
    "Show the person a project's images matching a filter: their project page opens with it once you've answered, "
    "where they see it as chips they can change. Say in a few words what you're showing.",
    FilterArgs,
    opens_page=True,
)
def show_images(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = _filtered(conn, args)  # checks the filter, so the page never opens with an error
    query = f"?{urlencode({'where': project['filter']})}" if project["filter"] else ""
    return {"open": f"/projects/{args.project_id}{query}", "matching_images": project["match_count"]}


@tool("List recent background jobs (image imports and more), newest first, with status and progress.")
def list_jobs(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return [
        {"id": j["id"], "title": j["title"], "project": j["project_name"], "status": j["status"],
         "done": j["done"], "total": j["total"], "summary": j["summary"]}
        for j in jobs.list_recent(conn, limit=20)
    ]


@tool("Show one background job: status, progress, result and error.", JobArgs)
def get_job(conn: sqlite3.Connection, args: JobArgs) -> dict:
    job = jobs.get(conn, args.job_id)
    if job is None:
        raise ToolError(f"There is no job {args.job_id}.")
    return {key: job[key] for key in ("id", "title", "project_name", "status", "done", "total", "result", "error",
                                      "created_at", "started_at", "finished_at")}
