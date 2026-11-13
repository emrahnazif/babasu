"""Tools the agent can call: babasu's own functions, the same ones the UI and the API use.

Each tool's arguments are a pydantic model, so the JSON schema the model sees is generated, never
written by hand. Tools that change things (downloading, loading models) don't ask for
confirmation yet; some may later (product.md "The agent").
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

from app import annotators, jobs, projects, runs, vision
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


class RunArgs(FilterArgs):
    workflow_id: str = Field(description="The workflow's id, from list_workflows.")
    count: int | None = Field(None, ge=1, le=1000, description="How many of the matching images, picked at random; empty for all of them.")
    prompts: list[str] | None = Field(None, description="What to find, as text prompts (one per class); empty for the project's class prompts.")
    name: str | None = Field(None, description="A name for the run; empty for one made from the workflow and the image count.")


class PromptArgs(ProjectArgs):
    prompts: dict[int, str] = Field(
        description="Class id (from get_project) -> its new prompt, the text annotators look for; an empty prompt goes "
        "back to the class's name. Classes left out keep theirs."
    )


class AutoAnnotateArgs(FilterArgs):
    count: int | None = Field(None, ge=1, description="How many of the images to do, picked at random among those with work left; empty for all.")
    class_ids: list[int] | None = Field(None, description="Which classes (ids from get_project); empty for all of them.")
    min_score: float = Field(0.0, ge=0, le=1, description="Leave out objects scoring below this (0..1), beyond the annotator's own threshold.")
    again: bool = Field(False, description="Also run images already auto-annotated for a class (new objects that are already there are still left out).")


class RunIdArgs(BaseModel):
    run_id: int = Field(description="The run's id, from start_run or list_runs.")


class ModelArgs(BaseModel):
    family: str = Field(description="The model family, from list_models, e.g. sam3.")
    variant: str = Field(description="The variant within it, from list_models, e.g. sam3.")


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
        {"id": p["id"], "name": p["name"], "description": p["description"], "task": p["task"],
         "image_count": p["image_count"], "labelled_count": p["labelled_count"], "classes": [c["name"] for c in p["classes"]]}
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
        "task": project["task"],
        "annotator": project["annotator"],
        "image_count": project["image_count"],
        "labelled_count": project["labelled_count"],
        "classes": [{"id": c["id"], "name": c["name"], "prompt": c["prompt"], "annotation_count": c["annotation_count"]}
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
        {"filename": i["filename"], "objects": len(i["annotations"]), **({"found": len(i["matches"])} if "matches" in i else {})}
        for i in project["images"][:20]
    ]
    return {"filter": project["filter"], "match_count": project["match_count"], "image_count": project["image_count"],
            "first": first}


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
    return {"open": f"/projects/{args.project_id}{query}", "match_count": project["match_count"]}


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


@tool(
    "List the vision models mozo publishes (for annotation and workflows), by family: task, description, and per "
    "variant its licence, size in bytes, bytes left to download, and state (not downloaded, downloading, downloaded, "
    "loading, loaded). Mention a model's licence when you suggest it."
)
def list_models(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return vision.catalogue(conn)


@tool("Download a vision model's files, as a background job (follow it with get_job). Large models take a while.", ModelArgs)
def download_model(conn: sqlite3.Connection, args: ModelArgs) -> dict:
    return {"job_id": vision.start_download(conn, args.family, args.variant)}


@tool("Load a downloaded vision model into memory, as a background job (follow it with get_job).", ModelArgs)
def load_model(conn: sqlite3.Connection, args: ModelArgs) -> dict:
    return {"job_id": vision.start_load(conn, args.family, args.variant)}


@tool("Unload a vision model from memory, freeing it (e.g. before training).", ModelArgs)
def unload_model(conn: sqlite3.Connection, args: ModelArgs) -> dict:
    return {"unloaded": vision.unload(args.family, args.variant)}


@tool(
    "List the workflows that can be run (mozo graphs: e.g. SAM 3, Grounding DINO, Grounding DINO → EdgeTAM), each "
    "with the models it uses: licence and bytes left to download (0: ready). A model must be downloaded before a run."
)
def list_workflows(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    keep = ("family", "variant", "license", "size")
    return [w | {"models": [{k: m[k] for k in keep} for m in w["models"]]} if "models" in w else w for w in runs.workflows()]


@tool(
    "Run a workflow over a project's images matching a filter (all, or `count` picked at random), as a background job. "
    "It finds the prompts (the project's class prompts unless given) and keeps what it found per image. Follow the job "
    "with get_job, then read the results with get_run.",
    RunArgs,
)
def start_run(conn: sqlite3.Connection, args: RunArgs) -> dict:
    media_ids = runs.on_project(conn, args.project_id, Filter.parse(args.where), args.count)
    prompts = args.prompts if args.prompts is not None else [c["prompt"] for c in projects.get(conn, args.project_id)["classes"]]
    if not prompts:
        raise ToolError("Nothing to find: the project has no classes, so give prompts.")
    run_id, job_id = runs.start(conn, args.workflow_id, media_ids, project_id=args.project_id, name=args.name, prompts=prompts)
    return {"run_id": run_id, "job_id": job_id, "images": len(media_ids), "prompts": prompts}


@tool("Change the prompts of some of a project's classes: the text its annotator looks for (e.g. 'red car' for a class "
      "named car). Returns the classes with their prompts.", PromptArgs)
def set_prompts(conn: sqlite3.Connection, args: PromptArgs) -> list[dict]:
    if projects.get(conn, args.project_id) is None:
        raise ToolError(f"There is no project {args.project_id}.")
    with conn:
        projects.set_prompts(conn, args.project_id, args.prompts)
    return [{"id": c["id"], "name": c["name"], "prompt": c["prompt"]} for c in projects.get(conn, args.project_id)["classes"]]


@tool(
    "Auto-annotate a project's images matching a filter with its annotator, as a background job: each class is found "
    "from its prompt and the objects are added to the images, next to the ones there (never replacing them; an object "
    "already there is left out). An image is done once per class, so running again only does what's new (a class "
    "added since, images added since). Follow the job with get_job; the run's results are in get_run. The annotator's "
    "models must be downloaded first (list_workflows).",
    AutoAnnotateArgs,
)
def auto_annotate(conn: sqlite3.Connection, args: AutoAnnotateArgs) -> dict:
    started = annotators.start(conn, args.project_id, Filter.parse(args.where), args.count, None, args.class_ids,
                               args.min_score, args.again)
    if started is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return started


@tool("List workflow runs, newest first: name, workflow, project, job status, and how many images have results.")
def list_runs(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return runs.list_all(conn)


@tool(
    "Read a run's results, per result label (a run can keep several, e.g. two models side by side): how many objects "
    "of each class were found with their average score, and for the first 20 images the objects found (class, score, "
    "box in pixels).",
    RunIdArgs,
)
def get_run(conn: sqlite3.Connection, args: RunIdArgs) -> dict:
    run = runs.summary(conn, args.run_id)
    if run is None:
        raise ToolError(f"There is no run {args.run_id}.")
    return run
