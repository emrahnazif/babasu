"""Tools the agent can call: babasu's own functions, the same ones the UI and the API use.

Each tool's arguments are a pydantic model, so the JSON schema the model sees is generated, never
written by hand. Tools that change things don't ask for confirmation: what they change in the
database, and the views and workflows they delete, can be undone for 30 days (app/history.py).
A tool registered with opens=True returns {"open": "/a/page", …}: that page is shown to the person
once the reply is done (app/agent.py, run_turn); after a tool that `writes`, the page they're on
reloads then instead (unless one opens). What tools return carries each thing's page, so the agent
can open it.
"""

import difflib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args
from urllib.parse import urlencode

from mozo.workflow import catalogue
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app import (
    annotators,
    config,
    history,
    jobs,
    projects,
    runs,
    segment,
    settings,
    views,
    vision,
)
from app.filters import Filter
from app.templating import filesize


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


class UndoArgs(Args):
    action_id: str = Field(description="The action's id, from read_api /api/history.")


class JobArgs(Args):
    job_id: int = Field(description="The job's id, from list_jobs.")


class RunArgs(FilterArgs):
    workflow_id: str = Field(description="The workflow's id, from list_workflows.")
    count: int | None = Field(None, ge=1, le=1000, description="How many of the matching images, picked at random; empty for all of them.")
    prompts: list[str] | None = Field(None, description="What to find, as text prompts (one per class); empty for the project's class prompts.")
    name: str | None = Field(None, description="A name for the run; empty for one made from the workflow and the image count.")


class NewProjectArgs(Args):
    name: projects.Name
    description: projects.Description = ""
    task: projects.Task = Field("detection", description="What its annotations are: boxes (detection), outlines "
                                "(segmentation) or points (keypoints).")
    classes: list[projects.NewClass] = Field([], description="The kinds of objects to annotate, in order.")


class UpdateProjectArgs(ProjectArgs):
    name: projects.Name | None = None
    description: projects.Description | None = None
    task: projects.Task | None = None
    annotator: annotators.Finder | None = Field(
        None, description="The workflow that auto-annotates it, from list_workflows; null for no annotator.")
    add_classes: list[projects.NewClass] = Field([], description="New classes, added after the others.")
    change_classes: list[projects.ClassEdit] = Field([], description="Classes to rename, or to give another prompt or colour.")
    remove_classes: list[projects.ClassRef] = Field([], description="Classes to remove (names or ids): their annotations are "
                                            "hidden, not deleted, and come back if a class by that name is added again.")


class DuplicateArgs(ProjectArgs):
    name: projects.Name | None = Field(None, description='The copy\'s name; empty: "<its name> copy".')


class AutoAnnotateArgs(FilterArgs):
    image_ids: list[int] | None = Field(None, description="Only these images (ids from find_images, or the image "
                                        "page's path: /projects/3/images/42 is image 42); empty for all that match.")
    count: int | None = Field(None, ge=1, description="How many of the images to do, picked at random among those with work left; empty for all.")
    class_ids: list[int] | None = Field(None, description="Which classes (ids from get_project); empty for all of them.")
    min_score: float = Field(0.0, ge=0, le=1, description="Leave out objects scoring below this (0..1), beyond the annotator's own threshold.")
    again: bool = Field(False, description="Also run images already auto-annotated for a class (new objects that are already there are still left out).")


class WaitArgs(JobArgs):
    seconds: int = Field(60, ge=1, le=300, description="How long to wait at most, in seconds.")


class NodesArgs(Args):
    names: list[str] | None = Field(None, description="Nodes to show in full (inputs, outputs, parameters); empty for "
                                    "the list of all nodes, each with its category and what it does.")


class WorkflowIdArgs(Args):
    workflow_id: str = Field(description="The workflow's id, from list_workflows (lowercase letters, digits and dashes).")


class WorkflowDocArgs(Args):
    document: dict[str, Any] = Field(
        description='A mozo workflow: {"name", "description", "nodes": [{"id", "type", "data": {"parameters": {…}}}], '
        '"edges": [{"source", "sourceHandle", "target", "targetHandle"}]}. Node types and their ports and parameters '
        "come from list_nodes; an edge joins an output (sourceHandle) to an input (targetHandle) of the same type.")


class SaveWorkflowArgs(WorkflowIdArgs, WorkflowDocArgs):
    pass


class DocsArgs(Args):
    topic: str = Field(description="A topic, from the list in read_docs' description.")


class ApiArgs(Args):
    path: str = Field(description="A GET endpoint from get_components, with its ids and query, e.g. /api/runs/3/summary.")


class ViewIdArgs(Args):
    view_id: str = Field(description="The view's id: lowercase letters, digits and dashes (e.g. model-comparison).")


class GetViewArgs(ViewIdArgs):
    version: int | None = Field(None, description="An older version (from view_history); empty for the newest.")


class WriteViewArgs(ViewIdArgs):
    template: str = Field(description="The whole page: a Jinja template (see get_components).")
    title: str | None = Field(None, description="The page's title; needed for a new view.")
    kind: Literal["report", "studio"] | None = Field(
        None, description="report (the default; kept as it was when written) or studio (a live page for one workflow).")
    workflow: str | None = Field(None, description="For a studio page: the workflow's id, from list_workflows.")


class EditViewArgs(ViewIdArgs):
    old: str = Field(description="A piece of the view's newest template, exactly as it is, found once in it.")
    new: str = Field(description="What replaces it (empty to remove it).")


class RestoreViewArgs(ViewIdArgs):
    version: int = Field(description="The version to make the newest again, from view_history.")


class NavArgs(Args):
    label: str = Field(description="The link's text, short (e.g. Model comparison).")
    href: str = Field(description="Where it goes in the app, e.g. /views/model-comparison.")


class HrefArgs(Args):
    href: str = Field(description="The link to remove, as list_nav shows it.")


class RunIdArgs(Args):
    run_id: int = Field(description="The run's id, from start_run or list_runs.")


class ModelArgs(Args):
    family: str = Field(description="The model family, from list_models, e.g. sam3.")
    variant: str = Field(description="The variant within it, from list_models, e.g. sam3.")


@dataclass(frozen=True)
class Tool:
    description: str
    args: type[BaseModel]
    func: Callable[[sqlite3.Connection, BaseModel], object]
    opens: bool = False  # it returns {"open": "/a/page", …}: that page is shown to the person after the reply
    writes: bool = False  # it changes what pages show: the page the person is on reloads after the reply


TOOLS: dict[str, Tool] = {}


class ToolError(Exception):
    """The model asked for a tool that doesn't exist, or gave arguments that don't fit."""


def tool(description: str, args: type[BaseModel] = NoArgs, opens: bool = False, writes: bool = False):
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


@tool("List all projects, with their type, image counts, how many images are labelled, and class names.")
def list_projects(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return [
        {"id": p["id"], "name": p["name"], "page": f"/projects/{p['id']}", "description": p["description"], "task": p["task"],
         "image_count": p["image_count"], "labelled_count": p["labelled_count"], "classes": [c["name"] for c in p["classes"]]}
        for p in projects.list_all(conn)
    ]


@tool("Show one project: its page and its settings page, its type and annotator, its classes (with the prompt sent "
      "to annotators and how many annotations each has), and its first images with their pages.", ProjectArgs)
def get_project(conn: sqlite3.Connection, args: ProjectArgs) -> dict:
    project = projects.detail(conn, args.project_id)
    if project is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return {
        "id": project["id"],
        "name": project["name"],
        "page": f"/projects/{project['id']}",
        "settings_page": f"/projects/{project['id']}/edit",
        "description": project["description"],
        "task": project["task"],
        "annotator": project["annotator"],
        "image_count": project["image_count"],
        "labelled_count": project["labelled_count"],
        "classes": [{"id": c["id"], "name": c["name"], "prompt": c["prompt"], "annotation_count": c["annotation_count"]}
                    for c in project["classes"]],
        "first_images": [_image(project, image) for image in project["images"][:20]],
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
    opens=True,
)
def show_images(conn: sqlite3.Connection, args: FilterArgs) -> dict:
    project = _filtered(conn, args)  # checks the filter, so the page never opens with an error
    query = f"?{urlencode({'where': project['filter']})}" if project["filter"] else ""
    return {"open": f"/projects/{args.project_id}{query}", "match_count": project["match_count"]}


@tool("List recent background jobs (image imports and more), newest first, with status and progress.")
def list_jobs(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return [
        {"id": j["id"], "page": f"/jobs/{j['id']}", "title": j["title"], "project": j["project_name"], "status": j["status"],
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


def _downloads(conn: sqlite3.Connection, job_id: int) -> dict:
    """What a job downloads before it runs, for the agent to tell the person: downloads_first
    (["sam3 sam3 (3.45 GB)"]) and a sentence saying so; nothing if it downloads nothing."""
    found = [f"{d['model']} ({filesize(d['size'])})" for d in vision.downloads_first(conn, job_id)]
    if not found:
        return {}
    return {"downloads_first": found,
            "note": f"Not downloaded yet: {', '.join(found)} downloads first, then this starts. Tell the person."}


@tool(
    "Download a vision model's files, as a background job (follow it with get_job). Large models take a while. "
    "Rarely needed: what uses a model (runs, auto-annotate, loading) downloads it first by itself.",
    ModelArgs,
    writes=True,
)
def download_model(conn: sqlite3.Connection, args: ModelArgs) -> dict:
    job_id = vision.ensure(conn, args.family, args.variant)
    return {"job_id": job_id} if job_id else {"already_downloaded": True}


@tool(
    "Load a vision model into memory, as a background job (follow it with get_job). One that isn't downloaded is "
    "downloaded first.",
    ModelArgs,
    writes=True,
)
def load_model(conn: sqlite3.Connection, args: ModelArgs) -> dict:
    job_id = vision.start_load(conn, args.family, args.variant)
    return {"job_id": job_id} | _downloads(conn, job_id)


def _click_tool(tool: dict) -> dict:
    """The click tool as segment.tool() has it, sizes for people."""
    return {k: tool[k] for k in ("on", "model", "in_use")} | {"models": [
        {k: m[k] for k in ("id", "name", "license", "downloaded", "downloading")}
        | ({} if m["downloaded"] else {"download_size": filesize(m["to_download"])})
        for m in tool["models"]]}


@tool(
    "Show the image editor's click tool (click to segment: point at an object and it's outlined): whether it's on, "
    "the model chosen and the one in use (another downloaded one answers while the chosen one downloads), and the "
    "models it can use, each with its licence and whether it's downloaded."
)
def get_click_tool(conn: sqlite3.Connection, args: NoArgs) -> dict:
    return _click_tool(segment.tool(conn))


@tool(
    "Change the image editor's click tool: turn it on or off, and/or choose its model. A model that isn't downloaded "
    "starts downloading (its download_size says how big: tell the person), and the tool switches to it when done.",
    segment.Change,
    writes=True,
)
def change_click_tool(conn: sqlite3.Connection, args: segment.Change) -> dict:
    if args.on is None and args.model is None:
        raise ToolError("Give on, model, or both.")
    return _click_tool(segment.change(conn, args))


@tool("Unload a vision model from memory, freeing it (e.g. before training).", ModelArgs, writes=True)
def unload_model(conn: sqlite3.Connection, args: ModelArgs) -> dict:
    return {"unloaded": vision.unload(args.family, args.variant)}


@tool(
    "List the workflows that can be run (mozo graphs: e.g. SAM 3, Grounding DINO, Grounding DINO → EdgeTAM), each "
    "with the models it uses: licence and bytes left to download (0: ready). A run downloads the ones it needs first."
)
def list_workflows(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    keep = ("family", "variant", "license", "size")
    return [w | {"page": f"/workflows/{w['id']}/edit", "studio": f"/studio/{w['id']}"}
            | ({"models": [{k: m[k] for k in keep} for m in w["models"]]} if "models" in w else {})
            for w in runs.workflows()]


@tool(
    "Run a workflow over a project's images matching a filter (all, or `count` picked at random), as a background job. "
    "It finds the prompts (the project's class prompts unless given) and keeps what it found per image. Follow the job "
    "with get_job, then read the results with get_run. Models it needs that aren't downloaded are downloaded first.",
    RunArgs,
    writes=True,
)
def start_run(conn: sqlite3.Connection, args: RunArgs) -> dict:
    media_ids = runs.on_project(conn, args.project_id, Filter.parse(args.where), args.count)
    prompts = args.prompts if args.prompts is not None else [c["prompt"] for c in projects.get(conn, args.project_id)["classes"]]
    if not prompts:
        raise ToolError("Nothing to find: the project has no classes, so give prompts.")
    run_id, job_id = runs.start(conn, args.workflow_id, media_ids, project_id=args.project_id, name=args.name, prompts=prompts)
    return {"run_id": run_id, "job_id": job_id, "images": len(media_ids), "prompts": prompts} | _downloads(conn, job_id)


def _project(conn: sqlite3.Connection, project_id: int) -> dict:
    project = projects.get(conn, project_id)
    if project is None:
        raise ToolError(f"There is no project {project_id}.")
    return project


@tool("Create a project, with its classes; only the name is needed (classes can be added later). It opens for "
      "the person once you've answered; images are imported from its page.", NewProjectArgs, opens=True)
def create_project(conn: sqlite3.Connection, args: NewProjectArgs) -> dict:
    classes = [projects.ClassChange(**c.model_dump()) for c in args.classes]
    try:
        with conn:
            project_id = projects.create(conn, args.name, args.description, args.task,
                                         annotators.SUGGESTED[args.task], classes)
    except ValueError as e:
        raise ToolError(str(e)) from None
    return {"id": project_id, "open": f"/projects/{project_id}"}


@tool("Change a project: any of its name, description, type, annotator and classes (add, rename, change the "
      "prompt or colour of, remove); what you leave out stays. Returns its classes. Renaming a class and "
      'its colour: {"change_classes": [{"name": "orange", "new_name": "citrus", "color": "yellow"}]}; its prompt: '
      '{"change_classes": [{"name": "apple", "prompt": "red or green apple"}]}.', UpdateProjectArgs, writes=True)
def update_project(conn: sqlite3.Connection, args: UpdateProjectArgs) -> dict:
    project = _project(conn, args.project_id)
    # Left out, it stays; null: none.
    annotator = args.annotator if "annotator" in args.model_fields_set else project["annotator"]
    try:
        with conn:
            projects.update(conn, args.project_id, args.name or project["name"],
                            project["description"] if args.description is None else args.description,
                            args.task or project["task"], annotator)
            projects.edit_classes(conn, args.project_id, args.add_classes, args.change_classes, args.remove_classes)
    except projects.FrozenError:
        raise ToolError("The project is frozen; duplicate it to change it.") from None
    except ValueError as e:
        raise ToolError(str(e)) from None
    return {"classes": [{"id": c["id"], "name": c["name"], "prompt": c["prompt"], "color": c["color"]}
                        for c in projects.get(conn, args.project_id)["classes"]]}


@tool("Duplicate a project: a new one with its classes, images and annotations (not its runs). The image files are "
      "shared, so it takes no space. It opens for the person once you've answered.", DuplicateArgs, opens=True)
def duplicate_project(conn: sqlite3.Connection, args: DuplicateArgs) -> dict:
    with conn:
        new_id = projects.duplicate(conn, args.project_id, args.name)
    if new_id is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return {"id": new_id, "open": f"/projects/{new_id}"}


@tool("Delete a project with its images, their annotations and its runs (undo brings it all back).", ProjectArgs,
      opens=True)
def delete_project(conn: sqlite3.Connection, args: ProjectArgs) -> dict:
    project = _project(conn, args.project_id)
    try:
        projects.remove(conn, args.project_id)
    except ValueError as e:
        raise ToolError(str(e)) from None
    return {"deleted": project["name"], "open": "/projects"}


@tool(
    "Auto-annotate a project's images matching a filter with its annotator, as a background job: each class is found "
    "from its prompt and the objects are added to the images, next to the ones there (never replacing them; an object "
    "already there is left out). An image is done once per class, so running again only does what's new (a class "
    "added since, images added since; `again` does them again). 'This image' is the image page the person is on, by "
    "its id in image_ids. Follow the job with get_job; the run's results are in get_run. Models the annotator "
    "needs that aren't downloaded are downloaded first.",
    AutoAnnotateArgs,
    writes=True,
)
def auto_annotate(conn: sqlite3.Connection, args: AutoAnnotateArgs) -> dict:
    started = annotators.start(conn, args.project_id, Filter.parse(args.where), args.count, None, args.class_ids,
                               args.min_score, args.again, args.image_ids)
    if started is None:
        raise ToolError(f"There is no project {args.project_id}.")
    return started | _downloads(conn, started["job_id"])


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


@tool("Wait for a background job to end (or for `seconds` to pass), then show it as get_job does.", WaitArgs)
def wait_for_job(conn: sqlite3.Connection, args: WaitArgs) -> dict:
    deadline = time.monotonic() + args.seconds
    while (job := get_job(conn, args)) and job["status"] in jobs.ACTIVE and time.monotonic() < deadline:
        time.sleep(1)
    return job


@tool("Undo an action: what one request, job or agent turn changed goes back as it was, deleted files included "
      "(read_api /api/history lists the last 30 days, newest first; your turns are labelled with the person's "
      "message). Undoing an undo does it again.", UndoArgs, writes=True)
def undo(conn: sqlite3.Connection, args: UndoArgs) -> dict:
    try:
        return history.undo(conn, args.action_id)
    except (LookupError, history.Refused) as e:
        raise ToolError(str(e)) from None


# --- Views ---


@tool("How to build a view (a page in the app you write) and the API a view can read. Read it before writing one.")
def get_components(conn: sqlite3.Connection, args: NoArgs) -> str:
    return views.reference()


API_PREVIEW = 8_000  # characters of an answer read_api shows


@tool(
    "Read the app's own JSON API at a path (GET), exactly as a view's api() gets it: the data of the page the person "
    "is on (its path under /api), or the field names to check before using them in a view.",
    ApiArgs,
)
def read_api(conn: sqlite3.Connection, args: ApiArgs) -> object:
    answer = views.api(args.path)
    text = json.dumps(answer, ensure_ascii=False)
    if len(text) <= API_PREVIEW:
        return answer
    return {"cut": f"The answer is {len(text):,} characters; its start:", "start": text[:API_PREVIEW]}


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


@tool("List the views (pages written by an agent): id, title, kind, workflow, newest version, page.")
def list_views(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return views.list_all()


def _view(view_id: str, version: int | None = None) -> dict:
    view = views.get(view_id, version)
    if view is None:
        raise ToolError(f"There is no view {view_id!r}" + (f" version {version}." if version else "."))
    return view


@tool("Show a view with its template (the newest version unless given).", GetViewArgs)
def get_view(conn: sqlite3.Connection, args: GetViewArgs) -> dict:
    view = _view(args.view_id, args.version)
    answers = view.pop("answers")  # a report's recorded data: the paths, not the answers themselves
    return view | ({"recorded": sorted(answers)} if answers is not None else {})


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
        raise ToolError("That piece isn't in the template; copy it exactly from get_view."
                        + (f" The closest lines:{hint}" if hint else ""))
    return _opened(views.save(args.view_id, template.replace(old, new)))


@tool("List a view's versions, newest first: number, when it was written, size.", ViewIdArgs)
def view_history(conn: sqlite3.Connection, args: ViewIdArgs) -> list[dict]:
    found = views.history(args.view_id)
    if found is None:
        raise ToolError(f"There is no view {args.view_id!r}.")
    return found


@tool("Make an older version of a view the newest again (saved as a new version).", RestoreViewArgs, opens=True)
def restore_view(conn: sqlite3.Connection, args: RestoreViewArgs) -> dict:
    restored = views.restore(args.view_id, args.version)
    if restored is None:
        raise ToolError(f"There is no view {args.view_id!r} version {args.version}.")
    return _opened(restored)


@tool("Delete a view and all its versions (and its link in the navigation).", ViewIdArgs, writes=True)
def delete_view(conn: sqlite3.Connection, args: ViewIdArgs) -> dict:
    if not views.delete(args.view_id):
        raise ToolError(f"There is no view {args.view_id!r}.")
    settings.remove_nav(f"/views/{args.view_id}")
    return {"deleted": args.view_id}


@tool("List the links added to the navigation (after the app's own sections).")
def list_nav(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return settings.nav()


@tool("Add a link to the navigation (e.g. to a view), or rename the one already going there.", NavArgs, writes=True)
def add_to_nav(conn: sqlite3.Connection, args: NavArgs) -> list[dict]:
    items = [i for i in settings.nav() if i["href"] != args.href.strip()]
    return settings.set_nav(items + [{"label": args.label, "href": args.href}])


@tool("Remove a link from the navigation.", HrefArgs, writes=True)
def remove_from_nav(conn: sqlite3.Connection, args: HrefArgs) -> list[dict]:
    return settings.remove_nav(args.href)


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


@tool("Show a workflow: its document (nodes and edges) and the models it uses, with bytes left to download.", WorkflowIdArgs)
def get_workflow(conn: sqlite3.Connection, args: WorkflowIdArgs) -> dict:
    workflow = runs.load(args.workflow_id)
    if workflow is None:
        raise ToolError(f"There is no workflow {args.workflow_id!r}.")
    return runs.detail(args.workflow_id, workflow)


@tool(
    "Check a workflow document without saving it: whether it loads (and why not), what a run keeps (its "
    "save_to_babasu labels), which parameters take the prompts, and its models (size: bytes left to download).",
    WorkflowDocArgs,
)
def validate_workflow(conn: sqlite3.Connection, args: WorkflowDocArgs) -> dict:
    return runs.check_workflow(args.document)


@tool(
    "Save a workflow in the app (new, or replacing a saved one; built-ins can't be replaced). Give it a name and a "
    "description in the document. A run keeps only what reaches a save_to_babasu node, so end each result there. "
    "It opens in the builder for the person once you've answered.",
    SaveWorkflowArgs,
    opens=True,
)
def save_workflow(conn: sqlite3.Connection, args: SaveWorkflowArgs) -> dict:
    saved = runs.save_workflow(args.workflow_id, args.document)
    return {"open": f"/workflows/{saved['id']}/edit"} | saved


@tool("Delete a saved workflow (built-ins can't be deleted). Runs made with it keep their copy.", WorkflowIdArgs, writes=True)
def delete_workflow(conn: sqlite3.Connection, args: WorkflowIdArgs) -> dict:
    if not runs.delete_workflow(args.workflow_id):
        raise ToolError(f"There is no saved workflow {args.workflow_id!r} (built-ins can't be deleted).")
    return {"deleted": args.workflow_id}
