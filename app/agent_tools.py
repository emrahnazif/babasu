"""Tools the agent can call: babasu's own functions, the same ones the UI and the API use.

Each tool's arguments are a pydantic model, so the JSON schema the model sees is generated, never
written by hand. Tools that change things (downloading, loading models) don't ask for
confirmation yet; some may later (product.md "The agent").
A tool registered with opens_page=True returns {"open": "/a/page", …}: that page is shown to the
person once the reply is done (app/agent.py, run_turn).
"""

import difflib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

from mozo.workflow import catalogue
from pydantic import BaseModel, Field, ValidationError

from app import annotators, jobs, projects, runs, settings, views, vision
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


class WaitArgs(JobArgs):
    seconds: int = Field(60, ge=1, le=300, description="How long to wait at most, in seconds.")


class NodesArgs(BaseModel):
    names: list[str] | None = Field(None, description="Nodes to show in full (inputs, outputs, parameters); empty for "
                                    "the list of all nodes, each with its category and what it does.")


class WorkflowIdArgs(BaseModel):
    workflow_id: str = Field(description="The workflow's id, from list_workflows (lowercase letters, digits and dashes).")


class WorkflowDocArgs(BaseModel):
    document: dict[str, Any] = Field(
        description='A mozo workflow: {"name", "description", "nodes": [{"id", "type", "data": {"parameters": {…}}}], '
        '"edges": [{"source", "sourceHandle", "target", "targetHandle"}]}. Node types and their ports and parameters '
        "come from list_nodes; an edge joins an output (sourceHandle) to an input (targetHandle) of the same type.")


class SaveWorkflowArgs(WorkflowIdArgs, WorkflowDocArgs):
    pass


class ApiArgs(BaseModel):
    path: str = Field(description="A GET endpoint from get_components, with its ids and query, e.g. /api/runs/3/summary.")


class ViewIdArgs(BaseModel):
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


class NavArgs(BaseModel):
    label: str = Field(description="The link's text, short (e.g. Model comparison).")
    href: str = Field(description="Where it goes in the app, e.g. /views/model-comparison.")


class HrefArgs(BaseModel):
    href: str = Field(description="The link to remove, as list_nav shows it.")


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


@tool("Wait for a background job to end (or for `seconds` to pass), then show it as get_job does.", WaitArgs)
def wait_for_job(conn: sqlite3.Connection, args: WaitArgs) -> dict:
    deadline = time.monotonic() + args.seconds
    while (job := get_job(conn, args)) and job["status"] in jobs.ACTIVE and time.monotonic() < deadline:
        time.sleep(1)
    return job


# --- Views ---


@tool("How to build a view (a page in the app you write) and the API a view can read. Read it before writing one.")
def get_components(conn: sqlite3.Connection, args: NoArgs) -> str:
    return views.reference()


API_PREVIEW = 8_000  # characters of an answer read_api shows


@tool(
    "Read what a view's api() gets for a path: the app's own JSON, exactly. Check the field names here before "
    "using them in a view.",
    ApiArgs,
)
def read_api(conn: sqlite3.Connection, args: ApiArgs) -> object:
    answer = views.api(args.path)
    text = json.dumps(answer, ensure_ascii=False)
    if len(text) <= API_PREVIEW:
        return answer
    return {"cut": f"The answer is {len(text):,} characters; its start:", "start": text[:API_PREVIEW]}


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
    opens_page=True,
)
def write_view(conn: sqlite3.Connection, args: WriteViewArgs) -> dict:
    return _opened(views.save(args.view_id, args.template, args.title, args.kind, args.workflow))


@tool(
    "Change part of a view: replace one exact piece of its newest template, saved as a new version (checked first, "
    "as write_view). Use it for small changes instead of writing the whole page again.",
    EditViewArgs,
    opens_page=True,
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


@tool("Make an older version of a view the newest again (saved as a new version).", RestoreViewArgs, opens_page=True)
def restore_view(conn: sqlite3.Connection, args: RestoreViewArgs) -> dict:
    restored = views.restore(args.view_id, args.version)
    if restored is None:
        raise ToolError(f"There is no view {args.view_id!r} version {args.version}.")
    return _opened(restored)


@tool("Delete a view and all its versions (and its link in the navigation).", ViewIdArgs)
def delete_view(conn: sqlite3.Connection, args: ViewIdArgs) -> dict:
    if not views.delete(args.view_id):
        raise ToolError(f"There is no view {args.view_id!r}.")
    settings.remove_nav(f"/views/{args.view_id}")
    return {"deleted": args.view_id}


@tool("List the links added to the navigation (after the app's own sections).")
def list_nav(conn: sqlite3.Connection, args: NoArgs) -> list[dict]:
    return settings.nav()


@tool("Add a link to the navigation (e.g. to a view), or rename the one already going there.", NavArgs)
def add_to_nav(conn: sqlite3.Connection, args: NavArgs) -> list[dict]:
    items = [i for i in settings.nav() if i["href"] != args.href.strip()]
    return settings.set_nav(items + [{"label": args.label, "href": args.href}])


@tool("Remove a link from the navigation.", HrefArgs)
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
    opens_page=True,
)
def save_workflow(conn: sqlite3.Connection, args: SaveWorkflowArgs) -> dict:
    saved = runs.save_workflow(args.workflow_id, args.document)
    return {"open": f"/workflows/{saved['id']}/edit"} | saved


@tool("Delete a saved workflow (built-ins can't be deleted). Runs made with it keep their copy.", WorkflowIdArgs)
def delete_workflow(conn: sqlite3.Connection, args: WorkflowIdArgs) -> dict:
    if not runs.delete_workflow(args.workflow_id):
        raise ToolError(f"There is no saved workflow {args.workflow_id!r} (built-ins can't be deleted).")
    return {"deleted": args.workflow_id}
