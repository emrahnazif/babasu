"""Workflows and running them, as JSON."""

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from mozo.workflow import WorkflowError
from pydantic import BaseModel, ConfigDict, Field

from app import db, runs
from app.formatting import plural
from app.projects import ClassName as ProjectClass
from app.routes import IMAGES, READS, Answer, CommaList, Done, Images, Started, started

router = APIRouter(prefix="/api")
DB = Annotated[sqlite3.Connection, Depends(db.get)]
# A workflow document, as the builder saves it: mozo's short form.
Document = Annotated[dict[str, Any], Body(description=(
    'A mozo workflow: {"name", "description", "nodes"}, each node {"id" (yours), "type" (from GET /api/workflows/nodes), '
    'its parameters}. An input not written takes the latest earlier node with an output of its type; to take another, '
    'write it under "inputs": {"image": "read"}. A run keeps what reaches its save nodes: save_to_babasu (the image\'s '
    'annotations), save_image_to_babasu (a new image), save_video_to_babasu (a video\'s frames as a new video). E.g. '
    '{"name": "Squares", "nodes": [{"id": "image", "type": "read_media"}, {"id": "find", "type": "grounding_dino", '
    '"text": ["square"]}, {"id": "keep", "type": "save_to_babasu"}]}'))]


# --- What the API answers ---


class ModelUse(Answer):
    node: str = Field(description="The node that uses it.")
    family: str = Field(description="The model family (GET /api/vision/models).")
    variant: str = Field(description="Which of the family's models.")
    license: str = Field(description="Its licence.")
    size: int = Field(description="Bytes still to download (0 once downloaded).")


class WorkflowListed(Answer):
    id: str = Field(description="The workflow's id (what runs and annotators name).")
    name: str
    description: str = Field(description="What it does, in a sentence or two; empty if it doesn't say.")
    builtin: bool = Field(description="Comes with the app (can't be changed or deleted).")
    default: bool | None = Field(None, description="What a run runs when no workflow is named.")
    prompts: bool | None = Field(None, description="It finds what prompts name (a run gives them).")
    keeps: list[str] | None = Field(None, description="What a run of it keeps, a save node at a time: annotations, or image (label).")
    to_download: int | None = Field(None, description="Bytes of models it downloads before its first run (0: ready).")
    models: list[ModelUse] | None = Field(None, description="The models it uses.")
    error: str | None = Field(None, description="Instead of models: why it doesn't load.")


class WorkflowSaved(WorkflowListed, Done):
    pass


class MissingFile(Answer):
    node: str
    parameter: str
    path: str = Field(description="The file it names, which isn't there.")


class WorkflowDetail(WorkflowListed):
    graph: dict[str, Any] = Field(description="The workflow document: name, description, nodes (as PUT takes it).")
    kept: list[str] = Field(description="What a run keeps, a save node at a time: annotations, or image (label).")
    prompt_parameters: list[str] = Field(description="The parameters that take the prompts, node.parameter.")
    missing_files: list[MissingFile] = Field(description="Files its nodes name that aren't there.")


@router.get("/workflows", response_model=list[WorkflowListed], response_model_exclude_unset=True)
def api_workflows() -> list[dict]:
    """Every workflow: what it does, whether it takes prompts, what a run keeps, what it still
    downloads, which one a run runs when none is named, and its models (or why it doesn't load)."""
    return runs.workflows()


class NodeOut(Answer):
    name: str
    category: str
    description: str = Field(description="What it does.")
    inputs: list[str] | None = Field(None, description="With ?names=: its inputs, name: type.")
    outputs: list[str] | None = Field(None, description="With ?names=: its outputs, name: type.")
    parameters: list[dict[str, Any]] | None = Field(None, description="With ?names=: its parameters (name, kind, "
                                                    "default, options; prompts: it takes the prompts).")


@router.get("/workflows/nodes", response_model=list[NodeOut], response_model_exclude_unset=True)
def api_nodes(names: Annotated[list[str] | None, CommaList, Query(
        description="Nodes to show in full (inputs, outputs, parameters): ?names=crop,resize or ?names=crop&names=resize.")] = None) -> list[dict]:
    """What workflows are built from: every node's name, category and what it does (?names= those
    in full). A run keeps what reaches its save nodes (save_to_babasu: the image's annotations;
    save_image_to_babasu: a new image; save_video_to_babasu: a video's frames as a new video); read_media is
    where each image or video comes in. 422 naming unknown nodes."""
    try:
        return runs.nodes(names)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetail, response_model_exclude_unset=True)
def api_workflow(workflow_id: str) -> dict:
    """The workflow's graph, the models it uses (licence, bytes left to download) and any missing files."""
    try:
        workflow = runs.load(workflow_id)
    except WorkflowError as e:
        raise HTTPException(422, f"The workflow doesn't load: {e}") from None
    if workflow is None:
        raise HTTPException(404, "Workflow not found")
    return runs.detail(workflow_id, workflow)


@router.post("/workflows", status_code=201, response_model=WorkflowSaved, response_model_exclude_unset=True)
def api_create_workflow(document: Document) -> dict:
    """Saves a new workflow document, its id made from its name (unique), and returns it as the
    list shows it. 422 with why if it has no name, doesn't load or keeps nothing."""
    try:
        saved = runs.create_workflow(document)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return saved | {"did": f"Saved workflow {saved['name']} (id {saved['id']})"}


@router.post("/workflows/check", tags=[READS])
def api_check_workflow(document: Document) -> dict:
    """Checks a workflow document without saving it: what a run keeps, which parameters take the
    prompts, its models and missing files. 422 with why if it doesn't load."""
    try:
        return runs.check_workflow(document)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.put("/workflows/{workflow_id}", response_model=WorkflowSaved, response_model_exclude_unset=True)
def api_save_workflow(workflow_id: str, document: Document) -> dict:
    """Saves a workflow document (new, or replacing a saved one; a built-in's id is refused). 422
    with why if it doesn't load or keeps nothing."""
    try:
        saved = runs.save_workflow(workflow_id, document)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return saved | {"did": f"Saved workflow {saved['name']} (id {saved['id']})"}


@router.delete("/workflows/{workflow_id}", response_model=Done)
def api_delete_workflow(workflow_id: str) -> dict:
    """Deletes a saved workflow (built-ins can't be); what its runs kept stays in the projects."""
    if not runs.delete_workflow(workflow_id):
        raise HTTPException(404, "No such saved workflow (built-ins can't be deleted)")
    return {"did": f"Deleted workflow {workflow_id}"}


# --- Running ---


class Annotate(BaseModel):
    """A workflow run over images, keeping what reaches its save nodes: what it finds as the images'
    annotations (in their projects), what it draws or crops as new images."""

    model_config = ConfigDict(extra="forbid")
    images: Images = Field(description=IMAGES)
    classes: list[ProjectClass] | None = Field(None, description="What to find, by class name, e.g. [\"car\", "
                                               "\"person\"]: each looked for by its prompt in the image's project "
                                               "(its name for a class the project hasn't had, which becomes one).")
    workflow_id: str | None = Field(None, description="The workflow, an id from GET /api/workflows; empty: the default "
                                    "one, which finds and outlines what the classes name; or give `workflow`.")
    workflow: dict[str, Any] | None = Field(None, description="Instead of workflow_id: a workflow not saved yet (its "
                                            "document, as POST /api/workflows takes it), to try it on a few images.")
    again: bool = Field(False, description="Also images the workflow has looked for these classes on: what its models "
                        "found then (objects with a score) is replaced; what people made or accepted stays.")
    min_score: float = Field(0.0, ge=0, le=1, description="Leave out objects scoring below this (0..1).")
    overrides: dict[str, Any] = Field({}, description="Parameter values by name, or one node's, e.g. {\"dino.threshold\": 0.4}.")


@router.post("/annotate", status_code=202, response_model=Started, response_model_exclude_unset=True)
def api_annotate(request: Annotate, conn: DB) -> dict:
    """Runs a workflow over images (the default finds and outlines the classes named), as a job: what
    it finds becomes the images' annotations, in their own projects; images it makes become new
    images. Each image runs only for the classes the workflow hasn't looked for on it, unless
    `again`. Returns {job_id, images, downloads_first: the models it downloads first}; follow the job
    at /api/jobs/{job_id} (its result: objects found per class). 422 with why it can't start."""
    try:
        media_ids = request.images.ids(conn)
        begun = runs.start(conn, media_ids, request.workflow_id, request.classes, None, request.again,
                           request.min_score, request.overrides, request.workflow)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    finding = f", finding {', '.join(request.classes)}" if request.classes else ""
    return begun | started(conn, begun["job_id"], f"Started a run on {plural(begun['images'], 'image')}{finding}")
