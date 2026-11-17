"""Workflows and their runs, as JSON."""

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from mozo.workflow import WorkflowError
from pydantic import BaseModel, Field, StringConstraints

from app import annotators, db, runs, vision
from app.filters import WHERE, Filter
from app.routes import AnnotationOut, Answer, ClassOut, ToDownload

router = APIRouter(prefix="/api")
DB = Annotated[sqlite3.Connection, Depends(db.get)]
# A workflow document, as the builder saves it.
Document = Annotated[dict[str, Any], Body(description=(
    'A mozo workflow: {"name", "description", "nodes": [{"id", "type", "data": {"parameters": {…}}}], '
    '"edges": [{"source", "sourceHandle", "target", "targetHandle"}]}. Node types with their ports and parameters '
    "are in the node catalogue (GET /api/workflows/nodes); an edge joins an output (sourceHandle) to an input "
    "(targetHandle) of the "
    "same type. A run keeps what reaches a save_to_babasu node."))]


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
    models: list[ModelUse] | None = Field(None, description="The models it uses.")
    error: str | None = Field(None, description="Instead of models: why it doesn't load.")


class MissingFile(Answer):
    node: str
    parameter: str
    path: str = Field(description="The file it names, which isn't there.")


class WorkflowDetail(WorkflowListed):
    graph: dict[str, Any] = Field(description="The workflow document: name, description, nodes, edges (as PUT takes it).")
    kept: list[str] = Field(description="What a run keeps: its save_to_babasu nodes' labels (a run's results are per label).")
    prompt_parameters: list[str] = Field(description="The parameters that take the prompts, node.parameter.")
    missing_files: list[MissingFile] = Field(description="Files its nodes name that aren't there.")


class RunOut(Answer):
    id: int = Field(description="The run's id.")
    name: str
    workflow_id: str | None = Field(description="The workflow it ran (its copy is kept with the run); null: one not saved, tried as a draft.")
    project_id: int | None = Field(description="The project its images came from; null: images given by id.")
    job_id: int | None = Field(description="The job that ran it (GET /api/jobs/{job_id}).")
    status: str | None = Field(description="The job's status: queued, running, succeeded, failed, cancelled, interrupted.")
    created_at: str = Field(description="When it started, UTC (ISO 8601).")
    classes: list[int] | None = Field(description="The project's class ids its prompts stood for, in order; null: other prompts.")
    annotates: bool = Field(description="It added its objects to the project (auto-annotate).")
    images: int = Field(description="How many images have results so far.")


class ClassFound(Answer):
    objects: int = Field(description="How many objects of it were found.")
    average_score: float = Field(description="Their average score, 0..1.")


class RunResult(Answer):
    media_id: int = Field(description="The image's id.")
    filename: str = Field(description="Its file name in the run's project (no project: the start of its sha256).")
    label: str = Field(description="Which of the workflow's save_to_babasu nodes kept it.")
    objects: dict[str, int] = Field(description="How many objects of each class were found on it.")
    width: int | None = Field(None, description="In pixels (with ?annotations=1).")
    height: int | None = Field(None, description="In pixels (with ?annotations=1).")
    annotations: list[AnnotationOut] | None = Field(None, description="The objects as a project stores them, class_id "
                                                    "(in `legend`), bbox [x, y, w, h], segments, score (with ?annotations=1).")


class LegendClass(ClassOut):
    prompt: str | None = Field(None, description="A project's class: its prompt.")
    active: bool | None = Field(None, description="A project's class: always true.")


class RunDetail(RunOut):
    found: dict[str, dict[str, ClassFound]] = Field(description="What it found, per label and class.")
    page: int = Field(description="This page's number, from 1.")
    pages: int = Field(description="How many pages its images make.")
    results: list[RunResult] = Field(description="This page of its images' results, by image (one a label).")
    legend: list[LegendClass] | None = Field(None, description="The classes the annotations are in: a project's run, "
                                             "the project's; any other, its prompts in order (with ?annotations=1).")


class RunStarted(Answer):
    run_id: int
    job_id: int = Field(description="Follow it at /api/jobs/{job_id}.")
    images: int = Field(description="How many images it runs on.")
    downloads_first: list[ToDownload] = Field(description="The models it downloads before it starts.")


@router.get("/workflows", response_model=list[WorkflowListed], response_model_exclude_unset=True)
def api_workflows() -> list[dict]:
    """Every workflow: id, name, whether it's built in, and its models (or why it doesn't load)."""
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
def api_nodes(names: Annotated[list[str] | None, Query(description="Nodes to show in full: inputs, outputs, parameters.")] = None) -> list[dict]:
    """What workflows are built from: every node's name, category and what it does (?names= those
    in full). A run keeps what reaches a save_to_babasu node, under its label; read_media is where
    each image comes in. 422 naming unknown nodes."""
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


@router.post("/workflows", status_code=201)
def api_create_workflow(document: Document) -> dict:
    """Saves a new workflow document, its id made from its name (unique), and returns it as the
    list shows it. 422 with why if it has no name, doesn't load or keeps nothing."""
    try:
        return runs.create_workflow(document)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/workflows/check")
def api_check_workflow(document: Document) -> dict:
    """Checks a workflow document without saving it: what a run keeps, which parameters take the
    prompts, its models and missing files. 422 with why if it doesn't load."""
    try:
        return runs.check_workflow(document)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.put("/workflows/{workflow_id}")
def api_save_workflow(workflow_id: str, document: Document) -> dict:
    """Saves a workflow document (new, or replacing a saved one; a built-in's id is refused). 422
    with why if it doesn't load or keeps nothing."""
    try:
        return runs.save_workflow(workflow_id, document)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/workflows/{workflow_id}", status_code=204)
def api_delete_workflow(workflow_id: str) -> None:
    """Deletes a saved workflow (built-ins can't be); its runs keep their copy."""
    if not runs.delete_workflow(workflow_id):
        raise HTTPException(404, "No such saved workflow (built-ins can't be deleted)")


# --- Runs ---


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class NewRun(BaseModel):
    """A run of a workflow over stored images: `media_ids`, or a project's images (`project_id`,
    `where`: its filter, `count`: that many at random). `prompts`: what to look for, given to every
    prompted node (e.g. ["car", "person"]); `overrides`: parameter values as mozo's run takes
    them, by name or one node's (e.g. {"dino.threshold": 0.4})."""

    workflow_id: str | None = Field(None, description="The workflow, an id from GET /api/workflows; or give `workflow`.")
    workflow: dict[str, Any] | None = Field(None, description="Instead of workflow_id: a workflow not saved yet (its "
                                            "document, as POST /api/workflows takes it), to try it on a few images.")
    media_ids: list[int] | None = Field(None, description="The images to run on, by id; or give project_id.")
    where: str = Field("", max_length=2000, description=WHERE)
    count: int | None = Field(None, ge=1, description="How many of the project's matching images, picked at random; empty for all.")
    prompts: list[str] | None = Field(None, description="What to find, as text prompts, e.g. [\"car\", \"person\"]; "
                                      "empty: a project's class prompts.")
    overrides: dict[str, Any] = Field({}, description="Parameter values by name, or one node's, e.g. {\"dino.threshold\": 0.4}.")
    project_id: int | None = Field(None, description="Run on this project's images (with where and count); its results are drawn with its classes.")
    name: Name | None = Field(None, description="A name for the run; empty for one made from the workflow and the image count.")


class RunChanges(BaseModel):
    """A run's new name."""

    name: Name = Field(description="Its name, e.g. night shift.")


def _found(run: dict | None) -> dict:
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


@router.post("/runs", status_code=202, response_model=RunStarted)
def api_start_run(new: NewRun, conn: DB) -> dict:
    """Starts a workflow run over images (`media_ids`, or a project's: `project_id`, `where`,
    `count`), finding `prompts` (a project's class prompts unless given); returns {run_id, job_id,
    images, downloads_first: the models it downloads before it starts}. Follow the job at
    /api/jobs/{job_id}, read the results at /api/runs/{run_id}. 422 with why it can't start."""
    try:
        media_ids = new.media_ids
        if media_ids is None:
            if new.project_id is None:
                raise ValueError("Give media_ids, or a project_id to run on its images")
            media_ids = runs.on_project(conn, new.project_id, Filter.parse(new.where), new.count)
        run_id, job_id = runs.start(conn, new.workflow_id, media_ids, new.overrides, new.project_id, new.name, new.prompts,
                                    new.workflow)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"run_id": run_id, "job_id": job_id, "images": len(media_ids),
            "downloads_first": vision.downloads_first(conn, job_id)}


@router.get("/runs", response_model=list[RunOut])
def api_runs(conn: DB, workflow_id: Annotated[str | None, Query(description="Only this workflow's runs.")] = None,
             limit: Annotated[int | None, Query(ge=1, description="Only the newest few.")] = None) -> list[dict]:
    """Every run (?workflow_id= one workflow's; ?limit= the newest few), newest first: name,
    workflow, project, classes, job status, images with results."""
    return runs.list_all(conn, workflow_id, limit)


@router.get("/runs/{run_id}", response_model=RunDetail, response_model_exclude_unset=True)
def api_run(run_id: int, conn: DB, page: Annotated[int, Query(ge=1, description="Which page of the results.")] = 1,
            annotations: Annotated[bool, Query(description="Each result's objects with their boxes, outlines and "
                                               "scores, not only how many of each class.")] = False) -> dict:
    """The run, what it found per label (objects per class, average score), and one page of its
    images' results, each how many objects of each class were found on the image (?annotations=1:
    the objects themselves, and the legend of their classes)."""
    run = _found(runs.results_page(conn, run_id, page, shapes=annotations))
    if annotations:
        run = annotators.annotated(conn, run)
    return run | {"results": [{k: r[k] for k in RunResult.model_fields if k in r} for r in run["results"]]}


@router.patch("/runs/{run_id}", status_code=204)
def api_rename_run(run_id: int, changes: RunChanges, conn: DB) -> None:
    """Renames a run."""
    if not runs.rename(conn, run_id, changes.name):
        raise HTTPException(404, "Run not found")


@router.delete("/runs/{run_id}", status_code=204)
def api_delete_run(run_id: int, conn: DB) -> None:
    """Deletes the run and its results, and the images no project or run uses any more; 409 while its job runs."""
    try:
        found = runs.delete(conn, run_id)
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    if not found:
        raise HTTPException(404, "Run not found")
