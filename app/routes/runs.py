"""Workflows and their runs, as JSON."""

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, UploadFile
from mozo.workflow import WorkflowError
from pydantic import BaseModel, Field, StringConstraints

from app import annotators, db, media, runs, vision
from app.filters import WHERE, Filter
from app.routes import UI_ONLY

router = APIRouter(prefix="/api")
DB = Annotated[sqlite3.Connection, Depends(db.get)]
# A workflow document, as the builder saves it.
Document = Annotated[dict[str, Any], Body(description=(
    'A mozo workflow: {"name", "description", "nodes": [{"id", "type", "data": {"parameters": {…}}}], '
    '"edges": [{"source", "sourceHandle", "target", "targetHandle"}]}. Node types with their ports and parameters '
    "are in mozo's node catalogue (GET /mozo/workflow/nodes); an edge joins an output (sourceHandle) to an input "
    "(targetHandle) of the "
    "same type. A run keeps what reaches a save_to_babasu node."))]


@router.get("/workflows")
def api_workflows() -> list[dict]:
    """Every workflow: id, name, whether it's built in, and its models (or why it doesn't load)."""
    return runs.workflows()


@router.get("/workflows/{workflow_id}")
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


# --- Images to run on ---


@router.post("/media", status_code=201, tags=[UI_ONLY])
def api_upload_media(files: list[UploadFile], conn: DB) -> list[dict]:
    """Stores images (e.g. dropped on a Studio page) to run a workflow on; each once, by its
    content. Returns them in order: media id, width, height, url, thumbnail. 422 if one isn't an
    image; none are kept then."""
    stored = []
    with conn:
        for upload in files:
            try:
                media_id = media.store(conn, upload.file.read())
            except media.NotAnImage:
                raise HTTPException(422, f"{upload.filename} isn't an image") from None
            row = conn.execute("SELECT sha256, ext, width, height FROM media WHERE id = ?", (media_id,)).fetchone()
            stored.append({"media_id": media_id, "filename": upload.filename, "width": row["width"],
                           "height": row["height"], "url": media.url(row["sha256"], row["ext"]),
                           "thumb_url": media.thumb_url(row["sha256"])})
    return stored


# --- Runs ---


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class NewRun(BaseModel):
    """A run of a workflow over stored images: `media_ids`, or a project's images (`project_id`,
    `where`: its filter, `count`: that many at random). `prompts`: what to look for, given to every
    prompted node (e.g. ["car", "person"]); `overrides`: parameter values as mozo's run takes
    them, by name or one node's (e.g. {"dino.threshold": 0.4})."""

    workflow_id: str = Field(description="The workflow, an id from GET /api/workflows.")
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


@router.post("/runs", status_code=202)
def api_start_run(new: NewRun, conn: DB) -> dict:
    """Starts a workflow run over images (`media_ids`, or a project's: `project_id`, `where`,
    `count`), finding `prompts` (a project's class prompts unless given); returns {run_id, job_id,
    images, downloads_first: the models it downloads before it starts}. Follow the job at
    /api/jobs/{job_id}, read the results at /api/runs/{run_id}/summary. 422 with why it can't start."""
    try:
        media_ids = new.media_ids
        if media_ids is None:
            if new.project_id is None:
                raise ValueError("Give media_ids, or a project_id to run on its images")
            media_ids = runs.on_project(conn, new.project_id, Filter.parse(new.where), new.count)
        run_id, job_id = runs.start(conn, new.workflow_id, media_ids, new.overrides, new.project_id, new.name, new.prompts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"run_id": run_id, "job_id": job_id, "images": len(media_ids),
            "downloads_first": vision.downloads_first(conn, job_id)}


@router.get("/runs")
def api_runs(conn: DB, workflow_id: Annotated[str | None, Query(description="Only this workflow's runs.")] = None,
             limit: Annotated[int | None, Query(ge=1, description="Only the newest few.")] = None) -> list[dict]:
    """Every run (?workflow_id= one workflow's; ?limit= the newest few), newest first: name,
    workflow, project, classes, job status, images with results."""
    return runs.list_all(conn, workflow_id, limit)


@router.get("/runs/{run_id}")
def api_run(run_id: int, conn: DB) -> dict:
    """The run with its results: per image and label, mozo's annotation line."""
    return _found(runs.get(conn, run_id))


@router.get("/runs/{run_id}/summary")
def api_run_summary(run_id: int, conn: DB, images: Annotated[int, Query(ge=0, le=1000, description="How many images' objects to show.")] = 20) -> dict:
    """The run in brief, per label: objects per class with average score, and the first ?images= images' objects."""
    return _found(runs.summary(conn, run_id, images))


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


@router.get("/runs/{run_id}/annotations")
def api_run_annotations(run_id: int, conn: DB, images: Annotated[int | None, Query(ge=1, description="Only the first few images'; empty for all.")] = None) -> dict:
    """The run's results as annotations, per image and result label (?images= only the first
    few images'), with `legend`: the classes to draw them with (a project's run: the project's
    classes; any other: its prompts, in order)."""
    return _found(annotators.results(conn, run_id, images))
