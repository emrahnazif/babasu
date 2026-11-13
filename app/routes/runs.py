"""Workflows and their runs, as JSON."""

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from mozo.workflow import WorkflowError
from pydantic import BaseModel, Field, StringConstraints

from app import annotators, db, media, runs
from app.filters import Filter

router = APIRouter(prefix="/api")
DB = Annotated[sqlite3.Connection, Depends(db.get)]


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


# --- Images to run on ---


@router.post("/media", status_code=201)
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

    workflow_id: str
    media_ids: list[int] | None = None
    where: str = Field("", max_length=2000)
    count: int | None = Field(None, ge=1)
    prompts: list[str] | None = None
    overrides: dict[str, Any] = {}
    project_id: int | None = None
    name: Name | None = None


class RunChanges(BaseModel):
    name: Name


def _found(run: dict | None) -> dict:
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


@router.post("/runs", status_code=202)
def api_start_run(new: NewRun, conn: DB) -> dict:
    """Starts a run; follow its job at /api/jobs/{job_id}, read its results at /api/runs/{run_id}."""
    try:
        media_ids = new.media_ids
        if media_ids is None:
            if new.project_id is None:
                raise ValueError("Give media_ids, or a project_id to run on its images")
            media_ids = runs.on_project(conn, new.project_id, Filter.parse(new.where), new.count)
        run_id, job_id = runs.start(conn, new.workflow_id, media_ids, new.overrides, new.project_id, new.name, new.prompts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"run_id": run_id, "job_id": job_id}


@router.get("/runs")
def api_runs(conn: DB, workflow_id: str | None = None, limit: Annotated[int | None, Query(ge=1)] = None) -> list[dict]:
    """Every run (?workflow_id= one workflow's; ?limit= the newest few), newest first: name,
    workflow, project, classes, job status, images with results."""
    return runs.list_all(conn, workflow_id, limit)


@router.get("/runs/{run_id}")
def api_run(run_id: int, conn: DB) -> dict:
    """The run with its results: per image and label, mozo's annotation line."""
    return _found(runs.get(conn, run_id))


@router.get("/runs/{run_id}/summary")
def api_run_summary(run_id: int, conn: DB, images: Annotated[int, Query(ge=0, le=1000)] = 20) -> dict:
    """The run in brief, per label: objects per class with average score, and the first ?images= images' objects."""
    return _found(runs.summary(conn, run_id, images))


@router.patch("/runs/{run_id}", status_code=204)
def api_rename_run(run_id: int, changes: RunChanges, conn: DB) -> None:
    if not runs.rename(conn, run_id, changes.name):
        raise HTTPException(404, "Run not found")


@router.delete("/runs/{run_id}", status_code=204)
def api_delete_run(run_id: int, conn: DB) -> None:
    """Deletes the run and its results, and the images no project or run uses any more; 409 while its job runs."""
    try:
        unused = runs.delete(conn, run_id)
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    if unused is None:
        raise HTTPException(404, "Run not found")
    for path in unused:
        path.unlink(missing_ok=True)


@router.get("/runs/{run_id}/annotations")
def api_run_annotations(run_id: int, conn: DB, images: Annotated[int | None, Query(ge=1)] = None) -> dict:
    """The run's results as annotations, per image and result label (?images= only the first
    few images'), with `legend`: the classes to draw them with (a project's run: the project's
    classes; any other: its prompts, in order)."""
    return _found(annotators.results(conn, run_id, images))
