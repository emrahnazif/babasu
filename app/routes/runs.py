"""Workflows and their runs, as JSON."""

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from mozo.workflow import WorkflowError
from pydantic import BaseModel, StringConstraints

from app import db, runs

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


# --- Runs ---


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class NewRun(BaseModel):
    """A run of a workflow over stored images. `prompts`: what to look for, given to every prompted
    node (e.g. ["car", "person"]); `overrides`: parameter values as mozo's run takes them, by name
    or one node's (e.g. {"dino.threshold": 0.4})."""

    workflow_id: str
    media_ids: list[int]
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
        run_id, job_id = runs.start(conn, new.workflow_id, new.media_ids, new.overrides, new.project_id, new.name, new.prompts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"run_id": run_id, "job_id": job_id}


@router.get("/runs")
def api_runs(conn: DB) -> list[dict]:
    return runs.list_all(conn)


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

