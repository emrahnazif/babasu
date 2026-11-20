"""Jobs: the drawer's list fragment, a page per job, and the JSON API shared by the UI and the agent.

Jobs are started by the feature they belong to (e.g. a project import); here they are listed,
followed, cancelled and retried.
"""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import Field

from app import db, jobs
from app.routes import Answer, Done
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


def _get(conn: sqlite3.Connection, job_id: int) -> dict:
    job = jobs.get(conn, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# --- Pages ---


@router.get("/jobs/panel", response_class=HTMLResponse)
def job_panel(request: Request, conn: DB):
    """The job list alone, for js/jobs.js to swap into the drawer."""
    return templates.TemplateResponse(request, "partials/jobs.html", jobs.overview(conn))


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: int, conn: DB):
    log, _ = jobs.read_log(job_id, offset=None)  # the tail
    return templates.TemplateResponse(request, "jobs/job.html", {"job": _get(conn, job_id), "log": log})


# --- API ---


class JobListed(Answer):
    id: int = Field(description="The job's id.")
    kind: str = Field(description="What it does: import, run, auto-annotate, download, load…")
    title: str = Field(description="Its kind in words.")
    project_id: int | None = Field(description="The project it works on; null: none.")
    project_name: str | None
    status: str = Field(description="queued, running, succeeded, failed, cancelled or interrupted.")
    active: bool = Field(description="Queued or running.")
    done: int = Field(description="How many of its items are done.")
    total: int | None = Field(description="How many items it has; null until it knows.")
    unit: str = Field(description="What its items are, e.g. images.")
    message: str = Field(description="What it's doing now.")
    summary: str = Field(description="The result or the error, in one line.")
    error: str | None = Field(description="Why it failed; read the log (GET /api/jobs/{id}/log) for the details.")
    stop_reason: str | None = Field(description="user or shutdown: it was asked to stop.")
    waits_for: list[int] = Field(description="The jobs it waits for before it starts (their ids).")
    can_cancel: bool
    can_retry: bool
    created_at: str = Field(description="UTC (ISO 8601), as are started_at and finished_at.")
    started_at: str | None
    finished_at: str | None


class JobOut(JobListed):
    params: dict = Field(description="What it was asked to do (its kind's arguments, e.g. a run's workflow and media_ids).")
    result: dict | None = Field(description="What it did, when it's finished (by kind, e.g. {images, failed}).")


class Log(Answer):
    text: str = Field(description="The log from `offset` on.")
    offset: int = Field(description="Where the next read starts (pass it as ?offset=).")


@router.get("/api/jobs", response_model=list[JobListed])
def job_list(conn: DB, limit: Annotated[int, Query(ge=1, le=50, description="How many.")] = 10) -> list[dict]:
    """Recent background jobs, the running and waiting ones first, then the newest: title, project,
    status, progress, summary (a job's own answer has what it was asked and what it did)."""
    return [{k: job[k] for k in JobListed.model_fields} for job in jobs.list_recent(conn, limit)]


@router.get("/api/jobs/{job_id}", response_model=JobOut)
def job_detail(job_id: int, conn: DB) -> dict:
    """One job: status, progress, result, error, when it started and finished."""
    return _get(conn, job_id)


class JobDone(JobOut, Done):
    pass


@router.post("/api/jobs/{job_id}/cancel", response_model=JobDone)
def job_cancel(job_id: int, conn: DB) -> dict:
    """Cancels a queued job, or asks a running one to stop (it stops at its next step). Returns the
    job as GET /api/jobs/{job_id} does."""
    before = _get(conn, job_id)
    jobs.cancel(conn, job_id)
    job = _get(conn, job_id)
    did = (f"Asked {job['title']} to stop (it stops at its next step)" if job["status"] == "running"
           else f"Cancelled {job['title']}" if before["status"] != job["status"] else f"{job['title']} had ended already ({job['status']})")
    return job | {"did": did}


@router.post("/api/jobs/{job_id}/retry", status_code=201, response_model=JobDone)
def job_retry(job_id: int, conn: DB) -> dict:
    """Runs a failed, cancelled or interrupted job again, as a new job with the same work (an
    import or a run skips what's already done). Returns the new job. 409 for any other job."""
    _get(conn, job_id)
    new_id = jobs.retry(conn, job_id)
    if new_id is None:
        raise HTTPException(409, "Only failed, cancelled or interrupted jobs can be retried")
    job = _get(conn, new_id)
    return job | {"did": f"Started {job['title']} again, as job {new_id}"}


@router.get("/api/jobs/{job_id}/log", response_model=Log)
def job_log(job_id: int, conn: DB, offset: Annotated[int, Query(ge=0, description="Where to start reading, in bytes: 0, or the offset the last read returned.")] = 0) -> dict:
    """The job's log (what it did, and the traceback of an error) from byte `offset`; pass the
    returned `offset` to read what comes next. Read it to find why a job failed."""
    _get(conn, job_id)
    text, next_offset = jobs.read_log(job_id, offset)
    return {"text": text, "offset": next_offset}
