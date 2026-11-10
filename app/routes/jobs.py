"""Jobs: the drawer's list fragment, a page per job, and the JSON API shared by the UI and the agent.

Jobs are started by the feature they belong to (e.g. a project import); here they are listed,
followed, cancelled and retried.
"""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app import db, jobs
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


@router.get("/api/jobs")
def job_list(conn: DB) -> list[dict]:
    return jobs.list_recent(conn)


@router.get("/api/jobs/{job_id}")
def job_detail(job_id: int, conn: DB) -> dict:
    return _get(conn, job_id)


@router.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: int, conn: DB) -> dict:
    _get(conn, job_id)
    jobs.cancel(conn, job_id)
    return _get(conn, job_id)


@router.post("/api/jobs/{job_id}/retry", status_code=201)
def job_retry(job_id: int, conn: DB) -> dict:
    _get(conn, job_id)
    new_id = jobs.retry(conn, job_id)
    if new_id is None:
        raise HTTPException(409, "Only failed, cancelled or interrupted jobs can be retried")
    return _get(conn, new_id)


@router.get("/api/jobs/{job_id}/log")
def job_log(job_id: int, conn: DB, offset: Annotated[int, Query(ge=0)] = 0) -> dict:
    """Log text from byte `offset`; pass the returned `offset` to read what comes next."""
    _get(conn, job_id)
    text, next_offset = jobs.read_log(job_id, offset)
    return {"text": text, "offset": next_offset}
