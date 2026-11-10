"""Projects: HTML pages (forms post, then redirect), and the same data and actions as JSON under /api."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, StringConstraints, field_validator

from app import db, imports, jobs, projects
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]
Page = Annotated[int, Query(ge=1)]


class NewProject(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""


class ImageImport(BaseModel):
    """Files and folders on this computer or a NAS. Blank entries (an empty path field) are ignored."""

    paths: list[str]

    @field_validator("paths")
    @classmethod
    def _not_blank(cls, paths: list[str]) -> list[str]:
        paths = [p.strip() for p in paths if p.strip()]
        if not paths:
            raise ValueError("give at least one file or folder")
        return paths


def _detail(conn: sqlite3.Connection, project_id: int, page: int) -> dict:
    project = projects.detail(conn, project_id, page)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


def _create(conn: sqlite3.Connection, new: NewProject) -> int:
    with conn:
        return projects.create(conn, new.name, new.description)


def _start_import(conn: sqlite3.Connection, project_id: int, request: ImageImport) -> int:
    """Start an import job; the paths are checked by the job, so a bad path shows up in its result."""
    if projects.get(conn, project_id) is None:
        raise HTTPException(404, "Project not found")
    params = {"project_id": project_id, "paths": request.paths}
    return jobs.start(conn, imports.KIND, params, project_id)


# --- Pages ---


@router.get("/projects", response_class=HTMLResponse)
def project_list(request: Request, conn: DB):
    return templates.TemplateResponse(request, "projects/list.html", {"projects": projects.list_all(conn)})


@router.post("/projects")
def create_project(new: Annotated[NewProject, Form()], conn: DB):
    return RedirectResponse(f"/projects/{_create(conn, new)}", status_code=303)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(request: Request, project_id: int, conn: DB, page: Page = 1):
    return templates.TemplateResponse(request, "projects/project.html", {"project": _detail(conn, project_id, page)})


@router.post("/projects/{project_id}/import")
def import_images(project_id: int, request: Annotated[ImageImport, Form()], conn: DB):
    """Starts the job and comes back to the project; progress shows in the jobs drawer.
    (With JS the form is sent in the background instead: data-background-form.)"""
    _start_import(conn, project_id, request)
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


# --- API ---


@router.get("/api/projects")
def api_project_list(conn: DB) -> list[dict]:
    return projects.list_all(conn)


@router.post("/api/projects", status_code=201)
def api_create_project(new: NewProject, conn: DB) -> dict:
    return {"id": _create(conn, new)}


@router.get("/api/projects/{project_id}")
def api_project(project_id: int, conn: DB, page: Page = 1) -> dict:
    return _detail(conn, project_id, page)


@router.post("/api/projects/{project_id}/import", status_code=202)
def api_import_images(project_id: int, request: ImageImport, conn: DB) -> dict:
    """Starts an import job for {"paths": [files and folders]}; follow it at /api/jobs/{job_id}."""
    return {"job_id": _start_import(conn, project_id, request)}
