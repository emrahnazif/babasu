"""Projects: HTML pages (forms post, then redirect), and the same data and actions as JSON under /api."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, StringConstraints

from app import db, projects
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]
Page = Annotated[int, Query(ge=1)]


class NewProject(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""


def _detail(conn: sqlite3.Connection, project_id: int, page: int) -> dict:
    project = projects.detail(conn, project_id, page)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


def _create(conn: sqlite3.Connection, new: NewProject) -> int:
    with conn:
        return projects.create(conn, new.name, new.description)


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
