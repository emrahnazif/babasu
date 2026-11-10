"""Projects: HTML pages, and the same data as JSON under /api."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app import db, projects
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]
Page = Annotated[int, Query(ge=1)]


def _detail(conn: sqlite3.Connection, project_id: int, page: int) -> dict:
    project = projects.detail(conn, project_id, page)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.get("/projects", response_class=HTMLResponse)
def project_list(request: Request, conn: DB):
    return templates.TemplateResponse(request, "projects/list.html", {"projects": projects.list_all(conn)})


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(request: Request, project_id: int, conn: DB, page: Page = 1):
    return templates.TemplateResponse(request, "projects/project.html", {"project": _detail(conn, project_id, page)})


@router.get("/api/projects")
def api_project_list(conn: DB) -> list[dict]:
    return projects.list_all(conn)


@router.get("/api/projects/{project_id}")
def api_project(project_id: int, conn: DB, page: Page = 1) -> dict:
    return _detail(conn, project_id, page)
