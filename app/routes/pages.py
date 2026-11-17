"""HTML page routes."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import annotators, db, runs
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Where the app opens: the way into Studio, the way into making a project, and examples to try."""
    return templates.TemplateResponse(request, "pages/home.html")


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: int, conn: DB, page: Annotated[int, Query(ge=1)] = 1):
    """A run: what it found per class, and a page of its images with what it found drawn."""
    run = runs.results_page(conn, run_id, page, shapes=True)
    if run is None:
        raise HTTPException(404, "No such run")
    run = annotators.annotated(conn, run)
    workflow = run["workflow_id"] and runs.load(run["workflow_id"])
    tiles = [{"src": r["thumb_url"], "width": r["width"], "height": r["height"], "annotations": r["annotations"],
              "caption": (", ".join(f"{name} {n}" for name, n in r["objects"].items()) or "nothing found")
                         + (f" · {r['filename']}" if run["project_id"] else "")} for r in run["results"]]
    return templates.TemplateResponse(request, "runs/run.html", {
        "run": run, "tiles": tiles, "workflow": workflow.name if workflow else run["workflow_id"] or "a workflow not saved"})


@router.get("/components", response_class=HTMLResponse)
def components(request: Request):
    """The components the app's pages are built from, each drawn, for designing and checking them."""
    return templates.TemplateResponse(request, "pages/components.html")


# Sections that aren't built yet: each shows a short note (its name is the breadcrumb's switcher).
def _coming(request: Request, title: str):
    return templates.TemplateResponse(request, "pages/coming.html", {"title": title})


@router.get("/workflows", response_class=HTMLResponse)
def workflows(request: Request):
    """Every workflow: Use opens its Studio page (not built yet); Edit, the builder."""
    return templates.TemplateResponse(request, "workflows/list.html", {"workflows": runs.workflows()})


@router.post("/workflows/{workflow_id}/delete")
def delete_workflow(workflow_id: str):
    """Delete a saved workflow (its runs keep their copy) and go back to the list."""
    if not runs.delete_workflow(workflow_id):
        raise HTTPException(404, "No such saved workflow (built-ins can't be deleted)")
    return RedirectResponse("/workflows", status_code=303)


def _builder(request: Request, workflow_id: str | None, document: dict):
    return templates.TemplateResponse(request, "workflows/edit.html", {
        "workflow_id": workflow_id, "document": document, "builtin": workflow_id in runs.BUILTIN})


@router.get("/workflows/new", response_class=HTMLResponse)
def new_workflow(request: Request):
    """The builder on an empty canvas; saving names the new workflow."""
    return _builder(request, None, {"name": "", "description": "", "nodes": [], "edges": []})


@router.get("/workflows/{workflow_id}/edit", response_class=HTMLResponse)
def edit_workflow(request: Request, workflow_id: str):
    """The builder: mozo's editor on the workflow, with our bar (name, Test, Save). A built-in is
    saved as a copy."""
    document = runs.document(workflow_id)
    if document is None:
        raise HTTPException(404, "No such workflow")
    return _builder(request, workflow_id, document)


@router.get("/studio", response_class=HTMLResponse)
def studio(request: Request):
    """The workflows to use, each a link to its Studio page (not built yet)."""
    return templates.TemplateResponse(request, "studio/list.html", {"workflows": runs.workflows()})


@router.get("/data-lake", response_class=HTMLResponse)
def data_lake(request: Request):
    return _coming(request, "Data lake")
