"""HTML page routes."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from mozo.workflow import WorkflowError

from app import runs, views
from app.templating import templates

router = APIRouter()


@router.get("/")
def home():
    return RedirectResponse("/projects")


# Sections that aren't built yet: each shows a short note (its name is the breadcrumb's switcher).
def _coming(request: Request, title: str):
    return templates.TemplateResponse(request, "pages/coming.html", {"title": title})


@router.get("/workflows", response_class=HTMLResponse)
def workflows(request: Request):
    """Every workflow: Use opens its Studio page; Edit, the builder."""
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
    """The Studio pages agents built, and every workflow's own."""
    built = [v for v in views.list_all() if v["kind"] == "studio"]
    return templates.TemplateResponse(request, "studio/list.html", {"built": built, "workflows": runs.workflows()})


@router.get("/studio/{workflow_id}", response_class=HTMLResponse)
def studio_page(request: Request, workflow_id: str):
    """A workflow's Studio page: the newest one an agent built for it, else the plain one (drop
    images in, see what it keeps)."""
    try:
        workflow = runs.load(workflow_id)
    except WorkflowError:
        workflow = None
    if workflow is None:
        raise HTTPException(404, "No such workflow, or it doesn't load")
    view = views.studio(workflow_id, workflow.name or workflow_id)
    body, error = views.page(view)
    crumbs = [("Studio", "/studio"), (view["title"], None)]
    return templates.TemplateResponse(request, "views/view.html", {"view": view, "body": body, "error": error, "crumbs": crumbs})


@router.get("/data-lake", response_class=HTMLResponse)
def data_lake(request: Request):
    return _coming(request, "Data lake")
