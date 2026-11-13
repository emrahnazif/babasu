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
    """Every workflow: Use opens its Studio page; Edit (the builder) comes later."""
    return templates.TemplateResponse(request, "workflows/list.html", {"workflows": runs.workflows()})


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
