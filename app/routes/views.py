"""Views the agent writes: their pages, the API to write, read and restore them, and the links
added to the navigation."""

from functools import cache

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app import settings, view_components, views
from app.templating import templates

router = APIRouter()


class ViewWrite(BaseModel):
    """A new version of a view: its whole template; title (needed for a new view), kind and
    workflow change only when given."""

    template: str = Field(max_length=500_000)
    title: str | None = Field(None, min_length=1, max_length=200)
    kind: views.Kind | None = None
    workflow: str | None = None


class Restore(BaseModel):
    version: int


def _found(found):
    if found is None:
        raise HTTPException(404, "View not found")
    return found


def view_error(request: Request, error: views.ViewError) -> JSONResponse:
    """For the API (registered in app/server.py): a view that can't be written or read is a 422
    with the reason (and the template's line) in words."""
    return JSONResponse({"detail": str(error)}, status_code=422)


# --- Pages ---


@router.get("/views", response_class=HTMLResponse)
def view_list(request: Request):
    return templates.TemplateResponse(request, "views/list.html", {"views": views.list_all()})


@cache
def _examples() -> list[tuple]:
    """Each component's name, call, what it does, example template and that example rendered
    (they never change)."""
    components, _ = view_components.docs()
    return [(name, call, about, example, views.render(example, {})) for name, call, about, example in components]


@router.get("/components", response_class=HTMLResponse)
def components_page(request: Request):
    """The components a view is built from, each with its example rendered."""
    return templates.TemplateResponse(request, "views/components.html", {
        "examples": _examples(), "layout": view_components.LAYOUT, "filters": view_components.docs()[1]})


@router.get("/views/{view_id}", response_class=HTMLResponse)
def view_page(request: Request, view_id: str, version: int | None = None):
    """The view (an older version with ?version=); a template that no longer renders (say, the
    data it reads is gone) shows why instead."""
    view = _found(views.get(view_id, version))
    body, error = views.page(view)
    return templates.TemplateResponse(request, "views/view.html", {"view": view, "body": body, "error": error})


# --- API ---


@router.get("/api/views")
def api_views() -> list[dict]:
    """Every view: id, title, kind, workflow, newest version, page."""
    return views.list_all()


@router.get("/api/views/{view_id}")
def api_view(view_id: str, version: int | None = None) -> dict:
    """The view with its template (the newest version, or ?version=), and for a report the API
    answers it recorded."""
    return _found(views.get(view_id, version))


@router.put("/api/views/{view_id}")
def api_write_view(view_id: str, write: ViewWrite) -> dict:
    """Adds a version (making the view if it's new). 422 if the template doesn't render, with the
    line and why."""
    return views.save(view_id, write.template, write.title, write.kind, write.workflow)


@router.get("/api/views/{view_id}/versions")
def api_view_versions(view_id: str) -> list[dict]:
    """The view's versions, newest first."""
    return _found(views.history(view_id))


@router.post("/api/views/{view_id}/restore")
def api_restore_view(view_id: str, restore: Restore) -> dict:
    """Makes an older version the newest again (as a new version)."""
    return _found(views.restore(view_id, restore.version))


@router.delete("/api/views/{view_id}", status_code=204)
def api_delete_view(view_id: str) -> None:
    """Deletes the view, all its versions and its link in the navigation."""
    _found(views.delete(view_id) or None)
    settings.remove_nav(f"/views/{view_id}")


@router.get("/api/nav")
def api_nav() -> list[dict]:
    """The links added to the navigation (label, href), after the app's own sections and Views."""
    return settings.nav()


@router.put("/api/nav")
def api_set_nav(items: list[settings.NavItem]) -> list[dict]:
    """Replaces the added links. Each is checked: a bad one refuses the whole list (422)."""
    try:
        return settings.set_nav([i.model_dump() for i in items])
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False)) from None
