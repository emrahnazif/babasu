"""Views the agent writes: their pages, the API to write, read and restore them, and the links
added to the navigation."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app import settings, views
from app.routes import Answer
from app.templating import templates

router = APIRouter()


class ViewWrite(BaseModel):
    """A new version of a view: its whole template, and the data it reads if any; its title
    (needed for a new view) and data change only when given."""

    template: str = Field(max_length=500_000, description="The whole page: a Jinja template (see get_components).")
    title: str | None = Field(None, min_length=1, max_length=200, description="The page's title; needed for a new view.")
    data: dict | list | None = Field(None, description="What your code worked out, as JSON: the template reads it "
                                     "as `data`, and the version keeps it as it keeps its API answers. Not given: "
                                     "the last version's.")


class Restore(BaseModel):
    """Which version to make the newest again."""

    version: int = Field(description="A version number, from GET /api/views/{view_id}/versions.")


class ViewListed(Answer):
    id: str = Field(description="The view's id; its page is /views/{id}.")
    title: str
    version: int = Field(description="Its newest version.")
    updated_at: str = Field(description="When its newest version was written, UTC (ISO 8601).")


class ViewDetail(ViewListed):
    shown: int = Field(description="The version this template is (the newest unless ?version= asked for another).")
    template: str = Field(description="The page: a Jinja template.")
    answers: dict = Field(description="The API answers it recorded when written, by path: what the page shows.")
    data: dict | list | None = Field(description="The data it was written with (the template's `data`); null if none.")


class ViewVersion(Answer):
    version: int
    written_at: str = Field(description="UTC (ISO 8601).")
    size: int = Field(description="Its template's size, in bytes.")


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


@router.get("/views/{view_id}", response_class=HTMLResponse)
def view_page(request: Request, view_id: str, version: int | None = None):
    """The view (an older version with ?version=); a version that no longer renders (say, a
    component it uses is gone) shows why instead."""
    view = _found(views.get(view_id, version))
    body, error = views.page(view)
    return templates.TemplateResponse(request, "views/view.html", {"view": view, "body": body, "error": error})


# --- API ---


@router.get("/api/views", response_model=list[ViewListed])
def api_views() -> list[dict]:
    """Every view, newest change first: id, title, newest version and when it was written."""
    return views.list_all()


@router.get("/api/views/{view_id}", response_model=ViewDetail)
def api_view(view_id: str, version: Annotated[int | None, Query(description="An older version; empty for the newest.")] = None) -> dict:
    """The view with its template (the newest version, or ?version=), the API answers it
    recorded and its data."""
    return _found(views.get(view_id, version))


@router.put("/api/views/{view_id}", response_model=ViewDetail)
def api_write_view(view_id: str, write: ViewWrite) -> dict:
    """Adds a version (making the view if it's new). 422 if the template doesn't render, with the
    line and why."""
    return views.save(view_id, write.template, write.title, write.data)


@router.get("/api/views/{view_id}/versions", response_model=list[ViewVersion])
def api_view_versions(view_id: str) -> list[dict]:
    """The view's versions, newest first."""
    return _found(views.history(view_id))


@router.post("/api/views/{view_id}/restore", response_model=ViewDetail)
def api_restore_view(view_id: str, restore: Restore) -> dict:
    """Makes an older version the newest again (as a new version)."""
    return _found(views.restore(view_id, restore.version))


@router.delete("/api/views/{view_id}", status_code=204)
def api_delete_view(view_id: str) -> None:
    """Deletes the view, all its versions and its link in the navigation. History can bring the
    view back for 30 days (not its link)."""
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
