"""The data lake: everything the person has, found in one place across projects (images, projects,
workflows, models), and every project's images worked with together. Things are managed on their
own pages; the lake finds them."""

import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import db, projects, runs, vision
from app.filters import Filter, FilterError
from app.routes.projects import ConditionForm, ObjectCountForm, Page, Where, parse_filter
from app.templating import query, templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]
Kind = Literal["all", "images", "projects", "workflows", "models"]
FIRST = 12  # images the All view shows; the rest are a click away (Images)
FIRST_ITEMS = 6  # projects, workflows and models the All view shows of each


def _projects(conn: sqlite3.Connection, q: str) -> list[dict]:
    return [p for p in projects.list_all(conn) if q.casefold() in p["name"].casefold()]


def _workflows(conn: sqlite3.Connection, q: str) -> list[dict]:
    named = {w["id"] for w in runs.named(q)}  # what the search finds, by id or name
    return [w for w in runs.workflows() if w["id"] in named]


def _models(conn: sqlite3.Connection, q: str) -> list[dict]:
    """The models downloaded, named like `q`."""
    return [{"name": name, "task": f["task"], "license": v["license"], "state": v["state"]}
            for f in vision.catalogue(conn) for v in f["variants"]
            if v["state"] in ("downloaded", "loaded") and q.casefold() in (name := f"{f['family']} {v['variant']}").casefold()]


# What each view lists besides images (the Images view: the projects, for its project menu).
LISTED = {"all": ("projects", "workflows", "models"), "images": ("projects",), "projects": ("projects",),
          "workflows": ("workflows",), "models": ("models",)}
LISTS = {"projects": _projects, "workflows": _workflows, "models": _models}


def _images_page(project: int | None, filter: Filter) -> RedirectResponse:
    return RedirectResponse("/data-lake" + query({"kind": "images", "project": project, "where": filter.text()}), status_code=303)


@router.get("/data-lake", response_class=HTMLResponse)
def lake(request: Request, conn: DB, kind: Kind = "all", q: str = "", where: Where = "", name: str | None = None,
         project: Annotated[int | None, Query()] = None, page: Page = 1):
    """All: a section per kind, its first few; Images: every project's images (or one's) with the
    filter bar, a page at a time; Projects, Workflows, Models: what there is. `q`: names containing it.
    A filter that can't be used shows why, over all the images."""
    q = q.strip()
    context = {"kind": kind, "q": q, "project": project, "error": None, "first_items": FIRST_ITEMS}
    context |= {listed: LISTS[listed](conn, "" if kind == "images" else q) for listed in LISTED[kind]}
    if kind in ("all", "images"):
        size = projects.PAGE_SIZE if kind == "images" else FIRST
        try:
            filter = parse_filter(where, name)
            if kind == "all" and q:
                filter = filter.set(name=q)
            found = projects.browse(conn, filter, page, project, size)
        except FilterError as e:
            filter, context["error"] = Filter(), str(e)
            found = projects.browse(conn, filter, 1, project, size)
        context |= {"filter": filter, "found": found}
    return templates.TemplateResponse(request, "lake/lake.html", context)


@router.post("/data-lake/conditions")
def set_condition(form: Annotated[ConditionForm, Form()], project: Annotated[int | None, Form()] = None):
    """Add or change one object condition of the Images filter (one part of it), and show the result."""
    return _images_page(project, form.apply(Filter.parse(form.where)))


@router.post("/data-lake/object-count")
def set_object_count(form: Annotated[ObjectCountForm, Form()], project: Annotated[int | None, Form()] = None):
    """Set (or clear) how many objects the Images filter asks an image to have."""
    return _images_page(project, form.apply(Filter.parse(form.where)))


@router.post("/data-lake/copy")
def copy(conn: DB, where: Annotated[str, Form()] = "", project: Annotated[int | None, Form()] = None,
         to_project: Annotated[str, Form()] = "", new_name: Annotated[str, Form()] = ""):
    """Copy the images the Images view shows (its filter, its project if one) into a project, with
    their annotations (projects.add_images): an existing one, or a new one named `new_name`; then
    open it."""
    try:
        media_ids = projects.sample(conn, Filter.parse(where), project)
    except FilterError as e:
        raise HTTPException(422, str(e)) from None
    with conn:
        target = int(to_project) if to_project else projects.create(conn, new_name.strip() or "Copied images")
        projects.add_images(conn, target, media_ids)
    return RedirectResponse(f"/projects/{target}", status_code=303)
