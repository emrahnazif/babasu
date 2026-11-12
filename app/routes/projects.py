"""Projects: HTML pages (forms post, then redirect), and the same data and actions as JSON under /api."""

import sqlite3
from contextlib import contextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, BeforeValidator, StringConstraints, field_validator

from app import annotators, db, imports, jobs, media, projects
from app.filters import Filter, FilterError, ObjectCondition, Range
from app.templating import query, templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]
Page = Annotated[int, Query(ge=1)]
# Which images: the filter's text form (app/filters.py), e.g. ?where=(car size:8..16) -person
Where = Annotated[str, Query(max_length=2000)]


# A number field of a form: empty means "no limit".
Number = Annotated[float | None, BeforeValidator(lambda v: v if v not in ("", None) else None)]


def _range(low: float | None, high: float | None) -> Range | None:
    return None if low is None and high is None else Range(min=low, max=high)


class ConditionForm(BaseModel):
    """A filter panel's form (project page): the current filter, which object condition to
    change (none: add one), and the `part` the panel shows; the condition's other parts stay.
      class   Has / Doesn't have, and the classes (several: any of them)
      size    width and height, for objects of one class (empty: any class)
      aspect  width / height, for objects of one class
      edge    touching the image's edge or not, for objects of one class"""

    where: str = ""
    index: int | None = None
    part: Literal["class", "size", "aspect", "edge"]
    negate: bool = False
    classes: list[str] = []
    of_class: str = ""
    w_min: Number = None
    w_max: Number = None
    h_min: Number = None
    h_max: Number = None
    aspect_min: Number = None
    aspect_max: Number = None
    edge: bool = True

    def apply(self, filter: Filter) -> Filter:
        found = self.index is not None and 0 <= self.index < len(filter.conditions)
        c = filter.conditions[self.index] if found else ObjectCondition()
        if self.part == "class":
            c = c.model_copy(update={"negate": self.negate, "classes": self.classes})
        else:
            changes = {"classes": [self.of_class] if self.of_class else []}
            if self.part == "size":
                changes |= {"w": _range(self.w_min, self.w_max), "h": _range(self.h_min, self.h_max)}
            elif self.part == "aspect":
                changes["aspect"] = _range(self.aspect_min, self.aspect_max)
            else:
                changes["edge"] = self.edge
            c = c.model_copy(update=changes)
        return filter.with_condition(c, self.index if found else None)


class ObjectCountForm(BaseModel):
    """The project page's form for how many objects an image has; both empty: any number."""

    where: str = ""
    objects_min: Number = None
    objects_max: Number = None


def _filter_page(project_id: int, filter: Filter) -> RedirectResponse:
    return RedirectResponse(f"/projects/{project_id}{query({'where': filter.text()})}", status_code=303)


def _parse(where: str, name: str | None) -> Filter:
    """The filter from ?where=, with ?name= (a search field's file name) put in it. Raises FilterError."""
    filter = Filter.parse(where)
    return filter if name is None else filter.set(name=name.strip())


def filter_error(request: Request, error: FilterError) -> JSONResponse:
    """For the API (registered in app/server.py): a filter that can't be used is a 422 with
    the reason in words. The pages show the reason above the images instead."""
    return JSONResponse({"detail": str(error)}, status_code=422)


class NewProject(BaseModel):
    """A new project: its name, description and type (it gets the annotator suggested for the type)."""

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""
    task: projects.Task


class ProjectChanges(NewProject):
    """A project's name, description, type and annotator, all of them, when it's changed.
    Left out (a form's empty choice), the annotator is none."""

    annotator: annotators.Finder | None = None


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


def _detail(conn: sqlite3.Connection, project_id: int, page: int, filter: Filter) -> dict:
    project = projects.detail(conn, project_id, page, filter)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


def _image(conn: sqlite3.Connection, project_id: int, media_id: int, filter: Filter) -> dict:
    image = projects.image(conn, project_id, media_id, filter)
    if image is None:
        raise HTTPException(404, "Image not found")
    return image


def _exists(conn: sqlite3.Connection, project_id: int) -> None:
    if projects.get(conn, project_id) is None:
        raise HTTPException(404, "Project not found")


@contextmanager
def _writing(conn: sqlite3.Connection, project_id: int):
    """A change to a project, committed at the end: 404 if there's no such project, 409 if it's frozen."""
    _exists(conn, project_id)
    try:
        with conn:
            yield
    except projects.FrozenError:
        raise HTTPException(409, "The project is frozen; duplicate it to change it") from None


def _update(conn: sqlite3.Connection, project_id: int, changed: ProjectChanges) -> None:
    with _writing(conn, project_id):
        projects.update(conn, project_id, changed.name, changed.description, changed.task, changed.annotator)


def _delete(conn: sqlite3.Connection, project_id: int) -> None:
    """Delete the project, then those of its media files no other project uses."""
    _exists(conn, project_id)
    with conn:
        unused = media.remove_unused(conn, projects.delete(conn, project_id))
    for path in unused:
        path.unlink(missing_ok=True)


def _create(conn: sqlite3.Connection, new: NewProject) -> int:
    with conn:
        return projects.create(conn, new.name, new.description, new.task)


def _set_annotations(conn: sqlite3.Connection, project_id: int, media_id: int, annotations: list[projects.Annotation]) -> None:
    try:
        with _writing(conn, project_id):
            found = projects.set_annotations(conn, project_id, media_id, annotations)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if not found:
        raise HTTPException(404, "Image not found")


def _start_import(conn: sqlite3.Connection, project_id: int, request: ImageImport) -> int:
    """Start an import job; the paths are checked by the job, so a bad path shows up in its result."""
    _exists(conn, project_id)
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
def project_page(request: Request, project_id: int, conn: DB, where: Where = "", name: str | None = None, page: Page = 1):
    """A filter that can't be used shows why, over all of the project's images."""
    try:
        filter, error = _parse(where, name), None
        project = _detail(conn, project_id, page, filter)
    except FilterError as e:
        filter, error = Filter(), str(e)
        project = _detail(conn, project_id, 1, filter)
    context = {"project": project, "filter": filter, "error": error, "where": where}
    return templates.TemplateResponse(request, "projects/project.html", context)


@router.post("/projects/{project_id}/conditions")
def set_condition(project_id: int, form: Annotated[ConditionForm, Form()]):
    """Add or change one object condition of the page's filter (one part of it), and show the result."""
    return _filter_page(project_id, form.apply(Filter.parse(form.where)))


@router.post("/projects/{project_id}/object-count")
def set_object_count(project_id: int, form: Annotated[ObjectCountForm, Form()]):
    """Set (or clear) how many objects the page's filter asks an image to have."""
    return _filter_page(project_id, Filter.parse(form.where).set(objects=_range(form.objects_min, form.objects_max)))


@router.get("/projects/{project_id}/images/{media_id}", response_class=HTMLResponse)
def image_page(request: Request, project_id: int, media_id: int, conn: DB, where: Where = ""):
    try:
        image = _image(conn, project_id, media_id, Filter.parse(where))
    except FilterError:  # the project page says why; here, step through all images instead
        image = _image(conn, project_id, media_id, Filter())
    return templates.TemplateResponse(request, "projects/image.html", {"image": image})


@router.post("/projects/{project_id}/edit")
def edit_project(project_id: int, changed: Annotated[ProjectChanges, Form()], conn: DB):
    _update(conn, project_id, changed)
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/delete")
def delete_project(project_id: int, conn: DB):
    _delete(conn, project_id)
    return RedirectResponse("/projects", status_code=303)


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
def api_project(project_id: int, conn: DB, where: Where = "", name: str | None = None, page: Page = 1) -> dict:
    """The project with one page of the images matching ?where= (app/filters.py) and ?name=."""
    return _detail(conn, project_id, page, _parse(where, name))


@router.get("/api/projects/{project_id}/stats")
def api_stats(project_id: int, conn: DB, field: str, where: Where = "") -> dict:
    """How ?field= (class, w, h, area, aspect or objects) is spread over the images matching ?where=."""
    result = projects.stats(conn, project_id, field, Filter.parse(where))
    if result is None:
        raise HTTPException(404, "Project not found")
    return result


@router.patch("/api/projects/{project_id}", status_code=204)
def api_update_project(project_id: int, changed: ProjectChanges, conn: DB) -> None:
    _update(conn, project_id, changed)


@router.delete("/api/projects/{project_id}", status_code=204)
def api_delete_project(project_id: int, conn: DB) -> None:
    _delete(conn, project_id)


@router.get("/api/projects/{project_id}/images/{media_id}")
def api_image(project_id: int, media_id: int, conn: DB, where: Where = "") -> dict:
    """One image with its annotations, and its previous and next among the images matching ?where=."""
    return _image(conn, project_id, media_id, Filter.parse(where))


@router.put("/api/projects/{project_id}/images/{media_id}/annotations", status_code=204)
def api_set_annotations(project_id: int, media_id: int, annotations: list[projects.Annotation], conn: DB) -> None:
    """Replaces the image's annotations (those of active classes, as the image's read returns them)."""
    _set_annotations(conn, project_id, media_id, annotations)


@router.post("/api/projects/{project_id}/import", status_code=202)
def api_import_images(project_id: int, request: ImageImport, conn: DB) -> dict:
    """Starts an import job for {"paths": [files and folders]}; follow it at /api/jobs/{job_id}."""
    return {"job_id": _start_import(conn, project_id, request)}
