"""Projects: HTML pages (forms post, then redirect), and the same data and actions as JSON under /api."""

import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, BeforeValidator, Field, StringConstraints, field_validator

from app import annotators, db, imports, jobs, media, projects, runs
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


class ProjectFields(BaseModel):
    """A project's name, description and type."""

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""
    task: projects.Task


class NewProject(ProjectFields):
    """A new project, with its classes; it gets the annotator suggested for its type."""

    classes: list[projects.ClassChange] = []


Prompt = Annotated[str, StringConstraints(max_length=500)]


class ProjectChanges(ProjectFields):
    """A project's name, description, type and annotator, all of them, when it's changed.
    Left out (a form's empty choice), the annotator is none."""

    annotator: annotators.Finder | None = None


class ProjectForm(ProjectChanges):
    """The project form (projects/form.html; the new project page has no annotator): the class
    rows, one value of each per row; an empty class_id is a new class."""

    class_id: list[Annotated[int | None, BeforeValidator(lambda v: v or None)]] = []
    class_name: list[str] = []
    class_color: list[str] = []
    class_prompt: list[str] = []

    def rows(self) -> list[dict]:
        return [{"id": i, "name": n, "color": c, "prompt": p}
                for i, n, c, p in zip(self.class_id, self.class_name, self.class_color, self.class_prompt, strict=True)]

    def classes(self) -> list[projects.ClassChange]:
        return [projects.ClassChange(**row) for row in self.rows()]


class AutoAnnotate(BaseModel):
    """Which of the project's images to auto-annotate, and how (see annotators.start)."""

    where: str = Field("", max_length=2000)
    count: int | None = Field(None, ge=1)
    annotator: annotators.Finder | None = None
    class_ids: list[int] | None = None
    min_score: float = Field(0.0, ge=0, le=1)
    again: bool = False
    media_ids: list[int] | None = None


class AutoAnnotateForm(BaseModel):
    """The auto-annotate dialog: each class's prompt (class_id and prompt in the same order), the
    classes to run (`use`), and whether to save the prompts, run (saving them too), or try them."""

    where: str = Field("", max_length=2000)
    class_id: list[int] = []
    prompt: list[Prompt] = []
    use: list[int] = []
    min_score: float = Field(0.0, ge=0, le=1)
    again: bool = False
    step: Literal["run", "save", "try"] = "run"  # not "action": a field by that name hides the form's action


class TryAnnotator(BaseModel):
    """Which images to try the annotator on, and with what (see annotators.try_)."""

    where: str = Field("", max_length=2000)
    annotator: annotators.Finder | None = None
    class_ids: list[int] | None = None
    prompts: dict[int, Prompt] | None = None


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


def _started(start: Callable[[], dict | None]) -> dict:
    """What an annotator's start (annotators.start, try_) returned: 404 if there's no such
    project, 409 if it's frozen, 422 with the reason it can't start."""
    try:
        started = start()
    except projects.FrozenError:
        raise HTTPException(409, "The project is frozen; duplicate it to change it") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if started is None:
        raise HTTPException(404, "Project not found")
    return started


def _update(conn: sqlite3.Connection, project_id: int, changed: ProjectChanges,
            classes: list[projects.ClassChange] | None = None) -> None:
    """Change the project, and set its classes if given. Raises ValueError (see projects.set_classes)."""
    with _writing(conn, project_id):
        projects.update(conn, project_id, changed.name, changed.description, changed.task, changed.annotator)
        if classes is not None:
            projects.set_classes(conn, project_id, classes)


def _delete(conn: sqlite3.Connection, project_id: int) -> None:
    """Delete the project and its runs, then the media files nothing else uses. 409 while its jobs run."""
    _exists(conn, project_id)
    try:
        with conn:
            unused = media.remove_unused(conn, projects.delete(conn, project_id))
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    for path in unused:
        path.unlink(missing_ok=True)


def _create(conn: sqlite3.Connection, new: ProjectFields, classes: list[projects.ClassChange]) -> int:
    """Create the project with its classes. Raises ValueError (see projects.set_classes)."""
    with conn:
        project_id = projects.create(conn, new.name, new.description, new.task, annotators.SUGGESTED[new.task])
        projects.set_classes(conn, project_id, classes)
    return project_id


def _form_page(request: Request, conn: sqlite3.Connection, project_id: int | None = None,
               form: ProjectForm | None = None, error: str | None = None) -> HTMLResponse:
    """The new project page, or a project's edit page; after a refused save, with what was typed and why."""
    project, values, counts, deletion = None, {"task": "detection", "classes": []}, {}, None
    if project_id is not None:
        project = values = projects.get(conn, project_id)
        if project is None:
            raise HTTPException(404, "Project not found")
        counts, deletion = projects.annotation_counts(conn, project_id), projects.deletion(conn, project_id)
    if form is not None:
        values = values | form.model_dump() | {"classes": form.rows()}
    return templates.TemplateResponse(
        request, "projects/form.html",
        {"project": project, "values": values, "counts": counts, "deletion": deletion, "palette": projects.PALETTE,
         "error": error},
        status_code=422 if error else 200,
    )


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


@router.get("/projects/new", response_class=HTMLResponse)
def new_project_page(request: Request, conn: DB):
    return _form_page(request, conn)


@router.post("/projects")
def create_project(request: Request, form: Annotated[ProjectForm, Form()], conn: DB):
    try:
        project_id = _create(conn, form, form.classes())
    except ValueError as e:
        return _form_page(request, conn, form=form, error=str(e))
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(request: Request, project_id: int, conn: DB, where: Where = "", name: str | None = None, page: Page = 1):
    """A filter that can't be used shows why, over all of the project's images."""
    try:
        filter, error = _parse(where, name), None
        project = _detail(conn, project_id, page, filter)
    except FilterError as e:
        filter, error = Filter(), str(e)
        project = _detail(conn, project_id, 1, filter)
    context = {"project": project, "filter": filter, "error": error, "where": where,
               "auto": annotators.overview(conn, project_id, filter), "tried": annotators.latest_try(conn, project_id)}
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
    annotator = image["project"]["annotator"]
    context = {"image": image, "to_download": runs.to_download(annotator) if annotator else []}
    return templates.TemplateResponse(request, "projects/image.html", context)


@router.post("/projects/{project_id}/images/{media_id}/annotate")
def auto_annotate_image(project_id: int, media_id: int, conn: DB):
    """Auto-annotate the image open in the editor, for all classes, even those it's done for (new
    objects that are already there are left out)."""
    _started(lambda: annotators.start(conn, project_id, again=True, media_ids=[media_id]))
    return RedirectResponse(f"/projects/{project_id}/images/{media_id}", status_code=303)


@router.get("/projects/{project_id}/edit", response_class=HTMLResponse)
def edit_project_page(request: Request, project_id: int, conn: DB):
    return _form_page(request, conn, project_id)


@router.post("/projects/{project_id}/edit")
def edit_project(request: Request, project_id: int, form: Annotated[ProjectForm, Form()], conn: DB):
    try:
        _update(conn, project_id, form, form.classes())
    except ValueError as e:
        return _form_page(request, conn, project_id, form, str(e))
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/annotate")
def auto_annotate(project_id: int, form: Annotated[AutoAnnotateForm, Form()], conn: DB):
    """Try the dialog's prompts on a few images (nothing saved), or save them, then (Run) start
    auto-annotating the classes ticked."""
    filter = Filter.parse(form.where)
    try:
        prompts = dict(zip(form.class_id, form.prompt, strict=True))
        if form.step == "try":
            _started(lambda: annotators.try_(conn, project_id, filter, class_ids=form.use, prompts=prompts))
        else:
            with _writing(conn, project_id):
                projects.set_prompts(conn, project_id, prompts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if form.step == "run":
        _started(lambda: annotators.start(conn, project_id, filter, class_ids=form.use, min_score=form.min_score,
                                          again=form.again))
    return RedirectResponse(f"/projects/{project_id}{query({'where': form.where})}", status_code=303)


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
    """Every project: id, name, type, image counts, how many are labelled, classes, covers."""
    return projects.list_all(conn)


@router.post("/api/projects", status_code=201)
def api_create_project(new: NewProject, conn: DB) -> dict:
    """Creates a project with its classes (see projects.set_classes)."""
    try:
        return {"id": _create(conn, new, new.classes)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


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


@router.get("/api/projects/{project_id}/deletion")
def api_deletion(project_id: int, conn: DB) -> dict:
    """What deleting the project would remove, and how many of its jobs still run (see projects.deletion)."""
    _exists(conn, project_id)
    return projects.deletion(conn, project_id)


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


@router.patch("/api/projects/{project_id}/classes/prompts", status_code=204)
def api_set_prompts(project_id: int, prompts: dict[int, Prompt], conn: DB) -> None:
    """Changes some classes' prompts ({class id: prompt}); see projects.set_prompts."""
    try:
        with _writing(conn, project_id):
            projects.set_prompts(conn, project_id, prompts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.put("/api/projects/{project_id}/classes", status_code=204)
def api_set_classes(project_id: int, classes: list[projects.ClassChange], conn: DB) -> None:
    """Sets the project's class list, in order; see projects.set_classes."""
    try:
        with _writing(conn, project_id):
            projects.set_classes(conn, project_id, classes)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.get("/api/projects/{project_id}/annotate")
def api_auto_annotate_overview(project_id: int, conn: DB, where: Where = "") -> dict:
    """Before auto-annotating the images matching ?where=: how many match, per class how many
    still need it, and the annotator's models still to download."""
    found = annotators.overview(conn, project_id, Filter.parse(where))
    if found is None:
        raise HTTPException(404, "Project not found")
    return found


@router.post("/api/projects/{project_id}/annotate", status_code=202)
def api_auto_annotate(project_id: int, request: AutoAnnotate, conn: DB) -> dict:
    """Starts auto-annotating; follow the job at /api/jobs/{job_id}, the run at /api/runs/{run_id}."""
    return _started(lambda: annotators.start(conn, project_id, Filter.parse(request.where), request.count, request.annotator,
                                             request.class_ids, request.min_score, request.again, request.media_ids))


@router.post("/api/projects/{project_id}/try", status_code=202)
def api_try_annotator(project_id: int, request: TryAnnotator, conn: DB) -> dict:
    """Starts a Try: the annotator on a few images, nothing written to the project. Follow the job
    at /api/jobs/{job_id}, then read the run at /api/runs/{run_id}."""
    return _started(lambda: annotators.try_(conn, project_id, Filter.parse(request.where), request.annotator,
                                            request.class_ids, request.prompts))


@router.post("/api/projects/{project_id}/import", status_code=202)
def api_import_images(project_id: int, request: ImageImport, conn: DB) -> dict:
    """Starts an import job for {"paths": [files and folders]}; follow it at /api/jobs/{job_id}."""
    return {"job_id": _start_import(conn, project_id, request)}
