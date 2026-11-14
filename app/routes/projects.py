"""Projects: HTML pages (forms post, then redirect), and the same data and actions as JSON under /api."""

import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from app import annotators, db, imports, jobs, projects, runs, segment, vision
from app.filters import WHERE, Filter, FilterError, ObjectCondition, Range
from app.templating import query, templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]
Page = Annotated[int, Query(ge=1, description="Which page of the matching images (100 a page), from 1.")]
# Which images: the filter's text form (app/filters.py), e.g. ?where=(car size:8..16) -person
Where = Annotated[str, Query(max_length=2000, description=WHERE)]
FINDER = "The workflow that auto-annotates it, from GET /api/workflows (e.g. sam3); null for none."
ANNOTATOR = "Another annotator than the project's, from GET /api/workflows; empty: the project's."
CLASS_IDS = "Which classes (ids from GET /api/projects/{id}); empty for all of them."


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


TASK = "What its annotations are: boxes (detection), outlines (segmentation) or points (keypoints)."


class ProjectFields(BaseModel):
    """A project's name, description and type."""

    name: projects.Name = Field(description="The project's name, e.g. Street cars.")
    description: projects.Description = Field("", description="What it's for, in a sentence or two.")
    task: projects.Task = Field(description=TASK)


class NewProject(ProjectFields):
    """A new project, with its classes; it gets the annotator suggested for its type."""

    classes: list[projects.ClassChange] = Field([], description="The kinds of objects to annotate, in order (no ids: all new).")


Prompt = projects.Prompt


class ProjectChanges(ProjectFields):
    """A project's name, description, type and annotator, all of them, when it's changed.
    Left out (a form's empty choice), the annotator is none."""

    annotator: annotators.Finder | None = Field(None, description=FINDER)


class ProjectPatch(BaseModel):
    """Changes to a project: only the fields given change; the others stay as they are."""

    model_config = ConfigDict(extra="forbid")
    name: projects.Name | None = Field(None, description="A new name.")
    description: projects.Description | None = Field(None, description="A new description.")
    task: projects.Task | None = Field(None, description=TASK)
    annotator: annotators.Finder | None = Field(None, description=FINDER)


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

    where: str = Field("", max_length=2000, description=WHERE)
    count: int | None = Field(None, ge=1, description="How many of the images, picked at random among those with work left; empty for all.")
    annotator: annotators.Finder | None = Field(None, description=ANNOTATOR)
    class_ids: list[int] | None = Field(None, description=CLASS_IDS)
    min_score: float = Field(0.0, ge=0, le=1, description="Leave out objects scoring below this (0..1), beyond the annotator's own threshold.")
    again: bool = Field(False, description="Also run images already auto-annotated for a class (objects already there are still not added twice).")
    media_ids: list[int] | None = Field(None, description="Only these images (their ids; /projects/3/images/42 is image 42); empty for all that match.")


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

    where: str = Field("", max_length=2000, description=WHERE)
    annotator: annotators.Finder | None = Field(None, description=ANNOTATOR)
    class_ids: list[int] | None = Field(None, description=CLASS_IDS)
    prompts: dict[int, Prompt] | None = Field(None, description="Other prompts to try, by class id, e.g. {\"3\": \"red car\"}; not saved.")


class ImageImport(BaseModel):
    """Files and folders on this computer or a NAS. Blank entries (an empty path field) are ignored."""

    paths: list[str] = Field(description="Image files and folders (searched with their subfolders), e.g. [\"~/Desktop/potholes\"].")

    @field_validator("paths")
    @classmethod
    def _not_blank(cls, paths: list[str]) -> list[str]:
        paths = [p.strip() for p in paths if p.strip()]
        if not paths:
            raise ValueError("give at least one file or folder")
        return paths


def _detail(conn: sqlite3.Connection, project_id: int, page: int, filter: Filter, page_size: int = projects.PAGE_SIZE) -> dict:
    project = projects.detail(conn, project_id, page, filter, page_size)
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
        projects.remove(conn, project_id)
    except ValueError as e:
        raise HTTPException(409, str(e)) from None


def _duplicate(conn: sqlite3.Connection, project_id: int, name: str | None) -> int:
    """Duplicate the project (projects.duplicate); 404 if there's none."""
    with conn:
        new_id = projects.duplicate(conn, project_id, name)
    if new_id is None:
        raise HTTPException(404, "Project not found")
    return new_id


def _create(conn: sqlite3.Connection, new: ProjectFields, classes: list[projects.ClassChange]) -> int:
    """Create the project with its classes. Raises ValueError (see projects.set_classes)."""
    with conn:
        return projects.create(conn, new.name, new.description, new.task, annotators.SUGGESTED[new.task], classes)


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
    context = {"image": image, "to_download": runs.to_download(annotator) if annotator else [],
               "click_tool": segment.tool(conn)}
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


@router.post("/projects/{project_id}/duplicate")
def duplicate_project(project_id: int, conn: DB):
    return RedirectResponse(f"/projects/{_duplicate(conn, project_id, None)}", status_code=303)


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
    """Every project, most recently changed first: id, name, type, how many images and how many of
    them labelled, and its classes."""
    return projects.list_all(conn)


@router.post("/api/projects", status_code=201)
def api_create_project(new: NewProject, conn: DB) -> dict:
    """Creates a project with its classes; it gets the annotator suggested for its type. Returns its id.
    422 with why if a class is given twice or a colour isn't one."""
    try:
        return {"id": _create(conn, new, new.classes)}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.get("/api/projects/{project_id}")
def api_project(project_id: int, conn: DB, where: Where = "",
                name: Annotated[str | None, Query(description="Only images whose file name contains this.")] = None,
                page: Page = 1,
                images: Annotated[int, Query(ge=0, le=projects.PAGE_SIZE, description="Images a page; 0 for the project alone.")] = projects.PAGE_SIZE) -> dict:
    """The project (name, type, annotator, classes with their prompts, colours and object counts,
    image and labelled counts) with one page of the images matching ?where=, each with its
    annotations. ?images=0 for the project alone. 422 if the filter names a class it doesn't have."""
    return _detail(conn, project_id, page, _parse(where, name), images)


@router.get("/api/projects/{project_id}/stats")
def api_stats(project_id: int, conn: DB, where: Where = "",
              field: str = Query(description="class: objects per class; w, h, area, aspect: a histogram of the "
                                 "objects (those the filter asks for in brackets, or all); objects: how many images "
                                 "have how many objects.")) -> dict:
    """Measures the images matching ?where=: how ?field= is spread over them."""
    result = projects.stats(conn, project_id, field, Filter.parse(where))
    if result is None:
        raise HTTPException(404, "Project not found")
    return result


@router.patch("/api/projects/{project_id}", status_code=204)
def api_update_project(project_id: int, changes: ProjectPatch, conn: DB) -> None:
    """Changes a project's name, description, type or annotator: only the fields given (classes:
    PATCH /api/projects/{id}/classes). 409 if it's frozen."""
    project = projects.get(conn, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    fields = {k: project[k] for k in ProjectChanges.model_fields} | changes.model_dump(exclude_unset=True)
    _update(conn, project_id, ProjectChanges(**fields))


class Duplicate(BaseModel):
    """The copy's name."""

    name: projects.Name | None = Field(None, description='Its name; empty: "<the project\'s name> copy".')


@router.post("/api/projects/{project_id}/duplicate", status_code=201)
def api_duplicate_project(project_id: int, conn: DB, body: Duplicate | None = None) -> dict:
    """A new project with this one's classes, images and annotations (not its runs), named `name` or "<its name> copy"."""
    return {"id": _duplicate(conn, project_id, body.name if body else None)}


@router.get("/api/projects/{project_id}/deletion")
def api_deletion(project_id: int, conn: DB) -> dict:
    """What deleting the project would remove (images, annotations, runs), and how many of its jobs
    are queued or running (deleting is refused while they are)."""
    _exists(conn, project_id)
    return projects.deletion(conn, project_id)


@router.delete("/api/projects/{project_id}", status_code=204)
def api_delete_project(project_id: int, conn: DB) -> None:
    """Deletes a project with its images (from it), their annotations and its runs; History can
    undo it for 30 days. 409 while one of its jobs is queued or running."""
    _delete(conn, project_id)


@router.get("/api/projects/{project_id}/images/{media_id}")
def api_image(project_id: int, media_id: int, conn: DB, where: Where = "") -> dict:
    """One image of the project: file name, size, its annotations (class, box, outline, points,
    score), and its previous and next among the images matching ?where=."""
    return _image(conn, project_id, media_id, Filter.parse(where))


class ImageAdd(BaseModel):
    """An image already stored (in another project or a run) to add to this one."""

    model_config = ConfigDict(extra="forbid")
    media_id: int = Field(description="The image's id, e.g. from GET /api/projects/{id} (its media_id).")
    filename: str = Field(min_length=1, max_length=500, description="Its file name in this project, e.g. the one it has elsewhere.")
    annotations: list[projects.Annotation] = Field([], description="Its objects here, with this project's class ids.")


@router.post("/api/projects/{project_id}/images", status_code=201)
def api_add_image(project_id: int, image: ImageAdd, conn: DB) -> dict:
    """Adds an image that's already stored to the project, with its objects (class ids of this
    project). The file isn't copied: projects share it. {added: false} if it's there already.
    422 for an unknown image or class; 409 if the project is frozen."""
    if conn.execute("SELECT 1 FROM media WHERE id = ?", (image.media_id,)).fetchone() is None:
        raise HTTPException(422, f"No image {image.media_id}")
    try:
        with _writing(conn, project_id):
            added = projects.add_image(conn, project_id, image.media_id, image.filename, image.annotations)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"added": added}


@router.put("/api/projects/{project_id}/images/{media_id}/annotations", status_code=204)
def api_set_annotations(project_id: int, media_id: int, annotations: list[projects.Annotation], conn: DB) -> None:
    """Replaces the image's annotations with this list (read them with GET …/images/{media_id},
    change the list, send it back: to remove one, leave it out). 422 for an unknown class."""
    _set_annotations(conn, project_id, media_id, annotations)


@contextmanager
def _clicking(conn: sqlite3.Connection, project_id: int, media_id: int):
    """The image click to segment works on (projects.image_file); 404 if there's none, 409 if no
    click model is downloaded."""
    image = projects.image_file(conn, project_id, media_id)
    if image is None:
        raise HTTPException(404, "Image not found")
    try:
        yield image
    except LookupError as e:
        raise HTTPException(409, f"{e}: choose one (PUT /api/click-tool) and it downloads") from None


@router.get("/api/click-tool")
def api_click_tool(conn: DB) -> dict:
    """The image editor's click to segment tool: on or off, the model chosen, the one in use (the
    chosen one, or while it downloads another that's downloaded), and each model with its licence
    and whether it's downloaded, downloading, or how many bytes it would download."""
    return segment.tool(conn)


@router.put("/api/click-tool")
def api_change_click_tool(change: segment.Change, conn: DB) -> dict:
    """Turns the click tool on or off and/or chooses its model. A model that isn't downloaded starts
    downloading (a job); until it's done, the first downloaded one answers. Returns the tool as GET
    does."""
    return segment.change(conn, change)


@router.post("/api/projects/{project_id}/images/{media_id}/prepare", status_code=204)
def api_prepare_image(project_id: int, media_id: int, conn: DB) -> None:
    """Gets the image ready for click to segment: encodes it now (the slow half, a fraction of a
    second), so each segment call on it afterwards takes milliseconds. Optional: the first segment
    call prepares it too."""
    with _clicking(conn, project_id, media_id) as image:
        segment.prepare(image["path"])


@router.post("/api/projects/{project_id}/images/{media_id}/segment", response_model=projects.Annotation | None)
def api_segment(project_id: int, media_id: int, prompt: segment.Prompt, conn: DB):
    """The object at some points and/or in a box, found by the click model in use, as an
    annotation of the class given, ready to add to the image's annotations: its box, and in a
    segmentation project its polygon too. Points are in the image's pixels; label 1 includes a point,
    0 leaves it out (e.g. one click on a car, then 0 on its wheel to leave the wheel out). null if
    nothing was found. Nothing is saved. Only the latest call runs, for a pointer that keeps moving:
    a call a newer one overtook while it waited is skipped and answers 204, with no content."""
    with _clicking(conn, project_id, media_id) as image:
        if prompt.class_id not in image["class_ids"]:
            raise HTTPException(422, f"No class {prompt.class_id} in this project")
        try:
            return segment.segment(image["path"], prompt, image["task"])
        except segment.Superseded:
            return Response(status_code=204)


@router.patch("/api/projects/{project_id}/classes/prompts", status_code=204)
def api_set_prompts(project_id: int, prompts: Annotated[dict[int, Prompt], Body(
        description="Prompts by class id, e.g. {\"3\": \"hole in the road surface\"}; empty: the class's name.")],
                    conn: DB) -> None:
    """Changes some classes' prompts: what annotators look for."""
    try:
        with _writing(conn, project_id):
            projects.set_prompts(conn, project_id, prompts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.put("/api/projects/{project_id}/classes", status_code=204)
def api_set_classes(project_id: int, classes: list[projects.ClassChange], conn: DB) -> None:
    """Sets the project's whole class list, in order: classes with an id are kept (renamed,
    recoloured), without one added, those left out removed (their annotations hidden, back if a
    class by that name returns). To change a few, PATCH …/classes instead."""
    try:
        with _writing(conn, project_id):
            projects.set_classes(conn, project_id, classes)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


class ClassEdits(BaseModel):
    """Classes to add, change and remove; the others stay (projects.edit_classes)."""

    add: list[projects.NewClass] = Field([], description="New classes, added after the others.")
    change: list[projects.ClassEdit] = Field([], description="Classes to rename, or to give another prompt or colour, "
                                             "e.g. [{\"name\": \"car\", \"new_name\": \"vehicle\"}].")
    remove: list[projects.ClassRef] = Field([], description="Classes to remove (names or ids): their annotations are "
                                            "hidden, not deleted, and come back if a class by that name is added again.")


@router.patch("/api/projects/{project_id}/classes", status_code=204)
def api_edit_classes(project_id: int, edits: ClassEdits, conn: DB) -> None:
    """Adds, changes and removes some of the project's classes; the others stay. 422 with why
    (an unknown class, a name taken)."""
    try:
        with _writing(conn, project_id):
            projects.edit_classes(conn, project_id, edits.add, edits.change, edits.remove)
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
    """Starts auto-annotating the images matching `where` with the project's annotator (or the one
    given): each class is found from its prompt and the objects are added next to the ones there
    (never replacing them; one already there is left out). An image is done once per class, so
    running again does only what's new (a class or images added since; `again` does them again).
    Returns {run_id, job_id, images, downloads_first: the models it downloads before it starts};
    follow the job at /api/jobs/{job_id}. 422 with why it can't start (no classes, no annotator)."""
    started = _started(lambda: annotators.start(conn, project_id, Filter.parse(request.where), request.count,
                                                request.annotator, request.class_ids, request.min_score, request.again,
                                                request.media_ids))
    return started | {"downloads_first": vision.downloads_first(conn, started["job_id"])}


@router.post("/api/projects/{project_id}/try", status_code=202)
def api_try_annotator(project_id: int, request: TryAnnotator, conn: DB) -> dict:
    """Starts a Try: the annotator on a few images, nothing written to the project. Follow the job
    at /api/jobs/{job_id}, then read the run at /api/runs/{run_id}."""
    return _started(lambda: annotators.try_(conn, project_id, Filter.parse(request.where), request.annotator,
                                            request.class_ids, request.prompts))


@router.post("/api/projects/{project_id}/import", status_code=202)
def api_import_images(project_id: int, request: ImageImport, conn: DB) -> dict:
    """Starts importing images into the project from files and folders on this computer; returns
    {job_id}: follow it at /api/jobs/{job_id} (its result says what was imported and skipped)."""
    return {"job_id": _start_import(conn, project_id, request)}
