"""Projects: HTML pages (forms post, then redirect), and the same data and actions as JSON under /api."""

import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from app import annotators, db, imports, jobs, projects, runs, segment
from app.filters import WHERE, Filter, FilterError, ObjectCondition, Range
from app.formatting import plural
from app.routes import IMAGES, READS, AnnotationOut, Answer, ClassOut, Done, Images, Started, ToDownload, started
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


def click_loading(request: Request, error: segment.Loading) -> JSONResponse:
    """For the API (registered in app/server.py): the click model isn't in memory yet, a 409
    saying so, with its load job (`loading`: id, status, ahead) for the editor to wait on."""
    return JSONResponse({"detail": str(error), "loading": error.job | {"model": error.name}}, status_code=409)


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
    """A new project, with its classes and images; it gets the annotator suggested for its type."""

    model_config = ConfigDict(extra="forbid")
    classes: list[projects.ClassChange] = Field([], description="The kinds of objects to annotate, in order (no ids: all new).")
    images: Images | None = Field(None, description="Images to start with. " + IMAGES)


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

    model_config = ConfigDict(extra="forbid")
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

    model_config = ConfigDict(extra="forbid")
    where: str = Field("", max_length=2000, description=WHERE)
    annotator: annotators.Finder | None = Field(None, description=ANNOTATOR)
    class_ids: list[int] | None = Field(None, description=CLASS_IDS)
    prompts: dict[int, Prompt] | None = Field(None, description="Other prompts to try, by class id, e.g. {\"3\": \"red car\"}; not saved.")


class ImageImport(BaseModel):
    """Files and folders on this computer or a NAS. Blank entries (an empty path field) are ignored."""

    model_config = ConfigDict(extra="forbid")
    paths: list[str] = Field(description="Image files and folders (searched with their subfolders), e.g. [\"~/Desktop/potholes\"].")

    @field_validator("paths")
    @classmethod
    def _not_blank(cls, paths: list[str]) -> list[str]:
        paths = [p.strip() for p in paths if p.strip()]
        if not paths:
            raise ValueError("give at least one file or folder")
        return paths


# --- What the API answers ---


class ClassCounted(ClassOut):
    annotation_count: int = Field(description="How many objects of this class the project's images have.")


class ProjectOut(Answer):
    id: int = Field(description="The project's id.")
    name: str
    description: str
    task: projects.Task = Field(description=TASK)
    annotator: str | None = Field(description=FINDER)
    root_id: int | None = Field(description="The project this is a version of; null for an original.")
    frozen_at: str | None = Field(description="When it was frozen (a frozen project never changes); null if it isn't.")
    created_at: str = Field(description="When it was made, UTC (ISO 8601).")
    updated_at: str = Field(description="When it last changed, UTC (ISO 8601).")


class ProjectWithClasses(ProjectOut):
    classes: list[ClassOut] = Field(description="Its classes, in order.")


class Counted(ProjectOut):
    image_count: int = Field(description="How many images it has.")
    labelled_count: int = Field(description="How many of them have at least one object.")


class ProjectListed(Answer):
    """A project in the list: what tells it apart and its counts (GET /api/projects/{id} has the rest)."""

    id: int = Field(description="The project's id.")
    name: str
    task: projects.Task = Field(description=TASK)
    image_count: int = Field(description="How many images it has.")
    labelled_count: int = Field(description="How many of them have at least one object.")
    class_count: int = Field(description="How many classes it has.")
    root_id: int | None = Field(description="The project this is a version of; null for an original.")
    frozen_at: str | None = Field(description="When it was frozen (a frozen project never changes); null if it isn't.")


class ImageListed(Answer):
    media_id: int = Field(description="The image's id (the same in every project and run that has it).")
    filename: str = Field(description="Its file name in this project.")
    objects: dict[str, int] = Field(description="How many objects of each class it has (by class name).")
    width: int | None = Field(None, description="In pixels (with ?annotations=1).")
    height: int | None = Field(None, description="In pixels (with ?annotations=1).")
    annotations: list[AnnotationOut] | None = Field(None, description="Its objects with their shapes, of the project's "
                                                    "classes (with ?annotations=1).")
    matches: list[int] | None = Field(None, description="With annotations, when the filter asks for objects (in "
                                      "brackets): the positions in `annotations` of the objects it found.")


class ImageOut(ImageListed):
    split: str | None = Field(description="train, val or test, if set; null if not.")
    width: int = Field(description="In pixels.")
    height: int = Field(description="In pixels.")
    url: str = Field(description="The full image.")
    thumb_url: str = Field(description="A small version of it.")
    annotations: list[AnnotationOut] = Field(description="Its objects with their shapes (of the project's classes).")


class ProjectDetail(Counted):
    classes: list[ClassCounted] = Field(description="Its classes, in order, each with its object count.")
    filter: str = Field(description="The filter as applied (?where=, its text form).")
    match_count: int = Field(description="How many images match the filter (all of them without one).")
    page: int = Field(description="This page's number, from 1.")
    pages: int = Field(description="How many pages the matching images make.")
    images: list[ImageListed] = Field(description="This page of the matching images, by file name.")


class ImageDetail(ImageOut):
    project: ProjectWithClasses = Field(description="The project it's in.")
    filter: str = Field(description="The filter as applied (?where=), which previous and next follow.")
    position: int = Field(description="Its place among the matching images, from 1.")
    match_count: int = Field(description="How many images match the filter.")
    previous_id: int | None = Field(description="The media_id of the matching image before it; null for the first.")
    next_id: int | None = Field(description="The media_id of the matching image after it; null for the last.")


class ClassCount(Answer):
    class_: str = Field(alias="class", description="The class's name.")
    objects: int = Field(description="How many objects of it.")


class ObjectsPerImage(Answer):
    objects: int = Field(description="A number of objects.")
    images: int = Field(description="How many images have that many.")


class Bin(Answer):
    from_: float = Field(alias="from", description="Where the bin starts (included).")
    to: float = Field(description="Where it ends.")
    objects: int = Field(description="How many objects measure in it.")


class Stats(Answer):
    field: str = Field(description="What was measured (?field=).")
    counts: list[ClassCount] | list[ObjectsPerImage] | None = Field(
        None, description="class: objects per class, most first; objects: images per number of objects.")
    objects: int | None = Field(None, description="w, h, area, aspect: how many objects were measured.")
    bins: list[Bin] | None = Field(None, description="w, h, area, aspect: 20 equal bins from the smallest to the largest.")


class Deletion(Answer):
    images: int = Field(description="Images the project has (the files stay while another project or run uses them).")
    annotations: int = Field(description="Their objects.")
    runs: int = Field(description="The project's runs.")
    jobs: int = Field(description="Its jobs queued or running; deleting is refused while there are any.")


class AnnotateOverview(Answer):
    annotator: str | None = Field(description="The project's annotator; null: none.")
    images: int = Field(description="How many images match the filter.")
    pending: dict[int, int] = Field(description="Per class id: how many of those images it hasn't been auto-annotated on.")
    to_download: list[ToDownload] = Field(description="The annotator's models still to download (a run downloads them first).")


class Created(Done):
    id: int = Field(description="The new project's id.")


class ProjectCreated(Created):
    classes: list[ClassOut] = Field(description="Its classes, with their ids.")


class ClassesOut(Done):
    classes: list[ClassOut] = Field(description="The project's classes now, in order, with their ids.")



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
def _writing(conn: sqlite3.Connection, project_id: int, dry_run: bool = False):
    """A change to a project, committed at the end (`dry_run`: done, then rolled back, so what it
    says it changed is exactly what it would): 404 if there's no such project, 409 if it's frozen."""
    _exists(conn, project_id)
    try:
        yield
    except projects.FrozenError:
        conn.rollback()
        raise HTTPException(409, "The project is frozen; duplicate it to change it") from None
    except BaseException:
        conn.rollback()
        raise
    if dry_run:
        conn.rollback()
    else:
        conn.commit()


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
    """Create the project with its classes. Raises ValueError (see projects.set_classes). Caller commits."""
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


def _set_annotations(conn: sqlite3.Connection, project_id: int, media_id: int, annotations: list[projects.Annotation]) -> str:
    """Set an image's annotations; its file name. 404 if the project hasn't the image, 422 for a class it hasn't."""
    try:
        with _writing(conn, project_id):
            filename = projects.set_annotations(conn, project_id, media_id, annotations)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if filename is None:
        raise HTTPException(404, "Image not found")
    return filename


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
        with conn:
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


class FoundProject(Answer):
    id: int
    name: str
    page: str = Field(description="Its page in the app.")


class FoundImage(Answer):
    media_id: int
    filename: str
    project_id: int
    project: str = Field(description="Its project's name.")
    page: str = Field(description="Its page in the app.")


class FoundWorkflow(Answer):
    id: str
    name: str
    page: str = Field(description="Its page in the app (the builder).")


class Found(Answer):
    projects: list[FoundProject]
    images: list[FoundImage] = Field(description="The first 20, by file name.")
    image_count: int = Field(description="How many images match in all.")
    workflows: list[FoundWorkflow]


class ProjectChange(BaseModel):
    """A change to a project's images matching a filter, all at once."""

    model_config = ConfigDict(extra="forbid")
    action: projects.Change = Field(description="delete_objects or set_class (the objects the filter names in "
                                    "brackets, e.g. (car w:<10)); remove_images; copy_images or move_images (to "
                                    "to_project_id, with their objects; classes it lacks are added).")
    where: str = Field("", max_length=2000, description=WHERE)
    class_id: int | None = Field(None, description="set_class: the class the objects become.")
    to_project_id: int | None = Field(None, description="copy_images, move_images: the project they go to.")
    dry_run: bool = Field(True, description="Only say what would change (how many, a few of the images), changing "
                          "nothing; false to change it.")


class ChangeOut(Done):
    action: str
    dry_run: bool = Field(description="Nothing was changed: this is what would be.")
    images: int = Field(description="How many images it changes (only those that change).")
    objects: int | None = Field(None, description="How many objects it deletes, gives a class, or copies.")
    examples: list[int] = Field(description="The first few of those images (media ids).")
    classes_added: list[str] | None = Field(None, description="Copying: the classes the other project gets.")
    already_in_target: int | None = Field(None, description="Copying: images the other project has already (left as they are).")


@router.post("/api/projects/{project_id}/changes", response_model=ChangeOut, response_model_exclude_unset=True)
def api_change(project_id: int, change: ProjectChange, conn: DB) -> dict:
    """Changes the project's images matching a filter all at once: deletes the objects it names or
    gives them a class, removes the images, or copies or moves them to another project. Unless
    dry_run is false, nothing changes: the answer says what would (how many images and objects, the
    first images). History undoes a change whole."""
    with _writing(conn, project_id, dry_run=change.dry_run):
        try:
            changed = projects.change(conn, project_id, Filter.parse(change.where), change.action, change.class_id,
                                      change.to_project_id)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return changed | {"dry_run": change.dry_run, "did": _change_done(conn, change, changed)}


def _change_done(conn: sqlite3.Connection, change: ProjectChange, changed: dict) -> str:
    """What a change did, or would (a preview): "Deleted 44 objects on 5 images"."""
    images, objects = plural(changed["images"], "image"), plural(changed.get("objects") or 0, "object")
    target = change.to_project_id and projects.get(conn, change.to_project_id)
    to = f" to {target['name']}" if target else ""
    would, did, what = {"delete_objects": ("delete", "Deleted", f"{objects} on {images}"),
                        "set_class": ("change the class of", "Changed the class of", f"{objects} on {images}"),
                        "remove_images": ("remove", "Removed", images),
                        "copy_images": ("copy", "Copied", images + to),
                        "move_images": ("move", "Moved", images + to)}[change.action]
    return f"Would {would} {what} (a preview: dry_run false changes it)" if change.dry_run else f"{did} {what}"


@router.get("/api/search", response_model=Found)
def api_search(conn: DB, q: Annotated[str, Query(min_length=1, description="Part of a name, e.g. street or img_0042.")]) -> dict:
    """Projects, images and workflows whose name contains `q` (ignoring case), each with its page."""
    text = q.strip()
    found = projects.named(conn, text)
    return {
        "projects": [p | {"page": f"/projects/{p['id']}"} for p in found["projects"]],
        "images": [i | {"page": f"/projects/{i['project_id']}/images/{i['media_id']}"} for i in found["images"]],
        "image_count": found["image_count"],
        "workflows": [w | {"page": f"/workflows/{w['id']}/edit"} for w in runs.named(text)],
    }


@router.get("/api/projects", response_model=list[ProjectListed], response_model_exclude_unset=True)
def api_project_list(conn: DB) -> list[dict]:
    """Every project, most recently changed first: id, name, type, how many images, how many of them
    labelled and how many classes, whether it's a version or frozen."""
    return [{k: p[k] for k in ProjectListed.model_fields} for p in projects.list_all(conn)]


@router.post("/api/projects", status_code=201, response_model=ProjectCreated)
def api_create_project(new: NewProject, conn: DB) -> dict:
    """Creates a project with its classes and images; it gets the annotator suggested for its type.
    Returns its id and classes. 422 with why (a class given twice, a colour that isn't one, images
    that aren't there)."""
    try:
        images = new.images.resolve(conn) if new.images else []
        with conn:
            project_id = _create(conn, new, new.classes)
            projects.add_images(conn, project_id, images)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    classes = projects.get(conn, project_id)["classes"]
    named = f": {', '.join(c['name'] for c in classes)}" if classes else ""
    return {"id": project_id, "classes": classes,
            "did": f"Created project {new.name} ({new.task}) with {plural(len(images), 'image')} and "
                    f"{plural(len(classes), 'class')}{named}"}


# What an image in the project's answer has: enough to know it and what's in it; with ?annotations=1
# also its objects themselves (and its size, which their coordinates are in).
LISTED = ("media_id", "filename", "objects")
WITH_SHAPES = ("width", "height", "annotations", "matches")


@router.get("/api/projects/{project_id}", response_model=ProjectDetail, response_model_exclude_unset=True)
def api_project(project_id: int, conn: DB, where: Where = "",
                name: Annotated[str | None, Query(description="Only images whose file name contains this.")] = None,
                page: Page = 1,
                images: Annotated[int, Query(ge=0, le=projects.PAGE_SIZE, description="Images a page; 0 for the project alone.")] = projects.PAGE_SIZE,
                annotations: Annotated[bool, Query(description="Each image's objects with their shapes (boxes, outlines, "
                                                   "keypoints), not only how many of each class.")] = False) -> dict:
    """The project (name, type, annotator, classes with their prompts, colours and object counts,
    image and labelled counts) with one page of the images matching ?where=, each with how many
    objects of each class it has (?annotations=1: the objects themselves). ?images=0 for the project
    alone. 422 if the filter names a class it doesn't have."""
    project = _detail(conn, project_id, page, _parse(where, name), images)
    kept = LISTED + WITH_SHAPES if annotations else LISTED
    project["images"] = [{k: image[k] for k in kept if k in image} for image in project["images"]]
    return project


@router.get("/api/projects/{project_id}/stats", response_model=Stats, response_model_exclude_unset=True)
def api_stats(project_id: int, conn: DB, where: Where = "",
              field: str = Query(description="class: objects per class; w, h, area, aspect: a histogram of the "
                                 "objects (those the filter asks for in brackets, or all); objects: how many images "
                                 "have how many objects.")) -> dict:
    """Measures the images matching ?where=: how ?field= is spread over them."""
    result = projects.stats(conn, project_id, field, Filter.parse(where))
    if result is None:
        raise HTTPException(404, "Project not found")
    return result


@router.patch("/api/projects/{project_id}", response_model=Done)
def api_update_project(project_id: int, changes: ProjectPatch, conn: DB) -> dict:
    """Changes a project's name, description, type or annotator: only the fields given (classes:
    PATCH /api/projects/{id}/classes). 409 if it's frozen."""
    project = projects.get(conn, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    given = changes.model_dump(exclude_unset=True)
    _update(conn, project_id, ProjectChanges(**{k: project[k] for k in ProjectChanges.model_fields} | given))
    changed = [f"{k} {v or 'none'}" for k, v in given.items() if v != project[k]]
    return {"did": f"Changed {project['name']}: {', '.join(changed)}" if changed else f"{project['name']} already had those"}


class Duplicate(BaseModel):
    """The copy's name."""

    model_config = ConfigDict(extra="forbid")
    name: projects.Name | None = Field(None, description='Its name; empty: "<the project\'s name> copy".')


@router.post("/api/projects/{project_id}/duplicate", status_code=201, response_model=Created)
def api_duplicate_project(project_id: int, conn: DB, body: Duplicate | None = None) -> dict:
    """A new project with this one's classes, images and annotations (not its runs), named `name` or "<its name> copy"."""
    new_id = _duplicate(conn, project_id, body.name if body else None)
    copy = projects.get(conn, new_id)
    return {"id": new_id, "did": f"Made {copy['name']}, with {plural(projects.image_count(conn, new_id), 'image')}"}


@router.get("/api/projects/{project_id}/deletion", response_model=Deletion)
def api_deletion(project_id: int, conn: DB) -> dict:
    """What deleting the project would remove (images, annotations, runs), and how many of its jobs
    are queued or running (deleting is refused while they are)."""
    _exists(conn, project_id)
    return projects.deletion(conn, project_id)


@router.delete("/api/projects/{project_id}", response_model=Done)
def api_delete_project(project_id: int, conn: DB) -> dict:
    """Deletes a project with its images (from it), their annotations and its runs; History can
    undo it for 30 days. 409 while one of its jobs is queued or running."""
    project = projects.get(conn, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    gone = projects.deletion(conn, project_id)
    _delete(conn, project_id)
    return {"did": f"Deleted project {project['name']} with {plural(gone['images'], 'image')}, "
                    f"{plural(gone['annotations'], 'object')} and {plural(gone['runs'], 'run')}"}


@router.get("/api/projects/{project_id}/images/{media_id}", response_model=ImageDetail, response_model_exclude_unset=True)
def api_image(project_id: int, media_id: int, conn: DB, where: Where = "") -> dict:
    """One image of the project: file name, size, its annotations (class, box, outline, points,
    score), and its previous and next among the images matching ?where=."""
    return _image(conn, project_id, media_id, Filter.parse(where))


class ImagesAdd(BaseModel):
    """Stored images to add to this project (attached to a message, another project's, a run's)."""

    model_config = ConfigDict(extra="forbid")
    images: Images = Field(description=IMAGES)


class ImagesAdded(Done):
    added: int = Field(description="How many it added.")
    already_there: int = Field(description="How many the project had already (left as they are).")
    image_count: int = Field(description="How many images the project has now.")


@router.post("/api/projects/{project_id}/images", status_code=201, response_model=ImagesAdded)
def api_add_images(project_id: int, body: ImagesAdd, conn: DB) -> dict:
    """Adds stored images to the project, without objects (their files aren't copied: projects
    share them); one it has already stays as it is. 422 with why (images that aren't there); 409
    if the project is frozen."""
    try:
        images = body.images.resolve(conn)
        with _writing(conn, project_id):
            added = projects.add_images(conn, project_id, images)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    there, name = len(images) - added, projects.get(conn, project_id)["name"]
    return {"added": added, "already_there": there, "image_count": projects.image_count(conn, project_id),
            "did": f"Added {plural(added, 'image')} to {name}" + (f" ({there} there already)" if there else "")}


@router.put("/api/projects/{project_id}/images/{media_id}/annotations", response_model=Done)
def api_set_annotations(project_id: int, media_id: int, annotations: list[projects.Annotation], conn: DB) -> dict:
    """Replaces the image's annotations with this list (read them with GET …/images/{media_id},
    change the list, send it back: to remove one, leave it out). 422 for an unknown class."""
    filename = _set_annotations(conn, project_id, media_id, annotations)
    return {"did": f"{filename} has {plural(len(annotations), 'object')} now"}


@contextmanager
def _clicking(conn: sqlite3.Connection, project_id: int, media_id: int):
    """The image click to segment works on (projects.image_file); 404 if there's none, 409 if no
    click model is downloaded, or while it loads (segment.Loading: click_loading answers it)."""
    image = projects.image_file(conn, project_id, media_id)
    if image is None:
        raise HTTPException(404, "Image not found")
    try:
        yield image
    except LookupError as e:
        raise HTTPException(409, f"{e}: choose one (PUT /api/click-tool) and it downloads") from None


class ClickModel(Answer):
    id: str = Field(description="What PUT /api/click-tool's model takes.")
    name: str
    license: str = Field(description="Its licence.")
    downloaded: bool
    downloading: bool
    to_download: int = Field(description="Bytes it would download (0: downloaded).")


class ClickTool(Answer):
    on: bool = Field(description="The editor's click to segment tool is on.")
    model: str = Field(description="The model chosen.")
    in_use: str | None = Field(description="The model it uses: the chosen one, or while that downloads one that's downloaded.")
    loaded: bool = Field(description="The one in use is in memory.")
    models: list[ClickModel] = Field(description="The models it can use.")


class ClickToolChanged(ClickTool, Done):
    pass


@router.get("/api/click-tool", response_model=ClickTool)
def api_click_tool(conn: DB) -> dict:
    """The image editor's click to segment tool: on or off, the model chosen, the one in use (the
    chosen one, or while it downloads another that's downloaded), and each model with its licence
    and whether it's downloaded, downloading, or how many bytes it would download."""
    return segment.tool(conn)


@router.put("/api/click-tool", response_model=ClickToolChanged)
def api_change_click_tool(change: segment.Change, conn: DB) -> dict:
    """Turns the click tool on or off and/or chooses its model. A model that isn't downloaded starts
    downloading (a job); until it's done, the first downloaded one answers. Returns the tool as GET
    does, and what changed (`did`)."""
    tool = segment.change(conn, change)
    return tool | {"did": f"The click tool is {'on' if tool['on'] else 'off'}, with {tool['model']}"
                           + ("" if tool["in_use"] == tool["model"] else f" (downloading; meanwhile {tool['in_use'] or 'none is ready'})")}


@router.post("/api/projects/{project_id}/images/{media_id}/prepare", response_model=Done)
def api_prepare_image(project_id: int, media_id: int, conn: DB) -> dict:
    """Gets the image ready for click to segment: encodes it now (the slow half, a fraction of a
    second), so each segment call on it afterwards takes milliseconds. Optional: the first segment
    call prepares it too. 409 while the click model loads, with its job (`loading`: id, status, and
    `ahead`, the job it waits behind): it's loaded by that job, never by this call."""
    with _clicking(conn, project_id, media_id) as image:
        segment.prepare(conn, image["path"])
    return {"did": "Ready for click to segment"}


@router.post("/api/projects/{project_id}/images/{media_id}/segment", response_model=projects.Annotation | None, tags=[READS])
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
            return segment.segment(conn, image["path"], prompt, image["task"])
        except segment.Superseded:
            return Response(status_code=204)


@router.patch("/api/projects/{project_id}/classes/prompts", response_model=ClassesOut)
def api_set_prompts(project_id: int, prompts: Annotated[dict[int, Prompt], Body(
        description="Prompts by class id, e.g. {\"3\": \"hole in the road surface\"}; empty: the class's name.")],
                    conn: DB) -> dict:
    """Changes some classes' prompts: what annotators look for. Returns the classes."""
    try:
        with _writing(conn, project_id):
            projects.set_prompts(conn, project_id, prompts)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    classes = projects.get(conn, project_id)["classes"]
    return {"classes": classes, "did": "Changed the prompts of " + ", ".join(c["name"] for c in classes if c["id"] in prompts)}


@router.get("/api/projects/{project_id}/classes", response_model=list[ClassOut])
def api_classes(project_id: int, conn: DB) -> list[dict]:
    """The project's classes, in order: as PUT …/classes takes them back."""
    if (project := projects.get(conn, project_id)) is None:
        raise HTTPException(404, "Project not found")
    return project["classes"]


@router.put("/api/projects/{project_id}/classes", response_model=ClassesOut)
def api_set_classes(project_id: int, classes: list[projects.ClassChange], conn: DB) -> dict:
    """Sets the project's whole class list, in order: classes with an id are kept (renamed,
    recoloured), without one added, those left out removed (their annotations hidden, back if a
    class by that name returns); returns the classes, with the new ones' ids. To change a few,
    PATCH …/classes instead."""
    try:
        with _writing(conn, project_id):
            now = projects.set_classes(conn, project_id, classes)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"classes": now, "did": f"Set {plural(len(now), 'class')}" + (f": {', '.join(c['name'] for c in now)}" if now else "")}


class ClassEdits(BaseModel):
    """Classes to add, change and remove; the others stay (projects.edit_classes)."""

    model_config = ConfigDict(extra="forbid")
    add: list[projects.NewClass] = Field([], description="New classes, added after the others.")
    change: list[projects.ClassEdit] = Field([], description="Classes to rename, or to give another prompt or colour, "
                                             "e.g. [{\"name\": \"car\", \"new_name\": \"vehicle\"}].")
    remove: list[projects.ClassRef] = Field([], description="Classes to remove (names or ids): their annotations are "
                                            "hidden, not deleted, and come back if a class by that name is added again.")


@router.patch("/api/projects/{project_id}/classes", response_model=ClassesOut)
def api_edit_classes(project_id: int, edits: ClassEdits, conn: DB) -> dict:
    """Adds, changes and removes some of the project's classes; the others stay. Returns the
    classes, with the new ones' ids; 422 with why (an unknown class, a name taken)."""
    project = projects.get(conn, project_id)
    before = {c["id"]: c for c in project["classes"]} if project else {}  # none: _writing answers 404
    try:
        with _writing(conn, project_id):
            now = projects.edit_classes(conn, project_id, edits.add, edits.change, edits.remove)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    after = {c["id"]: c for c in now}
    words = [f"added {c['name']} (id {i})" for i, c in after.items() if i not in before]
    words += [f"renamed {before[i]['name']} to {c['name']}" for i, c in after.items() if i in before and c["name"] != before[i]["name"]]
    words += [f"changed {c['name']}" for i, c in after.items() if i in before and c["name"] == before[i]["name"] and c != before[i]]
    words += [f"removed {c['name']}" for i, c in before.items() if i not in after]
    return {"classes": now, "did": ("Classes: " + "; ".join(words)) if words else "The classes were already so"}


@router.get("/api/projects/{project_id}/annotate", response_model=AnnotateOverview)
def api_auto_annotate_overview(project_id: int, conn: DB, where: Where = "") -> dict:
    """Before auto-annotating the images matching ?where=: how many match, per class how many
    still need it, and the annotator's models still to download."""
    found = annotators.overview(conn, project_id, Filter.parse(where))
    if found is None:
        raise HTTPException(404, "Project not found")
    return found


@router.post("/api/projects/{project_id}/annotate", status_code=202, response_model=Started, response_model_exclude_unset=True)
def api_auto_annotate(project_id: int, request: AutoAnnotate, conn: DB) -> dict:
    """Starts auto-annotating the images matching `where` with the project's annotator (or the one
    given): each class is found from its prompt and the objects are added next to the ones there
    (never replacing them; one already there is left out). An image is done once per class, so
    running again does only what's new (a class or images added since; `again` does them again).
    Returns {run_id, job_id, images, downloads_first: the models it downloads before it starts};
    follow the job at /api/jobs/{job_id}. 422 with why it can't start (no classes, no annotator)."""
    begun = _started(lambda: annotators.start(conn, project_id, Filter.parse(request.where), request.count,
                                              request.annotator, request.class_ids, request.min_score, request.again,
                                              request.media_ids))
    classes = [c["name"] for c in projects.get(conn, project_id)["classes"]
               if request.class_ids is None or c["id"] in request.class_ids]
    return begun | started(conn, begun["job_id"], f"Started auto-annotating {plural(begun['images'], 'image')} for {', '.join(classes)}")


@router.post("/api/projects/{project_id}/try", status_code=202, response_model=Started, response_model_exclude_unset=True)
def api_try_annotator(project_id: int, request: TryAnnotator, conn: DB) -> dict:
    """Starts a Try: the annotator on a few images, nothing written to the project. Follow the job
    at /api/jobs/{job_id}, then read the run at /api/runs/{run_id}."""
    begun = _started(lambda: annotators.try_(conn, project_id, Filter.parse(request.where), request.annotator,
                                             request.class_ids, request.prompts))
    return begun | started(conn, begun["job_id"], f"Started a Try on {plural(begun['images'], 'image')}")


@router.post("/api/projects/{project_id}/import", status_code=202, response_model=Started, response_model_exclude_unset=True)
def api_import_images(project_id: int, request: ImageImport, conn: DB) -> dict:
    """Starts importing images into the project from files and folders on this computer; returns
    {job_id}: follow it at /api/jobs/{job_id} (its result says what was imported and skipped)."""
    return {"job_id": _start_import(conn, project_id, request),
            "did": f"Started importing images from {plural(len(request.paths), 'path')} (the job says what it finds)"}
