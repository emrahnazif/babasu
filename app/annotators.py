"""Annotators: mozo workflows that find a project's objects from its class prompts (product.md
"Annotation"). A project names the one it uses.

Auto-annotating is a run of the annotator whose results also become the project's annotations.
The run records the class each prompt stood for, so an image is known to be auto-annotated for
a class once a run for that class has a result for it: a second run skips it, and a class added
later runs on its own. A Try is the same run on a few images, its results only looked at.
"""

import json
import random
import sqlite3
from contextlib import closing
from typing import Literal, get_args

import pixelflow as pf

from app import db, jobs, projects, runs
from app.filters import Filter
from app.projects import Task

KIND = "auto_annotate"  # job kind

# The built-in workflows that find objects from class prompts: id -> name.
Finder = Literal["sam3", "grounding-dino", "grounding-dino-edgetam"]
FINDERS: dict[Finder, str] = {id: runs.load(id).name for id in get_args(Finder)}

# The annotator a new project of each type gets. None: no model knows the user's own keypoints,
# so keypoint projects start by hand.
SUGGESTED: dict[Task, Finder | None] = {"detection": "sam3", "segmentation": "sam3", "keypoints": None}


def to_annotations(detections: pf.Detections, classes: list[int]) -> list[projects.Annotation]:
    """An annotator's detections as a project's annotations, box and polygon (from the mask, as
    pixelflow outlines it) with the confidence as score. `classes`: the project class id of each
    prompt, in the order given, since a detection's class_id is its prompt's index. Fills the
    detections' segments from their masks."""
    found = []
    for d in detections.masks_to_segments():
        if d.class_id is None or not 0 <= int(d.class_id) < len(classes):
            continue
        bbox = None
        if d.bbox:
            x1, y1, x2, y2 = d.bbox
            bbox = (x1, y1, x2 - x1, y2 - y1)
        if bbox is None and not d.segments:
            continue
        found.append(projects.Annotation(class_id=classes[int(d.class_id)], bbox=bbox, segments=d.segments,
                                         score=d.confidence))
    return found


def pending(conn: sqlite3.Connection, project_id: int, media_ids: list[int], class_ids: list[int],
            exclude_run: int | None = None) -> dict[int, list[int]]:
    """Image -> those of `class_ids` it hasn't been auto-annotated for (by the project's runs,
    apart from `exclude_run`), in their order; images with none left are left out."""
    done: dict[int, set[int]] = {}
    for media_id, class_id in conn.execute(
        """SELECT DISTINCT r.media_id, c.value FROM run_result r JOIN run ON run.id = r.run_id, json_each(run.classes) c
           WHERE run.project_id = ? AND run.annotates AND run.id IS NOT ? AND r.media_id IN (SELECT value FROM json_each(?))""",
        (project_id, exclude_run, json.dumps(media_ids)),
    ):
        done.setdefault(media_id, set()).add(class_id)
    todo = {m: [c for c in class_ids if c not in done.get(m, ())] for m in media_ids}
    return {m: classes for m, classes in todo.items() if classes}


def overview(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter()) -> dict | None:
    """Before auto-annotating the images matching `filter`: how many match, per class how many of
    them haven't been auto-annotated for it, and the models of the project's annotator still to
    download. None if there's no such project."""
    project = projects.get(conn, project_id)
    if project is None:
        return None
    media_ids = projects.sample(conn, project_id, filter)
    todo: dict[int, int] = {c["id"]: 0 for c in project["classes"]}
    for classes in pending(conn, project_id, media_ids, list(todo)).values():
        for c in classes:
            todo[c] += 1
    annotator = project["annotator"]
    return {"annotator": annotator, "images": len(media_ids), "pending": todo,
            "to_download": runs.to_download(annotator) if annotator else []}


def _prompts(conn: sqlite3.Connection, project_id: int, annotator: Finder | None, class_ids: list[int] | None,
             changed: dict[int, str] | None = None) -> tuple[dict, Finder, dict[int, str]] | None:
    """The project, the annotator to use (the project's unless given), and the prompts of the
    classes asked for (all of them unless given) by class id, with `changed` ones instead of the
    saved ones (projects.with_prompts). None if there's no such project. ValueError: no annotator,
    an unknown class, no classes."""
    project = projects.get(conn, project_id)
    if project is None:
        return None
    annotator = annotator or project["annotator"]
    if annotator is None:
        raise ValueError("The project has no annotator; choose one")
    asked = set(class_ids or ()) | set(changed or ())
    unknown = asked - {c["id"] for c in project["classes"]}
    if unknown:
        raise ValueError(f"unknown class ids: {sorted(unknown)}")
    classes = projects.with_prompts(project["classes"], changed or {})
    prompts = {c["id"]: c["prompt"] for c in classes if class_ids is None or c["id"] in class_ids}
    if not prompts:
        raise ValueError("Choose at least one class to find" if class_ids is not None else "The project has no classes to find")
    return project, annotator, prompts


def _launch(conn: sqlite3.Connection, project_id: int, annotator: Finder, label: str, media_ids: list[int],
            prompts: dict[int, str], kind: str, params: dict | None = None, annotates: bool = False) -> dict:
    """Create the annotator's run over the images with these prompts (class id -> prompt, in
    order) and start its job: {run_id, job_id, images}."""
    name = runs.run_name(f"{label} · {FINDERS[annotator]}", len(media_ids))
    run_id = runs.create(conn, annotator, media_ids, project_id, name, list(prompts), annotates)
    params = {"media_ids": media_ids, "prompts": list(prompts.values())} | (params or {})
    return {"run_id": run_id, "job_id": runs.launch(conn, run_id, kind, params, project_id), "images": len(media_ids)}


def start(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter(), count: int | None = None,
          annotator: Finder | None = None, class_ids: list[int] | None = None, min_score: float = 0.0,
          again: bool = False, media_ids: list[int] | None = None) -> dict | None:
    """Auto-annotate the project's images matching `filter` (`count` of them at random; of them,
    those in `media_ids` if given: say the one open in the editor), for its classes (`class_ids`:
    some of them), with its annotator unless another is given; returns {run_id, job_id, images}.
    Each image runs only for the classes it hasn't been auto-annotated for, unless `again`.
    Objects scoring below `min_score` are left out, beyond what the annotator's own threshold
    leaves out. None if there's no such project.

    Raises FrozenError; ValueError: as _prompts(), or nothing to do."""
    found = _prompts(conn, project_id, annotator, class_ids)
    if found is None:
        return None
    project, annotator, prompts = found
    if project["frozen_at"]:
        raise projects.FrozenError(f"project {project_id} is frozen")
    wanted = list(prompts)
    matching = projects.sample(conn, project_id, filter)
    if media_ids is not None:
        only = set(media_ids)
        outside = only - set(matching)
        if outside:
            raise ValueError(f"Images not in the project{' or not matching the filter' if filter.text() else ''}: {sorted(outside)}")
        matching = [m for m in matching if m in only]
    todo = {m: wanted for m in matching} if again else pending(conn, project_id, matching, wanted)
    if not todo:
        raise ValueError("Nothing to do: every matching image is auto-annotated for these classes")
    chosen = list(todo) if count is None or count >= len(todo) else random.sample(list(todo), count)
    needed = {c for m in chosen for c in todo[m]}
    return _launch(conn, project_id, annotator, "Auto-annotate", chosen,
                   {c: p for c, p in prompts.items() if c in needed}, KIND,
                   {"min_score": min_score, "again": again}, annotates=True)


TRY_IMAGES = 4


def try_(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter(), annotator: Finder | None = None,
         class_ids: list[int] | None = None, prompts: dict[int, str] | None = None) -> dict | None:
    """Try the annotator on TRY_IMAGES of the project's images matching `filter`, picked at
    random, writing nothing to the project; returns {run_id, job_id, images}. `prompts`: prompts
    to try instead of some classes' saved ones. None if there's no such project. ValueError: as
    _prompts(), a prompt for a class not tried, or no images."""
    if prompts and class_ids is not None and set(prompts) - set(class_ids):
        raise ValueError(f"prompts for classes not tried: {sorted(set(prompts) - set(class_ids))}")
    found = _prompts(conn, project_id, annotator, class_ids, prompts)
    if found is None:
        return None
    _, annotator, tried = found
    media_ids = projects.sample(conn, project_id, filter, TRY_IMAGES)
    if not media_ids:
        raise ValueError("No images match the filter")
    return _launch(conn, project_id, annotator, "Try", media_ids, tried, runs.KIND)


@jobs.kind(KIND, lane="gpu", title="Auto-annotate")
def annotate_job(ctx: jobs.Context, run_id: int, media_ids: list[int], prompts: list[str], min_score: float,
                 again: bool) -> dict:
    """The run, and each image's objects added to the project as it finishes, in the same commit
    as its results: of the classes the image needed, not already there (projects.add_annotations)."""
    with closing(db.connect()) as conn:
        run = conn.execute("SELECT project_id, classes FROM run WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise ValueError("The run was deleted")
        if run["project_id"] is None:
            raise ValueError("The project was deleted")
        project_id, classes = run["project_id"], json.loads(run["classes"])
        needed = ({m: classes for m in media_ids} if again
                  else pending(conn, project_id, media_ids, classes, exclude_run=run_id))
    added = 0

    def add(conn: sqlite3.Connection, image: sqlite3.Row, found: dict[str, pf.Detections]) -> None:
        nonlocal added
        want = needed.get(image["id"], ())
        new = [a for detections in found.values() for a in to_annotations(detections, classes)
               if a.class_id in want and (a.score is None or a.score >= min_score)]
        added += projects.add_annotations(conn, project_id, image["id"], new) or 0

    return runs.execute(ctx, run_id, media_ids, prompts=prompts, on_image=add) | {"objects": added}
