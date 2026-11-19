"""Annotators: mozo workflows that find a project's objects from its class prompts (product.md
"Annotation"). A project may name the one it uses; else its type's suggestion is used.

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

from app import db, jobs, media, projects, runs, vision
from app.filters import Filter
from app.projects import Task

KIND = "auto_annotate"  # job kind

# The built-in workflows that find objects from class prompts: id -> name.
Finder = Literal["sam3", "grounding-dino", "grounding-dino-edgetam"]
FINDERS: dict[Finder, str] = {id: runs.load(id).name for id in get_args(Finder)}

# The annotator suggested for each type of project. None: no model knows the user's own keypoints,
# so keypoint projects start by hand.
SUGGESTED: dict[Task, Finder | None] = {"detection": runs.FINDER, "segmentation": runs.FINDER, "keypoints": None}


def to_annotations(detections: pf.Detections, classes: list[str]) -> list[projects.Annotation]:
    """An annotator's detections as annotations, box and polygon (from the mask, as pixelflow
    outlines it) with the confidence as score. `classes`: the class name of each prompt, in the
    order given, since a detection's class_id is its prompt's index. Fills the detections'
    segments from their masks."""
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
        found.append(projects.Annotation(class_=classes[int(d.class_id)], bbox=bbox, segments=d.segments,
                                         score=d.confidence))
    return found


def pending(conn: sqlite3.Connection, project_id: int, media_ids: list[int], classes: list[str],
            exclude_run: int | None = None) -> dict[int, list[str]]:
    """Image -> those of `classes` (names) it hasn't been auto-annotated for (by the project's
    runs, apart from `exclude_run`), in their order; images with none left are left out."""
    done: dict[int, set[str]] = {}
    for media_id, name in conn.execute(
        """SELECT DISTINCT r.media_id, c.value FROM run_result r JOIN run ON run.id = r.run_id, json_each(run.classes) c
           WHERE run.project_id = ? AND run.annotates AND run.id IS NOT ? AND r.media_id IN (SELECT value FROM json_each(?))""",
        (project_id, exclude_run, json.dumps(media_ids)),
    ):
        done.setdefault(media_id, set()).add(name.casefold())
    todo = {m: [c for c in classes if c.casefold() not in done.get(m, ())] for m in media_ids}
    return {m: names for m, names in todo.items() if names}


def annotator_of(project: dict) -> Finder | None:
    """The annotator a project uses: its own, else the one suggested for its type (none for
    keypoints), else the default finder."""
    if project["annotator"]:
        return project["annotator"]
    return SUGGESTED[project["task"]] if project["task"] else runs.FINDER


def overview(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter()) -> dict | None:
    """Before auto-annotating the images matching `filter`: how many match, per class how many of
    them haven't been auto-annotated for it, and the models of the project's annotator still to
    download. None if there's no such project."""
    project = projects.get(conn, project_id)
    if project is None:
        return None
    media_ids = projects.sample(conn, filter, project_id)
    todo: dict[str, int] = {c["name"]: 0 for c in project["classes"]}
    for names in pending(conn, project_id, media_ids, list(todo)).values():
        for c in names:
            todo[c] += 1
    annotator = annotator_of(project)
    return {"annotator": annotator, "images": len(media_ids), "pending": todo,
            "to_download": runs.to_download(annotator) if annotator else []}


def _prompts(conn: sqlite3.Connection, project_id: int, annotator: Finder | None, classes: list[str] | None,
             changed: dict[str, str] | None = None) -> tuple[dict, Finder, dict[str, str]] | None:
    """The project, the annotator to use (the project's unless given), and the prompts of the
    classes asked for (all of the project's unless given; a name it hasn't yet is a new class,
    found by its name) by class name, with `changed` ones instead of the saved ones
    (projects.with_prompts). None if there's no such project. ValueError: no annotator, no classes."""
    project = projects.get(conn, project_id)
    if project is None:
        return None
    annotator = annotator or annotator_of(project)
    if annotator is None:
        raise ValueError("No model knows these keypoints, so the project has no annotator; choose one")
    listed = projects.with_prompts(projects.including(project["classes"], [*(classes or ()), *(changed or ())]),
                                   changed or {})
    wanted = None if classes is None else {n.casefold() for n in classes}
    prompts = {c["name"]: c["prompt"] for c in listed if wanted is None or c["name"].casefold() in wanted}
    if not prompts:
        raise ValueError("Choose at least one class to find" if classes is not None
                         else "The project has no classes yet: say which objects to find (classes)")
    return project, annotator, prompts


def _launch(conn: sqlite3.Connection, annotator: Finder, label: str, media_ids: list[int],
            prompts: dict[str, str], kind: str, params: dict | None = None, annotates: bool = False) -> dict:
    """Create the annotator's run over the images with these prompts (class name -> prompt, in
    order) and start its job: {run_id, job_id, images, downloads_first (the models it downloads
    before it runs, as vision.downloads_first)}."""
    name = runs.run_name(f"{label} · {FINDERS[annotator]}", len(media_ids))
    run_id = runs.create(conn, *runs.resolve(annotator), media_ids, name, list(prompts), annotates)
    params = {"media_ids": media_ids, "prompts": list(prompts.values())} | (params or {})
    job_id = runs.launch(conn, run_id, kind, params)
    return {"run_id": run_id, "job_id": job_id, "images": len(media_ids),
            "downloads_first": vision.downloads_first(conn, job_id)}


def start(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter(), count: int | None = None,
          annotator: Finder | None = None, classes: list[str] | None = None, min_score: float = 0.0,
          again: bool = False, media_ids: list[int] | None = None) -> dict | None:
    """Auto-annotate the project's images matching `filter` (`count` of them at random; of them,
    those in `media_ids` if given: say the one open in the editor), for its classes (`classes`:
    these, by name, new ones too), with its annotator unless another is given; returns {run_id,
    job_id, images}. Each image runs only for the classes it hasn't been auto-annotated for, unless
    `again`. Objects scoring below `min_score` are left out, beyond what the annotator's own
    threshold leaves out. None if there's no such project.

    Raises ValueError: as _prompts(), or nothing to do."""
    found = _prompts(conn, project_id, annotator, classes)
    if found is None:
        return None
    _, annotator, prompts = found
    wanted = list(prompts)
    matching = projects.sample(conn, filter, project_id)
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
    return _launch(conn, annotator, "Auto-annotate", chosen,
                   {c: p for c, p in prompts.items() if c in needed}, KIND,
                   {"min_score": min_score, "again": again}, annotates=True)


TRY_IMAGES = 4


def try_(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter(), annotator: Finder | None = None,
         classes: list[str] | None = None, prompts: dict[str, str] | None = None) -> dict | None:
    """Try the annotator on TRY_IMAGES of the project's images matching `filter`, picked at
    random, writing nothing to the project; returns {run_id, job_id, images}. `prompts`: prompts
    to try instead of some classes' saved ones (by class name). None if there's no such project.
    ValueError: as _prompts(), or no images."""
    found = _prompts(conn, project_id, annotator, classes, prompts)
    if found is None:
        return None
    _, annotator, tried = found
    media_ids = projects.sample(conn, filter, project_id, TRY_IMAGES)
    if not media_ids:
        raise ValueError("No images match the filter")
    return _launch(conn, annotator, "Try", media_ids, tried, runs.KIND)


def annotated(conn: sqlite3.Connection, run: dict) -> dict:
    """A run with its results (runs.get, or runs.results_page) as annotations: per image and
    result label, the image (media id, file name, width, height, thumbnail), its objects per class
    and the objects themselves with their scores; and `legend`, the classes to draw them with (name,
    colour). A run made from a project's classes (auto-annotate, Try) uses them; any other run's
    classes are its detections' names."""
    detections = [pf.Detections.from_dict(r["result"]["detections"]) for r in run["results"]]
    if run["classes"] is not None:
        names = run["classes"]
        drawn = {c["name"].casefold(): c for c in projects.classes(conn, run["project_id"])} if run["project_id"] else {}
    else:
        by_id = {int(d.class_id): d.class_name for found in detections for d in found if d.class_id is not None}
        names = [by_id.get(i) or f"class {i}" for i in range(max(by_id, default=-1) + 1)]
        drawn = {}
    legend = [{"name": n, "color": drawn[n.casefold()]["color"] if n.casefold() in drawn
               else projects.PALETTE[i % len(projects.PALETTE)]} for i, n in enumerate(names)]
    found = [{"media_id": r["media_id"], "filename": r["filename"], "label": r["label"], "objects": r["objects"],
              "width": r["result"]["width"], "height": r["result"]["height"], "thumb_url": media.thumb_url(r["sha256"]),
              "annotations": [a.stored() for a in to_annotations(d, names)]}
             for r, d in zip(run["results"], detections, strict=True)]
    return run | {"results": found, "legend": legend}


def latest_try(conn: sqlite3.Connection, project_id: int) -> dict | None:
    """The project's latest Try, with its results as annotations (annotated); None if it has none."""
    row = conn.execute("SELECT id FROM run WHERE project_id = ? AND classes IS NOT NULL AND NOT annotates "
                       "ORDER BY id DESC LIMIT 1", (project_id,)).fetchone()
    return row and annotated(conn, runs.get(conn, row[0]))


@jobs.kind(KIND, lane="gpu", title="Auto-annotate", needs=runs.needs)
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
        want = {c.casefold() for c in needed.get(image["id"], ())}
        new = [a for detections in found.values() for a in to_annotations(detections, classes)
               if a.class_.casefold() in want and (a.score is None or a.score >= min_score)]
        added += projects.add_annotations(conn, image["id"], new) or 0

    return runs.execute(ctx, run_id, media_ids, prompts=prompts, on_image=add) | {"objects": added}
