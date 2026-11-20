"""Annotators: mozo workflows that find a project's objects from its class prompts (product.md
"Annotation"). A project may name the one it uses; else its type's suggestion is used.

Auto-annotating is a run of the annotator (runs.start) over the project's images, looking for its
classes: what it finds becomes their annotations, and each image remembers the classes it was
looked at for, so running again does only what's new (a class or images added since). A Try is
the same run on a few images.
"""

import json
import sqlite3
from typing import Literal, get_args

from app import projects, runs
from app.filters import Filter
from app.projects import Task

# The built-in workflows that find objects from class prompts: id -> name.
Finder = Literal["sam3", "sam3-track", "grounding-dino", "grounding-dino-edgetam"]
FINDERS: dict[Finder, str] = {id: runs.load(id).name for id in get_args(Finder)}

# The annotator suggested for each type of project. None: no model knows the user's own keypoints,
# so keypoint projects start by hand.
SUGGESTED: dict[Task, Finder | None] = {"detection": runs.FINDER, "segmentation": runs.FINDER, "keypoints": None}

TRY_IMAGES = 4


def annotator_of(project: dict) -> Finder | None:
    """The annotator a project uses: its own, else the one suggested for its type (none for
    keypoints), else the default finder."""
    if project["annotator"]:
        return project["annotator"]
    return SUGGESTED[project["task"]] if project["task"] else runs.FINDER


def overview(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter()) -> dict | None:
    """Before auto-annotating the images matching `filter`: how many match, per class how many of
    them its annotator hasn't looked for it on, and the annotator's models still to download.
    None if there's no such project."""
    project = projects.get(conn, project_id)
    if project is None:
        return None
    annotator = annotator_of(project)
    media_ids = projects.sample(conn, filter, project_id)
    todo: dict[str, int] = {c["name"]: 0 for c in project["classes"]}
    if annotator is None:
        return {"annotator": None, "images": len(media_ids), "pending": todo, "to_download": []}
    for names in runs.pending(conn, media_ids, annotator, list(todo)).values():
        for c in names:
            todo[c] += 1
    return {"annotator": annotator, "images": len(media_ids), "pending": todo, "to_download": runs.to_download(annotator)}


def _prompts(conn: sqlite3.Connection, project_id: int, annotator: Finder | None, classes: list[str] | None,
             changed: dict[str, str] | None = None) -> tuple[Finder, dict[str, str]] | None:
    """The annotator to use (the project's unless given), and the prompts of the classes asked for
    (all of the project's unless given; a name it hasn't yet is a new class, found by its name) by
    class name, with `changed` ones instead of the saved ones (projects.with_prompts). None if
    there's no such project. ValueError: no annotator, no classes."""
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
    return annotator, prompts


def start(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter(), count: int | None = None,
          annotator: Finder | None = None, classes: list[str] | None = None, min_score: float = 0.0,
          again: bool = False, media_ids: list[int] | None = None, prompts: dict[str, str] | None = None,
          tried: bool = False) -> dict | None:
    """Auto-annotate the project's images matching `filter` (`count` of them at random, of those
    with work left; of them, those in `media_ids` if given: say the one open in the editor), for its
    classes (`classes`: these, by name, new ones too), with its annotator unless another is given;
    returns {job_id, images}. `prompts`: other prompts to look for than some
    classes' saved ones (by class name). `again` and `min_score`: as runs.start. `tried`: a Try (the
    project page shows its images). None if there's no such project. ValueError: as _prompts(),
    images not in the project, or nothing to do."""
    found = _prompts(conn, project_id, annotator, classes, prompts)
    if found is None:
        return None
    annotator, looked_for = found
    matching = projects.sample(conn, filter, project_id)
    if not matching:
        raise ValueError("No images match the filter")
    if media_ids is not None:
        only = set(media_ids)
        if outside := only - set(matching):
            raise ValueError(f"Images not in the project{' or not matching the filter' if filter.text() else ''}: {sorted(outside)}")
        matching = [m for m in matching if m in only]
    return runs.start(conn, matching, annotator, list(looked_for), list(looked_for.values()), again, min_score,
                      count=count, tried=tried)


def try_(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter(), annotator: Finder | None = None,
         classes: list[str] | None = None, prompts: dict[str, str] | None = None) -> dict | None:
    """Try the annotator on TRY_IMAGES of the project's images matching `filter`, picked at random,
    for its classes (or those named), with `prompts` instead of some classes' saved ones; what it
    finds is kept as any run's. Returns as start(). ValueError: as start()."""
    return start(conn, project_id, filter, TRY_IMAGES, annotator, classes, again=True, prompts=prompts, tried=True)


def latest_try(conn: sqlite3.Connection, project_id: int) -> dict | None:
    """The project's latest Try: its job's status and its images with their annotations
    (projects.images); None if it has had none."""
    row = conn.execute("SELECT status, params FROM job WHERE kind = ? AND project_id = ? AND params ->> 'tried' "
                       "ORDER BY id DESC LIMIT 1", (runs.KIND, project_id)).fetchone()
    if row is None:
        return None
    return {"status": row["status"]} | projects.images(conn, json.loads(row["params"])["media_ids"])

