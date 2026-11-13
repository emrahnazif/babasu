"""Projects and their media. Reads return plain dicts (JSON columns parsed), ready for templates and the API.

Inactive classes (deleted by the user, kept so their ids are never reused) are left
out of every read here, together with their annotations.
"""

import json
import math
import sqlite3
from typing import Annotated, Literal

from PIL import ImageColor
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app import db, jobs, media
from app.filters import Filter, FilterError, ObjectCondition, class_ids

# What a project is for. It suggests (an annotator: annotators.SUGGESTED; later what to train), it
# doesn't restrict: every project may hold every shape. Classification comes later.
Task = Literal["detection", "segmentation", "keypoints"]
TASKS: dict[Task, str] = {"detection": "Object detection", "segmentation": "Segmentation", "keypoints": "Keypoint detection"}

# Pixel coordinate on the original image, rounded to 1 decimal.
Coord = Annotated[float, AfterValidator(lambda v: round(v, 1))]


class Keypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    point: tuple[Coord, Coord]


class Annotation(BaseModel):
    """One thing in an image, with whichever shapes it has (like COCO and pixelflow's Detection)."""

    model_config = ConfigDict(extra="forbid")
    class_id: int
    bbox: tuple[Coord, Coord, Coord, Coord] | None = None  # x, y, w, h
    segments: list[tuple[Coord, Coord]] | None = None  # polygon points
    keypoints: list[Keypoint] | None = None
    # The model's confidence, on an object a model made and nobody has reviewed yet.
    score: Annotated[float, Field(ge=0, le=1), AfterValidator(lambda v: round(v, 3))] | None = None

    @model_validator(mode="after")
    def _has_a_shape(self):
        if self.bbox is None and not self.segments and not self.keypoints:
            raise ValueError("an annotation needs a bbox, segments or keypoints")
        return self


# What people type for a project and its classes, checked the same way wherever it comes in.
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]
ClassName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Prompt = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]
PROMPT = ("What annotators look for (not its name), e.g. 'hole in the road surface' for a class named pothole; "
          "empty: its name.")


def _color(value: str) -> str:
    """A colour as #rrggbb, from #rrggbb or as people name it (yellow, #ff0, rgb(…))."""
    try:
        return "#{:02x}{:02x}{:02x}".format(*ImageColor.getrgb(value.strip())[:3])
    except ValueError:
        raise ValueError(f"{value!r} isn't a colour: give a name (yellow) or #rrggbb") from None


Color = Annotated[str, AfterValidator(_color)]


class NewClass(BaseModel):
    """A class to add. No colour: it gets the first palette colour not in use; a blank prompt is the name."""

    model_config = ConfigDict(extra="forbid")
    name: ClassName
    color: Color | None = Field(None, description="A colour name or #rrggbb; empty: one not in use.")
    prompt: Prompt = Field("", description=PROMPT)


class ClassChange(NewClass):
    """One class of a project's class list, as it's set (set_classes): an existing one by its id
    (no colour: it keeps its own), or a new one (no id)."""

    id: int | None = None


# One of a project's classes, by its id or its name.
ClassRef = int | ClassName


class ClassEdit(BaseModel):
    """A change to one class (edit_classes): which one, by its id or else its name, and what changes.
    With an id, `name` is a new name; `new_name` is one either way."""

    model_config = ConfigDict(extra="forbid")
    id: int | None = Field(None, description="The class's id; without one, `name` says which class.")
    name: ClassName | None = Field(None, description="With an id: its new name. Without: the class's name.")
    new_name: ClassName | None = Field(None, description="Its new name, to rename it.")
    prompt: Prompt | None = Field(None, description=PROMPT)
    color: Color | None = Field(None, description="A colour name or #rrggbb.")


# Class colours, in the order new classes get them.
PALETTE = ["#e11d48", "#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#db2777", "#65a30d"]

PAGE_SIZE = 100  # images per page


class FrozenError(Exception):
    """Raised on any write to a frozen project."""


def _project(row: sqlite3.Row) -> dict:
    classes = [c for c in json.loads(row["classes"]) if c["active"]]
    return dict(row) | {"classes": classes}


COVER_SIZE = 4  # images shown on a project's card


def list_all(conn: sqlite3.Connection) -> list[dict]:
    """Every project, most recently changed first, with its image count, how many of its images
    are labelled (have an annotation) and the thumbnails of its first few images for its card."""
    rows = conn.execute(
        """SELECT project.*, count(project_media.media_id) AS image_count,
                  (SELECT json_group_array(sha256) FROM (
                      SELECT media.sha256 FROM project_media AS first JOIN media ON media.id = first.media_id
                      WHERE first.project_id = project.id ORDER BY first.media_id LIMIT ?)) AS covers
           FROM project LEFT JOIN project_media ON project_media.project_id = project.id
           GROUP BY project.id ORDER BY project.updated_at DESC""",
        (COVER_SIZE,),
    ).fetchall()
    labelled = _labelled_counts(conn, rows)
    return [
        _project(row)
        | {"covers": [media.thumb_url(sha) for sha in json.loads(row["covers"])], "labelled_count": labelled.get(row["id"], 0)}
        for row in rows
    ]


def _labelled(row: sqlite3.Row) -> tuple[str, list]:
    """SQL condition (and its parameters) on a project_media row of this project: the image is
    labelled, i.e. has an annotation of an active class."""
    active = json.dumps([c["id"] for c in _project(row)["classes"]])
    return f"EXISTS (SELECT 1 FROM annotation a WHERE {_OF_IMAGE} AND a.class_id IN (SELECT value FROM json_each(?)))", [active]


def _labelled_counts(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[int, int]:
    """Labelled images per project id: one count over all projects, then a recount of those
    with an inactive class (whose objects don't count)."""
    counts = dict(conn.execute("SELECT project_id, count(DISTINCT media_id) FROM annotation GROUP BY project_id").fetchall())
    for row in rows:
        if any(not c["active"] for c in json.loads(row["classes"])):
            counts[row["id"]] = _labelled_count(conn, row)
    return counts


def _labelled_count(conn: sqlite3.Connection, row: sqlite3.Row) -> int:
    condition, params = _labelled(row)
    return conn.execute(
        f"SELECT count(*) FROM project_media WHERE project_id = ? AND {condition}", (row["id"], *params)
    ).fetchone()[0]


def get(conn: sqlite3.Connection, project_id: int) -> dict | None:
    row = _row(conn, project_id)
    return _project(row) if row else None


def _row(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM project WHERE id = ?", (project_id,)).fetchone()


def _count(conn: sqlite3.Connection, where: str, params: list) -> int:
    return conn.execute(f"SELECT count(*) FROM project_media WHERE {where}", params).fetchone()[0]


def detail(conn: sqlite3.Connection, project_id: int, page: int = 1, filter: Filter = Filter()) -> dict | None:
    """A project with one page of the images matching `filter` (by file name) and their
    annotations (marking the objects the filter asks for), how many images it has and how many
    are labelled, and how many annotations each class has. Raises FilterError for a filter
    that names a class the project doesn't have."""
    row = _row(conn, project_id)
    if row is None:
        return None
    counts = annotation_counts(conn, project_id)
    project = _project(row)
    project["classes"] = [c | {"annotation_count": counts.get(c["id"], 0)} for c in project["classes"]]
    image_count = _count(conn, "project_media.project_id = ?", [project_id])
    where, params = _where(row, filter)
    match_count = image_count if filter == Filter() else _count(conn, where, params)
    pages = max(1, math.ceil(match_count / PAGE_SIZE))
    page = min(max(page, 1), pages)
    rows = conn.execute(
        f"""{_IMAGE_SELECT} WHERE {where} ORDER BY project_media.filename, project_media.media_id LIMIT ? OFFSET ?""",
        (*params, PAGE_SIZE, (page - 1) * PAGE_SIZE),
    ).fetchall()
    active = {c["id"] for c in project["classes"]}
    matches = _matches(conn, row, filter, [r["media_id"] for r in rows])
    return project | {
        "image_count": image_count,
        "labelled_count": _labelled_count(conn, row),
        "filter": filter.text(),
        "match_count": match_count,
        "page": page,
        "pages": pages,
        "images": [_image(r, active, matches and matches[r["media_id"]]) for r in rows],
    }


def named(conn: sqlite3.Connection, text: str, limit: int = 20) -> dict:
    """Projects and images whose name contains `text` (ignoring case): projects (id, name) and the
    first `limit` images (media id, file name, project id and name), with how many images match."""
    found = conn.execute("SELECT id, name FROM project WHERE instr(lower(name), lower(?)) > 0 ORDER BY name", (text,))
    images = """FROM project_media JOIN project ON project.id = project_media.project_id
                WHERE instr(lower(project_media.filename), lower(?)) > 0"""
    rows = conn.execute(
        f"""SELECT project_media.media_id, project_media.filename, project.id AS project_id, project.name AS project
            {images} ORDER BY project_media.filename, project.name LIMIT ?""", (text, limit))
    return {"projects": [dict(r) for r in found], "images": [dict(r) for r in rows],
            "image_count": conn.execute(f"SELECT count(*) {images}", (text,)).fetchone()[0]}


def sample(conn: sqlite3.Connection, project_id: int, filter: Filter = Filter(), count: int | None = None) -> list[int] | None:
    """The media ids of the project's images matching `filter`: all of them, or `count` picked at
    random. None if there's no such project; FilterError as detail()."""
    row = _row(conn, project_id)
    if row is None:
        return None
    where, params = _where(row, filter)
    order = "ORDER BY random() LIMIT ?" if count else "ORDER BY project_media.filename LIMIT ?"
    rows = conn.execute(f"SELECT media_id FROM project_media WHERE {where} {order}", (*params, count or -1))
    return [r[0] for r in rows]


def image(conn: sqlite3.Connection, project_id: int, media_id: int, filter: Filter = Filter()) -> dict | None:
    """One image of a project with its annotations, the project, and where the image is among
    the images matching `filter` (by file name): its position, how many match, and the
    previous and next ones."""
    row = _row(conn, project_id)
    found = row and conn.execute(
        f"{_IMAGE_SELECT} WHERE project_media.project_id = ? AND project_media.media_id = ?", (project_id, media_id)
    ).fetchone()
    if not found:
        return None
    where, params = _where(row, filter)
    key = (found["filename"], media_id)  # file names can repeat, so the id breaks ties

    def neighbour(before: bool) -> int | None:
        op, order = ("<", "DESC") if before else (">", "ASC")
        hit = conn.execute(
            f"""SELECT media_id FROM project_media WHERE {where} AND (filename, media_id) {op} (?, ?)
                ORDER BY filename {order}, media_id {order} LIMIT 1""",
            (*params, *key),
        ).fetchone()
        return hit[0] if hit else None

    match_count, position = conn.execute(
        f"SELECT count(*), total((filename, media_id) <= (?, ?)) FROM project_media WHERE {where}", (*key, *params)
    ).fetchone()
    project = _project(row)
    matches = _matches(conn, row, filter, [media_id])
    return _image(found, {c["id"] for c in project["classes"]}, matches and matches[media_id]) | {
        "project": project,
        "filter": filter.text(),
        "position": int(position),
        "match_count": match_count,
        "previous_id": neighbour(before=True),
        "next_id": neighbour(before=False),
    }


_IMAGE_SELECT = """SELECT project_media.media_id, project_media.filename, project_media.split,
                          project_media.annotations, media.sha256, media.ext, media.width, media.height
                   FROM project_media JOIN media ON media.id = project_media.media_id"""


def _image(row: sqlite3.Row, class_ids: set[int], matched: set[int] | None = None) -> dict:
    """An image row for templates and the API: its annotations (of active classes), and, when
    the filter asks for objects, `matches`: the positions in that list of the objects it found."""
    kept = [(i, a) for i, a in enumerate(json.loads(row["annotations"])) if a["class_id"] in class_ids]
    image = dict(row) | {
        "annotations": [a for _, a in kept],
        "url": media.url(row["sha256"], row["ext"]),
        "thumb_url": media.thumb_url(row["sha256"]),
        "aspect": round(row["width"] / row["height"], 3),  # width / height
    }
    if matched is not None:
        image["matches"] = [n for n, (i, _) in enumerate(kept) if i in matched]
    return image


# An annotation `a` of the project_media row being filtered.
_OF_IMAGE = "a.project_id = project_media.project_id AND a.media_id = project_media.media_id"


def _where(row: sqlite3.Row, filter: Filter) -> tuple[str, list]:
    """SQL condition (and parameters) on project_media for this project's images matching
    `filter`. Raises FilterError if it names a class the project doesn't have."""
    parts, params = ["project_media.project_id = ?"], [row["id"]]
    if filter.show != "all":
        labelled, more = _labelled(row)
        parts.append(labelled if filter.show == "labelled" else f"NOT ({labelled})")
        params += more
    if filter.name:
        parts.append("instr(lower(project_media.filename), lower(?)) > 0")
        params.append(filter.name)
    if filter.objects:
        condition, more = filter.objects.sql(*_object_count(row))
        parts.append(condition)
        params += more
    for c in filter.conditions:
        condition, more = _object_sql(row, c)
        parts.append(f"{'NOT ' if c.negate else ''}EXISTS (SELECT 1 FROM annotation a WHERE {_OF_IMAGE} AND {condition})")
        params += more
    return " AND ".join(parts), params


def _object_count(row: sqlite3.Row) -> tuple[str, tuple]:
    """SQL for how many objects (of active classes) the project_media row being filtered has, and its parameters."""
    active = json.dumps([c["id"] for c in _project(row)["classes"]])
    return f"(SELECT count(*) FROM annotation a WHERE {_OF_IMAGE} AND a.class_id IN (SELECT value FROM json_each(?)))", (active,)


def _wanted(row: sqlite3.Row, filter: Filter) -> tuple[str, list] | None:
    """SQL condition (and parameters) on an annotation `a`: it's one of the objects the filter
    asks for (meets one of its conditions that aren't negated), or None if it asks for none."""
    wanted = [_object_sql(row, c) for c in filter.conditions if not c.negate]
    if not wanted:
        return None
    return " OR ".join(f"({condition})" for condition, _ in wanted), [p for _, more in wanted for p in more]


def _object_sql(row: sqlite3.Row, c: ObjectCondition) -> tuple[str, list]:
    """SQL condition (and parameters) on an annotation `a`: it meets all of `c` (its sign aside)."""
    classes = _project(row)["classes"]
    ids = class_ids(c.classes, classes) if c.classes else [k["id"] for k in classes]
    parts, params = ["a.class_id IN (SELECT value FROM json_each(?))"], [json.dumps(ids)]
    for column, r in (("a.w", c.w), ("a.h", c.h), ("a.area", c.area), ("a.aspect", c.aspect)):
        if r:
            condition, more = r.sql(column)
            parts.append(condition)
            params += more
    if c.edge is not None:
        parts.append("a.edge = ?")
        params.append(int(c.edge))
    return " AND ".join(parts), params


def _matches(conn: sqlite3.Connection, row: sqlite3.Row, filter: Filter, media_ids: list[int]) -> dict[int, set[int]] | None:
    """For these images, the objects the filter asks for (positions in each image's annotations
    list), or None if it doesn't ask for any."""
    wanted = _wanted(row, filter)
    if wanted is None:
        return None
    condition, params = wanted
    found: dict[int, set[int]] = {m: set() for m in media_ids}
    rows = conn.execute(
        f"""SELECT a.media_id, a.idx FROM annotation a
            WHERE a.project_id = ? AND a.media_id IN (SELECT value FROM json_each(?)) AND ({condition})""",
        (row["id"], json.dumps(media_ids), *params),
    )
    for media_id, idx in rows:
        found[media_id].add(idx)
    return found


STAT_FIELDS = ("class", "w", "h", "area", "aspect", "objects")


def stats(conn: sqlite3.Connection, project_id: int, field: str, filter: Filter = Filter(), bins: int = 20) -> dict | None:
    """How a measure is spread over the images matching `filter`: "class" counts objects per
    class; "w", "h", "area" and "aspect" give a histogram of the objects (those the filter asks
    for, or all of them); "objects" gives a histogram of objects per image."""
    row = _row(conn, project_id)
    if row is None:
        return None
    where, params = _where(row, filter)
    if field == "objects":
        count, count_params = _object_count(row)
        rows = conn.execute(
            f"SELECT n, count(*) FROM (SELECT {count} AS n FROM project_media WHERE {where}) GROUP BY n ORDER BY n",
            (*count_params, *params),
        ).fetchall()
        return {"field": field, "counts": [{"objects": n, "images": k} for n, k in rows]}
    condition, more = _wanted(row, filter) or _object_sql(row, ObjectCondition())
    objects = f"FROM annotation a JOIN project_media ON {_OF_IMAGE} WHERE {where} AND ({condition})"
    params += more
    if field == "class":
        names = {c["id"]: c["name"] for c in _project(row)["classes"]}
        rows = conn.execute(f"SELECT a.class_id, count(*) {objects} GROUP BY a.class_id ORDER BY 2 DESC", params)
        return {"field": field, "counts": [{"class": names[k], "objects": n} for k, n in rows]}
    if field not in STAT_FIELDS:
        raise FilterError(f"There are no statistics for “{field}”; try one of {', '.join(STAT_FIELDS)}")
    column = f"a.{field}"
    low, high, total = conn.execute(f"SELECT min({column}), max({column}), count({column}) {objects}", params).fetchone()
    if not total:
        return {"field": field, "objects": 0, "bins": []}
    width = (high - low) / bins or 1
    rows = dict(conn.execute(
        f"SELECT min(CAST(({column} - ?) / ? AS INTEGER), ?) AS b, count(*) {objects} AND {column} IS NOT NULL GROUP BY b",
        (low, width, bins - 1, *params),
    ).fetchall())
    return {
        "field": field,
        "objects": total,
        "bins": [{"from": low + b * width, "to": low + (b + 1) * width, "objects": rows.get(b, 0)} for b in range(bins)],
    }


def annotation_counts(conn: sqlite3.Connection, project_id: int) -> dict[int, int]:
    """Number of annotations per class id, over all of the project's images."""
    rows = conn.execute("SELECT class_id, count(*) FROM annotation WHERE project_id = ? GROUP BY class_id", (project_id,))
    return dict(rows.fetchall())


def create(conn: sqlite3.Connection, name: str, description: str, task: Task, annotator: str | None,
           classes: list[ClassChange] = ()) -> int:
    """Create a project with an annotator (an id in annotators.FINDERS; None: none) and these
    classes (as set_classes sets them). Caller commits. Raises ValueError as set_classes."""
    project_id = conn.execute(
        "INSERT INTO project (name, description, task, annotator) VALUES (?, ?, ?, ?)",
        (name, description, task, annotator),
    ).lastrowid
    set_classes(conn, project_id, list(classes))
    return project_id


def duplicate(conn: sqlite3.Connection, project_id: int, name: str | None = None) -> int | None:
    """A new project named `name` ("<its name> copy" if none) with the project's description, type,
    annotator, classes (removed ones too, so their annotations come along hidden), images and
    annotations; not its runs. The image files are shared, not copied. Its id; None if there's no
    such project. Caller commits."""
    row = _row(conn, project_id)
    if row is None:
        return None
    new_id = conn.execute(
        """INSERT INTO project (name, description, classes, task, annotator)
           SELECT ?, description, classes, task, annotator FROM project WHERE id = ?""",
        (name or f"{row['name']} copy", project_id),
    ).lastrowid
    conn.execute(
        """INSERT INTO project_media (project_id, media_id, filename, split, annotations)
           SELECT ?, media_id, filename, split, annotations FROM project_media WHERE project_id = ?""",
        (new_id, project_id),
    )
    return new_id


def update(
    conn: sqlite3.Connection, project_id: int, name: str, description: str, task: Task, annotator: str | None
) -> None:
    """Change a project's name, description, type or annotator (None: none). Its annotations stay
    as they are. Caller commits. Raises FrozenError."""
    _editable(conn, project_id)
    conn.execute(
        "UPDATE project SET name = ?, description = ?, task = ?, annotator = ? WHERE id = ?",
        (name, description, task, annotator, project_id),
    )
    _touch(conn, project_id)


def _active_jobs(conn: sqlite3.Connection, project_id: int) -> int:
    return conn.execute(f"SELECT count(*) FROM job WHERE project_id = ? AND status IN {jobs.ACTIVE}", (project_id,)).fetchone()[0]


def deletion(conn: sqlite3.Connection, project_id: int) -> dict:
    """What deleting the project would remove: its `images`, their `annotations` (hidden ones
    too) and its `runs`. `jobs`: how many of its jobs are queued or running; while there are, it
    can't be deleted."""
    count = lambda sql: conn.execute(sql, (project_id,)).fetchone()[0]
    return {
        "images": count("SELECT count(*) FROM project_media WHERE project_id = ?"),
        "annotations": count("SELECT count(*) FROM annotation WHERE project_id = ?"),
        "runs": count("SELECT count(*) FROM run WHERE project_id = ?"),
        "jobs": _active_jobs(conn, project_id),
    }


def delete(conn: sqlite3.Connection, project_id: int) -> list[int]:
    """Delete a project, its rows of images and its runs (with their results), and return the ids
    of the media they used (the files are another matter: media.remove_unused). The run rows go
    here, not through runs.delete (its own transaction and file cleanup). Its versions stay: if
    it's an original, its oldest version becomes the original. Caller commits.

    Raises ValueError while one of its jobs is queued or running."""
    if _active_jobs(conn, project_id):
        raise ValueError("The project has jobs queued or running; cancel them first")
    media_ids = [r[0] for r in conn.execute(
        """SELECT media_id FROM project_media WHERE project_id = :project
           UNION SELECT r.media_id FROM run_result r JOIN run ON run.id = r.run_id WHERE run.project_id = :project""",
        {"project": project_id})]
    if conn.execute("SELECT root_id FROM project WHERE id = ?", (project_id,)).fetchone()["root_id"] is None:
        oldest = conn.execute(
            "SELECT id FROM project WHERE root_id = ? ORDER BY created_at, id LIMIT 1", (project_id,)
        ).fetchone()
        if oldest:
            conn.execute("UPDATE project SET root_id = NULL WHERE id = ?", (oldest["id"],))
            conn.execute("UPDATE project SET root_id = ? WHERE root_id = ?", (oldest["id"], project_id))
    conn.execute("DELETE FROM run WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM project WHERE id = ?", (project_id,))
    return media_ids


def remove(conn: sqlite3.Connection, project_id: int) -> None:
    """Delete the project as delete() does and commit, then the image files nothing uses any more.
    Raises ValueError as delete()."""
    with conn:
        unused = media.remove_unused(conn, delete(conn, project_id))
    for path in unused:
        path.unlink(missing_ok=True)


def add_image(
    conn: sqlite3.Connection,
    project_id: int,
    media_id: int,
    filename: str,
    annotations: list[dict] | None = None,
    split: str | None = None,
) -> bool:
    """Add a stored media file to a project. Returns False if it was already there. Caller commits.

    Raises pydantic.ValidationError for a malformed annotation, ValueError for an unknown class.
    """
    project = _editable(conn, project_id)
    objects = [Annotation.model_validate(a) for a in annotations or []]
    # Inactive classes count: their annotations are kept, only hidden.
    unknown = {o.class_id for o in objects} - {c["id"] for c in json.loads(project["classes"])}
    if unknown:
        raise ValueError(f"unknown class ids: {sorted(unknown)}")
    added = conn.execute(
        """INSERT INTO project_media (project_id, media_id, filename, split, annotations)
           VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
        (project_id, media_id, filename, split, json.dumps([o.model_dump(exclude_none=True) for o in objects])),
    ).rowcount == 1
    if added:
        _touch(conn, project_id)
    return added


def set_annotations(conn: sqlite3.Connection, project_id: int, media_id: int, annotations: list[Annotation]) -> bool:
    """Replace an image's annotations with these, as the editor and the agent see them: of active
    classes. Those of inactive classes are shown nowhere, so they're kept as they are. Returns
    False if the image isn't in the project. Caller commits.

    Raises FrozenError, or ValueError for a class the project doesn't have (or has deleted)."""
    active = _check_active(_editable(conn, project_id), {a.class_id for a in annotations})
    row = conn.execute(
        "SELECT annotations FROM project_media WHERE project_id = ? AND media_id = ?", (project_id, media_id)
    ).fetchone()
    if row is None:
        return False
    kept = [a for a in json.loads(row["annotations"]) if a["class_id"] not in active]
    conn.execute(
        f"UPDATE project_media SET annotations = ?, updated_at = {db.NOW} WHERE project_id = ? AND media_id = ?",
        (json.dumps([a.model_dump(exclude_none=True) for a in annotations] + kept), project_id, media_id),
    )
    _touch(conn, project_id)
    return True


DUPLICATE_IOU = 0.9  # an added object whose box overlaps one of its class this much is already there


def _box(a: dict) -> tuple[float, float, float, float] | None:
    """An annotation's box as corners: its bbox, or its polygon's bounds; None without either."""
    if a.get("bbox"):
        x, y, w, h = a["bbox"]
        return x, y, x + w, y + h
    if a.get("segments"):
        xs, ys = [p[0] for p in a["segments"]], [p[1] for p in a["segments"]]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def add_annotations(conn: sqlite3.Connection, project_id: int, media_id: int, annotations: list[Annotation]) -> int | None:
    """Add objects to an image, next to the ones it has, leaving out each one already there: its
    box overlaps an object of its class by DUPLICATE_IOU or more. Returns how many were added;
    None if the image isn't in the project. Caller commits.

    Raises FrozenError, or ValueError for a class the project doesn't have (or has deleted)."""
    _check_active(_editable(conn, project_id), {a.class_id for a in annotations})
    row = conn.execute(
        "SELECT annotations FROM project_media WHERE project_id = ? AND media_id = ?", (project_id, media_id)
    ).fetchone()
    if row is None:
        return None
    objects = json.loads(row["annotations"])
    boxes = [(a["class_id"], _box(a)) for a in objects]
    added = 0
    for a in annotations:
        new = a.model_dump(exclude_none=True)
        box = _box(new)
        if box and any(c == a.class_id and b and _iou(box, b) >= DUPLICATE_IOU for c, b in boxes):
            continue
        objects.append(new)
        boxes.append((a.class_id, box))
        added += 1
    if added:
        conn.execute(
            f"UPDATE project_media SET annotations = ?, updated_at = {db.NOW} WHERE project_id = ? AND media_id = ?",
            (json.dumps(objects), project_id, media_id),
        )
        _touch(conn, project_id)
    return added


def with_prompts(classes: list[dict], prompts: dict[int, str]) -> list[dict]:
    """The classes with some prompts changed (class id -> prompt); an empty one is the class's name."""
    return [c | {"prompt": prompts[c["id"]].strip() or c["name"]} if c["id"] in prompts else c for c in classes]


def set_prompts(conn: sqlite3.Connection, project_id: int, prompts: dict[int, str]) -> None:
    """Change the prompts of some classes (class id -> prompt), the text sent to annotators. An
    empty prompt goes back to the class's name. Caller commits.

    Raises FrozenError, or ValueError for a class the project doesn't have (or has deleted)."""
    row = _editable(conn, project_id)
    _check_active(row, set(prompts))
    classes = with_prompts(json.loads(row["classes"]), prompts)
    conn.execute("UPDATE project SET classes = ? WHERE id = ?", (json.dumps(classes), project_id))
    _touch(conn, project_id)


def set_classes(conn: sqlite3.Connection, project_id: int, classes: list[ClassChange]) -> None:
    """Set the project's class list to these, in this order (product.md "Classes"): a class left
    out becomes inactive (its annotations stay, hidden); a new one named like an inactive class
    brings that one back; any other new one gets the next id, never one used before. A class
    without a colour keeps its own, or gets the first palette colour no class has. Caller commits.

    Raises FrozenError, or ValueError for a repeated name or an id the project doesn't have."""
    current = {c["id"]: c for c in json.loads(_editable(conn, project_id)["classes"])}
    active = {i for i, c in current.items() if c["active"]}
    inactive = {c["name"].casefold(): i for i, c in current.items() if not c["active"]}
    taken = {c["color"] for c in current.values()} | {c.color.lower() for c in classes if c.color}
    free = [p for p in PALETTE if p not in taken]
    next_id = max(current, default=-1) + 1
    listed, seen = [], set()
    for c in classes:
        name = c.name.casefold()
        if name in seen:
            raise ValueError(f"Two classes are named {c.name}")
        seen.add(name)
        if c.id is None:
            id = inactive.pop(name, None)
            if id is None:
                id, next_id = next_id, next_id + 1
        elif c.id in active:
            id = c.id
            active.discard(id)
        else:
            raise ValueError(f"unknown class id: {c.id}")
        color = c.color or current.get(id, {}).get("color") or (free.pop(0) if free else PALETTE[id % len(PALETTE)])
        listed.append({"id": id, "name": c.name, "prompt": c.prompt or c.name, "color": color.lower(), "active": True})
    kept = {c["id"] for c in listed}
    listed += [c | {"active": False} for i, c in current.items() if i not in kept]
    conn.execute("UPDATE project SET classes = ? WHERE id = ?", (json.dumps(listed), project_id))
    _touch(conn, project_id)


def edit_classes(conn: sqlite3.Connection, project_id: int, add: list[NewClass] = (),
                 change: list[ClassEdit] = (), remove: list[ClassRef] = ()) -> None:
    """Add, change and remove some of the project's classes; the others stay as they are. Added
    ones go last; set_classes's rules hold (a removed class becomes inactive, one added by an
    inactive class's name brings it back). A prompt that is the class's name stays "the name", so
    it follows a rename. Caller commits.

    Raises FrozenError, or ValueError for a class it doesn't have (saying which it has) or as set_classes."""
    classes = _project(_editable(conn, project_id))["classes"]
    rows = {c["id"]: {"id": c["id"], "name": c["name"], "color": c["color"],
                      "prompt": "" if c["prompt"] == c["name"] else c["prompt"]} for c in classes}

    def find(ref: ClassRef) -> int:
        for id, row in rows.items():
            if ref == id if isinstance(ref, int) else ref.casefold() == row["name"].casefold():
                return id
        listed = ", ".join(f"{c['name']} ({c['id']})" for c in classes) or "none"
        raise ValueError(f"There is no class {ref!r}; the classes: {listed}")

    for e in change:
        if e.id is None and e.name is None:
            raise ValueError("Say which class to change: its id or its name")
        row = rows[find(e.name if e.id is None else e.id)]
        row |= e.model_dump(include={"prompt", "color"}, exclude_none=True)
        row["name"] = e.new_name or (e.name if e.id is not None and e.name else row["name"])
    for ref in remove:
        del rows[find(ref)]
    set_classes(conn, project_id, [ClassChange(**r) for r in rows.values()]
                + [ClassChange(**c.model_dump()) for c in add])


def _check_active(project: sqlite3.Row, class_ids: set[int]) -> set[int]:
    """The project's active class ids; ValueError if `class_ids` has others (unknown or deleted)."""
    active = {c["id"] for c in _project(project)["classes"]}
    if class_ids - active:
        raise ValueError(f"unknown class ids: {sorted(class_ids - active)}")
    return active


def _editable(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row:
    """The project row, for a write; raises FrozenError if it's frozen."""
    row = conn.execute("SELECT frozen_at, classes FROM project WHERE id = ?", (project_id,)).fetchone()
    if row["frozen_at"]:
        raise FrozenError(f"project {project_id} is frozen")
    return row


def _touch(conn: sqlite3.Connection, project_id: int) -> None:
    conn.execute(
        f"UPDATE project SET updated_at = {db.NOW} WHERE id = ?",
        (project_id,),
    )
