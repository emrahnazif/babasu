"""Projects and their media. Reads return plain dicts (JSON columns parsed), ready for templates and the API.

Inactive classes (deleted by the user, kept so their ids are never reused) are left
out of every read here, together with their annotations.
"""

import json
import math
import sqlite3
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from app import db, media
from app.filters import Filter, FilterError, ObjectCondition, class_ids

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

    @model_validator(mode="after")
    def _has_a_shape(self):
        if self.bbox is None and not self.segments and not self.keypoints:
            raise ValueError("an annotation needs a bbox, segments or keypoints")
        return self


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


def create(conn: sqlite3.Connection, name: str, description: str = "", classes: list[dict] | None = None) -> int:
    """Create a project. Caller commits."""
    return conn.execute(
        "INSERT INTO project (name, description, classes) VALUES (?, ?, ?)",
        (name, description, json.dumps(classes or [])),
    ).lastrowid


def update(conn: sqlite3.Connection, project_id: int, name: str, description: str) -> None:
    """Rename a project or change its description. Caller commits. Raises FrozenError."""
    _editable(conn, project_id)
    conn.execute("UPDATE project SET name = ?, description = ? WHERE id = ?", (name, description, project_id))
    _touch(conn, project_id)


def delete(conn: sqlite3.Connection, project_id: int) -> list[int]:
    """Delete a project and its rows of images, and return the ids of the media it used (the
    files are another matter: media.remove_unused). Its versions stay: if it's an original,
    its oldest version becomes the original. Caller commits."""
    media_ids = [r[0] for r in conn.execute("SELECT media_id FROM project_media WHERE project_id = ?", (project_id,))]
    if conn.execute("SELECT root_id FROM project WHERE id = ?", (project_id,)).fetchone()["root_id"] is None:
        oldest = conn.execute(
            "SELECT id FROM project WHERE root_id = ? ORDER BY created_at, id LIMIT 1", (project_id,)
        ).fetchone()
        if oldest:
            conn.execute("UPDATE project SET root_id = NULL WHERE id = ?", (oldest["id"],))
            conn.execute("UPDATE project SET root_id = ? WHERE root_id = ?", (oldest["id"], project_id))
    conn.execute("DELETE FROM project WHERE id = ?", (project_id,))
    return media_ids


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
