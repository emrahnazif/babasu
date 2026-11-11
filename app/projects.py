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


def _labelled_counts(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[int, int]:
    """Labelled images per project id. An image whose only annotations are of inactive classes
    isn't labelled. Checking that reads every annotation, so it's done only for the (rare)
    projects that have an inactive class; the others just count non-empty annotation lists."""
    counts = dict(conn.execute(
        "SELECT project_id, count(*) FROM project_media WHERE annotations != '[]' GROUP BY project_id"
    ).fetchall())
    for row in rows:
        inactive = [c["id"] for c in json.loads(row["classes"]) if not c["active"]]
        if inactive:
            counts[row["id"]] = conn.execute(
                f"""SELECT count(*) FROM project_media WHERE project_id = ? AND EXISTS (
                        SELECT 1 FROM json_each(annotations)
                        WHERE value->>'class_id' NOT IN ({",".join("?" * len(inactive))}))""",
                (row["id"], *inactive),
            ).fetchone()[0]
    return counts


def get(conn: sqlite3.Connection, project_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM project WHERE id = ?", (project_id,)).fetchone()
    return _project(row) if row else None


def detail(conn: sqlite3.Connection, project_id: int, page: int = 1) -> dict | None:
    """A project with one page of its images and their annotations."""
    project = get(conn, project_id)
    if project is None:
        return None
    image_count = conn.execute(
        "SELECT count(*) FROM project_media WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    pages = max(1, math.ceil(image_count / PAGE_SIZE))
    page = min(max(page, 1), pages)
    class_ids = {c["id"] for c in project["classes"]}
    images = _images(conn, project_id, class_ids, offset=(page - 1) * PAGE_SIZE)
    return project | {"image_count": image_count, "page": page, "pages": pages, "images": images}


def _images(conn: sqlite3.Connection, project_id: int, class_ids: set[int], offset: int) -> list[dict]:
    rows = conn.execute(
        """SELECT project_media.media_id, project_media.filename, project_media.split,
                  project_media.annotations, media.sha256, media.ext, media.width, media.height
           FROM project_media JOIN media ON media.id = project_media.media_id
           WHERE project_media.project_id = ? ORDER BY project_media.filename
           LIMIT ? OFFSET ?""",
        (project_id, PAGE_SIZE, offset),
    ).fetchall()
    return [
        dict(row)
        | {
            "annotations": [a for a in json.loads(row["annotations"]) if a["class_id"] in class_ids],
            "url": media.url(row["sha256"], row["ext"]),
            "thumb_url": media.thumb_url(row["sha256"]),
        }
        for row in rows
    ]


def annotation_counts(conn: sqlite3.Connection, project_id: int) -> dict[int, int]:
    """Number of annotations per class id, over all of the project's images."""
    rows = conn.execute(
        """SELECT a.value->>'class_id', count(*) FROM project_media, json_each(project_media.annotations) a
           WHERE project_media.project_id = ? GROUP BY 1""",
        (project_id,),
    ).fetchall()
    return {class_id: n for class_id, n in rows}


def create(conn: sqlite3.Connection, name: str, description: str = "", classes: list[dict] | None = None) -> int:
    """Create a project. Caller commits."""
    return conn.execute(
        "INSERT INTO project (name, description, classes) VALUES (?, ?, ?)",
        (name, description, json.dumps(classes or [])),
    ).lastrowid


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
