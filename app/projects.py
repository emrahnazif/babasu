"""Projects and their media. Reads return plain dicts (JSON columns parsed), ready for templates and the API.

Inactive classes (deleted by the user, kept so their ids are never reused) are left
out of every read here, together with their annotations.
"""

import json
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


class FrozenError(Exception):
    """Raised on any write to a frozen project."""


def _project(row: sqlite3.Row) -> dict:
    classes = [c for c in json.loads(row["classes"]) if c["active"]]
    return dict(row) | {"classes": classes}


def list_all(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT project.*, count(project_media.media_id) AS image_count
           FROM project LEFT JOIN project_media ON project_media.project_id = project.id
           GROUP BY project.id ORDER BY project.updated_at DESC"""
    ).fetchall()
    return [_project(row) for row in rows]


def detail(conn: sqlite3.Connection, project_id: int) -> dict | None:
    """A project with its images and their annotations."""
    row = conn.execute("SELECT * FROM project WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        return None
    project = _project(row)
    class_ids = {c["id"] for c in project["classes"]}
    return project | {"images": _images(conn, project_id, class_ids)}


def _images(conn: sqlite3.Connection, project_id: int, class_ids: set[int]) -> list[dict]:
    rows = conn.execute(
        """SELECT project_media.media_id, project_media.filename, project_media.split,
                  project_media.annotations, media.sha256, media.ext, media.width, media.height
           FROM project_media JOIN media ON media.id = project_media.media_id
           WHERE project_media.project_id = ? ORDER BY project_media.filename""",
        (project_id,),
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
) -> None:
    """Add a stored media file to a project (skipped if it's already there). Caller commits.

    Raises pydantic.ValidationError for a malformed annotation, ValueError for an unknown class.
    """
    project = _editable(conn, project_id)
    objects = [Annotation.model_validate(a) for a in annotations or []]
    # Inactive classes count: their annotations are kept, only hidden.
    unknown = {o.class_id for o in objects} - {c["id"] for c in json.loads(project["classes"])}
    if unknown:
        raise ValueError(f"unknown class ids: {sorted(unknown)}")
    conn.execute(
        """INSERT INTO project_media (project_id, media_id, filename, split, annotations)
           VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
        (project_id, media_id, filename, split, json.dumps([o.model_dump(exclude_none=True) for o in objects])),
    )
    _touch(conn, project_id)


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
