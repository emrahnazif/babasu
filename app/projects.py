"""Projects and their media. Reads return plain dicts (JSON columns parsed), ready for templates and the API.

Inactive classes (deleted by the user, kept so their ids are never reused) are left
out of every read here, together with their annotations.
"""

import json
import sqlite3

from app import media


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
    """Add a stored media file to a project (skipped if it's already there). Caller commits."""
    _check_editable(conn, project_id)
    conn.execute(
        """INSERT INTO project_media (project_id, media_id, filename, split, annotations)
           VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
        (project_id, media_id, filename, split, json.dumps(annotations or [])),
    )
    _touch(conn, project_id)


def _check_editable(conn: sqlite3.Connection, project_id: int) -> None:
    row = conn.execute("SELECT frozen_at FROM project WHERE id = ?", (project_id,)).fetchone()
    if row["frozen_at"]:
        raise FrozenError(f"project {project_id} is frozen")


def _touch(conn: sqlite3.Connection, project_id: int) -> None:
    conn.execute(
        "UPDATE project SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
        (project_id,),
    )
