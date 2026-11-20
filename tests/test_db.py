import pytest

from app import db, history, projects


def test_writes_are_recorded_again_after_migrating_again(conn):
    """migrate() drops the history triggers and makes them again, also with nothing to migrate."""
    with conn:
        project_id = projects.create(conn, "Cars")
    db.migrate()
    with db.action("request", "rename"), conn:
        projects.update(conn, project_id, "Trucks", "", None, None)
    history.undo(conn, history.latest(conn))
    assert projects.get(conn, project_id)["name"] == "Cars"


def test_a_database_from_a_newer_version_is_refused(conn):
    with conn:
        conn.execute("PRAGMA user_version = 999")
    with pytest.raises(RuntimeError):
        db.migrate()
