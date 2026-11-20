import pytest

from app import config, db, history, media, projects


def _project(conn, image, name="Cars", images=2) -> int:
    """A project with images, each with a car on it, made by one request."""
    with db.action("request", "make"), conn:
        project_id = projects.create(conn, name)
        for n in range(images):
            media_id, _ = media.store(conn, project_id, image(n), f"{n}.png")
            projects.save(conn, media_id, project_id, [{"class": "car", "bbox": [1, 1, 10, 10]}])
    return project_id


def _rename(conn, project_id, name) -> str:
    """Rename a project as one request; the action's id."""
    with db.action("request", f"rename {name}"), conn:
        projects.update(conn, project_id, name, "", None, None)
        return db.current_action().id


def _counts(conn) -> tuple[int, int, int]:
    return tuple(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("project", "media", "annotation"))


def test_undo_puts_a_deleted_project_back_and_redo_deletes_it_again(conn, image):
    project_id = _project(conn, image)
    made = _counts(conn)
    with db.action("request", "delete"):
        projects.remove(conn, project_id)
    deleted = history.latest(conn)

    undone = history.undo(conn, deleted)
    assert _counts(conn) == made  # the annotations are rebuilt from the images that came back
    assert all(media.path(r["sha256"], r["ext"]).exists() for r in conn.execute("SELECT sha256, ext FROM media"))

    history.undo(conn, undone["id"])
    assert _counts(conn)[0] == 0


def test_an_action_changed_again_since_is_undone_after_the_later_one(conn, image):
    project_id = _project(conn, image)
    first = _rename(conn, project_id, "Trucks")
    second = _rename(conn, project_id, "Buses")

    with pytest.raises(history.Refused):
        history.undo(conn, first)
    history.undo(conn, second)
    history.undo(conn, first)
    assert projects.get(conn, project_id)["name"] == "Cars"


def test_undo_and_redo_step_through_actions_as_an_editor_does(conn, image):
    project_id = _project(conn, image)
    first = _rename(conn, project_id, "Trucks")
    second = _rename(conn, project_id, "Buses")

    assert history.latest(conn) == second
    undo_second = history.undo(conn, second)["id"]
    assert history.latest(conn) == first  # undoing again goes further back
    assert history.latest(conn, redo=True) == undo_second

    _rename(conn, project_id, "Vans")
    assert history.latest(conn, redo=True) is None  # something new was done: nothing to redo


def test_undo_brings_a_trashed_file_back(conn):
    file = config.WORKFLOWS_DIR / "count.json"
    file.parent.mkdir(parents=True)
    file.write_text("{}")
    with db.action("request", "delete workflow"), conn:
        history.trash(conn, file)
        trashed = db.current_action().id
    assert not file.exists()

    history.undo(conn, trashed)
    assert file.read_text() == "{}"


def test_purge_removes_only_files_nothing_can_bring_back(conn, image):
    keeps = _project(conn, image, "Keeps", images=1)  # its image's file is also the other project's first
    gone = _project(conn, image, "Gone", images=2)
    files = {r["name"]: media.path(r["sha256"], r["ext"])
             for r in conn.execute("SELECT name, sha256, ext FROM media WHERE project_id = ?", (gone,))}
    with db.action("request", "delete"):
        projects.remove(conn, gone)
    with conn:
        conn.execute("UPDATE action SET at = '2000-01-01T00:00:00Z'")  # all of it older than history keeps

    history.purge()
    assert files["0.png"].exists()  # Keeps still has it
    assert not files["1.png"].exists()
    assert projects.get(conn, keeps) is not None
    assert conn.execute("SELECT count(*) FROM action").fetchone()[0] == 0


def test_an_action_undone_after_a_redo_is_undone_once(conn, image):
    """Undo, redo, undo: it's undone, and undoing goes on to the action before it."""
    project_id = _project(conn, image)
    made = history.latest(conn)
    _rename(conn, project_id, "Trucks")
    history.undo(conn, history.latest(conn))
    history.undo(conn, history.latest(conn, redo=True))
    history.undo(conn, history.latest(conn))

    assert projects.get(conn, project_id)["name"] == "Cars"
    assert history.latest(conn) == made


def test_history_offers_undo_on_a_redone_action(conn, image):
    """Its own undo and redo aren't changes since: the page offers what undo() does."""
    project_id = _project(conn, image)
    renamed = _rename(conn, project_id, "Trucks")
    history.undo(conn, history.undo(conn, renamed)["id"])

    shown = history.actions(conn, action_id=renamed)[0]
    assert not shown["undone_by"] and not shown["changed_since"]
    history.undo(conn, renamed)
    assert projects.get(conn, project_id)["name"] == "Cars"


def test_a_redo_is_refused_once_an_earlier_action_was_undone_after_it(conn, image):
    """Rename twice, undo both: redoing the second would write over the first's undo."""
    project_id = _project(conn, image)
    first = _rename(conn, project_id, "Trucks")
    second = _rename(conn, project_id, "Buses")
    undo_second = history.undo(conn, second)["id"]
    history.undo(conn, first)

    with pytest.raises(history.Refused):
        history.undo(conn, undo_second)
    assert projects.get(conn, project_id)["name"] == "Cars"
