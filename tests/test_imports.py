import pytest

from app import db, history, imports, jobs, projects


@pytest.fixture
def folder(tmp_path, image):
    """A folder of images as a camera or a NAS leaves it: subfolders, hidden files, other files."""
    files = {"a.png": image(1), "sub/b.PNG": image(2), ".hidden.png": image(3), "._a.png": b"mac metadata",
             ".cache/c.png": image(4), "notes.txt": b"not an image"}
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
    return tmp_path


def test_a_folder_gives_its_images_named_inside_it_without_hidden_files(folder):
    found, others, missing = imports.find_media([str(folder), str(folder / "nowhere")])
    assert [name for _, name in found] == ["a.png", "sub/b.PNG"]
    assert missing == [str(folder / "nowhere")]


def test_importing_again_adds_only_what_is_new(conn, workers, finished, folder):
    paths = [str(folder), str(folder / "notes.txt")]
    project_id, job_id = imports.into_new_project(conn, paths)
    assert finished(job_id)["result"] == {"added": 2, "already_there": 0, "skipped": 1}

    finished(jobs.start(conn, imports.KIND, {"paths": paths, "project_id": project_id}, project_id))
    assert projects.image_count(conn, project_id) == 2


def test_undo_takes_back_everything_an_import_added(conn, workers, finished, folder):
    """A job is one action in the history, however many transactions it commits."""
    with db.action("request", "make"):
        project_id, job_id = imports.into_new_project(conn, [str(folder)])
    finished(job_id)

    history.undo(conn, history.latest(conn))
    assert projects.image_count(conn, project_id) == 0
