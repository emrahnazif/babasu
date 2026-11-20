"""Every test gets an empty workspace of its own, never the real one: HOME (where config.DATA_DIR
comes from) points at a temporary folder before the app is imported, and the workspace is emptied
before each test."""

import atexit
import io
import os
import shutil
import tempfile
import threading
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image

_home = Path(tempfile.mkdtemp(prefix="babasu-tests-"))
atexit.register(shutil.rmtree, _home, ignore_errors=True)
os.environ["HOME"] = str(_home)
os.environ["XDG_DATA_HOME"] = str(_home / "data")

from app import config, db, jobs  # noqa: E402  (after HOME is set)

if not config.DATA_DIR.is_relative_to(_home):  # e.g. Windows, whose folders don't come from HOME
    pytest.exit(f"The workspace would be {config.DATA_DIR}, not a temporary folder: not running tests there")


@pytest.fixture
def conn():
    """A connection to a new, empty database."""
    shutil.rmtree(config.DATA_DIR, ignore_errors=True)
    db.migrate()
    with closing(db.connect()) as c:
        yield c


@pytest.fixture
def workers(conn):
    """The job workers running, as in the app (crash recovery first); stopped after the test."""
    jobs.run_workers()
    yield
    jobs.shutdown()


@pytest.fixture
def finished(conn):
    """finished(job_id): the job once it has ended (or as it is after 10 s)."""
    def wait(job_id: int) -> dict:
        *_, job = jobs.watched(conn, job_id, threading.Event(), seconds=10)
        return job
    return wait


@pytest.fixture
def image():
    """image(n): the bytes of a small PNG, a different file for each n."""
    def make(n: int = 0, size: tuple[int, int] = (64, 48)) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", size, (n % 256, n // 256 % 256, 7)).save(buffer, format="PNG")
        return buffer.getvalue()
    return make
