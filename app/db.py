"""The app's SQLite database: one file in DATA_DIR, schema from numbered migrations."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from app import config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# SQL expression for "now" in the format all timestamp columns use (UTC, ISO 8601).
NOW = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"


def connect() -> sqlite3.Connection:
    # check_same_thread=False: FastAPI may open the connection in one worker thread
    # and use it in another; each request still has its own connection.
    conn = sqlite3.connect(config.DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # With WAL this stays crash-safe (a power cut can lose the last commits, never corrupt),
    # and commits don't wait for a disk sync, which matters for jobs that commit per item.
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def migrate() -> None:
    """Bring the schema up to date: run each NNN_*.sql newer than PRAGMA user_version."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(connect()) as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # stored in the file; set once
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            number = int(path.name.split("_")[0])
            if number > version:
                # One transaction per migration, including the version bump.
                conn.executescript(f"BEGIN; {path.read_text()} PRAGMA user_version = {number}; COMMIT;")


def get() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: a connection for one request."""
    with closing(connect()) as conn:
        yield conn
