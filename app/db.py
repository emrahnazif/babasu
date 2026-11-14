"""The app's SQLite database: one file in DATA_DIR, schema from numbered migrations."""

import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from app import config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# The format all timestamp columns use (UTC, ISO 8601), and the SQL expression for "now" in it.
TIME = "%Y-%m-%dT%H:%M:%SZ"
NOW = f"strftime('{TIME}', 'now')"

# The user's data: every write to these tables is kept in `history` (app/history.py) by triggers
# migrate() makes from their columns. Not `annotation` (its triggers rebuild it from project_media).
HISTORY_TABLES = ("project", "media", "project_media", "run", "run_result", "view_media")


@dataclass(frozen=True)
class Action:
    """What is being done now, whose writes are undone together: one request, job or agent turn."""

    source: str  # request, job, agent, app
    label: str  # "DELETE /api/projects/7", the job, the person's message
    id: str = field(default_factory=lambda: secrets.token_hex(8))


# The current action of this thread or task; outside any (startup, scripts), one per process.
_action: ContextVar[Action] = ContextVar("action")
_APP_ACTION = Action("app", "app")


@contextmanager
def action(source: str, label: str) -> Iterator[None]:
    """Make the writes inside this one action in the history."""
    token = _action.set(Action(source, label[:200]))
    try:
        yield
    finally:
        _action.reset(token)


def current_action() -> Action:
    return _action.get(_APP_ACTION)


def _current_action(name: str) -> str:
    """current_action('id' | 'source' | 'label'), for the history triggers."""
    return getattr(current_action(), name)


def connect() -> sqlite3.Connection:
    # check_same_thread=False: FastAPI may open the connection in one worker thread
    # and use it in another; each request still has its own connection.
    conn = sqlite3.connect(config.DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.create_function("current_action", 1, _current_action)
    conn.execute("PRAGMA foreign_keys = ON")
    # With WAL this stays crash-safe (a power cut can lose the last commits, never corrupt),
    # and commits don't wait for a disk sync, which matters for jobs that commit per item.
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def migrate() -> None:
    """Bring the schema up to date: run each NNN_*.sql newer than PRAGMA user_version.

    Foreign keys are off while migrating, as SQLite's procedure for schema changes has it: a table
    rebuilt (new table, copy, drop the old, rename) would otherwise cascade the drop into the rows
    referring to it. Each migration is checked before its commit instead.

    The history triggers are dropped first and made again after, from the columns then: migrations
    aren't recorded, and a column added or dropped is recorded with no trigger to rewrite."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(connect()) as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # stored in the file; set once
        conn.execute("PRAGMA foreign_keys = OFF")  # outside a transaction, or it does nothing
        with conn:
            for (name,) in conn.execute(r"SELECT name FROM sqlite_schema WHERE type = 'trigger' "
                                        r"AND name LIKE 'history\_%' ESCAPE '\'").fetchall():
                conn.execute(f"DROP TRIGGER {name}")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            number = int(path.name.split("_")[0])
            if number > version:
                # One transaction per migration, including the version bump.
                conn.executescript(f"BEGIN; {path.read_text()}")
                if broken := conn.execute("PRAGMA foreign_key_check").fetchall():
                    conn.rollback()
                    raise RuntimeError(f"Migration {path.name} leaves {len(broken)} broken references")
                conn.execute(f"PRAGMA user_version = {number}")
                conn.commit()
        with conn:
            for table in HISTORY_TABLES:
                for sql in _history_triggers(conn, table):
                    conn.execute(sql)


def _history_triggers(conn: sqlite3.Connection, table: str) -> list[str]:
    """The triggers that keep a table's writes in history: an insert's key, the row as it was
    before an update or a delete (the key of an update is the new one), each under the current
    action, whose row the first write makes. A row already in the action isn't kept again on an
    update: undoing goes back to how it was first (an import touches its project once per image)."""
    columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
    keys = [c["name"] for c in sorted((c for c in columns if c["pk"]), key=lambda c: c["pk"])]

    def row(which: str, names: list[str]) -> str:
        return "json_object(" + ", ".join(f"'{n}', {which}.{n}" for n in names) + ")"

    whole = [c["name"] for c in columns]
    kept = {"insert": (row("NEW", keys), "NULL"),
            "update": (row("NEW", keys), row("OLD", whole)),
            "delete": (row("OLD", keys), row("OLD", whole))}
    once = {"update": f"""WHEN NOT EXISTS (SELECT 1 FROM history WHERE action_id = current_action('id')
                                         AND tbl = '{table}' AND key = {row("NEW", keys)})"""}
    return [
        f"""CREATE TRIGGER history_{table}_{op} AFTER {op.upper()} ON {table} {once.get(op, "")} BEGIN
              INSERT OR IGNORE INTO action (id, source, label)
                VALUES (current_action('id'), current_action('source'), current_action('label'));
              INSERT INTO history (action_id, tbl, op, key, before)
                VALUES (current_action('id'), '{table}', '{op}', {key}, {before});
            END"""
        for op, (key, before) in kept.items()
    ]


def get() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: a connection for one request."""
    with closing(connect()) as conn:
        yield conn
