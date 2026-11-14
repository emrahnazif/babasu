"""History: what was done to the user's data in the last 30 days, so it can be undone.

SQLite triggers keep every write to the user's tables (db.HISTORY_TABLES) under the action that
made it: one request, job or agent turn (db.action). Files are kept too: a deleted image's file
stays in place while an entry can put its row back; a deleted view or workflow moves to the trash.
Not kept: settings.json (models, agents, nav) and a saved workflow written over.
"""

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from app import config, db, media

DAYS = 30  # how long anything done can be undone
OLD = f"SELECT id FROM action WHERE at < strftime('{db.TIME}', 'now', '-{DAYS} days')"


class Refused(Exception):
    """An action that can't be undone now, and why."""


def trash(conn: sqlite3.Connection, path: Path) -> Path:
    """Move a file or folder in the workspace to the trash, under the current action, which undoing
    moves it back; returns where it went. Caller commits (a move whose entry isn't committed is
    removed by purge)."""
    moved = config.TRASH_DIR / db.current_action().id / path.relative_to(config.DATA_DIR)
    moved.parent.mkdir(parents=True, exist_ok=True)
    path.rename(moved)
    _record_file(conn, "trash", path, moved)
    return moved


def _record_file(conn: sqlite3.Connection, op: str, path: Path, moved: Path) -> None:
    """A file's move in history, as the triggers record a row's write."""
    action = db.current_action()
    conn.execute("INSERT OR IGNORE INTO action (id, source, label) VALUES (?, ?, ?)",
                 (action.id, action.source, action.label))
    conn.execute("INSERT INTO history (action_id, tbl, op, key, before) VALUES (?, 'file', ?, "
                 "json_object('path', ?), json_object('trash', ?))",
                 (action.id, op, *(p.relative_to(config.DATA_DIR).as_posix() for p in (path, moved))))


_columns_of: dict[str, set[str]] = {}  # the schema doesn't change once the app runs (db.migrate)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if table not in _columns_of:
        _columns_of[table] = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    return _columns_of[table]


def _where(key: dict) -> tuple[str, dict]:
    """A WHERE clause for a row's key, and its parameters."""
    return " AND ".join(f"{k} = :k_{k}" for k in key), {f"k_{k}": v for k, v in key.items()}


# --- Reading ---

# How the history page and the agent name what an action changed.
_NAMES = {"project": ("project", "projects"), "project_media": ("image in a project", "images in projects"),
          "media": ("stored image", "stored images"), "run": ("run", "runs"), "run_result": ("run result", "run results"),
          "view_media": ("image of a report", "images of reports"), "file": ("file", "files")}
_DONE = {"insert": "added", "update": "changed", "delete": "deleted", "trash": "moved to the trash",
         "untrash": "back from the trash"}
_KIND_LABEL = {"do": "", "undo": "Undo: ", "redo": "Redo: "}

# The undo that undid action `a`, unless that undo was undone in turn.
_UNDONE_BY = """(SELECT u.id FROM action u WHERE u.undoes = a.id
                 AND NOT EXISTS (SELECT 1 FROM action r WHERE r.undoes = u.id))"""
# An action's entries (mine) joined to the later entries of other actions on the same rows and files (h).
_LATER = """history mine JOIN history h ON h.tbl = mine.tbl AND h.key = mine.key AND h.id > mine.id
            AND h.action_id != mine.action_id"""


def actions(conn: sqlite3.Connection, limit: int = 100, action_id: str | None = None) -> list[dict]:
    """The latest actions (or the one given), newest first: id, source (request, job, agent, app),
    kind (do, undo, redo), label, at, what it changed (in words), the undo that undid it or the
    action it undid, if any, and whether a later action changed the same things (`changed_since`:
    it can't be undone before that one is)."""
    rows = conn.execute(
        f"""WITH RECURSIVE chain(id, depth) AS (  -- how many undos deep: 0 done, odd an undo, even a redo
              SELECT id, 0 FROM action WHERE undoes IS NULL
              UNION ALL SELECT a.id, depth + 1 FROM action a JOIN chain ON a.undoes = chain.id)
            SELECT a.id, source, label, at, undoes, depth, {_UNDONE_BY} AS undone_by,
                   EXISTS (SELECT 1 FROM {_LATER} WHERE mine.action_id = a.id) AS changed_since
            FROM action a JOIN chain USING (id) WHERE ? IS NULL OR a.id = ?
            ORDER BY a.rowid DESC LIMIT ?""", (action_id, action_id, limit)).fetchall()
    found = []
    for r in rows:
        kind = "do" if r["depth"] == 0 else "undo" if r["depth"] % 2 else "redo"
        found.append({"id": r["id"], "source": r["source"], "kind": kind, "label": _KIND_LABEL[kind] + r["label"],
                      "at": r["at"], "changed": _changed(conn, r["id"]), "undoes": r["undoes"],
                      "undone_by": r["undone_by"], "changed_since": bool(r["changed_since"])})
    return found


def _changed(conn: sqlite3.Connection, action_id: str) -> str:
    """What an action changed, per row: added (its first entry an insert), deleted (its last a
    delete) or changed; "project “Cats” deleted, 1,240 images in projects deleted"."""
    groups = conn.execute(
        """WITH touched AS (SELECT tbl, key, min(id) AS first, max(id) AS last FROM history
                            WHERE action_id = ? GROUP BY tbl, key)
           SELECT t.tbl, CASE f.op WHEN 'insert' THEN 'insert' ELSE l.op END AS op, count(*) AS n,
                  max(f.key) AS key, max(f.before) AS before  -- the row's, when there's one
           FROM touched t JOIN history f ON f.id = t.first JOIN history l ON l.id = t.last
           GROUP BY 1, 2 ORDER BY min(t.first)""", (action_id,)).fetchall()
    words = []
    for g in groups:
        one, many = _NAMES.get(g["tbl"], (g["tbl"], g["tbl"]))
        name = _name(conn, g["tbl"], json.loads(g["key"]), json.loads(g["before"] or "{}")) if g["n"] == 1 else None
        what = f"{one} “{name}”" if name else one if g["n"] == 1 else f"{g['n']:,} {many}"
        words.append(f"{what} {_DONE[g['op']]}")
    return ", ".join(words)


def _name(conn: sqlite3.Connection, table: str, key: dict, before: dict) -> str | None:
    """What a row or file is called: its path, or its name as it is now (as it was, once deleted)."""
    if table == "file":
        return key["path"]
    if "name" not in _columns(conn, table):
        return None
    where, params = _where(key)
    row = conn.execute(f"SELECT name FROM {table} WHERE {where}", params).fetchone()
    return row["name"] if row else before.get("name")


# --- Undoing ---


def undo(conn: sqlite3.Connection, action_id: str) -> dict:
    """Undo an action: in one transaction, its entries in reverse order (the inserted rows deleted,
    the old values written back, the deleted rows inserted, the trashed files moved back), as a new
    action that names it, which undoing redoes. Returns the new action as actions() shows it.
    LookupError if there's no such action; Refused if a later action changed the same rows (undo
    that first), or a row or file can't go back."""
    action = conn.execute(f"SELECT label, {_UNDONE_BY} AS undone_by FROM action a WHERE id = ?", (action_id,)).fetchone()
    if action is None:
        raise LookupError(f"No action {action_id} in the last {DAYS} days")
    if action["undone_by"]:
        raise Refused("It's undone already; undoing that undo does it again.")
    later = conn.execute(f"SELECT DISTINCT l.id FROM {_LATER} JOIN action l ON l.id = h.action_id "
                         "WHERE mine.action_id = ?", (action_id,)).fetchall()
    if later:
        named = [actions(conn, action_id=r["id"])[0] for r in later]
        raise Refused("Changed again since by " + "; ".join(f"“{a['label']}” ({a['at']})" for a in named)
                      + ": undo that first.")

    moved: list[tuple[Path, Path]] = []  # files moved (where they are, where they were), put back if this fails
    with db.action(db.current_action().source, action["label"]):  # the kind (undo, redo) comes from `undoes`
        undoing = db.current_action()
        try:
            conn.execute("BEGIN")
            conn.execute("PRAGMA defer_foreign_keys = ON")  # rows go back in any order; checked at the commit
            conn.execute("INSERT INTO action (id, source, label, undoes) VALUES (?, ?, ?, ?)",
                         (undoing.id, undoing.source, undoing.label, action_id))
            for entry in conn.execute("SELECT * FROM history WHERE action_id = ? ORDER BY id DESC", (action_id,)).fetchall():
                if entry["tbl"] == "file":
                    _undo_file(conn, entry, moved)
                else:
                    _undo_row(conn, entry)
            conn.commit()
        except Exception as e:
            conn.rollback()
            for now, was in reversed(moved):
                now.rename(was)
            if isinstance(e, sqlite3.IntegrityError):
                raise Refused(f"A row can't go back ({e}): something like it was made since.") from None
            raise
    return actions(conn, action_id=undoing.id)[0]


def _undo_row(conn: sqlite3.Connection, entry: sqlite3.Row) -> None:
    table = entry["tbl"]
    where, params = _where(json.loads(entry["key"]))
    if entry["op"] == "insert":
        conn.execute(f"DELETE FROM {table} WHERE {where}", params)
        return
    columns = _columns(conn, table)
    before = {k: v for k, v in json.loads(entry["before"]).items() if k in columns}  # a column dropped since is left out
    if entry["op"] == "update":
        sets = ", ".join(f"{k} = :{k}" for k in before)
        conn.execute(f"UPDATE {table} SET {sets} WHERE {where}", params | before)
    else:
        conn.execute(f"INSERT INTO {table} ({', '.join(before)}) VALUES ({', '.join(':' + k for k in before)})", before)


def _undo_file(conn: sqlite3.Connection, entry: sqlite3.Row, moved: list[tuple[Path, Path]]) -> None:
    path = config.DATA_DIR / json.loads(entry["key"])["path"]
    if entry["op"] == "untrash":  # it was brought back: back to the trash
        moved.append((trash(conn, path), path))
        return
    in_trash = config.DATA_DIR / json.loads(entry["before"])["trash"]
    if path.exists():
        raise Refused(f"{path.relative_to(config.DATA_DIR)} was made again since; delete it first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    in_trash.rename(path)
    moved.append((path, in_trash))
    _record_file(conn, "untrash", path, in_trash)


# --- Forgetting ---


def purge() -> None:
    """Forget actions older than DAYS with their entries (at start), then remove what only they
    could bring back: image files no media row or remaining entry names, and their trash."""
    with closing(db.connect()) as conn:
        with conn:
            images = conn.execute(f"""SELECT DISTINCT before ->> 'sha256', before ->> 'ext' FROM history
                                      WHERE tbl = 'media' AND op = 'delete' AND action_id IN ({OLD})""").fetchall()
            conn.execute(f"DELETE FROM action WHERE id IN ({OLD})")
        named = {r[0] for r in conn.execute("SELECT sha256 FROM media UNION SELECT before ->> 'sha256' FROM history "
                                            "WHERE tbl = 'media' AND before IS NOT NULL")}
        kept = {r[0] for r in conn.execute("SELECT id FROM action")}
    for sha256, ext in images:
        if sha256 not in named:
            media.path(sha256, ext).unlink(missing_ok=True)
            media.path(sha256, "thumb.webp").unlink(missing_ok=True)
    if config.TRASH_DIR.is_dir():
        for folder in config.TRASH_DIR.iterdir():
            if folder.name not in kept:
                shutil.rmtree(folder)
